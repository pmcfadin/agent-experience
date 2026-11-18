"""Model clients (SPEC §6.3). Responses are normalised to plain dicts so traces are exact."""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from datetime import datetime, timezone
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import anthropic


@dataclass
class ModelResponse:
    content: list[dict]
    stop_reason: str | None
    usage: dict
    model: str
    request_id: str | None = None
    latency_s: float = 0.0
    served: dict = field(default_factory=dict)  # backend identity as reported by the gateway
    cache_retry: dict | None = None  # set when a degenerate reply was retried without the prompt cache
    attempt: dict | None = field(default=None, repr=False)  # request payload + timing, kept for degenerate-reply capture


class InfraError(RuntimeError):
    """The episode could not run for reasons outside the agent's control; it is re-queued."""


class BackendUnreachableError(InfraError):
    """The backend (or its gateway) cannot be reached: connection failure, timeout, HTTP 5xx after the client's
    retries, or the /props fetch failing. `run_batch` aborts at once instead of skipping episodes."""


class DegenerateOutputError(InfraError):
    """The backend returned a degenerate reply (a server fault, not agent behaviour); the episode is skipped."""

    def __init__(self, msg: str, cache_retry: dict | None = None):
        super().__init__(msg)
        self.cache_retry = cache_retry


# Degenerate-reply detector. A reply is degenerate when a text block of at least
# DEGENERATE_MIN_CHARS characters is made up >= DEGENERATE_SHARE of one single character (the
# self-hosted server's fault state returns 8192 '?'), or, when generation hit the token limit
# (stop_reason max_tokens), when the same covers a short repeated pattern of at most
# DEGENERATE_MAX_PERIOD characters. Real code, prose and tables never put 90% of 200+ characters
# on one character (the commonest one, a space, is ~15-30% even in padded tables), so the
# thresholds are deliberately far from legitimate output.
DEGENERATE_MIN_CHARS = 200
DEGENERATE_SHARE = 0.9
DEGENERATE_MAX_PERIOD = 8


def degenerate_reason(text: str, stop_reason: str | None = None) -> str | None:
    """Why `text` (one text block of a model reply) is degenerate, or None."""
    if len(text) < DEGENERATE_MIN_CHARS:
        return None
    from collections import Counter

    ch, n = Counter(text).most_common(1)[0]
    if n / len(text) >= DEGENERATE_SHARE:
        return f"{n} of {len(text)} characters are {ch!r}"
    if stop_reason == "max_tokens":
        for period in range(2, DEGENERATE_MAX_PERIOD + 1):
            unit = text[:period]
            covered = sum(1 for i, c in enumerate(text) if c == unit[i % period])
            if covered / len(text) >= DEGENERATE_SHARE:
                return f"{len(text)} characters repeat {unit!r} until the token limit"
    return None


class ModelClient(Protocol):
    model_id: str

    def create(self, *, system: str, messages: list[dict], tools: list[dict], max_tokens: int) -> ModelResponse: ...


class AnthropicClient:
    """Pinned-model client. Refusal fallbacks are deliberately NOT enabled (SPEC §1): a fallback
    would switch models mid-episode and break the frozen-model control."""

    def __init__(self, model_id: str, effort: str, max_retries: int = 4, timeout_s: float = 600):
        self.model_id = model_id
        self.effort = effort
        self.client = anthropic.Anthropic(max_retries=max_retries, timeout=timeout_s)

    def create(self, *, system: str, messages: list[dict], tools: list[dict], max_tokens: int) -> ModelResponse:
        t0 = time.perf_counter()
        try:
            resp = self.client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens,
                system=system,
                tools=tools,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                cache_control={"type": "ephemeral"},
            )
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as exc:
            raise InfraError(f"{type(exc).__name__}: {exc}") from exc
        latency = time.perf_counter() - t0
        if resp.model != self.model_id:
            raise InfraError(f"served model {resp.model!r} != pinned {self.model_id!r}")
        return ModelResponse(
            content=[b.model_dump(mode="json", exclude_none=True) for b in resp.content],
            stop_reason=resp.stop_reason,
            usage=resp.usage.model_dump(mode="json", exclude_none=True),
            model=resp.model,
            request_id=resp._request_id,
            latency_s=latency,
        )


@dataclass
class ScriptedClient:
    """Replays a fixed sequence of turns; each turn is a list of (tool_name, input) calls.
    Used by tests and dry runs so the whole pipeline runs without API spend."""

    turns: list[list[tuple[str, dict]]]
    model_id: str = "scripted"
    fake_usage: dict = field(default_factory=lambda: {"input_tokens": 1000, "output_tokens": 200})
    _i: int = 0

    def create(self, *, system: str, messages: list[dict], tools: list[dict], max_tokens: int) -> ModelResponse:
        if self._i >= len(self.turns):
            return ModelResponse([{"type": "text", "text": "Done."}], "end_turn", dict(self.fake_usage), self.model_id)
        calls = self.turns[self._i]
        content = [{"type": "tool_use", "id": f"toolu_scripted_{self._i}_{j}", "name": name, "input": inp}
                   for j, (name, inp) in enumerate(calls)]
        self._i += 1
        return ModelResponse(content, "tool_use", dict(self.fake_usage), self.model_id, f"req_scripted_{self._i}")


class ContextLimitError(RuntimeError):
    """The transcript no longer fits the model's context window; the episode ends as budget_exhausted."""


# Function-tool schemas used when a provider has no Anthropic-defined bash tool.
BASH_FUNCTION = {
    "name": "bash",
    "description": "Run a bash command in the task workspace and return its stdout, stderr and exit code.",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The bash command to run."}},
        "required": ["command"],
    },
}


def to_openai_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        if t.get("type", "").startswith("bash_"):
            fn = BASH_FUNCTION
        else:
            fn = {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}
        out.append({"type": "function", "function": fn})
    return out


def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    """Translate the loop's Anthropic-shaped, append-only history into chat-completions messages."""
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "assistant":
            text = "".join(b["text"] for b in content if b.get("type") == "text")
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})}}
                     for b in content if b.get("type") == "tool_use"]
            msg: dict = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
            continue
        texts = []
        for b in content:
            if b.get("type") == "tool_result":
                body = b["content"] if isinstance(b["content"], str) else json.dumps(b["content"])
                out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": body})
            elif b.get("type") == "text":
                texts.append(b["text"])
        if texts:
            out.append({"role": "user", "content": "\n\n".join(texts)})
    return out


# Gateway response headers that identify what actually served a request (LiteLLM over llama.cpp).
SERVED_HEADERS = {"x-litellm-model-group": "alias", "x-litellm-model-name": "deployment",
                  "x-litellm-model-id": "deployment_id", "x-litellm-model-api-base": "api_base",
                  "llm_provider-server": "server", "x-litellm-version": "gateway_version"}

# Fields that must not change within an episode; a change means the backend was swapped mid-run.
IDENTITY_FIELDS = ("alias", "deployment", "deployment_id", "system_fingerprint")


@contextmanager
def alias_lock(alias: str):
    """Hold an exclusive per-alias lock for one request: each study alias serves one request at a
    time, so concurrent workers queue here instead of queueing (and timing out) at the gateway."""
    lock_dir = Path(os.environ.get("ELH_LOCK_DIR", Path.home() / ".cache" / "elh" / "locks"))
    lock_dir.mkdir(parents=True, exist_ok=True)
    with open(lock_dir / f"{alias.replace('/', '_')}.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def degenerate_dir() -> Path:
    """Where degenerate-reply captures go: $ELH_DEGENERATE_DIR, else runs/degenerate under the cwd."""
    return Path(os.environ.get("ELH_DEGENERATE_DIR") or "runs/degenerate")


def fetch_backend_props(url: str, timeout_s: float = 10) -> bytes:
    """GET a llama.cpp /props document (model path, build, n_ctx, default sampling) verbatim.
    No credentials: the relay serving it is read-only and must not see the gateway key."""
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        return resp.read()


FINISH_TO_STOP = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens",
                  "content_filter": "refusal", "function_call": "tool_use"}


class OpenAICompatClient:
    """Self-hosted model behind an OpenAI-compatible gateway (e.g. LiteLLM over llama.cpp).

    Base URL and key come from environment variables named in the registry, never from the
    registry itself."""

    def __init__(self, model_id: str, base_url: str, api_key: str, temperature: float | None = None,
                 top_p: float | None = None, max_retries: int = 4, timeout_s: float = 900,
                 props_url: str | None = None):
        import openai

        self._openai = openai
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.props_url = props_url
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries, timeout=timeout_s)

    def create(self, *, system: str, messages: list[dict], tools: list[dict], max_tokens: int) -> ModelResponse:
        """One request. A degenerate reply (a corrupted reused prompt-cache prefix on the server) is retried
        ONCE with cache_prompt=false; a clean retry is the response (recorded in `cache_retry`), a second
        degenerate reply raises DegenerateOutputError."""
        resp, why = self._request(system, messages, tools, max_tokens, None)
        if not why:
            return resp
        retry, why2 = self._request(system, messages, tools, max_tokens, {"cache_prompt": False})
        info = {"first_request_id": resp.request_id, "first_output_tokens": resp.usage["output_tokens"],
                "first_finish": resp.stop_reason, "retry_request_id": retry.request_id, "retry_clean": not why2,
                "retry_cached_tokens": retry.usage["cache_read_input_tokens"]}
        path = self._capture(resp, retry, why, why2)
        if path:
            info["capture"] = str(path)
        print(f"DEGENERATE reply (request {resp.request_id}): retried without prompt cache -> "
              f"{'degenerate' if why2 else 'clean'} (request {retry.request_id})"
              + (f"; saved {path}" if path else ""), file=sys.stderr, flush=True)
        if why2:
            raise DegenerateOutputError(
                f"degenerate reply from {self.model_id} (stop {retry.stop_reason}, {retry.usage['output_tokens']} "
                f"output tokens, also without prompt cache; first request {resp.request_id}"
                + (f"; saved {path}" if path else "") + f"): {why2}", cache_retry=info)
        retry.cache_retry = info
        return retry

    def _capture(self, first, retry, why1, why2) -> Path | None:
        """Save the failing request (exact payload, both attempts' ids, UTC timestamps, finish reasons, usage,
        reply head) so the server admin can match it against the backend log. Never fatal; holds no API key."""
        try:
            attempts = []
            for label, r, w in (("first", first, why1), ("retry_no_cache", retry, why2)):
                a = dict(r.attempt or {})
                a.pop("payload", None)
                a.update(attempt=label, request_id=r.request_id, degenerate=bool(w), degenerate_reason=w or None,
                         latency_s=round(r.latency_s, 3), stop_reason=r.stop_reason,
                         usage=a.pop("usage_raw", None) or r.usage)
                attempts.append(a)
            doc = {"model": self.model_id, "served": first.served, "payload": (first.attempt or {}).get("payload"),
                   "retry_extra_body": {"cache_prompt": False}, "attempts": attempts,
                   "backend": self._backend_build()}
            d = degenerate_dir()
            d.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")[:-4] + "Z"
            path = d / f"{stamp}-{str(first.request_id).replace('/', '_')}.json"
            path.write_text(json.dumps(doc, indent=2, default=str))
            return path
        except Exception as exc:  # capture must never break an episode
            print(f"degenerate capture failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return None

    def _backend_build(self) -> dict | None:
        if not self.props_url:
            return None
        try:
            props = json.loads(fetch_backend_props(self.props_url, timeout_s=5))
            return {k: props.get(k) for k in ("build_info", "model_path", "n_ctx") if k in props} or None
        except Exception:
            return None

    def _request(self, system, messages, tools, max_tokens, extra_body) -> tuple[ModelResponse, str | None]:
        openai = self._openai
        kwargs: dict = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        payload = {"model": self.model_id, "messages": to_openai_messages(system, messages),
                   "tools": to_openai_tools(tools), "max_tokens": max_tokens, **kwargs,
                   **({"extra_body": extra_body} if extra_body else {})}
        try:
            with alias_lock(self.model_id):
                sent_at = utc_iso()
                t0 = time.perf_counter()
                raw = self.client.chat.completions.with_raw_response.create(**payload)
        except openai.BadRequestError as exc:
            if "context" in str(exc).lower() or "too long" in str(exc).lower():
                raise ContextLimitError(str(exc)) from exc
            raise
        except (openai.InternalServerError, openai.APIConnectionError, openai.APITimeoutError) as exc:
            raise BackendUnreachableError(f"{type(exc).__name__}: {exc}") from exc
        except openai.RateLimitError as exc:
            raise InfraError(f"{type(exc).__name__}: {exc}") from exc
        resp = raw.parse()
        latency = time.perf_counter() - t0
        received_at = utc_iso()
        served = {v: raw.headers[k] for k, v in SERVED_HEADERS.items() if k in raw.headers}
        served["system_fingerprint"] = getattr(resp, "system_fingerprint", None)
        if served.get("alias", self.model_id) != self.model_id:
            raise InfraError(f"gateway served alias {served['alias']!r} != pinned {self.model_id!r}")
        timings = getattr(resp, "timings", None) or (resp.model_extra or {}).get("timings")

        choice = resp.choices[0]
        msg = choice.message
        content: list[dict] = []
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            # Kept in the trace only; the translator never sends it back to the model.
            content.append({"type": "reasoning", "text": reasoning})
        if msg.content:
            content.append({"type": "text", "text": msg.content})
        for call in msg.tool_calls or []:
            try:
                args = json.loads(call.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
            except (json.JSONDecodeError, ValueError):
                args = {"_unparsed_arguments": call.function.arguments}
            content.append({"type": "tool_use", "id": call.id, "name": call.function.name, "input": args})

        stop = FINISH_TO_STOP.get(choice.finish_reason or "", choice.finish_reason)
        if stop == "end_turn" and msg.tool_calls:
            stop = "tool_use"
        why = degenerate_reason(msg.content, stop) if msg.content else None
        u = resp.usage
        cached = 0
        if u is not None and getattr(u, "prompt_tokens_details", None) is not None:
            cached = u.prompt_tokens_details.cached_tokens or 0
        usage = {
            "input_tokens": ((u.prompt_tokens if u else 0) or 0) - cached,
            "output_tokens": (u.completion_tokens if u else 0) or 0,
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0,
        }
        if timings:
            served["timings"] = timings
        request_id = raw.headers.get("x-litellm-call-id") or raw.headers.get("x-request-id") or resp.id
        # kept on every response (references, no copy); written to disk only for degenerate replies
        attempt = {"payload": payload, "sent_at_utc": sent_at, "received_at_utc": received_at,
                   "finish_reason": choice.finish_reason, "usage_raw": u.model_dump() if u is not None else None,
                   "reply_head": (msg.content or "")[:200], "gateway_id": resp.id}
        return ModelResponse(content=content, stop_reason=stop, usage=usage, model=resp.model or self.model_id,
                             request_id=request_id, latency_s=latency, served=served, attempt=attempt), why


def make_client(model_cfg) -> "ModelClient":
    """Build the agent's client from the registry's ModelConfig."""
    import os

    if model_cfg.provider == "anthropic":
        return AnthropicClient(model_cfg.id, model_cfg.effort)
    base_url = os.environ.get(model_cfg.base_url_env)
    api_key = os.environ.get(model_cfg.api_key_env)
    if not base_url or not api_key:
        raise RuntimeError(f"set {model_cfg.base_url_env} and {model_cfg.api_key_env} for the local model gateway")
    return OpenAICompatClient(model_cfg.id, base_url, api_key, model_cfg.temperature, model_cfg.top_p,
                              props_url=getattr(model_cfg, "props_url", None)
                              or (os.environ.get(model_cfg.props_url_env, "") if getattr(model_cfg, "props_url_env", None) else None) or None)
