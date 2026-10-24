"""Model clients (SPEC §6.3). Responses are normalised to plain dicts so traces are exact."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
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


class InfraError(RuntimeError):
    """The episode could not run for reasons outside the agent's control; it is re-queued."""


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


FINISH_TO_STOP = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens",
                  "content_filter": "refusal", "function_call": "tool_use"}


class OpenAICompatClient:
    """Self-hosted model behind an OpenAI-compatible gateway (e.g. LiteLLM over llama.cpp).

    Base URL and key come from environment variables named in the registry, never from the
    registry itself."""

    def __init__(self, model_id: str, base_url: str, api_key: str, temperature: float | None = None,
                 top_p: float | None = None, max_retries: int = 4, timeout_s: float = 900):
        import openai

        self._openai = openai
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries, timeout=timeout_s)

    def create(self, *, system: str, messages: list[dict], tools: list[dict], max_tokens: int) -> ModelResponse:
        openai = self._openai
        kwargs: dict = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        t0 = time.perf_counter()
        try:
            raw = self.client.chat.completions.with_raw_response.create(
                model=self.model_id, messages=to_openai_messages(system, messages),
                tools=to_openai_tools(tools), max_tokens=max_tokens, **kwargs)
        except openai.BadRequestError as exc:
            if "context" in str(exc).lower() or "too long" in str(exc).lower():
                raise ContextLimitError(str(exc)) from exc
            raise
        except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError,
                openai.APITimeoutError) as exc:
            raise InfraError(f"{type(exc).__name__}: {exc}") from exc
        resp = raw.parse()
        latency = time.perf_counter() - t0

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
        return ModelResponse(content=content, stop_reason=stop, usage=usage, model=resp.model or self.model_id,
                             request_id=raw.headers.get("x-request-id") or resp.id, latency_s=latency)


def make_client(model_cfg) -> "ModelClient":
    """Build the agent's client from the registry's ModelConfig."""
    import os

    if model_cfg.provider == "anthropic":
        return AnthropicClient(model_cfg.id, model_cfg.effort)
    base_url = os.environ.get(model_cfg.base_url_env)
    api_key = os.environ.get(model_cfg.api_key_env)
    if not base_url or not api_key:
        raise RuntimeError(f"set {model_cfg.base_url_env} and {model_cfg.api_key_env} for the local model gateway")
    return OpenAICompatClient(model_cfg.id, base_url, api_key, model_cfg.temperature, model_cfg.top_p)
