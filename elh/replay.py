"""Replay a saved degenerate request with log-probabilities (`uv run elh replay-degenerate <file.json>`).

The input is either a capture written by OpenAICompatClient (elh.runner.models, `payload` key) or a bare
OpenAI chat-completions body. NaN or +/-inf logits show up as NaN / -inf log-probabilities on the first
bad token, so each run reports the first non-finite value it sees. Needs WINTERMUTE_MODEL_API_BASE and
WINTERMUTE_MODEL_API_KEY (or the ELH_MODEL_* names); the key is never printed or written."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from elh.runner.models import FINISH_TO_STOP, alias_lock, degenerate_reason, utc_iso


def build_body(doc: dict, *, cache_prompt: bool, top_logprobs: int, max_tokens: int) -> dict:
    src = doc.get("payload", doc)
    body = {k: v for k, v in src.items() if k != "extra_body"}
    body.update(src.get("extra_body") or {})
    body.update(max_tokens=max_tokens, logprobs=True, top_logprobs=top_logprobs, cache_prompt=cache_prompt)
    body.pop("stream", None)
    return body


def _bad(x) -> str | None:
    if x is None:
        return "null"
    if isinstance(x, (int, float)) and not math.isfinite(x):
        return "NaN" if math.isnan(x) else ("-inf" if x < 0 else "+inf")
    return None


def numerics_flags(tokens: list[dict]) -> list[str]:
    """One flag per token holding a non-finite log-probability (own or in its top list), in token order."""
    flags = []
    for i, t in enumerate(tokens):
        kinds = []
        b = _bad(t.get("logprob"))
        if b:
            kinds.append(f"{b} logprob")
        for alt in t.get("top_logprobs") or []:
            b = _bad(alt.get("logprob"))
            if b:
                kinds.append(f"{b} top-logprob for {alt.get('token')!r}")
        if kinds:
            flags.append(f"NUMERICS: {'; '.join(kinds)} at token {i}")
    return flags


class _Reply:
    def __init__(self, status, headers, text):
        self.status_code, self.headers, self.text = status, headers, text


def _post(url: str, key: str, body: dict, timeout_s: float) -> _Reply:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"authorization": f"Bearer {key}", "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return _Reply(r.status, {k.lower(): v for k, v in r.headers.items()}, r.read().decode())
    except urllib.error.HTTPError as e:
        return _Reply(e.code, {k.lower(): v for k, v in e.headers.items()}, e.read().decode(errors="replace"))


def one_run(url: str, key: str, body: dict, timeout_s: float) -> dict:
    with alias_lock(body["model"]):
        sent = utc_iso()
        t0 = time.perf_counter()
        r = _post(url, key, body, timeout_s)
        latency = time.perf_counter() - t0
    received = utc_iso()
    run = {"sent_at_utc": sent, "received_at_utc": received, "latency_s": round(latency, 3), "http_status": r.status_code,
           "request_id": r.headers.get("x-litellm-call-id") or r.headers.get("x-request-id")}
    if r.status_code != 200:
        return {**run, "error": r.text[:500]}
    data = json.loads(r.text)  # accepts NaN / Infinity tokens
    choice = data["choices"][0]
    content = (choice.get("message") or {}).get("content") or ""
    stop = FINISH_TO_STOP.get(choice.get("finish_reason") or "", choice.get("finish_reason"))
    usage = data.get("usage") or {}
    lp = (choice.get("logprobs") or {}).get("content")
    run.update(request_id=run["request_id"] or data.get("id"), finish_reason=choice.get("finish_reason"), usage=usage,
               cached_tokens=((usage.get("prompt_tokens_details") or {}).get("cached_tokens")),
               degenerate=degenerate_reason(content, stop) if content else None, reply_head=content[:200],
               logprobs_returned=bool(lp), logprobs=lp[:64] if lp else None,
               numerics=numerics_flags(lp) if lp else [])
    return run


def _fmt_lp(x) -> str:
    return "null" if x is None else f"{x:.4f}"


def print_run(i: int, n: int, run: dict, show_tokens: int = 5) -> None:
    print(f"--- run {i}/{n}")
    print(f"sent {run['sent_at_utc']}  received {run['received_at_utc']}  latency {run['latency_s']}s")
    if "error" in run:
        print(f"HTTP {run['http_status']} error: {run['error']}")
        return
    print(f"request id {run['request_id']}  finish {run['finish_reason']}  usage {json.dumps(run['usage'])}  "
          f"cached_tokens {run['cached_tokens']}")
    print(f"degenerate: {run['degenerate'] or 'no'}  reply head: {run['reply_head'][:80]!r}")
    if not run["logprobs_returned"]:
        print("logprobs not returned")
        return
    for j, t in enumerate(run["logprobs"][:show_tokens]):
        alts = ", ".join(f"{a.get('token')!r}:{_fmt_lp(a.get('logprob'))}" for a in t.get("top_logprobs") or [])
        print(f"  token {j} {t.get('token')!r} logprob {_fmt_lp(t.get('logprob'))}  top: {alts}")
    for f in run["numerics"][:3]:
        print(f)
    if not run["numerics"]:
        print("numerics: all log-probabilities finite")


def replay(path: Path, n: int = 3, cache_prompt: bool = False, top_logprobs: int = 5, max_tokens: int = 512,
           timeout_s: float = 900) -> tuple[Path, list[dict]]:
    url = os.environ.get("WINTERMUTE_MODEL_API_BASE") or os.environ.get("ELH_MODEL_BASE_URL")
    key = os.environ.get("WINTERMUTE_MODEL_API_KEY") or os.environ.get("ELH_MODEL_API_KEY")
    if not url or not key:
        raise SystemExit("set WINTERMUTE_MODEL_API_BASE and WINTERMUTE_MODEL_API_KEY")
    doc = json.loads(Path(path).read_text())
    body = build_body(doc, cache_prompt=cache_prompt, top_logprobs=top_logprobs, max_tokens=max_tokens)
    runs = []
    for i in range(1, n + 1):
        try:
            run = one_run(url, key, body, timeout_s)
        except Exception as exc:
            run = {"sent_at_utc": utc_iso(), "received_at_utc": utc_iso(), "latency_s": 0, "http_status": None,
                   "error": f"{type(exc).__name__}: {exc}"}
        runs.append(run)
        print_run(i, n, run)
    out = Path(path).with_name(Path(path).name.removesuffix(".json") + f".replay-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    request = {k: v for k, v in body.items() if k not in ("messages", "tools")}
    out.write_text(json.dumps({"source": str(path), "request_params": request, "n_messages": len(body["messages"]),
                               "runs": runs}, indent=2))
    print(f"wrote {out}", file=sys.stderr)
    return out, runs
