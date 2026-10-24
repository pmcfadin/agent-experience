"""Agent loop (SPEC §6.1): manual, append-only tool-use loop with identical budgets for every arm."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from elh.registry import Budgets
from elh.runner.models import ContextLimitError, ModelClient
from elh.runner.prompts import SYSTEM_PROMPT, TOOLS, first_user_message
from elh.sandbox.base import ExecResult, Sandbox
from elh.tasks.base import TaskInstance

TRACE_OUTPUT_CAP = 200_000

TRUNCATED_CONTINUE = "Your previous response hit the output limit and was cut off. Continue."


@dataclass
class EpisodeResult:
    outcome: str  # submitted | implicit_submit | budget_exhausted | refused
    budget_reason: str | None = None
    turns: int = 0
    tool_calls: int = 0
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0,
                                                 "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})
    calls: list[dict] = field(default_factory=list)  # per-API-call records (for llm_calls + ledger)
    trace: list[dict] = field(default_factory=list)
    submit_summary: str | None = None
    wall_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return sum(self.usage.values())


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n\n[... {len(text) - limit} characters omitted ...]\n\n{text[-half:]}"


def format_exec(res: ExecResult, timeout_s: float, limit: int) -> str:
    out = res.stdout
    if res.stderr:
        out += ("\n" if out and not out.endswith("\n") else "") + res.stderr
    out = _clip(out, limit)
    if res.timed_out:
        return out + f"\n[command timed out after {timeout_s:.0f}s and was killed]"
    return out + (f"\n[exit code {res.exit_code}]" if res.exit_code else "")


def run_episode(task: TaskInstance, sandbox: Sandbox, client: ModelClient, budgets: Budgets,
                max_tokens: int, memory_block: str = "") -> EpisodeResult:
    """Run one episode in an already-started sandbox. The caller owns sandbox lifecycle."""
    t0 = time.perf_counter()
    first = first_user_message(task, sandbox.workdir, memory_block)
    messages: list[dict] = [{"role": "user", "content": first}]
    result = EpisodeResult(outcome="budget_exhausted")
    result.trace.append({"event": "episode_start", "task_id": task.task_id, "task_hash": task.content_hash,
                         "model": client.model_id, "system": SYSTEM_PROMPT, "tools": TOOLS,
                         "first_message": first})

    for turn in range(budgets.max_turns):
        if time.perf_counter() - t0 > budgets.wall_clock_seconds:
            result.budget_reason = "wall_clock"
            break
        if result.total_tokens > budgets.max_total_tokens:
            result.budget_reason = "tokens"
            break

        try:
            resp = client.create(system=SYSTEM_PROMPT, messages=messages, tools=TOOLS, max_tokens=max_tokens)
        except ContextLimitError:
            result.budget_reason = "context"
            break
        result.turns = turn + 1
        for k in result.usage:
            result.usage[k] += resp.usage.get(k) or 0
        result.calls.append({"turn": turn, "request_id": resp.request_id, "model": resp.model,
                             "stop_reason": resp.stop_reason, "usage": resp.usage, "latency_s": resp.latency_s})
        result.trace.append({"event": "response", "turn": turn, "request_id": resp.request_id,
                             "stop_reason": resp.stop_reason, "usage": resp.usage,
                             "latency_s": round(resp.latency_s, 3), "content": resp.content})
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "refusal":
            result.outcome = "refused"
            break

        tool_uses = [b for b in resp.content if b.get("type") == "tool_use"]
        if not tool_uses:
            if resp.stop_reason == "max_tokens":
                messages.append({"role": "user", "content": TRUNCATED_CONTINUE})
                continue
            result.outcome = "implicit_submit"
            break

        tool_results = []
        submitted = False
        for tu in tool_uses:
            result.tool_calls += 1
            name, inp = tu.get("name"), tu.get("input") or {}
            if name == "submit":
                submitted = True
                result.submit_summary = str(inp.get("summary", ""))
                content, is_error = "Submitted for grading.", False
            elif name == "bash" and inp.get("restart"):
                content, is_error = "Shell restarted. (Each command already runs in a fresh shell.)", False
            elif name == "bash" and isinstance(inp.get("command"), str):
                res = sandbox.exec(inp["command"], budgets.bash_timeout_seconds)
                content = format_exec(res, budgets.bash_timeout_seconds, budgets.tool_output_chars)
                is_error = False
                result.trace.append({"event": "tool_exec", "turn": turn, "tool_use_id": tu["id"],
                                     "command": inp["command"], "exit_code": res.exit_code,
                                     "timed_out": res.timed_out, "wall_s": round(res.wall_s, 3),
                                     "stdout": res.stdout[:TRACE_OUTPUT_CAP], "stderr": res.stderr[:TRACE_OUTPUT_CAP]})
            else:
                content, is_error = f"Invalid tool call: {name} {inp!r}", True
            block = {"type": "tool_result", "tool_use_id": tu["id"], "content": content}
            if is_error:
                block["is_error"] = True
            tool_results.append(block)

        messages.append({"role": "user", "content": tool_results})
        result.trace.append({"event": "tool_results", "turn": turn, "content": tool_results})
        if submitted:
            result.outcome = "submitted"
            break
    else:
        result.budget_reason = "turns"

    result.wall_s = time.perf_counter() - t0
    result.trace.append({"event": "episode_end", "outcome": result.outcome, "budget_reason": result.budget_reason,
                         "turns": result.turns, "tool_calls": result.tool_calls, "usage": result.usage,
                         "wall_s": round(result.wall_s, 3)})
    return result
