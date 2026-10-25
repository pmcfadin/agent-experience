"""HindsightMemory: arm B's memory, backed by the dedicated Hindsight server (SPEC §10).

One bank per (run, history, arm). Recall uses the ticket text as the query. Retain stores one
post-episode record built only from what the agent may learn from: the ticket, its own summary
and the judge's verified feedback (checks and timings), never the hidden oracle.

Hindsight consolidates in the background after a retain; `wait_idle()` blocks until that work is
done, so it never overlaps the next episode's timed grading.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from elh.memory.base import MemoryItem
from elh.runner.models import InfraError
from elh.tasks.base import TaskInstance

ACTIVE_STATES = {"pending", "processing", "running", "queued"}


class HindsightMemory:
    def __init__(self, base_url: str, bank_id: str, *, readonly: bool = False, timeout_s: float = 600,
                 recall_max_tokens: int = 2048):
        self.base = base_url.rstrip("/") + f"/v1/default/banks/{bank_id}"
        self.bank_id = bank_id
        self.readonly = readonly
        self.timeout_s = timeout_s
        self.recall_max_tokens = recall_max_tokens
        self.last_usage: dict | None = None  # token usage of the last retain (memory LLM)

    # -- http --------------------------------------------------------------------------------
    def _call(self, method: str, path: str, body: dict | None = None, timeout_s: float | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout_s or self.timeout_s) as resp:
                return json.loads(resp.read() or b"{}")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise InfraError(f"hindsight {method} {path}: {exc}") from exc

    def ensure_bank(self) -> None:
        self._call("PUT", "", {})

    # -- contract ----------------------------------------------------------------------------
    def recall(self, task: TaskInstance, k: int) -> list[MemoryItem]:
        out = self._call("POST", "/memories/recall", {"query": task.prompt, "max_tokens": self.recall_max_tokens})
        return [MemoryItem(memory_id=r["id"], text=r["text"]) for r in out.get("results", [])[:k]]

    def retain(self, episode: dict) -> list[str]:
        if self.readonly:
            raise RuntimeError(f"bank {self.bank_id} is read-only for this episode")
        out = self._call("POST", "/memories", {"items": [{
            "content": render_episode(episode),
            "context": "a coding-agent episode and the judge's verified result",
            "document_id": episode["episode_id"],
            "tags": [f"family:{episode['family']}"],
        }]})
        self.last_usage = out.get("usage")
        return [episode["episode_id"]]

    def wait_idle(self, timeout_s: float = 900, poll_s: float = 1.0) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            ops = self._call("GET", "/operations").get("operations", [])
            if not any(op.get("status") in ACTIVE_STATES for op in ops):
                return
            if time.monotonic() > deadline:
                raise InfraError(f"hindsight bank {self.bank_id} still busy after {timeout_s:.0f}s")
            time.sleep(poll_s)


def render_episode(ep: dict) -> str:
    verdict = ep["verdict"]
    checks = ", ".join(f"{k}={'pass' if v else 'FAIL'}" for k, v in verdict["checks"].items())
    lines = [
        f"Task family {ep['family']}. Outcome: {'PASSED' if verdict['passed'] else 'FAILED'} ({checks}).",
        f"Ticket:\n{ep['prompt'].strip()}",
        f"Agent's own summary of its change:\n{(ep.get('submit_summary') or '(none)').strip()}",
    ]
    q = verdict.get("detail", {}).get("quality") or {}
    if q:
        lines.append(f"Judge timing on the large input: agent {q.get('agent_s')}s vs budget {q.get('budget_s')}s "
                     f"(status {q.get('large_status')}).")
    lines.append(f"Episode used {ep['turns']} turns and {ep['tool_calls']} tool calls.")
    return "\n\n".join(lines)


def usage_to_ledger(usage: dict | None) -> dict:
    """Hindsight reports OpenAI-style usage (input includes cached); map to the ledger's keys."""
    if not usage:
        return {}
    cached = usage.get("cached_tokens") or 0
    return {"input_tokens": (usage.get("input_tokens") or 0) - cached,
            "output_tokens": (usage.get("output_tokens") or 0) + (usage.get("thoughts_tokens") or 0),
            "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0}
