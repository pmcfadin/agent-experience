"""HindsightMemory: arm B's memory, backed by the dedicated Hindsight server (SPEC §10).

One bank per (run, history, arm). Recall uses the ticket text as the query. Retain stores one
post-episode record built only from what the agent may learn from: the judge's verified feedback
(checks and timings), its own summary and code diff, with the ticket passed as extraction context;
never the hidden oracle.

Hindsight consolidates in the background after a retain; `wait_idle()` blocks until that work is
done, so it never overlaps the next episode's timed grading.

Checkpoint snapshots (SPEC §10) use Hindsight's bank transfer: `export_snapshot()` returns the
transfer archive, `restore()` writes it into a fresh bank (re-embedded, no LLM call). A restored
bank has new memory ids and recomputed similarity links, so banks are compared by
`fingerprint()`, a hash of their memory contents, never by ids.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
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
        self.snapshot_id: str | None = None    # set when this bank is a checkpoint snapshot

    # -- http --------------------------------------------------------------------------------
    def _raw(self, method: str, url: str, data: bytes | None = None, content_type: str = "application/json",
             timeout_s: float | None = None) -> bytes:
        req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": content_type})
        try:
            with urllib.request.urlopen(req, timeout=timeout_s or self.timeout_s) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise InfraError(f"hindsight {method} {url}: {exc}") from exc

    def _call(self, method: str, path: str, body: dict | None = None, timeout_s: float | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        return json.loads(self._raw(method, self.base + path, data, timeout_s=timeout_s) or b"{}")

    @property
    def _root(self) -> str:
        return self.base.split("/v1/default/banks/")[0]

    def ensure_bank(self) -> None:
        self._call("PUT", "", {})

    def exists(self) -> bool:
        try:
            urllib.request.urlopen(self.base + "/stats", timeout=self.timeout_s).close()
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            raise InfraError(f"hindsight GET {self.base}/stats: {exc}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise InfraError(f"hindsight GET {self.base}/stats: {exc}") from exc

    def delete_bank(self) -> None:
        self._call("DELETE", "")

    def _wait_op(self, op_id: str, timeout_s: float = 900, poll_s: float = 0.5) -> dict:
        deadline = time.monotonic() + timeout_s
        while True:
            op = self._call("GET", f"/operations/{op_id}")
            if op.get("status") == "completed":
                return op
            if op.get("status") not in ACTIVE_STATES:
                raise InfraError(f"hindsight operation {op_id} on {self.bank_id}: {op.get('status')} "
                                 f"{op.get('error_message') or ''}".strip())
            if time.monotonic() > deadline:
                raise InfraError(f"hindsight operation {op_id} on {self.bank_id} still running after {timeout_s:.0f}s")
            time.sleep(poll_s)

    # -- contract ----------------------------------------------------------------------------
    def recall(self, task: TaskInstance, k: int) -> list[MemoryItem]:
        out = self._call("POST", "/memories/recall", {"query": task.prompt, "max_tokens": self.recall_max_tokens})
        return [MemoryItem(memory_id=r["id"], text=r["text"]) for r in out.get("results", [])[:k]]

    def retain(self, episode: dict) -> list[str]:
        if self.readonly:
            raise RuntimeError(f"bank {self.bank_id} is read-only for this episode")
        out = self._call("POST", "/memories", {"items": [{
            "content": render_episode(episode),
            "context": render_context(episode),
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


    # -- snapshots ---------------------------------------------------------------------------
    def fingerprint(self) -> str:
        """Hash of what the bank holds: every memory unit's type, state, tags and text, plus the
        document count. Ids, timestamps and similarity links are left out (a restore changes them)."""
        units, offset = [], 0
        while True:
            page = self._call("GET", f"/memories/list?limit=100&offset={offset}").get("items", [])
            units += [[u.get("fact_type"), u.get("state"), sorted(u.get("tags") or []), u.get("text")] for u in page]
            if len(page) < 100:
                break
            offset += 100
        docs = self._call("GET", "/stats").get("total_documents")
        blob = json.dumps({"documents": docs, "units": sorted(units, key=json.dumps)}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def export_snapshot(self) -> bytes:
        """The bank as a transfer archive (data and bank config; no audit or LLM history)."""
        self.wait_idle()
        op = self._call("POST", "/transfer/export?include_data=true&include_bank_config=true&include_history=false")
        meta = self._wait_op(op["operation_id"]).get("result_metadata") or {}
        return self._raw("GET", self._root + meta["download_url"])

    def restore(self, archive: bytes, target_bank_id: str) -> "HindsightMemory":
        return restore_bank(self._root, archive, target_bank_id, timeout_s=self.timeout_s,
                            recall_max_tokens=self.recall_max_tokens)


# Hindsight records an import against an existing bank; this one only ever hosts those operations.
TRANSFER_BANK = "elh-transfer"


def restore_bank(base_url: str, archive: bytes, target_bank_id: str, **kw) -> HindsightMemory:
    """Write a transfer archive into a new bank (which must not exist) and return it read-only."""
    host = HindsightMemory(base_url, TRANSFER_BANK)
    host.ensure_bank()
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"bank.zip\"\r\n"
            f"Content-Type: application/zip\r\n\r\n").encode() + archive + f"\r\n--{boundary}--\r\n".encode()
    out = json.loads(host._raw("POST", host.base + f"/transfer/import?mode=restore&target_bank_id={target_bank_id}",
                               body, content_type=f"multipart/form-data; boundary={boundary}"))
    host._wait_op(out["operation_id"])
    target = HindsightMemory(base_url, target_bank_id, readonly=True, **kw)
    target.wait_idle()
    return target


CHECK_WORDS = {"integrity": "left the tests and harness untouched", "public_tests": "public tests",
               "correctness": "hidden correctness tests", "performance": "performance budget on the large input"}
OUTCOME_WORDS = {"submitted": "submitted its fix", "implicit_submit": "stopped without calling submit",
                 "budget_exhausted": "ran out of its turn/token budget before submitting"}
MAX_DIFF_LINES = 80
MAX_DIFF_CHARS = 6000
SOURCE_SUFFIXES = (".py",)


def workspace_diff(before: dict[str, str], after: dict[str, str], max_lines: int = MAX_DIFF_LINES,
                   max_chars: int = MAX_DIFF_CHARS) -> str:
    """The agent's own changes: a unified diff of source files it edited or created, and the
    names of every other file it created or deleted (scratch data, backups, outputs), truncated."""
    import difflib

    lines: list[str] = []
    other: list[str] = []
    for rel in sorted(set(before) | set(after)):
        a, b = before.get(rel), after.get(rel)
        if a == b:
            continue
        if not rel.endswith(SOURCE_SUFFIXES):
            other.append(f"{rel} ({'deleted' if b is None else 'created' if a is None else 'modified'}, "
                         f"{len(b or '')} bytes)")
            continue
        lines += difflib.unified_diff((a or "").splitlines(), (b or "").splitlines(),
                                      f"a/{rel}" if a is not None else "/dev/null",
                                      f"b/{rel}" if b is not None else "/dev/null", lineterm="", n=1)
    lines = [ln if len(ln) <= 200 else ln[:200] + " ..." for ln in lines]
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more diff lines)"]
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (diff truncated)"
    if other:
        text += ("\n" if text else "No source file was changed.\n") + "Other files changed: " + "; ".join(other[:20]) + \
            (f"; and {len(other) - 20} more" if len(other) > 20 else "")
    return text


def render_episode(ep: dict) -> str:
    """The retained record leads with what happened and what was tried; the ticket goes in
    `render_context`, so extraction yields lessons rather than restatements of the ticket."""
    verdict = ep["verdict"]
    failed = [CHECK_WORDS.get(k, k) for k, v in verdict["checks"].items() if not v]
    passed = [CHECK_WORDS.get(k, k) for k, v in verdict["checks"].items() if v]
    head = "PASSED: the fix was verified correct and fast enough." if verdict["passed"] else \
        f"FAILED: {', '.join(failed)}."
    lines = [f"Outcome of an attempt at a {ep['family']} ticket ({ep.get('title') or 'untitled'}): {head}"]
    if not verdict["passed"] and passed:
        lines.append(f"Checks that passed: {', '.join(passed)}.")
    q = verdict.get("detail", {}).get("quality") or {}
    if q.get("agent_s") is not None and q.get("budget_s"):
        status = q.get("large_status")
        lines.append(f"On the large input the agent's code took {q['agent_s']:.2f}s against a budget of "
                     f"{q['budget_s']:.2f}s ({q['agent_s'] / q['budget_s']:.0%} of budget"
                     f"{', ' + status if status and status != 'ok' else ''}).")
    lines.append(f"The agent {OUTCOME_WORDS.get(ep.get('outcome'), ep.get('outcome') or 'finished')} after "
                 f"{ep['turns']} turns and {ep['tool_calls']} tool calls.")
    lines.append(f"What the agent said it changed: {(ep.get('submit_summary') or '(no summary)').strip()}")
    diff = ep.get("diff")
    lines.append(f"The agent's code change:\n{diff}" if diff else "The agent made no code change.")
    return "\n\n".join(lines)


def render_context(ep: dict) -> str:
    return (f"One coding agent's attempt at a performance ticket, with the result verified by an automated "
            f"judge. The ticket was:\n{ep['prompt'].strip()}")


def usage_to_ledger(usage: dict | None) -> dict:
    """Hindsight reports OpenAI-style usage (input includes cached); map to the ledger's keys."""
    if not usage:
        return {}
    cached = usage.get("cached_tokens") or 0
    return {"input_tokens": (usage.get("input_tokens") or 0) - cached,
            "output_tokens": (usage.get("output_tokens") or 0) + (usage.get("thoughts_tokens") or 0),
            "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0}
