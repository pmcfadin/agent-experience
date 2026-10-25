"""HindsightMemory: arm B's memory, backed by the dedicated Hindsight server (SPEC §10).

One bank per (run, history, arm). Recall uses the ticket text as the query. Retain stores one
post-episode record built only from what the agent may learn from: the ticket, its own summary
and the judge's verified feedback (checks and timings), never the hidden oracle.

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
