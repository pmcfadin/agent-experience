"""HindsightMemory: group B's memory, backed by the dedicated Hindsight server (SPEC §10).

One bank per (run, history, group). Recall uses the ticket text as the query (optionally restricted
to fact types, plus a fixed pre-registered lesson query). Retain stores one post-episode record
built only from what the agent may learn from: the judge's verified feedback (checks and timings),
its own summary and code diff (and, in lesson-v3, a distilled WHEN/THEN rule), with the ticket
passed as extraction context (lesson-v1/v2) or as metadata outside the reranked text (lesson-v3);
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
FACT_TYPES = {"world", "experience", "observation"}


class HindsightMemory:
    """Every option defaults to the behaviour of registration v6 (lesson-v2, one ticket-text recall
    over all fact types), so existing registrations reproduce exactly.

    Retain (record format, see RECORD_FORMATS):
      record_format      "lesson-v2" (default), "lesson-v1", "ticket-v0" or "lesson-v3".
      include_diff       lesson-v3 only: keep the agent's raw diff in the record (Moves-to-Memory
                         finds raw trajectories transfer negatively, so this is an ablation switch).
      ticket_metadata    lesson-v3 only: send the ticket as item `metadata` (returned with results and
                         shown to the extraction LLM, never to the reranker) instead of omitting it.
    Recall:
      recall_types       fact types for the ticket query, e.g. ["experience"]; None = all types.
      lesson_query       a fixed, pre-registered query whose results take the last `lesson_k` of the
                         k slots (after the ticket results, without duplicates); None = off.
      lesson_k           slots reserved for the lesson query (only with lesson_query).
      lesson_types       fact types for the lesson query; None = recall_types.
    Bank (applied by ensure_bank through PATCH /config, only when set and the bank is writable):
      retain_mission, observations_mission, retain_extraction_mode.
    Rules (Learning on the Job, arXiv 2607.22157):
      rule_store         "extract" (default): an episode's rule is part of its record and goes
                         through LLM extraction. "verbatim": the record is sent without the rule and
                         the rule is its own item under the bank strategy RULE_STRATEGY (chunks mode:
                         stored as written, fact type "world", tagged "rule"), document
                         "<episode_id>:rule".
      rule_k             verbatim only: recall and search fill their first rule_k slots from a
                         rule-only recall (tags ["rule"], strict), the rest from the usual queries.
    Snapshots:
      fingerprint_version  1 = v6 content hash (unit type/state/tags/text + document count, plus
                         unit metadata, bank config overrides, mental models and directives when any
                         exist, which no v6 bank has); 2 = also every unit's `context`, which the
                         reranker scores. None = 2 for lesson-v3, else 1.
    """

    def __init__(self, base_url: str, bank_id: str, *, readonly: bool = False, timeout_s: float = 600,
                 recall_max_tokens: int = 2048, record_format: str = "lesson-v2",
                 include_diff: bool = True, ticket_metadata: bool = True,
                 recall_types: list[str] | None = None, lesson_query: str | None = None, lesson_k: int = 2,
                 lesson_types: list[str] | None = None,
                 retain_mission: str | None = None, observations_mission: str | None = None,
                 retain_extraction_mode: str | None = None, fingerprint_version: int | None = None,
                 rule_store: str = "extract", rule_k: int = 2):
        if record_format not in RECORD_FORMATS:
            raise ValueError(f"unknown record format {record_format!r}; known: {sorted(RECORD_FORMATS)}")
        if not include_diff and record_format != "lesson-v3":
            raise ValueError(f"include_diff=False needs record_format lesson-v3, not {record_format!r}")
        for types in (recall_types, lesson_types):
            if types is not None and (not types or not set(types) <= FACT_TYPES):
                raise ValueError(f"fact types must be a non-empty subset of {sorted(FACT_TYPES)}, got {types!r}")
        if lesson_query is not None and (not lesson_query.strip() or lesson_k < 1):
            raise ValueError("lesson_query needs non-empty text and lesson_k >= 1")
        if fingerprint_version not in (None, 1, 2):
            raise ValueError(f"unknown fingerprint_version {fingerprint_version!r}")
        if rule_store not in ("extract", "verbatim") or rule_k < 0:
            raise ValueError(f"rule_store must be 'extract' or 'verbatim' and rule_k >= 0; got {rule_store!r}, {rule_k}")
        self.base = base_url.rstrip("/") + f"/v1/default/banks/{bank_id}"
        self.bank_id = bank_id
        self.readonly = readonly
        self.timeout_s = timeout_s
        self.recall_max_tokens = recall_max_tokens
        self.record_format = record_format
        self.include_diff = include_diff
        self.ticket_metadata = ticket_metadata
        self.recall_types = list(recall_types) if recall_types else None
        self.lesson_query = lesson_query
        self.lesson_k = lesson_k
        self.lesson_types = list(lesson_types) if lesson_types else self.recall_types
        self.bank_config = {k: v for k, v in (("retain_mission", retain_mission),
                                               ("observations_mission", observations_mission),
                                               ("retain_extraction_mode", retain_extraction_mode)) if v is not None}
        self.rule_store = rule_store
        self.rule_k = rule_k
        if rule_store == "verbatim":
            self.bank_config["retain_strategies"] = {RULE_STRATEGY: {"retain_extraction_mode": "chunks"}}
        self.fingerprint_version = fingerprint_version or (2 if record_format == "lesson-v3" else 1)
        self.last_usage: dict | None = None  # token usage of the last retain (memory LLM)
        self.snapshot_id: str | None = None    # set when this bank is a checkpoint snapshot

    def options(self) -> dict:
        """The constructor options (besides url, bank and readonly), so a restored copy behaves the same."""
        return {"timeout_s": self.timeout_s, "recall_max_tokens": self.recall_max_tokens,
                "record_format": self.record_format, "include_diff": self.include_diff,
                "ticket_metadata": self.ticket_metadata, "recall_types": self.recall_types,
                "lesson_query": self.lesson_query, "lesson_k": self.lesson_k, "lesson_types": self.lesson_types,
                "retain_mission": self.bank_config.get("retain_mission"),
                "observations_mission": self.bank_config.get("observations_mission"),
                "retain_extraction_mode": self.bank_config.get("retain_extraction_mode"),
                "fingerprint_version": self.fingerprint_version,
                "rule_store": self.rule_store, "rule_k": self.rule_k}

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
        if self.bank_config and not self.readonly:
            self._call("PATCH", "/config", {"updates": self.bank_config})
            got = self._call("GET", "/config").get("overrides") or {}
            wrong = {k for k, v in self.bank_config.items() if got.get(k) != v}
            if wrong:
                raise InfraError(f"hindsight bank {self.bank_id} did not take config {sorted(wrong)}")

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
    def _recall(self, query: str, types: list[str] | None, tags: list[str] | None = None) -> list[MemoryItem]:
        """One recall, every result in Hindsight's ranked order (it packs results up to max_tokens).
        With tags, only memories carrying one of them (untagged memories excluded)."""
        body = {"query": query, "max_tokens": self.recall_max_tokens}
        if types:
            body["types"] = types
        if tags:
            body["tags"], body["tags_match"] = tags, "any_strict"
        return [_item(r) for r in self._call("POST", "/memories/recall", body).get("results", [])]

    @property
    def verbatim_rules(self) -> bool:
        return self.rule_store == "verbatim"

    def recall_rules(self, query: str, n: int | None = None) -> list[MemoryItem]:
        """Verbatim rules ranked for `query` (first n; default rule_k). Empty unless rule_store is
        "verbatim"."""
        n = self.rule_k if n is None else n
        if not self.verbatim_rules or n <= 0:
            return []
        return self._recall(query, [RULE_FACT_TYPE], [RULE_TAG])[:n]

    def _records(self, task: TaskInstance, k: int) -> tuple[list[MemoryItem], list[MemoryItem]]:
        """(top k for the ticket, the ticket query's full ranked list). With a lesson query, the
        ticket fills the first k - lesson_k slots and the lesson query the rest (skipping memories
        already chosen); short lists backfill from the ticket results."""
        ranked = self._recall(task.prompt, self.recall_types)
        if not self.lesson_query:
            return ranked[:k], ranked
        n_ticket = max(k - self.lesson_k, 0)
        out = ranked[:n_ticket]
        seen = {it.memory_id for it in out}
        _fill(out, seen, self._recall(self.lesson_query, self.lesson_types), k)
        _fill(out, seen, ranked[n_ticket:], k)
        return out, ranked

    def recall(self, task: TaskInstance, k: int) -> list[MemoryItem]:
        """Top k for the ticket (see `_records`). With verbatim rules, the first min(rule_k, k) slots
        go to rules recalled by the ticket text, the rest to the records; duplicates are skipped and
        a short list backfills from the remaining rules, then the ticket results."""
        if not self.verbatim_rules:
            return self._records(task, k)[0]
        all_rules = self.recall_rules(task.prompt, k)
        out = all_rules[:min(self.rule_k, k)]
        seen = {it.memory_id for it in out}
        records, ranked = self._records(task, k - len(out))
        _fill(out, seen, records, k)
        _fill(out, seen, all_rules, k)
        _fill(out, seen, ranked, k)
        return out

    def recall_candidates(self, task: TaskInstance, n: int | None = None) -> list[MemoryItem]:
        """The ticket query's full ranked list (first n; all when n is None), with document ids and
        per-stage scores, for a learner that reranks relevance candidates (group C). Verbatim rules
        are not candidates: the controller shows `recall_rules` first and reranks the rest."""
        ranked = self._recall(task.prompt, self.recall_types)
        return ranked if n is None else ranked[:n]

    def search(self, query: str, k: int) -> list[MemoryItem]:
        """Top k for a free-text query (an agent-issued memory search), over recall_types. With
        verbatim rules, up to min(k, rule_k) rule hits come first, then the other hits, no duplicates."""
        out = self.recall_rules(query, min(k, self.rule_k))
        _fill(out, {it.memory_id for it in out}, self._recall(query, self.recall_types), k)
        return out

    def retain(self, episode: dict) -> list[str]:
        """Store the episode record (document id = episode id). With verbatim rules the record is
        rendered without the rule, and the rule, when there is one, is a second item (document
        "<episode_id>:rule") stored as written. Returns the document ids written."""
        if self.readonly:
            raise RuntimeError(f"bank {self.bank_id} is read-only for this episode")
        rule = _rule(episode) if self.verbatim_rules else None
        record = {k: v for k, v in episode.items() if k != "rule"} if self.verbatim_rules else episode
        if self.record_format == "lesson-v3":
            item = lesson_v3_item(record, include_diff=self.include_diff, ticket_metadata=self.ticket_metadata)
        else:
            content, context = RECORD_FORMATS[self.record_format]
            item = {"content": content(record), "context": context(record),
                    "tags": [f"family:{record['family']}"]}
        items = [{**item, "document_id": episode["episode_id"]}]
        if rule is not None:
            items.append(rule_item(episode, rule))
        out = self._call("POST", "/memories", {"items": items})
        self.last_usage = out.get("usage")
        return [it["document_id"] for it in items]

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
        document count. Ids, timestamps and similarity links are left out (a restore changes them).

        Also hashed, but only when present so that v6 banks keep their recorded hash: unit metadata,
        the bank's config overrides (missions, recall settings), mental models and directives.
        Version 2 also hashes every unit's `context` (the reranker scores "{context}: {text}")."""
        v2 = self.fingerprint_version >= 2
        units = []
        for u in self.list_units():
            row = [u.get("fact_type"), u.get("state"), sorted(u.get("tags") or []), u.get("text")]
            if v2:
                row += [u.get("context") or "", u.get("metadata") or {}]
            elif u.get("metadata"):
                row.append(u["metadata"])
            units.append(row)
        docs = self._call("GET", "/stats").get("total_documents")
        content: dict = {"documents": docs, "units": sorted(units, key=json.dumps)}
        overrides = self._call("GET", "/config").get("overrides") or {}
        if overrides or v2:
            content["config"] = overrides
        models = [[m.get("id"), m.get("name"), m.get("source_query"), m.get("content"), sorted(m.get("tags") or [])]
                  for m in self._call("GET", "/mental-models").get("items", [])]
        directives = [[d.get("name"), d.get("content"), d.get("priority"), d.get("is_active"), sorted(d.get("tags") or [])]
                      for d in self._call("GET", "/directives").get("items", [])]
        if models or v2:
            content["mental_models"] = sorted(models, key=json.dumps)
        if directives or v2:
            content["directives"] = sorted(directives, key=json.dumps)
        if v2:
            content["fingerprint_version"] = 2
        blob = json.dumps(content, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def list_units(self) -> list[dict]:
        """Every memory unit the bank holds, as Hindsight lists them."""
        units, offset = [], 0
        while True:
            page = self._call("GET", f"/memories/list?limit=100&offset={offset}").get("items", [])
            units += page
            if len(page) < 100:
                return units
            offset += 100

    def record_units(self) -> list[dict]:
        """The units the ticket and search queries can return besides verbatim rules: every unit of
        `recall_types` (all types when None) that is not tagged as a rule."""
        return [u for u in self.list_units() if RULE_TAG not in (u.get("tags") or [])
                and (self.recall_types is None or u.get("fact_type") in self.recall_types)]

    def export_snapshot(self) -> bytes:
        """The bank as a transfer archive (data and bank config; no audit or LLM history)."""
        self.wait_idle()
        op = self._call("POST", "/transfer/export?include_data=true&include_bank_config=true&include_history=false")
        meta = self._wait_op(op["operation_id"]).get("result_metadata") or {}
        return self._raw("GET", self._root + meta["download_url"])

    def restore(self, archive: bytes, target_bank_id: str) -> "HindsightMemory":
        return restore_bank(self._root, archive, target_bank_id, **self.options())


# Hindsight records an import against an existing bank; this one only ever hosts those operations.
TRANSFER_BANK = "elh-transfer"


def restore_bank(base_url: str, archive: bytes, target_bank_id: str, **kw) -> HindsightMemory:
    """Write a transfer archive into a new bank (which must not exist) and return it read-only."""
    host = HindsightMemory(base_url, TRANSFER_BANK, timeout_s=kw.get("timeout_s", 600))
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


# lesson-v1 described every check by one phrase, so a failed integrity check read as "FAILED: left the
# tests and harness untouched" and was stored as the opposite lesson. Kept only to reproduce v1 runs.
CHECK_WORDS = {"integrity": "left the tests and harness untouched", "public_tests": "public tests",
               "correctness": "hidden correctness tests", "performance": "performance budget on the large input"}
# lesson-v2: what a passed and a failed check each mean, stated plainly.
CHECK_PASSED = {"integrity": "left protected files unchanged", "public_tests": "public tests passed",
                "correctness": "hidden correctness tests passed", "performance": "fast enough on the large input"}
CHECK_FAILED = {"integrity": "modified files the ticket says must not be changed",
                "public_tests": "public tests failed", "correctness": "hidden correctness tests failed",
                "performance": "too slow on the large input (over the time budget)"}
# Family wording. cpu_mem's is what lesson-v1 to v3 always said and stays byte-identical; other
# families get neutral wording. Their oracle states each family-specific check in plain words in
# the verdict's `check_text` ({check: {"label", "passed", "failed"}}), so raw check ids such as
# house_lib's `lesson:<id>` never reach memory or the rule writer.
FAMILY_WORDS = {"cpu_mem": {"ticket": "performance ticket", "passed": "the fix was verified correct and fast enough."}}
NEUTRAL_WORDS = {"ticket": "ticket", "passed": "the change passed every check."}


def family_name(family: str) -> str:
    """`house_lib/L16-c50-w1` -> `house_lib`: the world id stays out of what the agent reads."""
    return family.split("/")[0]


def family_words(family: str) -> dict[str, str]:
    return FAMILY_WORDS.get(family_name(family), NEUTRAL_WORDS)


def check_phrase(name: str, ok: bool, detail: dict | None, *, passed_words: dict = CHECK_PASSED,
                 failed_words: dict = CHECK_FAILED) -> str:
    text = ((detail or {}).get("check_text") or {}).get(name)
    if text:
        return text["passed" if ok else "failed"]
    return (passed_words if ok else failed_words).get(name, name)


def check_label(name: str, detail: dict | None) -> str:
    text = ((detail or {}).get("check_text") or {}).get(name)
    return text["label"] if text else name


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


def render_episode(ep: dict, *, version: int = 2, include_diff: bool = True) -> str:
    """The retained record leads with what happened and what was tried; the ticket goes in
    `render_context`, so extraction yields lessons rather than restatements of the ticket.
    Version 3 (lesson-v3) uses v2's wording but leaves out the ticket title, and may drop the diff."""
    verdict = ep["verdict"]
    fail_words, pass_words = (CHECK_WORDS, CHECK_WORDS) if version == 1 else (CHECK_FAILED, CHECK_PASSED)
    detail = verdict.get("detail")
    failed = [check_phrase(k, False, detail, failed_words=fail_words) for k, v in verdict["checks"].items() if not v]
    passed = [check_phrase(k, True, detail, passed_words=pass_words) for k, v in verdict["checks"].items() if v]
    head = f"PASSED: {family_words(ep['family'])['passed']}" if verdict["passed"] else f"FAILED: {', '.join(failed)}."
    family = family_name(ep["family"])
    if version >= 3:
        lines = [f"Outcome of an attempt at a {family} ticket: {head}"]
    else:
        lines = [f"Outcome of an attempt at a {family} ticket ({ep.get('title') or 'untitled'}): {head}"]
    violations = verdict.get("detail", {}).get("integrity_violations") or []
    if version >= 2 and violations:
        lines.append(f"Protected files the agent changed (the ticket forbids this): {', '.join(violations)}.")
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
    if include_diff:
        diff = ep.get("diff")
        lines.append(f"The agent's code change:\n{diff}" if diff else "The agent made no code change.")
    return "\n\n".join(lines)


def render_context(ep: dict) -> str:
    return (f"One coding agent's attempt at a {family_words(ep['family'])['ticket']}, with the result verified by an automated "
            f"judge. The ticket was:\n{ep['prompt'].strip()}")


# lesson-v3 (2026-10-25). Hindsight 1.1.0 reranks every unit as "{context}: {text}" with a 512-token
# cross-encoder, so lesson-v2's context (preamble + whole ticket) made recall rank by ticket
# similarity and truncated the lesson itself. v3's context is one short sentence stating the
# verified outcome (plus the rule's situation key when there is a rule); the ticket never enters
# the reranked string (it goes in item metadata, or is omitted). Outcome tags are added at retain.
RULE_HEADS = {"verified": "VERIFIED RULE", "avoid": "AVOID"}


def _rule(ep: dict) -> dict | None:
    rule = ep.get("rule")
    if rule is None:
        return None
    if rule.get("kind") not in RULE_HEADS or not str(rule.get("when") or "").strip() \
            or not str(rule.get("then") or "").strip():
        raise ValueError(f"episode {ep.get('episode_id')}: rule needs kind in {sorted(RULE_HEADS)}, "
                         f"non-empty when and then; got {rule!r}")
    return rule


def render_rule(rule: dict) -> str:
    """Learning on the Job's one-rule-per-episode form: `VERIFIED RULE: WHEN … THEN …` (success) or
    `AVOID: WHEN … THEN …` (failure, experience-grade feedback)."""
    return f"{RULE_HEADS[rule['kind']]}: WHEN {rule['when'].strip()} THEN {rule['then'].strip()}"


def situation_key(rule: dict) -> str:
    """The rule's retrieval hook: its `query` (the question a future agent would ask), else its WHEN."""
    return str(rule.get("query") or "").strip() or rule["when"].strip()


def failed_checks(ep: dict) -> list[str]:
    """Failed checks by label: the check id, or the family's plain label for it (`check_text`)."""
    detail = ep["verdict"].get("detail")
    return [check_label(k, detail) for k, v in ep["verdict"]["checks"].items() if not v]


def render_context_v3(ep: dict) -> str:
    failed = failed_checks(ep)
    result = "PASSED" if ep["verdict"]["passed"] else f"FAILED ({', '.join(failed) or 'not verified'})"
    ctx = f"Verified result of a coding agent's attempt at a {family_name(ep['family'])} {family_words(ep['family'])['ticket']}: {result}"
    rule = _rule(ep)
    return f"{ctx}. Situation: {situation_key(rule)}" if rule else ctx


def render_episode_v3(ep: dict, *, include_diff: bool = True) -> str:
    """The rule first (when a distiller produced one), with its situation key; then the v2 outcome
    record without the ticket title, optionally without the raw diff."""
    rule = _rule(ep)
    lines = [render_rule(rule), f"Situation key: {situation_key(rule)}"] if rule else []
    return "\n\n".join(lines + [render_episode(ep, version=3, include_diff=include_diff)])


def lesson_v3_tags(ep: dict) -> list[str]:
    tags = [f"family:{ep['family']}", f"outcome:{'passed' if ep['verdict']['passed'] else 'failed'}"]
    tags += [f"failed-check:{c}" for c in failed_checks(ep)]
    rule = _rule(ep)
    return tags + ([f"rule:{rule['kind']}"] if rule else [])


def lesson_v3_item(ep: dict, *, include_diff: bool = True, ticket_metadata: bool = True) -> dict:
    """The retain item (without document_id). Metadata is returned with recall results and shown to
    the extraction LLM, but is not part of the reranker's input."""
    item = {"content": render_episode_v3(ep, include_diff=include_diff), "context": render_context_v3(ep),
            "tags": lesson_v3_tags(ep)}
    if ticket_metadata:
        item["metadata"] = {"ticket": ep["prompt"].strip(),
                            "outcome": "passed" if ep["verdict"]["passed"] else "failed"}
    return item


# Verbatim rules (rule_store="verbatim"). Hindsight 1.1.0's "chunks" extraction mode stores each
# chunk of the item's content as one "world" unit with the item's tags/context/metadata and no LLM
# call; a per-item `strategy` names a bank `retain_strategies` entry whose overrides apply to that
# item only. A chunk holds up to retain_chunk_size characters (default 3000), so a rule stays one
# unit unless the agent wrote more than that.
RULE_STRATEGY = "rule-verbatim"
RULE_TAG = "rule"
RULE_FACT_TYPE = "world"
RULE_DOC_SUFFIX = ":rule"
RULE_CONTEXT = "a rule an agent wrote after a verified attempt"


def render_rule_verbatim(rule: dict) -> str:
    return f"{render_rule(rule)}\nSituation: {situation_key(rule)}"


def rule_item(ep: dict, rule: dict) -> dict:
    """The retain item for one verbatim rule (content stored as written)."""
    return {"content": render_rule_verbatim(rule), "context": RULE_CONTEXT,
            "tags": [f"family:{ep['family']}", RULE_TAG, f"rule:{rule['kind']}"],
            "strategy": RULE_STRATEGY, "document_id": ep["episode_id"] + RULE_DOC_SUFFIX}


class RulesOnlyMemory:
    """Group B-rules (H6): a read-only B snapshot where every start slot and every search hit is a
    verbatim rule, so retrieval selects among rules only (no episode records)."""

    verbatim_rules = True
    readonly = True

    def __init__(self, bank: "HindsightMemory"):
        if not bank.verbatim_rules or not bank.readonly:
            raise ValueError("B-rules needs a read-only bank with verbatim rules")
        self.bank, self.bank_id, self.snapshot_id = bank, bank.bank_id, bank.snapshot_id

    def recall(self, task: TaskInstance, k: int) -> list[MemoryItem]:
        return self.bank.recall_rules(task.prompt, k)

    def search(self, query: str, k: int) -> list[MemoryItem]:
        return self.bank.recall_rules(query, k)

    def retain(self, episode: dict) -> list[str]:
        raise RuntimeError("group B-rules never retains")

    def fingerprint(self) -> str:
        return self.bank.fingerprint()

    def wait_idle(self) -> None:
        self.bank.wait_idle()


def is_rule_item(it: MemoryItem) -> bool:
    return (it.document_id or "").endswith(RULE_DOC_SUFFIX)


def _fill(out: list[MemoryItem], seen: set[str], items, k: int) -> None:
    """Append items not yet seen until out holds k."""
    for it in items:
        if len(out) >= k:
            break
        if it.memory_id not in seen:
            out.append(it)
            seen.add(it.memory_id)


def _item(r: dict) -> MemoryItem:
    scores = {k: v for k, v in (r.get("scores") or {}).items() if v is not None}
    return MemoryItem(memory_id=r["id"], text=r["text"], score=float(scores.get("final") or 0.0),
                      document_id=r.get("document_id"), scores=scores)


def render_episode_ticket_v0(ep: dict) -> str:
    """`ticket-v0`: the first record format (git 95b83ec), ticket first. Kept for comparison runs."""
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


TICKET_V0_CONTEXT = "a coding-agent episode and the judge's verified result"

# memory_service.record_format -> (content renderer, context renderer). lesson-v3's retain item
# (tags, metadata, include_diff) is built by lesson_v3_item.
RECORD_FORMATS = {
    "lesson-v3": (render_episode_v3, render_context_v3),
    "lesson-v2": (render_episode, render_context),
    "lesson-v1": (lambda ep: render_episode(ep, version=1), render_context),
    "ticket-v0": (render_episode_ticket_v0, lambda ep: TICKET_V0_CONTEXT),
}


def usage_to_ledger(usage: dict | None) -> dict:
    """Hindsight reports OpenAI-style usage (input includes cached); map to the ledger's keys."""
    if not usage:
        return {}
    cached = usage.get("cached_tokens") or 0
    return {"input_tokens": (usage.get("input_tokens") or 0) - cached,
            "output_tokens": (usage.get("output_tokens") or 0) + (usage.get("thoughts_tokens") or 0),
            "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0}
