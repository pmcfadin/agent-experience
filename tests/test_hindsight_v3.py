"""lesson-v3 records, recall options, bank missions and the extended fingerprint.

Stub-server tests check request shapes; the live tests restore the fixture bank into throwaway
banks (deleted afterwards) and are skipped when the Hindsight server is not running."""

import hashlib
import json
import os
import threading
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from elh.memory.hindsight import (HindsightMemory, render_context, render_context_v3, render_episode,
                                  render_episode_v3, restore_bank)
from elh.runner.models import InfraError
from elh.tasks.base import get_family

LQ = "What mistakes made a coding agent fail a performance ticket? Which files or rules did the agent violate?"


def _r(i, prefix="t"):
    return {"id": f"{prefix}{i}", "text": f"fact {prefix}{i}", "document_id": f"doc-{prefix}{i}", "type": "experience",
            "scores": {"final": 1 - i / 10, "reranker": 0.5, "semantic": 0.7, "keyword": None}}


class _Stub:
    """Answers recall by query (ticket -> t0..t6, lesson query -> `lesson`), and serves list/config."""

    def __init__(self, lesson=None, units=None, overrides=None):
        self.requests, self.overrides = [], dict(overrides or {})
        self.lesson = lesson if lesson is not None else [_r(i, "l") for i in range(4)]
        self.units = units if units is not None else [
            {"fact_type": "experience", "state": "valid", "tags": ["family:cpu_mem"], "text": "a", "context": "c",
             "metadata": {}}]
        stub = self

        class H(BaseHTTPRequestHandler):
            def _reply(self, body):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self):
                n = int(self.headers.get("content-length") or 0)
                return json.loads(self.rfile.read(n)) if n else None

            def do_PUT(self):
                stub.requests.append(("PUT", self.path, self._body()))
                self._reply({})

            def do_PATCH(self):
                body = self._body()
                stub.requests.append(("PATCH", self.path, body))
                stub.overrides.update(body["updates"])
                self._reply({})

            def do_POST(self):
                body = self._body()
                stub.requests.append(("POST", self.path, body))
                if self.path.endswith("/recall"):
                    self._reply({"results": stub.lesson if body["query"] == LQ else [_r(i) for i in range(7)]})
                else:
                    self._reply({"success": True})

            def do_GET(self):
                stub.requests.append(("GET", self.path, None))
                if "/memories/list" in self.path:
                    self._reply({"items": stub.units if self.path.endswith("offset=0") else []})
                elif self.path.endswith("/stats"):
                    self._reply({"total_documents": 1})
                elif self.path.endswith("/config"):
                    self._reply({"overrides": stub.overrides})
                elif self.path.endswith("/mental-models") or self.path.endswith("/directives"):
                    self._reply({"items": []})
                else:
                    self._reply({"operations": []})

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def recalls(self):
        return [b for m, p, b in self.requests if p.endswith("/recall")]


@pytest.fixture
def task():
    return get_family("cpu_mem").generate("queue_pop_front", "billing", 1)


def _episode(task, *, passed=False, checks=None, rule=None, violations=None):
    ep = {"episode_id": "e1", "family": task.family, "prompt": task.prompt, "title": task.prompt.splitlines()[0],
          "outcome": "submitted", "submit_summary": "used a deque", "diff": "--- a/job/core.py\n+++ b/job/core.py",
          "verdict": {"passed": passed, "checks": checks or {"integrity": True, "performance": passed},
                      "detail": {"quality": {"agent_s": 3.0, "budget_s": 3.0, "large_status": "timeout"},
                                 **({"integrity_violations": violations} if violations else {})}},
          "turns": 12, "tool_calls": 12}
    if rule:
        ep["rule"] = rule
    return ep


# -- recall ------------------------------------------------------------------------------------
def test_default_recall_request_is_unchanged_and_items_carry_scores(task):
    stub = _Stub()
    items = HindsightMemory(stub.url, "b1").recall(task, 5)
    assert stub.recalls() == [{"query": task.prompt, "max_tokens": 2048}]
    assert [it.memory_id for it in items] == ["t0", "t1", "t2", "t3", "t4"]
    assert items[0].document_id == "doc-t0" and items[0].score == 1.0
    assert items[0].scores == {"final": 1.0, "reranker": 0.5, "semantic": 0.7}  # null stages dropped


def test_recall_types_are_sent(task):
    stub = _Stub()
    HindsightMemory(stub.url, "b1", recall_types=["experience"]).recall(task, 5)
    assert stub.recalls() == [{"query": task.prompt, "max_tokens": 2048, "types": ["experience"]}]


def test_lesson_query_takes_the_last_slots(task):
    stub = _Stub()
    mem = HindsightMemory(stub.url, "b1", recall_types=["experience"], lesson_query=LQ, lesson_k=2,
                          lesson_types=["experience", "observation"])
    assert [it.memory_id for it in mem.recall(task, 5)] == ["t0", "t1", "t2", "l0", "l1"]
    assert stub.recalls()[1] == {"query": LQ, "max_tokens": 2048, "types": ["experience", "observation"]}


def test_lesson_query_skips_duplicates_and_backfills(task):
    stub = _Stub(lesson=[_r(1), _r(0, "l")])  # t1 is already a ticket result
    mem = HindsightMemory(stub.url, "b1", lesson_query=LQ, lesson_k=2)
    assert [it.memory_id for it in mem.recall(task, 5)] == ["t0", "t1", "t2", "l0", "t3"]
    assert "types" not in stub.recalls()[1]  # lesson_types defaults to recall_types (all)


def test_candidates_and_search(task):
    stub = _Stub()
    mem = HindsightMemory(stub.url, "b1", recall_types=["experience"])
    assert len(mem.recall_candidates(task)) == 7 and len(mem.recall_candidates(task, 3)) == 3
    assert all(c.document_id and "final" in c.scores for c in mem.recall_candidates(task))
    assert [x.memory_id for x in mem.search("protected files", 2)] == ["t0", "t1"]
    assert stub.recalls()[-1] == {"query": "protected files", "max_tokens": 2048, "types": ["experience"]}


@pytest.mark.parametrize("kw", [{"recall_types": ["facts"]}, {"recall_types": []}, {"lesson_types": ["opinion"]},
                                {"lesson_query": "  "}, {"lesson_query": LQ, "lesson_k": 0},
                                {"include_diff": False}, {"fingerprint_version": 3}])
def test_bad_options_are_rejected(kw):
    with pytest.raises(ValueError):
        HindsightMemory("http://127.0.0.1:9", "b1", **kw)


# -- bank missions -----------------------------------------------------------------------------
def test_missions_are_patched_into_bank_config():
    stub = _Stub()
    HindsightMemory(stub.url, "b1", retain_mission="Record lessons.", observations_mission="Track causes.").ensure_bank()
    patches = [b for m, p, b in stub.requests if m == "PATCH"]
    assert patches == [{"updates": {"retain_mission": "Record lessons.", "observations_mission": "Track causes."}}]
    assert stub.requests[1][1] == "/v1/default/banks/b1/config"


def test_no_mission_and_readonly_banks_send_no_config():
    stub = _Stub()
    HindsightMemory(stub.url, "b1").ensure_bank()
    HindsightMemory(stub.url, "b2", readonly=True, retain_mission="x").ensure_bank()
    assert [m for m, p, b in stub.requests] == ["PUT", "PUT"]


# -- lesson-v3 records -------------------------------------------------------------------------
def test_v3_context_is_short_outcome_without_ticket(task):
    ep = _episode(task, checks={"integrity": False, "performance": True}, violations=["examples/x.json"])
    ctx = render_context_v3(ep)
    assert ctx == "Verified result of a coding agent's attempt at a cpu_mem performance ticket: FAILED (integrity)"
    text = render_episode_v3(ep)
    assert task.prompt.strip() not in ctx + text and ep["title"] not in ctx + text
    assert "modified files the ticket says must not be changed" in text and "examples/x.json" in text
    assert render_context_v3(_episode(task, passed=True)).endswith(": PASSED")


def test_v3_retain_item_tags_metadata_and_rule(task):
    rule = {"kind": "avoid", "when": "the ticket forbids editing examples/", "then": "do NOT edit examples/",
            "query": "may I regenerate the example output?"}
    ep = _episode(task, checks={"integrity": False, "performance": True}, rule=rule)
    stub = _Stub()
    HindsightMemory(stub.url, "b1", record_format="lesson-v3").retain(ep)
    item = stub.requests[-1][2]["items"][0]
    assert item["document_id"] == "e1"
    assert item["tags"] == ["family:cpu_mem", "outcome:failed", "failed-check:integrity", "rule:avoid"]
    assert item["metadata"] == {"ticket": task.prompt.strip(), "outcome": "failed"}
    assert item["content"].startswith("AVOID: WHEN the ticket forbids editing examples/ THEN do NOT edit examples/\n\n"
                                      "Situation key: may I regenerate the example output?")
    assert item["context"].endswith("FAILED (integrity). Situation: may I regenerate the example output?")
    assert "a/job/core.py" in item["content"]


def test_v3_verified_rule_without_query_keys_on_when(task):
    rule = {"kind": "verified", "when": "pop(0) on a growing list", "then": "use collections.deque.popleft()"}
    ep = _episode(task, passed=True, checks={"integrity": True, "performance": True}, rule=rule)
    assert render_episode_v3(ep).startswith("VERIFIED RULE: WHEN pop(0) on a growing list THEN use "
                                            "collections.deque.popleft()\n\nSituation key: pop(0) on a growing list")
    assert render_context_v3(ep).endswith("PASSED. Situation: pop(0) on a growing list")


def test_v3_options_drop_diff_and_ticket(task):
    stub = _Stub()
    HindsightMemory(stub.url, "b1", record_format="lesson-v3", include_diff=False, ticket_metadata=False
                    ).retain(_episode(task, passed=True, checks={"integrity": True, "performance": True}))
    item = stub.requests[-1][2]["items"][0]
    assert "metadata" not in item and "a/job/core.py" not in item["content"] and "code change" not in item["content"]
    assert item["tags"] == ["family:cpu_mem", "outcome:passed"]


@pytest.mark.parametrize("rule", [{"kind": "maybe", "when": "x", "then": "y"}, {"kind": "avoid", "when": "", "then": "y"},
                                  {"kind": "verified", "when": "x"}])
def test_v3_malformed_rule_is_rejected(task, rule):
    with pytest.raises(ValueError, match="rule"):
        render_episode_v3(_episode(task, rule=rule))


def test_v2_ignores_rule(task):
    ep = _episode(task, rule={"kind": "avoid", "when": "x", "then": "y"})
    stub = _Stub()
    HindsightMemory(stub.url, "b1").retain(ep)
    item = stub.requests[-1][2]["items"][0]
    assert (item["content"], item["context"], item["tags"]) == (render_episode(ep), render_context(ep),
                                                                ["family:cpu_mem"])
    assert "metadata" not in item


# -- fingerprint -------------------------------------------------------------------------------
def _v6_hash(units, docs=1):
    rows = [[u["fact_type"], u["state"], sorted(u["tags"]), u["text"]] for u in units]
    blob = json.dumps({"documents": docs, "units": sorted(rows, key=json.dumps)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def test_v1_fingerprint_matches_v6_hash_for_v6_banks():
    stub = _Stub()
    assert HindsightMemory(stub.url, "b1").fingerprint() == _v6_hash(stub.units)


def test_fingerprint_covers_config_metadata_and_v2_context():
    stub = _Stub()
    mem = HindsightMemory(stub.url, "b1")
    plain = mem.fingerprint()
    stub.overrides["observations_mission"] = "x"
    with_config = mem.fingerprint()
    stub.units[0]["metadata"] = {"ticket": "t"}
    with_meta = mem.fingerprint()
    assert len({plain, with_config, with_meta}) == 3
    v2 = HindsightMemory(stub.url, "b1", record_format="lesson-v3")
    assert v2.fingerprint_version == 2 and v2.fingerprint() != with_meta
    before = v2.fingerprint()
    stub.units[0]["context"] = "other"
    assert v2.fingerprint() != before and mem.fingerprint() == with_meta  # v1 ignores context


def test_options_round_trip():
    mem = HindsightMemory("http://x", "b", record_format="lesson-v3", include_diff=False, recall_types=["experience"],
                          lesson_query=LQ, lesson_k=3, retain_mission="m", fingerprint_version=1)
    again = HindsightMemory("http://x", "b", **mem.options())
    assert again.options() == mem.options() and again.fingerprint_version == 1


# -- live server -------------------------------------------------------------------------------
URL = os.environ.get("ELH_HINDSIGHT_URL", "http://127.0.0.1:8901")
FIXTURE = (Path(__file__).parent / "data" / "hindsight-bank.zip").read_bytes()


def _up() -> bool:
    try:
        urllib.request.urlopen(URL + "/health", timeout=2).close()
        return True
    except OSError:
        return False


needs_server = pytest.mark.skipif(not _up(), reason="Hindsight server not running")


@pytest.fixture
def banks():
    """Throwaway bank ids; only these are ever deleted."""
    made: list[str] = []
    yield made
    for b in made:
        try:
            HindsightMemory(URL, b).delete_bank()
        except InfraError:
            pass


def _new_bank(banks) -> str:
    b = f"elh-test-v3-{uuid.uuid4().hex[:10]}"
    banks.append(b)
    return b


@needs_server
def test_live_recall_options(banks, task):
    mem = restore_bank(URL, FIXTURE, _new_bank(banks), recall_types=["experience"], lesson_query=LQ, lesson_k=2,
                       lesson_types=["experience", "observation"])
    assert mem.readonly and mem.lesson_query == LQ  # restore keeps the options
    before = mem.fingerprint()
    units = mem._call("GET", "/memories/list?limit=100&offset=0")["items"]
    types = {u["id"]: u["fact_type"] for u in units}
    cands = mem.recall_candidates(task)
    assert cands and all(types.get(c.memory_id) == "experience" for c in cands)
    assert all(c.document_id and "final" in c.scores for c in cands)
    assert [c.scores["final"] for c in cands] == sorted((c.scores["final"] for c in cands), reverse=True)
    items = mem.recall(task, 3)
    ids = [it.memory_id for it in items]
    assert 2 <= len(ids) == len(set(ids)) <= 3
    assert ids[0] == cands[0].memory_id  # the ticket fills the first k - lesson_k slots
    lesson = [it.memory_id for it in mem._recall(LQ, ["experience", "observation"])]
    assert ids[1] in lesson
    assert mem.search(LQ, 2) and mem.fingerprint() == before  # recall is read-only


@needs_server
def test_live_missions_survive_snapshot_and_change_fingerprint(banks):
    src = _new_bank(banks)
    restore_bank(URL, FIXTURE, src)
    plain = HindsightMemory(URL, src).fingerprint()
    mem = HindsightMemory(URL, src, retain_mission="Record what made each attempt pass or fail.",
                          observations_mission="Track which agent actions caused verified failures.")
    mem.ensure_bank()
    overrides = mem._call("GET", "/config")["overrides"]
    assert overrides["retain_mission"].startswith("Record") and overrides["observations_mission"].startswith("Track")
    assert mem.fingerprint() != plain
    for version in (1, 2):
        mem.fingerprint_version = version
        live = mem.fingerprint()
        snap = mem.restore(mem.export_snapshot(), _new_bank(banks))
        assert snap.fingerprint_version == version and snap.fingerprint() == live
