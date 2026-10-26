"""Verbatim rule store (Learning on the Job, arXiv 2607.22157): the agent's rule is retained as its
own chunks-mode item, recalled by a rule-only query into reserved slots, and shown first in groups B
and C. Stub-server tests check request shapes; nothing here talks to a live Hindsight."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from elh.batch import load_experiment
from elh.controller import hindsight_options, run_single_episode
from elh.learner.utility import UtilityConfig, UtilityLearner
from elh.memory.base import MemoryItem, render_memory_block
from elh.memory.hindsight import (RULE_CONTEXT, RULE_STRATEGY, HindsightMemory, lesson_v3_item, render_rule,
                                  situation_key)
from elh.registry import RulesConfig, load_config
from elh.runner.scripts import scripted_client
from elh.store import Store
from elh.tasks.base import get_family

ROOT = Path(__file__).parent.parent
CONFIG = ROOT / "experiments" / "phase0-local.yaml"
RULE = {"kind": "avoid", "when": "the ticket says examples/sample_output.json is a fixture",
        "then": "do NOT rewrite examples/sample_output.json -- verified wrong against the ground-truth outcome.",
        "query": "batch job slow; may I regenerate the sample output?", "title": "AVOID: touching fixtures"}


def _u(i, prefix="t", doc=None):
    return {"id": f"{prefix}{i}", "text": f"fact {prefix}{i}", "document_id": doc or f"doc-{prefix}{i}",
            "scores": {"final": 1 - i / 10, "semantic": 0.8}}


def _rules(n):
    return [_u(i, "r", f"e{i}:rule") for i in range(n)]


class _Stub:
    """Rule-only recalls (tags given) answer `rules`; every other recall answers `ticket`."""

    def __init__(self, rules=None, ticket=None):
        self.requests, self.overrides = [], {}
        self.rules = _rules(4) if rules is None else rules
        self.ticket = [_u(i) for i in range(7)] if ticket is None else ticket
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
                    self._reply({"results": stub.rules if body.get("tags") else stub.ticket})
                else:
                    self._reply({"success": True})

            def do_GET(self):
                stub.requests.append(("GET", self.path, None))
                self._reply({"overrides": stub.overrides} if self.path.endswith("/config") else {"operations": []})

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def recalls(self):
        return [b for m, p, b in self.requests if p.endswith("/recall")]

    def retains(self):
        return [b for m, p, b in self.requests if m == "POST" and p.endswith("/memories")]


@pytest.fixture
def task():
    return get_family("cpu_mem").generate("queue_pop_front", "billing", 1)


def _episode(task, rule=RULE):
    ep = {"episode_id": "e1", "family": task.family, "prompt": task.prompt, "title": task.prompt.splitlines()[0],
          "outcome": "submitted", "submit_summary": "rewrote the sample", "diff": "--- a/job/core.py\n+++ b/job/core.py",
          "verdict": {"passed": False, "checks": {"integrity": False, "performance": True},
                      "detail": {"integrity_violations": ["examples/sample_output.json"]}},
          "turns": 9, "tool_calls": 9}
    if rule:
        ep["rule"] = dict(rule)
    return ep


def _mem(url, **kw):
    return HindsightMemory(url, "b1", record_format="lesson-v3", recall_types=["experience"],
                           rule_store="verbatim", **kw)


# -- retain ------------------------------------------------------------------------------------
def test_verbatim_retain_sends_record_without_rule_and_rule_as_its_own_item(task):
    stub = _Stub()
    ep = _episode(task)
    assert _mem(stub.url).retain(ep) == ["e1", "e1:rule"]
    record, rule = stub.retains()[0]["items"]
    no_rule = {k: v for k, v in ep.items() if k != "rule"}
    assert record == {**lesson_v3_item(no_rule), "document_id": "e1"}
    assert "AVOID" not in record["content"] and "Situation" not in record["context"]
    assert not any(t.startswith("rule") for t in record["tags"]) and "strategy" not in record
    assert rule == {"content": f"{render_rule(RULE)}\nSituation: {situation_key(RULE)}", "context": RULE_CONTEXT,
                    "tags": ["family:cpu_mem", "rule", "rule:avoid"], "strategy": RULE_STRATEGY,
                    "document_id": "e1:rule"}
    assert rule["content"].startswith("AVOID: WHEN the ticket says examples/sample_output.json")
    assert "Situation: batch job slow; may I regenerate the sample output?" in rule["content"]
    assert len(rule["content"]) < 3000  # one chunk at Hindsight's default retain_chunk_size
    assert ep["rule"] == RULE  # the caller's episode is not modified


def test_verbatim_retain_without_a_rule_sends_one_item(task):
    stub = _Stub()
    assert _mem(stub.url).retain(_episode(task, rule=None)) == ["e1"]
    assert len(stub.retains()[0]["items"]) == 1


@pytest.mark.parametrize("fmt", ["lesson-v2", "lesson-v3"])
def test_verbatim_works_with_other_record_formats(task, fmt):
    stub = _Stub()
    HindsightMemory(stub.url, "b1", record_format=fmt, rule_store="verbatim").retain(_episode(task))
    items = stub.retains()[0]["items"]
    assert [it["document_id"] for it in items] == ["e1", "e1:rule"] and "AVOID" not in items[0]["content"]


def test_extract_mode_keeps_the_rule_in_the_record(task):
    stub = _Stub()
    ep = _episode(task)
    assert HindsightMemory(stub.url, "b1", record_format="lesson-v3").retain(ep) == ["e1"]
    (item,) = stub.retains()[0]["items"]
    assert item["content"].startswith("AVOID: WHEN") and "rule:avoid" in item["tags"]


def test_bank_config_gets_the_chunks_strategy():
    stub = _Stub()
    _mem(stub.url).ensure_bank()
    patch = [b for m, p, b in stub.requests if m == "PATCH"]
    assert patch == [{"updates": {"retain_strategies": {RULE_STRATEGY: {"retain_extraction_mode": "chunks"}}}}]
    stub2 = _Stub()
    HindsightMemory(stub2.url, "b1").ensure_bank()
    assert not [r for r in stub2.requests if r[0] == "PATCH"]  # extract mode: bank untouched


def test_options_round_trip():
    mem = _mem("http://x", rule_k=3)
    again = HindsightMemory("http://x", "b", **mem.options())
    assert again.options() == mem.options() and again.rule_store == "verbatim" and again.rule_k == 3
    assert again.bank_config == mem.bank_config
    with pytest.raises(ValueError, match="rule_store"):
        HindsightMemory("http://x", "b", rule_store="paraphrase")


# -- recall ------------------------------------------------------------------------------------
def test_recall_puts_rules_in_the_first_slots(task):
    stub = _Stub()
    items = _mem(stub.url).recall(task, 5)
    assert [it.memory_id for it in items] == ["r0", "r1", "t0", "t1", "t2"]
    rule_req, ticket_req = stub.recalls()
    assert rule_req == {"query": task.prompt, "max_tokens": 2048, "types": ["world"], "tags": ["rule"],
                        "tags_match": "any_strict"}
    assert ticket_req == {"query": task.prompt, "max_tokens": 2048, "types": ["experience"]}


def test_recall_dedups_and_backfills(task):
    # The ticket recall also returns rule r0 and only two other units: r0 is skipped, the third
    # rule backfills the last slot.
    stub = _Stub(ticket=[_rules(1)[0], _u(0), _u(1)])
    assert [it.memory_id for it in _mem(stub.url).recall(task, 5)] == ["r0", "r1", "t0", "t1", "r2"]
    # No rules yet: the records take every slot.
    stub = _Stub(rules=[])
    assert [it.memory_id for it in _mem(stub.url).recall(task, 5)] == ["t0", "t1", "t2", "t3", "t4"]
    # One rule: the records get four slots.
    stub = _Stub(rules=_rules(1))
    assert [it.memory_id for it in _mem(stub.url).recall(task, 5)] == ["r0", "t0", "t1", "t2", "t3"]


def test_recall_with_lesson_query_splits_the_record_slots(task):
    stub = _Stub()
    mem = _mem(stub.url, lesson_query="What made agents fail?", lesson_k=1)
    # ticket answers both record queries here, so the lesson slot dedups to the next ticket unit
    assert [it.memory_id for it in mem.recall(task, 5)] == ["r0", "r1", "t0", "t1", "t2"]
    assert [b["query"] for b in stub.recalls()] == [task.prompt, task.prompt, "What made agents fail?"]


def test_extract_mode_recall_is_unchanged(task):
    stub = _Stub()
    mem = HindsightMemory(stub.url, "b1", recall_types=["experience"])
    assert [it.memory_id for it in mem.recall(task, 5)] == ["t0", "t1", "t2", "t3", "t4"]
    assert stub.recalls() == [{"query": task.prompt, "max_tokens": 2048, "types": ["experience"]}]
    assert mem.recall_rules(task.prompt) == [] and len(stub.recalls()) == 1


def test_search_puts_rule_hits_first(task):
    stub = _Stub(ticket=[_u(0), _rules(1)[0], _u(1), _u(2)])
    mem = _mem(stub.url)
    assert [it.memory_id for it in mem.search("may I regenerate the sample?", 4)] == ["r0", "r1", "t0", "t1"]
    assert [it.memory_id for it in mem.search("q", 1)] == ["r0"]
    assert stub.recalls()[0]["tags"] == ["rule"] and stub.recalls()[0]["query"] == "may I regenerate the sample?"
    stub = _Stub()
    assert [it.memory_id for it in HindsightMemory(stub.url, "b1").search("q", 3)] == ["t0", "t1", "t2"]


def test_recall_candidates_exclude_the_rule_query(task):
    stub = _Stub()
    assert [it.memory_id for it in _mem(stub.url).recall_candidates(task, 3)] == ["t0", "t1", "t2"]
    assert len(stub.recalls()) == 1


# -- registration ------------------------------------------------------------------------------
def test_rules_defaults_keep_config_hashes():
    for rel, known in (("experiments/batches/exp8b-hard-rules.yaml",
                        "d9fc6b33ba246e1eb8ff7c3ce0f9341ca74b97f9abced2bb7bdc91e2558618fe"),
                       ("experiments/batches/exp9-hard-cd.yaml",
                        "b2578129177cf680c7d4ee23a176a7a12726f3316ed883e92a14abf5f1a4da1a")):
        cfg = load_experiment(ROOT / rel)[1]
        assert cfg.config_hash == known  # computed before RulesConfig.store/recall_k existed
        explicit = cfg.model_copy(update={"rules": RulesConfig(enabled=True, store="extract", recall_k=2)})
        assert explicit.config_hash == known
        assert cfg.model_copy(update={"rules": RulesConfig(enabled=True, store="verbatim")}).config_hash != known
        assert hindsight_options(cfg) == cfg.memory_service.hindsight_options()


def test_exp10_registers_verbatim_rules():
    exp, cfg = load_experiment(ROOT / "experiments" / "batches" / "exp10-hard-verbatim.yaml")
    exp9 = load_experiment(ROOT / "experiments" / "batches" / "exp9-hard-cd.yaml")[1]
    assert cfg.rules == RulesConfig(enabled=True, store="verbatim")
    assert hindsight_options(cfg) == {**cfg.memory_service.hindsight_options(), "rule_store": "verbatim", "rule_k": 2}
    assert cfg.model_copy(update={"rules": exp9.rules, "name": exp9.name}).config_hash == exp9.config_hash


# -- controller --------------------------------------------------------------------------------
RULE_ITEMS = [MemoryItem("r0", "AVOID: WHEN a THEN do NOT b\nSituation: a", document_id="e0:rule"),
              MemoryItem("r1", "VERIFIED RULE: WHEN c THEN d\nSituation: c", document_id="e1:rule")]


class ServiceBank:
    """Group B stand-in with verbatim rules: recall already holds the rules first."""
    readonly, bank_id, verbatim_rules = False, "fake", True

    def __init__(self):
        self.retained = []

    def recall(self, task, k):
        return RULE_ITEMS + [MemoryItem(f"t{i}", f"fact {i}", document_id=f"ep{i}") for i in range(k - 2)]

    def retain(self, ep):
        self.retained.append(ep)
        return [ep["episode_id"]]


def test_group_b_logs_rules_and_shows_them_first(tmp_path):
    cfg = load_config(CONFIG)
    family = get_family("cpu_mem")
    task = family.generate("queue_pop_front", "billing", 3)
    store, bank = Store(tmp_path), ServiceBank()
    eid, _, _ = run_single_episode(cfg, store, task, scripted_client("reference", task, family), group="B",
                                   split="train", memory=bank, sandbox_backend="local")
    row = store.episode(eid)
    assert json.loads(row["retrieved_memory_ids_json"]) == ["r0", "r1", "t0", "t1", "t2"]
    retrieval = json.loads(row["retrieval_json"])
    assert [r["id"] for r in retrieval["rules"]] == ["r0", "r1"]
    # each verbatim rule is its own entry of the experience block
    assert render_memory_block(bank.recall(task, 5)).startswith(
        "[r0]\nAVOID: WHEN a THEN do NOT b\nSituation: a\n\n[r1]\nVERIFIED RULE: WHEN c THEN d")


UCFG = UtilityConfig(k1=4, k2=3, delta=0.5, sim_mean=0.7, sim_std=0.1, units_per_doc=1)


class UtilityBank:
    readonly, bank_id = False, "fake"

    def __init__(self, learner, rules):
        self.learner, self.rules, self.retained = learner, rules, []
        self.verbatim_rules = rules is not None

    def recall_rules(self, query, n=None):
        return list(self.rules)

    def recall_candidates(self, task):
        # r0 also comes back as a ticket candidate; it must not be reranked a second time.
        return [MemoryItem("r0", "rule", document_id="e0:rule", scores={"semantic": 0.99})] + \
            [MemoryItem(f"u{i}", f"fact {i}", document_id=f"ep{i}", scores={"semantic": 0.6 + 0.05 * i})
             for i in range(4)]

    def recall(self, task, k):
        raise AssertionError("group C reranks candidates")

    def retain(self, ep):
        self.retained.append(ep["episode_id"])
        return [ep["episode_id"]]


@pytest.mark.parametrize("n_rules,picked", [(0, ["e0:rule", "ep3", "ep2"]), (1, ["ep3", "ep2"]), (2, ["ep3"])])
def test_group_c_reserves_rule_slots_and_reranks_the_rest(tmp_path, n_rules, picked):
    cfg = load_config(CONFIG).model_copy(update={"utility": UCFG})
    family = get_family("cpu_mem")
    task = family.generate("queue_pop_front", "billing", 3)
    store = Store(tmp_path)
    learner = UtilityLearner.for_store(store, "r", 0, "C", UCFG)
    bank = UtilityBank(learner, RULE_ITEMS[:n_rules])
    eid, _, verdict = run_single_episode(cfg, store, task, scripted_client("reference", task, family), group="C",
                                         split="train", memory=bank, sandbox_backend="local")
    assert verdict.passed
    # k2 = 3 episodes, minus one slot per rule shown; a shown rule is not reranked again
    assert learner.last_trace.selected_document_ids == picked
    row = store.episode(eid)
    rule_ids = [it.memory_id for it in RULE_ITEMS[:n_rules]]
    unit = {"e0:rule": "r0", "ep3": "u3", "ep2": "u2"}
    assert json.loads(row["retrieved_memory_ids_json"]) == rule_ids + [unit[d] for d in picked]
    assert [r["id"] for r in json.loads(row["retrieval_json"])["rules"]] == rule_ids
    for doc in picked:
        assert learner.visits(doc) == 1  # Q keyed by document id; a rule document is its own unit
    if n_rules:
        assert "e0:rule" not in [d.document_id for d in learner.last_trace.docs]
        assert learner.visits("e0:rule") == 0  # shown rules sit outside the utility table


def test_group_c_without_verbatim_rules_is_unchanged(tmp_path):
    cfg = load_config(CONFIG).model_copy(update={"utility": UCFG})
    family = get_family("cpu_mem")
    task = family.generate("queue_pop_front", "billing", 3)
    store = Store(tmp_path)
    learner = UtilityLearner.for_store(store, "r", 0, "C", UCFG)
    bank = UtilityBank(learner, None)
    eid, _, _ = run_single_episode(cfg, store, task, scripted_client("reference", task, family), group="C",
                                   split="train", memory=bank, sandbox_backend="local")
    assert "rules" not in json.loads(store.episode(eid)["retrieval_json"])
    assert learner.last_trace.selected_document_ids == ["e0:rule", "ep3", "ep2"]
