"""Learning on the Job in the controller: a train episode reflects, the rule is retained first in
the record, its calls are charged as learning, and nothing happens on read-only splits."""

import json
from pathlib import Path

from elh.controller import run_single_episode
from elh.memory.base import MemoryItem
from elh.registry import RulesConfig, load_config
from elh.runner.scripts import scripted_client
from elh.store import Store
from elh.tasks.base import get_family

CONFIG = Path(__file__).parent.parent / "experiments" / "phase0-local.yaml"


class Bank:
    readonly, bank_id = False, "fake"

    def __init__(self):
        self.retained = []

    def recall(self, task, k):
        return [MemoryItem("m1", "AVOID: WHEN x THEN y", document_id="e0")]

    def retain(self, ep):
        self.retained.append(ep)
        return [ep["episode_id"]]


def test_train_episode_writes_and_retains_a_rule(tmp_path):
    cfg = load_config(CONFIG).model_copy(update={"rules": RulesConfig(enabled=True, max_retries=0)})
    family = get_family("cpu_mem")
    task = family.generate("queue_pop_front", "billing", 3)
    client = scripted_client("reference", task, family)
    client.turns.append([("memory_create", {
        "title": "VERIFIED RULE: use a deque for front pops",
        "content": "WHEN a batch job pops from the front of a list THEN use collections.deque.popleft",
        "query": "batch job slow popping from the front of a list"})])
    store, bank = Store(tmp_path), Bank()
    eid, _, verdict = run_single_episode(cfg, store, task, client, arm="B", split="train", memory=bank,
                                         sandbox_backend="local")
    assert verdict.passed
    rule = bank.retained[0]["rule"]
    assert rule["kind"] == "verified" and "deque" in rule["then"]
    row = store.episode(eid)
    assert json.loads(row["rule_json"])["rule"]["query"].startswith("batch job")
    cats = [r[0] for r in store.db.execute("SELECT category FROM ledger WHERE episode_id = ?", (eid,))]
    assert "learning" in cats


def test_rules_off_by_default(tmp_path):
    cfg = load_config(CONFIG)
    family = get_family("cpu_mem")
    task = family.generate("queue_pop_front", "billing", 3)
    store, bank = Store(tmp_path), Bank()
    eid, _, _ = run_single_episode(cfg, store, task, scripted_client("reference", task, family), arm="B",
                                   split="train", memory=bank, sandbox_backend="local")
    assert "rule" not in bank.retained[0] and store.episode(eid)["rule_json"] is None
