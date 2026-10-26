"""Group C in the controller: utility rerank at recall, Q update after a train episode, frozen
learner on read-only splits, learner state recorded with the episode's retrieval."""

import json
from pathlib import Path

import pytest

from elh.controller import budget_cost_usd, run_single_episode
from elh.learner.utility import UtilityConfig, UtilityLearner
from elh.memory.base import MemoryItem
from elh.registry import load_config
from elh.runner.scripts import scripted_client
from elh.store import Store
from elh.tasks.base import get_family

CONFIG = Path(__file__).parent.parent / "experiments" / "phase0-local.yaml"
UCFG = UtilityConfig(k1=4, k2=2, delta=0.5, sim_mean=0.7, sim_std=0.1, units_per_doc=1)


class FakeBank:
    """Stands in for HindsightMemory: a fixed candidate list, retain recorded."""

    def __init__(self, learner, readonly=False):
        self.learner, self.readonly, self.bank_id, self.retained = learner, readonly, "fake", []

    def recall_candidates(self, task):
        return [MemoryItem(f"u{i}", f"fact {i}", document_id=f"ep{i}", scores={"semantic": 0.6 + 0.05 * i})
                for i in range(4)]

    def recall(self, task, k):
        raise AssertionError("group C must rerank candidates, not take the service's top k")

    def retain(self, episode):
        self.retained.append(episode["episode_id"])
        return [episode["episode_id"]]


@pytest.fixture
def setup(tmp_path):
    cfg = load_config(CONFIG).model_copy(update={"utility": UCFG})
    store = Store(tmp_path)
    family = get_family("cpu_mem")
    task = family.generate("queue_pop_front", "billing", 3)
    return cfg, store, family, task


def test_train_episode_reranks_then_updates_q(setup):
    cfg, store, family, task = setup
    learner = UtilityLearner.for_store(store, "r", 0, "C", UCFG)
    bank = FakeBank(learner)
    eid, _, verdict = run_single_episode(cfg, store, task, scripted_client("reference", task, family),
                                         group="C", split="train", memory=bank, sandbox_backend="local")
    assert verdict.passed and bank.retained == [eid]
    trace = learner.last_trace
    assert trace.selected_document_ids == ["ep3", "ep2"]  # all Q equal at start: similarity decides
    for doc in trace.selected_document_ids:
        assert learner.q(doc) > 0 and learner.visits(doc) == 1
    assert learner.q("ep0") == 0 and learner.visits("ep0") == 0
    row = store.episode(eid)
    assert json.loads(row["retrieved_memory_ids_json"]) == ["u3", "u2"]
    assert json.loads(row["retrieval_json"])["ranker"] == "utility"


def test_failed_train_episode_demotes(setup):
    cfg, store, family, task = setup
    learner = UtilityLearner.for_store(store, "r", 0, "C", UCFG)
    run_single_episode(cfg, store, task, scripted_client("wrong", task, family), group="C", split="train",
                       memory=FakeBank(learner), sandbox_backend="local")
    assert learner.q("ep3") < 0  # spec-signed: a failure pulls Q towards -1


def test_read_only_split_uses_frozen_learner(setup):
    cfg, store, family, task = setup
    live = UtilityLearner.for_store(store, "r", 0, "C", UCFG)
    live.observe(["ep0"], 1.0)
    view = live.readonly_view()
    before = view.state_sha256()
    eid, _, _ = run_single_episode(cfg, store, task, scripted_client("reference", task, family), group="C",
                                   split="dev", memory=FakeBank(view, readonly=True), sandbox_backend="local")
    assert view.state_sha256() == before and view.last_trace is not None
    with pytest.raises(ValueError, match="read-only view"):
        run_single_episode(cfg, store, task, scripted_client("reference", task, family), group="C",
                           split="train", memory=FakeBank(view), sandbox_backend="local")


def test_budget_cost_is_positive(setup):
    cfg, *_ = setup
    assert budget_cost_usd(cfg) == pytest.approx(600_000 * 0.05 / 1e6)
