"""Arm C UtilityLearner (MemRL two-phase retrieval, C-SPEC variant). No server needed."""

import math

import pytest
from pydantic import ValidationError

from elh.learner.utility import (
    ReadOnlyLearnerError, UtilityConfig, UtilityLearner, doc_similarities, estimate_similarity_stats,
    group_by_document, normalised_cost, reward,
)
from elh.memory.base import MemoryItem
from elh.store import Store


def unit(uid, doc, sem):
    return MemoryItem(memory_id=uid, text=f"text {uid}", document_id=doc,
                      scores={"semantic": sem, "final": 0.5})


CANDS = [
    unit("u1", "A", 0.84),
    unit("u2", "A", 0.81),
    unit("u3", "B", 0.83),
    unit("u4", "C", 0.85),
    unit("u5", "D", 0.79),    # below delta
    unit("u6", "E", 0.805),   # passes delta, 4th by sim -> beyond k1
    unit("u7", None, 0.86),   # observation: no document
    unit("u8", "F", None),    # semantic arm did not surface it
]

CFG = UtilityConfig(k1=3, k2=2, delta=0.80, sim_mean=0.82, sim_std=0.02)


class Task:
    content_hash = "t1"


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path)


def learner(store, cfg=CFG, arm="C", history=0):
    return UtilityLearner.for_store(store, "run1", history, arm, cfg)


def test_hand_computed_example(store):
    L = learner(store)
    L.observe(["A"], 1.0)    # 0 + 0.3 (1 - 0)      = 0.3
    L.observe(["A"], 1.0)    # 0.3 + 0.3 (1 - 0.3)  = 0.51
    L.observe(["B"], -1.0)   # 0 + 0.3 (-1 - 0)     = -0.3
    assert L.q("A") == pytest.approx(0.51) and L.q("B") == pytest.approx(-0.3)
    assert L.q("C") == 0.0 and L.visits("A") == 2 and L.visits("C") == 0

    items = L.rank(Task(), CANDS)
    t = L.last_trace
    # Phase A: D below delta, F unscored, E fourth by sim; survivors by sim C .85, A .84, B .83.
    assert t.phase1_document_ids == ["C", "A", "B"]
    status = {r.document_id: r.status for r in t.docs}
    assert status == {"A": "selected", "B": "not_selected", "C": "selected", "D": "below_delta",
                      "E": "beyond_k1", "F": "no_sim"}
    assert t.unattributed_ids == ["u7"]
    # Phase B by hand: Q = (.51, -.3, 0); mean .07; pstd sqrt(((.44)^2 + (.37)^2 + (.07)^2) / 3) = .334365
    assert t.q_mean == pytest.approx(0.07)
    assert t.q_std == pytest.approx(0.334365, abs=1e-6)
    r = {x.document_id: x for x in t.docs}
    assert r["A"].sim_z == pytest.approx(1.0) and r["B"].sim_z == pytest.approx(0.5) and r["C"].sim_z == pytest.approx(1.5)
    assert r["A"].q_z == pytest.approx(1.315927, abs=1e-5)
    assert r["B"].q_z == pytest.approx(-1.106575, abs=1e-5)
    assert r["C"].q_z == pytest.approx(-0.209352, abs=1e-5)
    assert r["A"].score == pytest.approx(1.157964, abs=1e-5)
    assert r["C"].score == pytest.approx(0.645324, abs=1e-5)
    assert r["B"].score == pytest.approx(-0.303288, abs=1e-5)
    # Q lifts A over C, which pure similarity would put first.
    assert t.ranked_document_ids == ["A", "C", "B"]
    assert t.selected_document_ids == ["A", "C"]
    assert [i.memory_id for i in items] == ["u1", "u2", "u4"] == t.injected_ids
    assert t.candidate_ids == [c.memory_id for c in CANDS]


def test_equal_q_ranks_by_similarity(store):
    items = learner(store).rank(Task(), CANDS)  # all Q = q_init -> q_z = 0
    assert [i.memory_id for i in items] == ["u4", "u1", "u2"]
    assert all(r.q_z == 0 for r in learner(store).rank_with_trace(Task(), CANDS)[1].docs if r.q_z is not None)


def test_q_z_is_clipped(store):
    cfg = CFG.model_copy(update={"k1": 11, "k2": 11, "delta": 0.0})
    L = learner(store, cfg)
    L.observe(["d0"], 1.0)
    cands = [unit(f"u{i}", f"d{i}", 0.82) for i in range(11)]  # one outlier among 11: z = sqrt(10) > 3
    t = L.rank_with_trace(Task(), cands)[1]
    zs = {r.document_id: r.q_z for r in t.docs}
    assert zs["d0"] == 3.0
    assert zs["d1"] == pytest.approx(-1 / math.sqrt(10))


def test_gate_strict_vs_inclusive(store):
    cands = [unit("x", "X", 0.80), unit("y", "Y", 0.81)]
    assert learner(store).rank_with_trace(Task(), cands)[1].phase1_document_ids == ["Y"]
    inc = CFG.model_copy(update={"gate_inclusive": True})
    assert learner(store, inc).rank_with_trace(Task(), cands)[1].phase1_document_ids == ["Y", "X"]


def test_grouping_and_units_per_doc(store):
    docs, unattributed = group_by_document(CANDS)
    assert [d.document_id for d in docs] == ["A", "B", "C", "D", "E", "F"]
    assert [u.memory_id for u in docs[0].units] == ["u1", "u2"] and docs[0].sim == 0.84
    assert docs[-1].sim is None and unattributed == ["u7"]
    mean_docs, _ = group_by_document(CANDS, agg="mean")
    assert mean_docs[0].sim == pytest.approx(0.825)
    assert doc_similarities(CANDS, CFG) == [0.84, 0.83, 0.85, 0.79, 0.805]
    one = CFG.model_copy(update={"units_per_doc": 1})
    assert [i.memory_id for i in learner(store, one).rank(Task(), CANDS)] == ["u4", "u1"]


def test_q_floor_and_q_min(store):
    L = learner(store, CFG.model_copy(update={"q_floor": -0.1}))
    L.observe(["A"], -1.0)
    assert L.q("A") == pytest.approx(-0.1)
    L2 = learner(store, CFG.model_copy(update={"q_min": -0.2}), arm="C2")
    L2.observe(["C"], -1.0)
    t = L2.rank_with_trace(Task(), CANDS)[1]
    assert {r.document_id: r.status for r in t.docs}["C"] == "below_q_min"
    assert t.selected_document_ids == ["A", "B"]


def test_observe_dedupes_persists_and_isolates(store):
    L = learner(store)
    out = L.observe(["A", "A", "B"], 0.5, episode_id="e1")
    assert out == {"A": pytest.approx(0.15), "B": pytest.approx(0.15)}
    assert L.visits("A") == 1
    again = learner(store)
    assert again.table() == L.table() and again.state_sha256() == L.state_sha256()
    assert learner(store, arm="D").table() == {}
    assert learner(store, history=1).table() == {}
    log = store.db.execute("SELECT document_id, q_before, q_after FROM learner_utility_updates "
                           "WHERE episode_id = 'e1' ORDER BY document_id").fetchall()
    assert [tuple(r) for r in log] == [("A", 0.0, pytest.approx(0.15)), ("B", 0.0, pytest.approx(0.15))]


def test_hash_stability(store, tmp_path):
    L = learner(store)
    empty = L.state_sha256()
    L.observe(["A"], 1.0)
    L.observe(["B"], -1.0)
    h = L.state_sha256()
    assert h == L.state_sha256() != empty
    other = UtilityLearner.for_store(Store(tmp_path / "other"), "run1", 0, "C", CFG)
    other.observe(["B"], -1.0)
    other.observe(["A"], 1.0)
    assert other.state_sha256() == h                   # insertion order does not matter
    L.rank(Task(), CANDS)
    assert L.state_sha256() == h                       # ranking never changes state
    assert learner(store, CFG.model_copy(update={"alpha": 0.2})).state_sha256() != h  # config is hashed


def test_snapshot_is_readonly(store):
    L = learner(store)
    L.observe(["A"], 1.0)
    sha = L.snapshot(store)
    assert sha == L.state_sha256()
    snap = UtilityLearner.load_snapshot(store, sha, config=CFG)
    assert snap.readonly and snap.state_sha256() == sha and snap.q("A") == pytest.approx(0.3)
    L.observe(["A"], 1.0)                               # training continues; the snapshot is frozen
    assert snap.q("A") == pytest.approx(0.3)
    items = snap.rank(Task(), CANDS)
    assert items and snap.last_trace.readonly and snap.state_sha256() == sha
    with pytest.raises(ReadOnlyLearnerError):
        snap.observe(["A"], 1.0)
    view = L.readonly_view()
    with pytest.raises(ReadOnlyLearnerError):
        view.observe(["A"], 1.0)
    assert view.state_sha256() == L.state_sha256()
    with pytest.raises(ValueError):
        UtilityLearner.load_snapshot(store, sha, config=CFG.model_copy(update={"k2": 1}))


def test_epsilon_is_seeded(store):
    cfg = CFG.model_copy(update={"epsilon": 1.0, "k1": 4, "k2": 2})
    a = learner(store, cfg).rank_with_trace(Task(), CANDS)[1]
    b = learner(store, cfg).rank_with_trace(Task(), CANDS)[1]
    assert a.epsilon_fired and a.selected_document_ids == b.selected_document_ids
    assert len(a.selected_document_ids) == 2


def test_reward_modes():
    assert reward(True, 0.25, "spec") == 0.75
    assert reward(False, 0.25, "spec") == 0.0
    assert reward(True, 0.25, "memrl") == 1.0
    assert reward(False, 0.25, "memrl") == -1.0
    assert reward(True, 0.25) == 0.75                  # default spec-signed
    assert reward(False, 0.25) == -1.0
    with pytest.raises(ValueError):
        reward(True, 1.5)
    with pytest.raises(ValueError):
        reward(True, 0.5, "other")


def test_normalised_cost():
    assert normalised_cost(0.5, 2.0) == 0.25
    assert normalised_cost(3.0, 2.0) == 1.0
    assert normalised_cost(-1.0, 2.0) == 0.0
    with pytest.raises(ValueError):
        normalised_cost(1.0, 0.0)


def test_spec_reward_never_demotes(store):
    """The reason spec-signed is the default: under "spec" a failure-linked memory stays at Q_init."""
    L = learner(store)
    for _ in range(5):
        L.observe(["bad"], reward(False, 0.1, "spec"))
    assert L.q("bad") == 0.0 == L.q("never-used")
    L2 = learner(store, arm="C-signed")
    L2.observe(["bad"], reward(False, 0.1))
    assert L2.q("bad") < L2.q("never-used")


def test_observe_episode(store):
    L = learner(store)
    r, new = L.observe_episode(["A"], passed=True, episode_cost=1.0, budget_cost=4.0)
    assert r == 0.75 and new["A"] == pytest.approx(0.225)


def test_estimate_similarity_stats():
    s = estimate_similarity_stats([5, 1, 3, 2, 4])
    assert s.mean == 3 and s.std == pytest.approx(math.sqrt(2)) and s.delta == pytest.approx(4.2) and s.n == 5
    cfg = UtilityConfig(delta=s.delta, sim_mean=s.mean, sim_std=s.std)
    assert cfg.lam == 0.5 and cfg.alpha == 0.3 and cfg.q_init == 0.0 and cfg.epsilon == 0.0
    assert cfg.q_floor is None and cfg.q_min is None and cfg.reward_mode == "spec-signed"


def test_config_validation():
    with pytest.raises(ValidationError):
        UtilityConfig(k1=2, k2=3, delta=0.8, sim_mean=0.8, sim_std=0.1)
    with pytest.raises(ValidationError):
        UtilityConfig(delta=0.8, sim_mean=0.8)        # sim_std must be registered
