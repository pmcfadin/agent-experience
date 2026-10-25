"""Arm D end to end (elh/batch.py, elh/controller.py): scripted agents, an in-memory stand-in for
Hindsight, AWM induction and GEPA, the paired dev gate, snapshot-pinned libraries, the retention
rollback hook, the batch report, and the config hashes of the registrations already in use."""

import hashlib
import json
import uuid
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from elh import controller
from elh.analysis import format_summary, summarize_batch
from elh.batch import load_experiment, print_schedule, retention_rollback, run_batch, schedule
from elh.learner import workflows as wf
from elh.memory.base import MemoryItem
from elh.memory.hindsight import HindsightMemory
from elh.registry import load_config
from elh.runner.models import InfraError, ScriptedClient
from elh.runner.scripts import (SCRIPTED_WORKFLOWS, ScriptedInductionClient, scripted_client,
                                scripted_reflection)
from elh.store import Store
from elh.tasks.base import get_family

ROOT = Path(__file__).parent.parent
BASE = ROOT / "experiments" / "phase0-local.yaml"
UTILITY = {"k1": 10, "k2": 5, "delta": 0.8533, "sim_mean": 0.8651, "sim_std": 0.01716, "units_per_doc": 1,
           "reward_mode": "spec-signed"}
FAMILY = get_family("cpu_mem")


# -- an in-memory Hindsight --------------------------------------------------------------------
class FakeHindsight(HindsightMemory):
    """HindsightMemory's contract over a dict of banks: retain stores the record, recall returns
    every record as one unit (document_id = episode id), snapshots copy the bank."""

    banks: dict[str, list[dict]] = {}

    def ensure_bank(self):
        self.banks.setdefault(self.bank_id, [])

    def exists(self):
        return self.bank_id in self.banks

    def delete_bank(self):
        self.banks.pop(self.bank_id, None)

    def recall_candidates(self, task, n=None):
        items = [MemoryItem(f"{self.bank_id}:{i}", f"{d['outcome']} on {d['title']}", document_id=d["episode_id"],
                            scores={"semantic": 0.9}) for i, d in enumerate(self.banks[self.bank_id])]
        return items if n is None else items[:n]

    def recall(self, task, k):
        return self.recall_candidates(task, k)

    def retain(self, episode):
        if self.readonly:
            raise RuntimeError(f"bank {self.bank_id} is read-only")
        self.banks[self.bank_id].append({k: episode[k] for k in ("episode_id", "title", "outcome")})
        self.last_usage = None
        return [episode["episode_id"]]

    def wait_idle(self, *a, **kw):
        pass

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.banks[self.bank_id], sort_keys=True).encode()).hexdigest()

    def export_snapshot(self):
        return json.dumps(self.banks[self.bank_id]).encode()

    def restore(self, archive, target_bank_id):
        return fake_restore(self._root, archive, target_bank_id)


def fake_restore(base_url, archive, target_bank_id, **kw):
    FakeHindsight.banks[target_bank_id] = json.loads(archive)
    return FakeHindsight(base_url, target_bank_id, readonly=True)


@pytest.fixture
def fake_hindsight(monkeypatch):
    FakeHindsight.banks = {}
    monkeypatch.setattr(controller, "HindsightMemory", FakeHindsight)
    monkeypatch.setattr(controller, "restore_bank", fake_restore)
    return FakeHindsight.banks


# -- scripted agents that react to the procedure block ----------------------------------------
class Aware:
    """Wraps a scripted agent. With procedures in its first message it spends fewer tokens (so the
    candidate library wins the gate on reward); `fail_without` tasks fail unless procedures are shown."""

    model_id = "scripted"

    def __init__(self, task, fail_without=False):
        self.task, self.fail_without, self.inner, self.saw = task, fail_without, None, None

    def create(self, *, system, messages, tools, max_tokens):
        if self.inner is None:
            first = messages[0]["content"]
            first = first if isinstance(first, str) else json.dumps(first)
            self.saw = wf.PROCEDURE_HEADER in first
            kind = "wrong" if self.fail_without and not self.saw else "reference"
            self.inner = scripted_client(kind, self.task, FAMILY)
            self.inner.fake_usage = {"input_tokens": 300, "output_tokens": 50} if self.saw else \
                {"input_tokens": 3000, "output_tokens": 500}
        return self.inner.create(system=system, messages=messages, tools=tools, max_tokens=max_tokens)


def _experiment(tmp_path, **kw) -> Path:
    doc = {"name": f"armd-{uuid.uuid4().hex[:8]}", "base_config": str(BASE), "family": "cpu_mem", "histories": 1,
           "arms": ["A", "B", "C", "D"],
           "overrides": {"utility": UTILITY, "workflows": {"induce_every": 1, "repeats": 1},
                         "memory_service.llm.props_url": ""},  # no /props fetch from scripted runs
           "phases": [{"train": {"tasks": {"causes": ["queue_pop_front"], "themes": ["billing", "logistics"],
                                           "seeds": [1000, 1001]}}},
                      {"eval": {"tasks": {"causes": ["queue_pop_front"], "themes": ["inventory"],
                                          "seeds": [2000, 2001]}}}],
           "dev": {"tasks": {"causes": ["queue_pop_front"], "themes": ["billing"], "seeds": [3000, 3001]}}}
    for k, v in kw.items():
        if k == "overrides":
            doc["overrides"] = {**doc["overrides"], **v}
        else:
            doc[k] = v
    path = tmp_path / f"{doc['name']}.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def _first_message(store, row) -> str:
    return json.loads(store.get_blob("traces", row["trace_sha256"]).split(b"\n", 1)[0])["first_message"]


def _rows(store, sql, *args):
    return store.db.execute(sql, args).fetchall()


# -- the scripted end-to-end batch -------------------------------------------------------------
def test_arm_d_batch_end_to_end(tmp_path, fake_hindsight):
    exp, cfg = load_experiment(_experiment(tmp_path))
    store = Store(tmp_path / "runs")
    clients = []

    def make_client(task):  # the logistics train ticket fails for every arm
        c = scripted_client("wrong", task, FAMILY) if task.theme == "logistics" else Aware(task)
        clients.append((task, c))
        return c

    lines: list[str] = []
    results = run_batch(exp, cfg, store, make_client, sandbox_backend="local", out=lines.append,
                        make_induction_client=ScriptedInductionClient)
    run_id = controller.run_id_for(cfg)
    assert all(r["error"] is None for r in results)

    # Induction saw only D's verified train success: one induction, from the billing episode.
    d_train = {r["theme"]: r for r in _rows(store, "SELECT * FROM episodes WHERE arm = 'D' AND split = 'train'")}
    assert d_train["billing"]["passed"] == 1 and d_train["logistics"]["passed"] == 0
    skills = _rows(store, "SELECT * FROM skills WHERE run_id = ?", run_id)
    assert {r["revision"] for r in skills} == {1}
    assert json.loads(skills[0]["source_episode_ids_json"]) == [d_train["billing"]["episode_id"]]
    assert "induction 1" in "\n".join(lines)

    # The paired gate: incumbent (no library) vs candidate, 1 dev task x 1 repeat, charged as learning.
    dev = _rows(store, "SELECT * FROM episodes WHERE split = 'dev' ORDER BY ordinal")
    assert len(dev) == 2 and {r["arm"] for r in dev} == {"D"} and {r["batch_id"] for r in dev} == {f"{exp.name}/0"}
    assert {r["seed"] for r in dev} == {3000}
    revs = sorted(json.loads(r["retrieval_json"])["procedure_revision"] or 0 for r in dev)
    assert revs == [0, 1]
    induced_after = 1 if d_train["billing"]["ordinal"] == 0 else 2
    assert {r["checkpoint"] for r in dev} == {induced_after}
    cats = {r[0] for r in store.db.execute(
        f"SELECT category FROM ledger WHERE episode_id IN ({','.join('?' * len(dev))})", [r["episode_id"] for r in dev])}
    assert cats == {"learning"}
    train_cats = {r[0] for r in store.db.execute("SELECT category FROM ledger WHERE episode_id = ?",
                                                 (d_train["billing"]["episode_id"],))}
    assert "learning" not in train_cats and "tool_compute" in train_cats

    # Promotion recorded, with the gate's evidence.
    assert wf.revision_status(store, run_id=run_id, history=0, arm="D", family="cpu_mem", revision=1) == "promoted"
    v = json.loads(skills[0]["validation_json"])
    assert v["promote"] and v["source"] == "awm" and v["after_train_episodes"] == induced_after
    assert v["candidate_mean_reward"] > v["incumbent_mean_reward"]
    assert sorted(v["incumbent_episode_ids"] + v["candidate_episode_ids"]) == sorted(r["episode_id"] for r in dev)

    # Snapshot pins revision 1; its learner state covers the utility table and the library.
    snap = store.db.execute("SELECT * FROM memory_snapshots WHERE arm = 'D'").fetchone()
    assert snap["library_revision"] == 1
    state = json.loads(store.get_blob("learner", snap["learner_state_sha256"]))
    lib1 = wf.load_library(store, run_id=run_id, history=0, arm="D", family="cpu_mem", revision=1)
    assert snap["learner_state_sha256"] == wf.arm_d_state_sha256(state["utility_state_sha256"], 1, lib1)
    assert store.get_blob("learner", state["utility_state_sha256"])  # the utility table, restorable
    c_snap = store.db.execute("SELECT * FROM memory_snapshots WHERE arm = 'C'").fetchone()
    assert c_snap["library_revision"] is None

    # D's eval episode saw the snapshot-pinned library first in its experience block; A-C saw none.
    evals = {r["arm"]: r for r in _rows(store, "SELECT * FROM episodes WHERE split = 'checkpoint'")}
    rd = json.loads(evals["D"]["retrieval_json"])
    assert rd["procedure_revision"] == 1 and rd["procedure_source"] == "snapshot"
    assert rd["procedure_ids"] == wf.procedure_ids(lib1, 1) and rd["ranker"] == "utility"
    first = _first_message(store, evals["D"])
    assert first.startswith("<experience>") and wf.render_procedure_block(lib1) in first
    assert first.index(wf.PROCEDURE_HEADER) < first.index("[elh-")  # procedures before recalled notes
    for arm in "ABC":
        assert wf.PROCEDURE_HEADER not in _first_message(store, evals[arm])
        assert "procedure_ids" not in json.loads(evals[arm]["retrieval_json"])

    # The batch report shows the library revisions, the decision and the learning cost.
    text = format_summary(summarize_batch(store, exp.name, resamples=0))
    assert "arm D procedures" in text and "promoted" in text and "dev episodes" in text
    assert "2 dev episodes" in text
    assert all(p["phase"] != "dev" for p in summarize_batch(store, exp.name, resamples=0)["episode_rows"])

    # Later promotions do not reach the snapshot: a new revision is promoted, the snapshot still pins 1.
    ind = wf.Induction(wf.assemble_library("cpu_mem", SCRIPTED_WORKFLOWS.replace("Verify the fix", "Check it"),
                                           max_library_tokens=10_000)[0], "x", "p", ["e"], {}, "m", 0, [])
    r2 = wf.record_candidate(store, ind, run_id=run_id, history=0)
    wf.decide_candidate(store, run_id=run_id, history=0, arm="D", family="cpu_mem", revision=r2,
                        decision=wf.GateDecision(True, ("test",), {}))
    mem = controller.open_snapshot(cfg, store, snap["snapshot_id"])
    assert mem.procedures[0] == 1 and mem.procedures[1] == lib1
    task = FAMILY.generate("queue_pop_front", "inventory", 2005)
    eid, _, _ = controller.run_single_episode(cfg, store, task, Aware(task), arm="D", split="checkpoint",
                                              checkpoint=2, memory=mem, sandbox_backend="local")
    assert json.loads(store.episode(eid)["retrieval_json"])["procedure_revision"] == 1
    assert wf.promoted_revision(store, run_id=run_id, history=0, arm="D", family="cpu_mem") == r2

    # A tampered pinned library is refused when the snapshot is opened.
    store.db.execute("UPDATE skills SET body = body || ' ' WHERE revision = 1 AND position = 0")
    store.db.commit()
    with pytest.raises((ValueError, InfraError)):
        controller.open_snapshot(cfg, store, snap["snapshot_id"])


def test_arm_d_with_gepa(tmp_path, fake_hindsight):
    """GEPA seeded with the (empty) incumbent, searching on the dev panel's reflection/validation
    pools; the winner is gated by workflows.promote on the promotion pool."""
    path = _experiment(tmp_path, arms=["A", "D"],
                       overrides={"procedure_source": "gepa",
                                  "gepa": {"max_metric_calls": 4, "reflection_minibatch_size": 1}},
                       phases=[{"train": {"tasks": {"causes": ["queue_pop_front"], "themes": ["billing"],
                                                    "seeds": [1000, 1001]}}},
                               {"eval": {"tasks": {"causes": ["queue_pop_front"], "themes": ["inventory"],
                                                   "seeds": [2000, 2001]}}}],
                       dev={"tasks": {"causes": ["queue_pop_front"], "themes": ["billing"], "seeds": [3000, 3003]}})
    exp, cfg = load_experiment(path)
    store = Store(tmp_path / "runs")
    lines: list[str] = []
    # Dev tickets fail without a procedure and pass with one; train and eval tickets always pass.
    run_batch(exp, cfg, store, lambda task: Aware(task, fail_without=task.seed >= 3000), sandbox_backend="local",
              out=lines.append, make_reflection_client=lambda: scripted_reflection)
    run_id = controller.run_id_for(cfg)
    skills = _rows(store, "SELECT * FROM skills WHERE run_id = ?", run_id)
    assert len(skills) == 1 and skills[0]["title"] == wf.GEPA_TITLE and skills[0]["status"] == "promoted"
    prov = json.loads(skills[0]["induction_json"])
    assert prov["source"] == "gepa" and prov["gepa"]["best_idx"] >= 1
    v = json.loads(skills[0]["validation_json"])
    pools = prov["gepa"]["pools"]
    assert v["gate_task_ids"] == pools["promotion"] and v["candidate_passes"] == 1 and v["incumbent_passes"] == 0
    gepa_eps = [r for r in _rows(store, "SELECT * FROM episodes WHERE split = 'dev'")
                if json.loads(r["retrieval_json"])["procedure_source"] == "explicit"]
    assert len(gepa_eps) >= 4
    cats = {r[0] for r in store.db.execute("SELECT DISTINCT l.category FROM ledger l JOIN episodes e "
                                           "USING (episode_id) WHERE e.split = 'dev'")}
    assert cats == {"learning"}
    lib = wf.load_library(store, run_id=run_id, history=0, arm="D", family="cpu_mem", revision=1)
    assert wf.procedure_text(lib).startswith("1. Reproduce the slowness")
    ev = store.db.execute("SELECT * FROM episodes WHERE split = 'checkpoint' AND arm = 'D'").fetchone()
    assert json.loads(ev["retrieval_json"])["procedure_revision"] == 1


# -- experiment-file validation and dry run ----------------------------------------------------
def test_dev_panel_rules(tmp_path):
    dev = lambda themes, seeds=(3000, 3002): {"tasks": {"causes": ["queue_pop_front"], "themes": themes,
                                                         "seeds": list(seeds)}}
    with pytest.raises(ValueError, match="only use train cells"):
        load_experiment(_experiment(tmp_path, dev=dev(["telemetry"])))
    with pytest.raises(ValueError, match="eval/retention phase measures"):
        load_experiment(_experiment(tmp_path, dev=dev(["billing"]), phases=[
            {"train": {"tasks": {"causes": ["queue_pop_front"], "themes": ["billing"], "seeds": [1000, 1001]}}},
            {"eval": {"tasks": {"causes": ["queue_pop_front"], "themes": ["billing"], "seeds": [2000, 2001]}}}]))
    with pytest.raises(ValueError, match="dev seeds"):
        load_experiment(_experiment(tmp_path, dev=dev(["billing"], (2000, 2002))))
    with pytest.raises(ValueError, match="needs a `dev` section"):
        load_experiment(_experiment(tmp_path, dev=None))
    with pytest.raises(ValidationError, match="arm D needs"):
        load_experiment(_experiment(tmp_path, overrides={"workflows": None}))
    with pytest.raises(ValidationError, match="gepa"):
        load_experiment(_experiment(tmp_path, overrides={"procedure_source": "gepa"}))
    with pytest.raises(ValueError, match="retention phases may only use train cells"):
        load_experiment(_experiment(tmp_path, phases=[
            {"train": {"tasks": {"causes": ["queue_pop_front"], "themes": ["billing"], "seeds": [1000, 1001]}}},
            {"retention": {"tasks": {"causes": ["queue_pop_front"], "themes": ["inventory"], "seeds": [1500, 1501]}}}]))


def test_exp7_dry_run_shows_dev_panel_and_induction_points(capsys):
    exp, cfg = load_experiment(ROOT / "experiments" / "batches" / "exp7-arm-d.yaml")
    steps = schedule(exp, cfg)
    assert len(steps) == 15 * 3 + 10 * 4
    print_schedule(exp, cfg, steps)
    out = capsys.readouterr().out
    assert "dev panel (6 tasks" in out and "cpu_mem/queue_pop_front/telemetry/3001" in out
    assert "induction points after train episodes [5, 10, 15]" in out
    assert "24 dev episodes per induction; up to 72 in all" in out
    assert cfg.utility == load_experiment(ROOT / "experiments" / "batches" / "exp6-papers-bc.yaml")[1].utility


# Registrations already in use: adding arm D's fields must not move their hashes (run ids, banks).
KNOWN_HASHES = {
    "experiments/phase0-local.yaml": "5390cd5fb2fcddb59e438125765deb3a5efed7a81284a3f85c484360094d7e0f",
    "experiments/phase0-sonnet.yaml": "a55fad880d8708ceda2914d41f17b42d979bc23b46556e76302cd5858226dc6b",
    "experiments/batches/exp1-memory-helps.yaml": "10112455fc8163a41e2d25dcd729f7eed36f376afe56e26e67f4c32bdbc7a181",
    "experiments/batches/exp2-transfer.yaml": "53f9703a5a26e72cb7efa3fe75705bf4c9ed767db676d9e80af972c10e100d42",
    "experiments/batches/exp3-own-mistakes.yaml": "44963a0766e3312fa68af3b251efb54ac2552344e6e2512afbf3999e5c450177",
    "experiments/batches/exp4-record-format.yaml": "1dd1351e8725c84b151e1e3f5a596160a9085b11aecb18a0b4646292e86418e6",
    "experiments/batches/exp5-dose.yaml": "a6c22d611cbb97d2774eb0548f6b1505216bbe178f9af9e301be84a24765e7ad",
    "experiments/batches/exp6-papers-bc.yaml": "749350de06672d88c324703eeb2b861c847b23d7822cc3d8813c2317a9add792",
    # pinned when RulesConfig.store / recall_k were added (values from before that change)
    "experiments/batches/exp7-arm-d.yaml": "1e8a415510bd70e87e4ca13337d914b34b0fa8c051192d1005c50dd2185bc343",
    "experiments/batches/exp8a-hard-memory.yaml": "45ef717de862efdf36b705319fe3c5e3a35c9454d1c507bd20a973e9e2381e4e",
    "experiments/batches/exp8b-hard-rules.yaml": "d9fc6b33ba246e1eb8ff7c3ce0f9341ca74b97f9abced2bb7bdc91e2558618fe",
    "experiments/batches/exp9-hard-cd.yaml": "b2578129177cf680c7d4ee23a176a7a12726f3316ed883e92a14abf5f1a4da1a",
    "experiments/batches/probe-difficulty.yaml": "a8051420eea2169d474d9583b152fa50d7e5500060bd70c3874e0a423102eb64",
}


@pytest.mark.parametrize("rel", sorted(KNOWN_HASHES))
def test_existing_config_hashes_unchanged(rel):
    path = ROOT / rel
    cfg = load_experiment(path)[1] if "batches" in rel else load_config(path)
    assert cfg.config_hash == KNOWN_HASHES[rel]


# -- retention rollback hook -------------------------------------------------------------------
def _retention_row(store, run_id, ck, task_id, passed, revision):
    store.insert("episodes", {
        "episode_id": uuid.uuid4().hex, "run_id": run_id, "history": 0, "arm": "D", "split": "retention",
        "checkpoint": ck, "task_id": task_id, "family": "cpu_mem", "outcome": "submitted", "passed": int(passed),
        "verdict_json": json.dumps({"passed": passed, "checks": {"integrity": True}}), "input_tokens": 1000,
        "output_tokens": 100, "cache_read_tokens": 0, "cache_write_tokens": 0, "batch_id": "x/0",
        "retrieval_json": json.dumps({"procedure_revision": revision}), "started_at": f"2026-10-25T00:{ck:02d}:00"})


def _promote(store, run_id, text):
    lib = wf.assemble_library("cpu_mem", text, max_library_tokens=10_000)[0]
    rev = wf.record_candidate(store, wf.Induction(lib, text, "p", ["e"], {}, "m", 0, []), run_id=run_id, history=0)
    wf.decide_candidate(store, run_id=run_id, history=0, arm="D", family="cpu_mem", revision=rev,
                        decision=wf.GateDecision(True, ("test",), {}))
    return rev


def test_retention_rollback_hook(tmp_path):
    cfg = load_config(BASE).model_copy(update={"workflows": wf.WorkflowConfig(rollback_margin=0.1)})
    store, run_id = Store(tmp_path), "r"
    scope = dict(run_id=run_id, history=0, family="cpu_mem", batch="x/0")
    r1 = _promote(store, run_id, SCRIPTED_WORKFLOWS)
    for t in ("t1", "t2"):
        _retention_row(store, run_id, 5, t, True, r1)
    assert retention_rollback(store, cfg, checkpoint=5, **scope) is None  # no earlier panel
    r2 = _promote(store, run_id, SCRIPTED_WORKFLOWS.replace("Verify the fix", "Check it"))
    _retention_row(store, run_id, 10, "t1", True, r2)
    _retention_row(store, run_id, 10, "t2", False, r2)
    d = retention_rollback(store, cfg, checkpoint=10, **scope)
    assert d.promote and "rolled back" in d.reasons[-1] and d.stats["active_revision"] == r1
    assert wf.revision_status(store, run_id=run_id, history=0, arm="D", family="cpu_mem", revision=r2) == "rolled_back"
    assert wf.promoted_revision(store, run_id=run_id, history=0, arm="D", family="cpu_mem") == r1
    for t in ("t1", "t2"):  # back on revision 1: the panel recovers, nothing to roll back
        _retention_row(store, run_id, 15, t, True, r1)
    d = retention_rollback(store, cfg, checkpoint=15, **scope)
    assert not d.promote and d.reasons == ("retention held",)
    # A drop with the same library at both checkpoints is noise, not a bad promotion: no rollback.
    for t in ("t1", "t2"):
        _retention_row(store, run_id, 20, t, False, r1)
    d = retention_rollback(store, cfg, checkpoint=20, **scope)
    assert not d.promote and "did not change" in d.reasons[-1]
    assert wf.promoted_revision(store, run_id=run_id, history=0, arm="D", family="cpu_mem") == r1


def test_run_single_episode_guards(tmp_path, fake_hindsight):
    exp, cfg = load_experiment(_experiment(tmp_path))
    store = Store(tmp_path / "runs")
    task = FAMILY.generate("queue_pop_front", "billing", 1000)
    mem = controller.memory_for(cfg, "D", 0, "train", store=store)
    with pytest.raises(ValueError, match="charge_as"):
        controller.run_single_episode(cfg, store, task, Aware(task), arm="D", split="train", memory=mem,
                                      sandbox_backend="local", charge_as="eval_overhead")
    with pytest.raises(ValueError, match="dev validation"):
        controller.run_single_episode(cfg, store, task, Aware(task), arm="D", split="train", memory=mem,
                                      sandbox_backend="local", procedures=(None, wf.WorkflowLibrary("cpu_mem")))
    with pytest.raises(ValueError, match="workflows"):
        controller.memory_for(cfg.model_copy(update={"workflows": None}), "D", 0, "train", store=store)
    # A dev episode on the live bank leaves bank, learner and library as they were.
    view = controller.memory_for(cfg, "D", 0, "dev", store=store)
    eid, _, _ = controller.run_single_episode(cfg, store, task, ScriptedClient([[("submit", {"summary": "x"})]]),
                                              arm="D", split="dev", memory=view, sandbox_backend="local",
                                              charge_as="learning")
    r = json.loads(store.episode(eid)["retrieval_json"])
    assert r["procedure_revision"] is None and r["procedure_ids"] == [] and r["procedure_source"] == "promoted"
