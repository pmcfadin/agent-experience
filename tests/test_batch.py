"""Batch runner (elh/batch.py): experiment files, schedule expansion, scripted end-to-end runs."""

import json
import os
import urllib.request
import uuid
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from elh import batch
from elh.batch import load_experiment, run_batch, schedule
from elh.cli import main
from elh.controller import bank_id_for, run_id_for
from elh.memory.hindsight import HindsightMemory
from elh.runner.models import InfraError
from elh.runner.prompts import system_prompt, tools
from elh.runner.scripts import scripted_client
from elh.store import Store
from elh.tasks.base import get_family

ROOT = Path(__file__).parent.parent
BASE = ROOT / "experiments" / "phase0-local.yaml"
URL = os.environ.get("ELH_HINDSIGHT_URL", "http://127.0.0.1:8901")


def _write(tmp_path, **kw) -> Path:
    doc = {"name": f"batchtest-{uuid.uuid4().hex[:8]}", "base_config": str(BASE), "family": "cpu_mem",
           "histories": 1, "groups": ["A"],
           "phases": [{"train": {"tasks": {"causes": ["queue_pop_front"], "themes": ["billing"], "seeds": [1000, 1002]}}},
                      {"eval": {"tasks": {"causes": ["nested_scan_join"], "themes": ["inventory"], "seeds": [2000, 2002]}}}]}
    doc.update(kw)
    path = tmp_path / f"{doc['name']}.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def _client(task):
    return scripted_client("reference", task, get_family("cpu_mem"))


@pytest.mark.parametrize("path", sorted((ROOT / "experiments" / "batches").glob("*.yaml")), ids=lambda p: p.stem)
def test_bundled_experiments_load(path):
    exp, cfg = load_experiment(path)
    assert cfg.name == f"phase0-local-{exp.name}"
    steps = schedule(exp, cfg)
    assert {s.group for s in steps if s.kind == "train"} <= set(exp.groups) - set(batch.NON_LEARNING)
    assert all(s.task.seed >= 2000 if s.kind == "eval" else 1000 <= s.task.seed < 2000 for s in steps)
    assert all(3000 <= t.seed < 4000 for t in batch.dev_tasks(exp, cfg, 0))
    if "D" in exp.groups:  # a dev panel without group D is unused (experiments 11-13 kept experiment 10's)
        assert exp.dev is not None


def test_schedule_is_deterministic_and_differs_across_histories(tmp_path):
    exp, cfg = load_experiment(_write(tmp_path, histories=2, groups=["A", "B"], phases=[
        {"train": {"tasks": {"causes": ["queue_pop_front"], "themes": ["billing", "logistics"], "seeds": [1000, 1005]}}},
        {"eval": {"tasks": {"causes": ["queue_pop_front"], "themes": ["inventory"], "seeds": [2000, 2004]}}}]))
    a, b = schedule(exp, cfg), schedule(exp, cfg)
    key = lambda steps: [(s.history, s.phase, s.group, s.task.task_id, s.after, s.ordinal) for s in steps]
    assert key(a) == key(b)
    train = {h: [s.task.task_id for s in a if s.history == h and s.kind == "train"] for h in (0, 1)}
    assert len(train[0]) == 10 and sorted(train[0]) == sorted(train[1]) and train[0] != train[1]
    evals = [s for s in a if s.history == 0 and s.kind == "eval"]
    # Groups interleave per task, all at checkpoint 10 (train episodes before the phase).
    assert [s.group for s in evals] == ["A", "B"] * 4 and {s.after for s in evals} == {10}
    assert evals[0].task.task_id == evals[1].task.task_id
    assert {s.phase for s in evals} == {"after-10"}


def test_overrides_make_a_distinct_validated_run(tmp_path):
    _, plain = load_experiment(_write(tmp_path, name="ovr-plain"))
    _, cfg = load_experiment(_write(tmp_path, name="ovr-turns", overrides={"budgets.max_turns": 7}))
    assert cfg.name == "phase0-local-ovr-turns" and cfg.budgets.max_turns == 7
    assert run_id_for(cfg) != run_id_for(plain)
    with pytest.raises(ValidationError):
        load_experiment(_write(tmp_path, overrides={"memory_service.record_format": "no-such-format"}))
    with pytest.raises(KeyError):
        load_experiment(_write(tmp_path, overrides={"nosection.x": 1}))
    with pytest.raises(ValidationError):
        load_experiment(_write(tmp_path, groups=["A", "Z"]))


def test_dry_run_prints_without_touching_the_store(tmp_path, capsys):
    runs = tmp_path / "runs"
    assert main(["run-batch", str(_write(tmp_path)), "--dry-run", "--runs-dir", str(runs)]) == 0
    out = capsys.readouterr().out
    assert "cpu_mem/nested_scan_join/inventory/2001" in out and "after-2" in out
    assert "queue_pop_front" not in out  # group A has no train phase
    assert not runs.exists()


def test_a_only_batch_end_to_end(tmp_path, capsys):
    path = _write(tmp_path)
    runs = tmp_path / "runs"
    assert main(["run-batch", str(path), "--client", "scripted-reference", "--sandbox", "local",
                 "--runs-dir", str(runs)]) == 0
    out = capsys.readouterr().out
    assert out.count("passed=1") == 2 and "after-2" in out
    exp, cfg = load_experiment(path)
    store = Store(runs)
    rows = store.db.execute("SELECT * FROM episodes ORDER BY ordinal").fetchall()
    assert [r["batch_id"] for r in rows] == [f"{exp.name}/0"] * 2
    assert {(r["run_id"], r["test_group"], r["split"], r["checkpoint"]) for r in rows} == {(run_id_for(cfg), "A", "checkpoint", 2)}
    # A second run of the same experiment is refused (no resume).
    with pytest.raises(RuntimeError, match="already has"):
        run_batch(exp, cfg, store, _client, sandbox_backend="local")


def test_infra_error_skips_the_episode(tmp_path, monkeypatch):
    exp, cfg = load_experiment(_write(tmp_path))
    real = batch.run_single_episode
    calls = []

    def flaky(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise InfraError("backend went away")
        return real(*a, **kw)

    monkeypatch.setattr(batch, "run_single_episode", flaky)
    lines: list[str] = []
    results = run_batch(exp, cfg, Store(tmp_path / "runs"), _client, sandbox_backend="local", out=lines.append)
    assert [r["error"] is None for r in results] == [False, True]
    assert "INFRA ERROR" in lines[0]
    _, phase, group, n, rate, _, infra = batch.summarize(results)[1].split()
    assert (phase, group, n, rate, infra) == ("after-2", "A", "1", "1.00", "1")


def _up() -> bool:
    try:
        urllib.request.urlopen(URL + "/health", timeout=2).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _up(), reason="Hindsight server not running")
def test_group_b_trains_then_evaluates_on_a_snapshot(tmp_path):
    """One train episode (one memory-LLM retain, ~30 s), then A and B on one task."""
    exp, cfg = load_experiment(_write(tmp_path, groups=["A", "B"], phases=[
        {"train": {"tasks": {"causes": ["queue_pop_front"], "themes": ["billing"], "seeds": [1000, 1001]}}},
        {"eval": {"label": "after-1", "tasks": {"causes": ["queue_pop_front"], "themes": ["inventory"], "seeds": [2000, 2001]}}}]))
    bank = bank_id_for(cfg, 0, "B")
    store = Store(tmp_path / "runs")
    try:
        results = run_batch(exp, cfg, store, _client, sandbox_backend="local", out=lambda _: None)
        assert [(r["kind"], r["group"], r["passed"]) for r in results] == [
            ("train", "B", True), ("eval", "A", True), ("eval", "B", True)]
        rows = store.db.execute("SELECT test_group, split, checkpoint, memory_snapshot_id FROM episodes "
                                "WHERE split = 'checkpoint' ORDER BY test_group").fetchall()
        snap = store.db.execute("SELECT snapshot_id, after_episode FROM memory_snapshots").fetchone()
        assert snap["after_episode"] == 1
        assert [tuple(r) for r in rows] == [("A", "checkpoint", 1, None), ("B", "checkpoint", 1, snap["snapshot_id"])]
    finally:
        for b in (bank, f"{bank}-e1"):
            try:
                HindsightMemory(URL, b).delete_bank()
            except InfraError:
                pass


# --- static-prompt groups (P-all, A0; SPEC §6.2) -------------------------------------------------

def _source_rules(store, name: str, history: int, n: int, *, skip: int | None = None) -> None:
    """Fake a learning group's train stream: n episodes in `<name>/<history>`, one rule each."""
    store.ensure_run("src-run", "src", 1, "x", "{}", None, None)
    for i in reversed(range(n)):  # inserted out of order: the block must follow the ordinal
        rule = None if i == skip else {"kind": "verified", "when": f"case {i}", "then": f"do {i}",
                                        "query": f"q{i}", "title": f"T{i}"}
        store.insert("episodes", {"episode_id": f"ep{history}-{i}", "run_id": "src-run", "history": history,
                                  "test_group": "B", "split": "train", "ordinal": i,
                                  "batch_id": f"{name}/{history}"})
        store.db.execute("UPDATE episodes SET rule_json = ? WHERE episode_id = ?",
                         (json.dumps({"rule": rule}), f"ep{history}-{i}"))
    store.db.commit()


def _static_exp(tmp_path, **kw):
    return _write(tmp_path, groups=["P-all", "A0"], static={"from_experiment": "src", "group": "B", "after": 3},
                  overrides={"memory_service.agent_search": True},
                  phases=[{"eval": {"label": "after-3", "tasks": {"causes": ["queue_pop_front"], "themes": ["inventory"],
                                                                   "seeds": [2000, 2002]}}}], **kw)


def test_p_all_needs_a_static_source(tmp_path):
    with pytest.raises(ValidationError, match="static"):
        load_experiment(_write(tmp_path, groups=["P-all"]))


def test_static_groups_never_train(tmp_path):
    exp, cfg = load_experiment(_write(tmp_path, groups=["A", "B", "P-all", "A0"],
                                      static={"from_experiment": "src", "after": 2}))
    assert {s.group for s in schedule(exp, cfg) if s.kind == "train"} == {"B"}


def test_p_all_block_is_the_source_rules_in_training_order(tmp_path):
    store = Store(tmp_path / "runs")
    _source_rules(store, "src", 0, 4, skip=1)
    exp, _ = load_experiment(_static_exp(tmp_path))
    mem = batch.static_memory(store, exp, 0, "P-all")
    assert [it.memory_id for it in mem.recall(None, 1)] == ["rule-00", "rule-02"]  # k ignored; no rule at 1; stops at after
    assert mem.items[1].text == "VERIFIED RULE: WHEN case 2 THEN do 2\nSituation: q2"
    assert mem.search("anything", 5) == []
    with pytest.raises(RuntimeError):
        mem.retain({})
    assert batch.static_memory(store, exp, 0, "A0").items == []
    with pytest.raises(RuntimeError, match="ordinals"):
        batch.static_memory(store, exp, 1, "P-all")  # no source stream for history 1


def test_static_groups_end_to_end(tmp_path):
    store = Store(tmp_path / "runs")
    _source_rules(store, "src", 0, 3)
    exp, cfg = load_experiment(_static_exp(tmp_path))
    results = run_batch(exp, cfg, store, _client, sandbox_backend="local", out=lambda _: None)
    assert [r["group"] for r in results] == ["P-all", "A0"] * 2 and all(r["passed"] for r in results)
    rows = store.db.execute("SELECT test_group, checkpoint, trace_sha256, retrieval_json FROM episodes "
                            "WHERE batch_id LIKE ? ORDER BY test_group", (f"{exp.name}/%",)).fetchall()
    for group, checkpoint, sha, retrieval in rows:
        assert checkpoint == 3 and json.loads(retrieval)["ranker"] == "static"
        start = json.loads(store.get_blob("traces", sha).decode().splitlines()[0])
        assert start["system"] == system_prompt(True) and start["tools"] == tools(True)  # memory-group instructions
        assert ("rule-02" in start["first_message"]) == (group == "P-all")
        assert ("<experience>" in start["first_message"]) == (group == "P-all")
