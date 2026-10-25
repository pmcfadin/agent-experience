"""Batch runner (elh/batch.py): experiment files, schedule expansion, scripted end-to-end runs."""

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
from elh.runner.scripts import scripted_client
from elh.store import Store
from elh.tasks.base import get_family

ROOT = Path(__file__).parent.parent
BASE = ROOT / "experiments" / "phase0-local.yaml"
URL = os.environ.get("ELH_HINDSIGHT_URL", "http://127.0.0.1:8901")


def _write(tmp_path, **kw) -> Path:
    doc = {"name": f"batchtest-{uuid.uuid4().hex[:8]}", "base_config": str(BASE), "family": "cpu_mem",
           "histories": 1, "arms": ["A"],
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
    assert {s.arm for s in steps if s.kind == "train"} == set(exp.arms) - {"A"}
    assert all(s.task.seed >= 2000 if s.kind == "eval" else 1000 <= s.task.seed < 2000 for s in steps)


def test_schedule_is_deterministic_and_differs_across_histories(tmp_path):
    exp, cfg = load_experiment(_write(tmp_path, histories=2, arms=["A", "B"], phases=[
        {"train": {"tasks": {"causes": ["queue_pop_front"], "themes": ["billing", "logistics"], "seeds": [1000, 1005]}}},
        {"eval": {"tasks": {"causes": ["queue_pop_front"], "themes": ["inventory"], "seeds": [2000, 2004]}}}]))
    a, b = schedule(exp, cfg), schedule(exp, cfg)
    key = lambda steps: [(s.history, s.phase, s.arm, s.task.task_id, s.after, s.ordinal) for s in steps]
    assert key(a) == key(b)
    train = {h: [s.task.task_id for s in a if s.history == h and s.kind == "train"] for h in (0, 1)}
    assert len(train[0]) == 10 and sorted(train[0]) == sorted(train[1]) and train[0] != train[1]
    evals = [s for s in a if s.history == 0 and s.kind == "eval"]
    # Arms interleave per task, all at checkpoint 10 (train episodes before the phase).
    assert [s.arm for s in evals] == ["A", "B"] * 4 and {s.after for s in evals} == {10}
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
        load_experiment(_write(tmp_path, arms=["A", "Z"]))


def test_dry_run_prints_without_touching_the_store(tmp_path, capsys):
    runs = tmp_path / "runs"
    assert main(["run-batch", str(_write(tmp_path)), "--dry-run", "--runs-dir", str(runs)]) == 0
    out = capsys.readouterr().out
    assert "cpu_mem/nested_scan_join/inventory/2001" in out and "after-2" in out
    assert "queue_pop_front" not in out  # arm A has no train phase
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
    assert {(r["run_id"], r["arm"], r["split"], r["checkpoint"]) for r in rows} == {(run_id_for(cfg), "A", "checkpoint", 2)}
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
    _, phase, arm, n, rate, _, infra = batch.summarize(results)[1].split()
    assert (phase, arm, n, rate, infra) == ("after-2", "A", "1", "1.00", "1")


def _up() -> bool:
    try:
        urllib.request.urlopen(URL + "/health", timeout=2).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _up(), reason="Hindsight server not running")
def test_arm_b_trains_then_evaluates_on_a_snapshot(tmp_path):
    """One train episode (one memory-LLM retain, ~30 s), then A and B on one task."""
    exp, cfg = load_experiment(_write(tmp_path, arms=["A", "B"], phases=[
        {"train": {"tasks": {"causes": ["queue_pop_front"], "themes": ["billing"], "seeds": [1000, 1001]}}},
        {"eval": {"label": "after-1", "tasks": {"causes": ["queue_pop_front"], "themes": ["inventory"], "seeds": [2000, 2001]}}}]))
    bank = bank_id_for(cfg, 0, "B")
    store = Store(tmp_path / "runs")
    try:
        results = run_batch(exp, cfg, store, _client, sandbox_backend="local", out=lambda _: None)
        assert [(r["kind"], r["arm"], r["passed"]) for r in results] == [
            ("train", "B", True), ("eval", "A", True), ("eval", "B", True)]
        rows = store.db.execute("SELECT arm, split, checkpoint, memory_snapshot_id FROM episodes "
                                "WHERE split = 'checkpoint' ORDER BY arm").fetchall()
        snap = store.db.execute("SELECT snapshot_id, after_episode FROM memory_snapshots").fetchone()
        assert snap["after_episode"] == 1
        assert [tuple(r) for r in rows] == [("A", "checkpoint", 1, None), ("B", "checkpoint", 1, snap["snapshot_id"])]
    finally:
        for b in (bank, f"{bank}-e1"):
            try:
                HindsightMemory(URL, b).delete_bank()
            except InfraError:
                pass
