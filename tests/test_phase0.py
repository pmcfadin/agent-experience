import json
from pathlib import Path

import pytest

from elh.controller import regrade_episode, run_single_episode
from elh.judge import grade, integrity_violations
from elh.ledger import usage_entries
from elh.registry import load_config
from elh.runner.agent import format_exec
from elh.runner.scripts import scripted_client
from elh.sandbox.base import ExecResult
from elh.sandbox.local import LocalSandbox
from elh.store import Store
from elh.tasks.base import get_family

CONFIG = Path(__file__).parent.parent / "experiments" / "phase0-local.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def family():
    return get_family("cpu_mem")


def test_generation_is_deterministic_and_varies(family):
    a = family.generate("quadratic_membership", "telemetry", 11)
    b = family.generate("quadratic_membership", "telemetry", 11)
    c = family.generate("quadratic_membership", "telemetry", 12)
    d = family.generate("quadratic_membership", "billing", 11)
    assert a.content_hash == b.content_hash
    assert len({a.content_hash, c.content_hash, d.content_hash}) == 3
    assert "reading_mv" in a.workspace["job/core.py"]
    assert "amount_cents" in d.workspace["job/core.py"]


def test_oracle_never_in_workspace(family):
    for cause in family.causes:
        task = family.generate(cause, "inventory", 1)
        joined = "\n".join(task.workspace.values())
        assert "hidden_seed" not in joined
        assert "def reference(" not in joined


def test_integrity_detects_protected_edits(family):
    task = family.generate("queue_pop_front", "billing", 2)
    ws = dict(task.workspace)
    ws["tests/test_job.py"] = "import unittest\n"
    assert integrity_violations(task, ws) == ["tests/test_job.py"]
    verdict = grade(task, ws, LocalSandbox)
    assert not verdict.passed and verdict.checks["integrity"] is False


def test_ledger_prices_from_registry():
    cfg = load_config(CONFIG.parent / "phase0-sonnet.yaml")
    entries = usage_entries(cfg, "claude-sonnet-5-5", {
        "input_tokens": 1_000_000, "output_tokens": 100_000,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 2_000_000,
    })
    by = {e.component.split(":")[1]: e.usd for e in entries}
    assert by == pytest.approx({"input": 2.0, "output": 1.0, "cache_read": 0.4})
    with pytest.raises(KeyError):
        usage_entries(cfg, "some-other-model", {"input_tokens": 1})


def test_format_exec_clips_and_reports():
    res = ExecResult(1, "x" * 50_000, "boom", 0.1)
    out = format_exec(res, 120, 1000)
    assert "characters omitted" in out and out.endswith("[exit code 1]")
    assert "timed out" in format_exec(ExecResult(None, "", "", 120, True), 120, 1000)


@pytest.mark.parametrize("kind,expect_pass", [("reference", True), ("wrong", False), ("noop", False)])
def test_scripted_episode_end_to_end_and_replay(tmp_path, cfg, family, kind, expect_pass):
    store = Store(tmp_path)
    task = family.generate("nested_scan_join", "course_registration", 4)
    client = scripted_client(kind, task, family)
    episode_id, result, verdict = run_single_episode(cfg, store, task, client, sandbox_backend="local")
    assert result.outcome == "submitted"
    assert verdict.passed is expect_pass
    row = store.episode(episode_id)
    assert row["passed"] == int(expect_pass) and row["sandbox_backend"] == "local"
    trace = store.get_blob("traces", row["trace_sha256"]).decode().splitlines()
    assert json.loads(trace[0])["event"] == "episode_start"
    assert json.loads(trace[-1])["event"] == "verdict"
    same, _, _ = regrade_episode(cfg, store, episode_id, sandbox_backend="local")
    assert same


def test_registration_is_immutable(tmp_path, cfg):
    store = Store(tmp_path)
    store.ensure_run("r1", cfg.name, cfg.config_hash, "{}", None, "0")
    with pytest.raises(RuntimeError):
        store.ensure_run("r2", cfg.name, "different-hash", "{}", None, "0")
