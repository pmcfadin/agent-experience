"""Checkpoint snapshots against the dedicated Hindsight server (SPEC §10, §11).

The fixture is a small real bank export (one scripted-wrong cpu_mem episode), so these tests
restore banks without calling the memory LLM. Skipped when the server is not running.
"""

import os
import urllib.request
import uuid
from pathlib import Path

import pytest

from elh.controller import memory_for, open_snapshot, run_single_episode, snapshot_memory
from elh.memory.hindsight import HindsightMemory, restore_bank
from elh.registry import load_config
from elh.runner.models import InfraError
from elh.runner.scripts import scripted_client
from elh.store import Store
from elh.tasks.base import get_family

URL = os.environ.get("ELH_HINDSIGHT_URL", "http://127.0.0.1:8901")
FIXTURE = (Path(__file__).parent / "data" / "hindsight-bank.zip").read_bytes()
CONFIG = Path(__file__).parent.parent / "experiments" / "phase0-local.yaml"


def _up() -> bool:
    try:
        urllib.request.urlopen(URL + "/health", timeout=2).close()
        return True
    except OSError:
        return False


@pytest.fixture
def cfg():
    return load_config(CONFIG).model_copy(update={"name": f"snaptest-{uuid.uuid4().hex[:8]}"})


@pytest.fixture
def banks():
    made: list[str] = []
    yield made
    for b in made:
        try:
            HindsightMemory(URL, b).delete_bank()
        except InfraError:
            pass


def _training_bank(cfg, banks) -> HindsightMemory:
    from elh.controller import bank_id_for

    bank_id = bank_id_for(cfg, 0, "B")
    banks += [bank_id, f"{bank_id}-e1"]
    mem = restore_bank(URL, FIXTURE, bank_id)
    mem.readonly = False  # stands in for a bank built by train episodes
    return mem


def test_scored_splits_need_a_snapshot(cfg):
    with pytest.raises(ValueError, match="snapshot"):
        memory_for(cfg, "B", 0, "checkpoint")
    with pytest.raises(ValueError, match="read-only"):
        memory_for(cfg, "B", 0, "train", store=object(), snapshot_id="x")


needs_server = pytest.mark.skipif(not _up(), reason="Hindsight server not running")


@needs_server
def test_snapshot_round_trip(cfg, banks, tmp_path):
    store = Store(tmp_path)
    live = _training_bank(cfg, banks)
    sid = snapshot_memory(cfg, store, live, history=0, arm="B", after_episode=1)
    row = store.snapshot(sid)
    assert row["bank_id"] == f"{live.bank_id}-e1" and row["content_sha256"] == live.fingerprint()
    assert store.get_blob("snapshots", row["bank_export_sha256"])

    snap = memory_for(cfg, "B", 0, "checkpoint", store=store, snapshot_id=sid)
    assert snap.readonly and snap.snapshot_id == sid
    task = get_family("cpu_mem").generate("queue_pop_front", "billing", 1)
    assert [m.text for m in snap.recall(task, 5)] == [m.text for m in live.recall(task, 5)]

    # A lost snapshot bank comes back from the stored archive, with the same contents.
    snap.delete_bank()
    again = open_snapshot(cfg, store, sid)
    assert again.fingerprint() == row["content_sha256"]


@needs_server
def test_tampered_snapshot_is_refused(cfg, banks, tmp_path):
    store = Store(tmp_path)
    sid = snapshot_memory(cfg, store, _training_bank(cfg, banks), history=0, arm="B", after_episode=1)
    snap = open_snapshot(cfg, store, sid)
    doc = snap._call("GET", "/documents")["items"][0]["id"]
    snap._call("DELETE", f"/documents/{doc}")
    snap.wait_idle()
    with pytest.raises(InfraError, match="no longer matches"):
        open_snapshot(cfg, store, sid)


@needs_server
def test_checkpoint_episode_uses_snapshot_and_leaves_it_unchanged(cfg, banks, tmp_path):
    store = Store(tmp_path)
    sid = snapshot_memory(cfg, store, _training_bank(cfg, banks), history=0, arm="B", after_episode=1)
    family = get_family("cpu_mem")
    task = family.generate("queue_pop_front", "billing", 7)
    snap = memory_for(cfg, "B", 0, "checkpoint", store=store, snapshot_id=sid)
    before = snap.fingerprint()
    episode_id, _, verdict = run_single_episode(
        cfg, store, task, scripted_client("reference", task, family), arm="B", split="checkpoint",
        checkpoint=1, memory=snap)
    row = store.episode(episode_id)
    assert verdict.passed and row["memory_snapshot_id"] == sid
    assert snap.fingerprint() == before
