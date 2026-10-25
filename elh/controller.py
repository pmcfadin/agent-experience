"""Experiment controller (SPEC §11). Phase 0: run and re-grade single episodes."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from functools import partial

from elh import __version__
from elh.judge import Verdict, grade
from elh.ledger import compute_entry, usage_entries
from elh.memory.base import Memory, NullMemory, render_memory_block
from elh.memory.hindsight import HindsightMemory, restore_bank, usage_to_ledger, workspace_diff
from elh.registry import ExperimentConfig
from elh.runner.agent import EpisodeResult, run_episode
from elh.runner.models import InfraError, ModelClient, fetch_backend_props
from elh.sandbox.base import make_sandbox
from elh.store import Store, utcnow
from elh.tasks.base import TaskInstance, get_family

# Splits whose episodes may never write to memory or learner state (SPEC §10).
READ_ONLY_SPLITS = {"checkpoint", "sealed_familiar", "sealed_novel_combo", "sealed_novel_cause", "retention", "dev"}
# Read-only splits that are scored against a frozen snapshot, never the live training bank.
SNAPSHOT_SPLITS = READ_ONLY_SPLITS - {"dev"}


def run_id_for(cfg: ExperimentConfig) -> str:
    return f"{cfg.name}-v{cfg.version}-{cfg.config_hash[:10]}"


def git_rev() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def sandbox_factory(cfg: ExperimentConfig, backend: str | None = None):
    s = cfg.sandbox
    return partial(make_sandbox, backend or s.backend, image=s.image, cpus=s.cpus,
                   memory_mb=s.memory_mb, network=s.network)


def capture_props(store: Store, m, split: str, what: str = "backend") -> str | None:
    """Fetch the serving backend's /props, check it against the pin and store it verbatim; returns
    its blob hash. Outside the dev split a self-hosted model must have it, so every scored episode
    names the exact weights and flags, and a backend swap is caught before the episode runs."""
    url = m.props_url or (os.environ.get(m.props_url_env, "") if m.props_url_env else "")
    if not url:
        if split != "dev":
            raise ValueError(f"{what} /props is required outside the dev split; set props_url for {m.id}")
        return None
    try:
        raw = fetch_backend_props(url)
        props = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise InfraError(f"could not fetch {what} /props from {url}: {exc}") from exc
    if m.pin.complete:
        bad = m.pin.props_mismatches(props)
        if bad:
            raise InfraError(f"{what} {m.id} does not match its pin: " + "; ".join(bad))
    return store.put_blob("props", raw)


def capture_backend_props(cfg: ExperimentConfig, store: Store, split: str) -> str | None:
    return capture_props(store, cfg.model, split)


def _hindsight_url(cfg: ExperimentConfig) -> str:
    ms = cfg.memory_service
    url = ms.base_url or os.environ.get(ms.base_url_env, "")
    if not url:
        raise ValueError(f"arm B needs memory_service.base_url or ${ms.base_url_env}")
    return url


def bank_id_for(cfg: ExperimentConfig, history: int, arm: str) -> str:
    return f"elh-{run_id_for(cfg)}-h{history}-{arm}"


def memory_for(cfg: ExperimentConfig, arm: str, history: int, split: str, *,
               store: Store | None = None, snapshot_id: str | None = None) -> Memory:
    """Arm wiring (SPEC §10). One Hindsight bank per (run, history, arm); read-only off `train`.
    Checkpoint, sealed and retention episodes must name a snapshot; `dev` may read the live bank."""
    if arm == "A":
        return NullMemory()
    if arm == "B":
        if snapshot_id is not None:
            if split not in READ_ONLY_SPLITS:
                raise ValueError(f"a snapshot is read-only; split {split!r} writes to memory")
            if store is None:
                raise ValueError("opening a snapshot needs the store")
            return open_snapshot(cfg, store, snapshot_id)
        if split in SNAPSHOT_SPLITS:
            raise ValueError(f"split {split!r} must run against a memory snapshot (snapshot_id)")
        mem = HindsightMemory(_hindsight_url(cfg), bank_id_for(cfg, history, arm), readonly=split in READ_ONLY_SPLITS)
        mem.ensure_bank()
        return mem
    raise ValueError(f"arm {arm} is not implemented yet")


def snapshot_memory(cfg: ExperimentConfig, store: Store, memory: Memory, *, history: int, arm: str,
                    after_episode: int) -> str | None:
    """Freeze a training bank after `after_episode` train episodes (SPEC §11): store its transfer
    archive as a blob, restore it into a snapshot bank, and check the copy holds the same memories.
    Returns the snapshot id (None for arms without memory)."""
    if not isinstance(memory, HindsightMemory):
        return None
    if memory.readonly:
        raise ValueError(f"bank {memory.bank_id} is a read-only view; snapshot the training bank")
    run_id = run_id_for(cfg)
    snapshot_id = f"{run_id}-h{history}-{arm}-e{after_episode}"
    target = f"{memory.bank_id}-e{after_episode}"
    memory.wait_idle()
    live = memory.fingerprint()
    archive = memory.export_snapshot()
    if memory.fingerprint() != live:
        raise InfraError(f"bank {memory.bank_id} changed while it was being exported")
    archive_sha = store.put_blob("snapshots", archive)
    snap = memory.restore(archive, target)
    if snap.fingerprint() != live:
        raise InfraError(f"snapshot bank {target} does not hold the same memories as {memory.bank_id}")
    store.insert("memory_snapshots", {
        "snapshot_id": snapshot_id, "run_id": run_id, "history": history, "arm": arm,
        "after_episode": after_episode, "source_bank_id": memory.bank_id, "bank_id": target,
        "bank_export_sha256": archive_sha, "content_sha256": live, "learner_state_sha256": None,
        "created_at": utcnow()})
    return snapshot_id


def open_snapshot(cfg: ExperimentConfig, store: Store, snapshot_id: str) -> HindsightMemory:
    """Read-only view of a snapshot bank. If the bank is gone it is restored again from the stored
    archive; either way its contents must match the hash recorded when the snapshot was taken."""
    row = store.snapshot(snapshot_id)
    if row["run_id"] != run_id_for(cfg):
        raise ValueError(f"snapshot {snapshot_id} belongs to run {row['run_id']}, not {run_id_for(cfg)}")
    url = _hindsight_url(cfg)
    mem = HindsightMemory(url, row["bank_id"], readonly=True)
    if not mem.exists():
        archive = store.get_blob("snapshots", row["bank_export_sha256"])
        mem = restore_bank(url, archive, row["bank_id"])
    if mem.fingerprint() != row["content_sha256"]:
        raise InfraError(f"snapshot bank {row['bank_id']} no longer matches snapshot {snapshot_id}")
    mem.snapshot_id = snapshot_id
    return mem


def run_single_episode(cfg: ExperimentConfig, store: Store, task: TaskInstance, client: ModelClient, *,
                       arm: str = "A", split: str = "dev", history: int = 0, ordinal: int = 0,
                       checkpoint: int | None = None, memory: Memory | None = None,
                       sandbox_backend: str | None = None, memory_k: int = 5,
                       batch_id: str | None = None) -> tuple[str, EpisodeResult, Verdict]:
    run_id = run_id_for(cfg)
    store.ensure_run(run_id, cfg.name, cfg.version, cfg.config_hash, json.dumps(cfg.model_dump(mode="json"), sort_keys=True),
                     git_rev(), __version__)
    if client.model_id != "scripted" and client.model_id != cfg.model.id:
        raise ValueError(f"client model {client.model_id} != registered model {cfg.model.id}")
    if (client.model_id != "scripted" and cfg.model.provider == "openai_compat"
            and split != "dev" and not cfg.model.pin.complete):
        raise ValueError("self-hosted model is not pinned (model.pin in the registration file); "
                         "only the dev split may run against an unpinned model")

    props_sha = None
    if client.model_id != "scripted" and cfg.model.provider == "openai_compat":
        props_sha = capture_backend_props(cfg, store, split)
    memory_props_sha = None
    if isinstance(memory, HindsightMemory):
        llm = cfg.memory_service.llm
        if split not in READ_ONLY_SPLITS and client.model_id != "scripted" and not llm.pin.complete:
            raise ValueError(f"memory LLM {llm.id} is not pinned; train episodes with a real model retain through it")
        memory_props_sha = capture_props(store, llm, split if client.model_id != "scripted" else "dev", "memory LLM")

    memory = memory or NullMemory()
    if split in SNAPSHOT_SPLITS and not isinstance(memory, NullMemory) and getattr(memory, "snapshot_id", None) is None:
        raise ValueError(f"split {split!r} must run against a memory snapshot")
    if split not in READ_ONLY_SPLITS and getattr(memory, "readonly", False):
        raise ValueError(f"split {split!r} retains, but bank {memory.bank_id} is read-only")
    if hasattr(memory, "wait_idle"):
        memory.wait_idle()  # no background memory work may overlap a timed episode
    # Read-only episodes must leave the bank exactly as they found it (SPEC §10).
    guard = memory.fingerprint() if split in READ_ONLY_SPLITS and hasattr(memory, "fingerprint") else None
    items = memory.recall(task, memory_k)
    make_sbx = sandbox_factory(cfg, sandbox_backend)
    episode_id = uuid.uuid4().hex
    started = utcnow()

    sbx = make_sbx()
    try:
        sbx.start(task.workspace)
        result = run_episode(task, sbx, client, cfg.budgets, cfg.model.max_tokens_per_turn,
                             memory_block=render_memory_block(items))
        final_ws = sbx.snapshot_workspace()
        usage = sbx.resource_usage()
        backend, image = sbx.backend, sbx.image
    finally:
        sbx.close()

    verdict = grade(task, final_ws, make_sbx)
    if guard is not None:
        memory.wait_idle()
        if memory.fingerprint() != guard:
            raise InfraError(f"read-only bank {memory.bank_id} changed during a {split} episode")

    trace_sha = store.put_trace(result.trace + [{"event": "verdict", **verdict.to_dict()}])
    ws_sha = store.put_workspace(final_ws)
    u = result.usage
    store.insert("episodes", {
        "episode_id": episode_id, "run_id": run_id, "history": history, "arm": arm, "split": split,
        "checkpoint": checkpoint, "ordinal": ordinal, "task_id": task.task_id, "family": task.family,
        "cause": task.cause, "theme": task.theme, "seed": task.seed, "task_hash": task.content_hash,
        "model_id": client.model_id, "sandbox_backend": backend, "sandbox_image": image,
        "outcome": result.outcome, "passed": int(verdict.passed), "verdict_json": json.dumps(verdict.to_dict()),
        "turns": result.turns, "tool_calls": result.tool_calls,
        "input_tokens": u["input_tokens"], "output_tokens": u["output_tokens"],
        "cache_read_tokens": u["cache_read_input_tokens"], "cache_write_tokens": u["cache_creation_input_tokens"],
        "wall_s": result.wall_s, "sandbox_cpu_s": usage.cpu_s,
        "trace_sha256": trace_sha, "final_workspace_sha256": ws_sha,
        "retrieved_memory_ids_json": json.dumps([it.memory_id for it in items]),
        "memory_snapshot_id": getattr(memory, "snapshot_id", None),
        "backend_props_sha256": props_sha, "memory_props_sha256": memory_props_sha,
        "batch_id": batch_id,
        "started_at": started, "finished_at": utcnow(),
    })
    for c in result.calls:
        cu = c["usage"]
        store.insert("llm_calls", {
            "episode_id": episode_id, "turn": c["turn"], "request_id": c["request_id"], "model": c["model"],
            "stop_reason": c["stop_reason"], "input_tokens": cu.get("input_tokens"),
            "output_tokens": cu.get("output_tokens"), "cache_read_tokens": cu.get("cache_read_input_tokens"),
            "cache_write_tokens": cu.get("cache_creation_input_tokens"), "latency_s": c["latency_s"],
            "served_json": json.dumps(c.get("served") or {}, sort_keys=True),
        })
    if client.model_id != "scripted":
        # Priced under the registered model id: a gateway may report the served model by another name.
        entries = [e for c in result.calls for e in usage_entries(cfg, cfg.model.id, c["usage"])]
    else:
        entries = []
    entries.append(compute_entry(cfg, usage.cpu_s))

    if split not in READ_ONLY_SPLITS and result.outcome != "refused":
        memory.retain(episode_record(episode_id, task, result, verdict, final_ws))
        mem_usage = usage_to_ledger(getattr(memory, "last_usage", None))
        if mem_usage:
            entries += usage_entries(cfg, cfg.memory_service.llm.id, mem_usage, category="learning")
        if hasattr(memory, "wait_idle"):
            memory.wait_idle()
    for e in entries:
        store.insert("ledger", {"run_id": run_id, "episode_id": episode_id, "category": e.category,
                                "component": e.component, "units": e.units, "unit": e.unit, "usd": e.usd})
    return episode_id, result, verdict


def episode_record(episode_id: str, task: TaskInstance, result: EpisodeResult, verdict: Verdict,
                   final_ws: dict[str, str]) -> dict:
    """What a memory may retain from an episode: the agent's own work and the judge's feedback."""
    return {"episode_id": episode_id, "family": task.family, "prompt": task.prompt,
            "title": task.prompt.strip().splitlines()[0], "outcome": result.outcome,
            "submit_summary": result.submit_summary, "verdict": verdict.to_dict(),
            "diff": workspace_diff(task.workspace, final_ws), "turns": result.turns, "tool_calls": result.tool_calls}


def regrade_episode(cfg: ExperimentConfig, store: Store, episode_id: str,
                    sandbox_backend: str | None = None) -> tuple[bool, Verdict, dict]:
    """Replay check (Phase 0 exit criterion): regenerate the task from its id, confirm the content
    hash matches, re-grade the stored final workspace and compare with the recorded verdict."""
    row = store.episode(episode_id)
    family = get_family(row["family"])
    task = family.generate(row["cause"], row["theme"], row["seed"])
    if task.content_hash != row["task_hash"]:
        raise RuntimeError(f"task regeneration is not deterministic for {row['task_id']}")
    final_ws = store.get_workspace(row["final_workspace_sha256"])
    verdict = grade(task, final_ws, sandbox_factory(cfg, sandbox_backend))
    recorded = json.loads(row["verdict_json"])
    same = verdict.passed == recorded["passed"] and verdict.checks == recorded["checks"]
    return same, verdict, recorded
