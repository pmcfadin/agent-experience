"""Experiment controller (SPEC §11). Phase 0: run and re-grade single episodes."""

from __future__ import annotations

import json
import subprocess
import uuid
from functools import partial

from elh import __version__
from elh.judge import Verdict, grade
from elh.ledger import compute_entry, usage_entries
from elh.memory.base import Memory, NullMemory, render_memory_block
from elh.registry import ExperimentConfig
from elh.runner.agent import EpisodeResult, run_episode
from elh.runner.models import ModelClient
from elh.sandbox.base import make_sandbox
from elh.store import Store, utcnow
from elh.tasks.base import TaskInstance, get_family

# Splits whose episodes may never write to memory or learner state (SPEC §10).
READ_ONLY_SPLITS = {"checkpoint", "sealed_familiar", "sealed_novel_combo", "sealed_novel_cause", "retention", "dev"}


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


def run_single_episode(cfg: ExperimentConfig, store: Store, task: TaskInstance, client: ModelClient, *,
                       arm: str = "A", split: str = "dev", history: int = 0, ordinal: int = 0,
                       checkpoint: int | None = None, memory: Memory | None = None,
                       sandbox_backend: str | None = None, memory_k: int = 5) -> tuple[str, EpisodeResult, Verdict]:
    run_id = run_id_for(cfg)
    store.ensure_run(run_id, cfg.name, cfg.config_hash, json.dumps(cfg.model_dump(mode="json"), sort_keys=True),
                     git_rev(), __version__)
    if client.model_id != "scripted" and client.model_id != cfg.model.id:
        raise ValueError(f"client model {client.model_id} != registered model {cfg.model.id}")
    if (client.model_id != "scripted" and cfg.model.provider == "openai_compat"
            and split != "dev" and not cfg.model.pin.complete):
        raise ValueError("self-hosted model is not pinned (model.pin in the registration file); "
                         "only the dev split may run against an unpinned model")

    memory = memory or NullMemory()
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
    if split not in READ_ONLY_SPLITS and result.outcome != "refused":
        memory.retain({"episode_id": episode_id, "task": task.signature, "passed": verdict.passed})

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
        "retrieved_memory_ids_json": json.dumps([it.memory_id for it in items]), "memory_snapshot_id": None,
        "started_at": started, "finished_at": utcnow(),
    })
    for c in result.calls:
        cu = c["usage"]
        store.insert("llm_calls", {
            "episode_id": episode_id, "turn": c["turn"], "request_id": c["request_id"], "model": c["model"],
            "stop_reason": c["stop_reason"], "input_tokens": cu.get("input_tokens"),
            "output_tokens": cu.get("output_tokens"), "cache_read_tokens": cu.get("cache_read_input_tokens"),
            "cache_write_tokens": cu.get("cache_creation_input_tokens"), "latency_s": c["latency_s"],
        })
    if client.model_id != "scripted":
        # Priced under the registered model id: a gateway may report the served model by another name.
        entries = [e for c in result.calls for e in usage_entries(cfg, cfg.model.id, c["usage"])]
    else:
        entries = []
    entries.append(compute_entry(cfg, usage.cpu_s))
    for e in entries:
        store.insert("ledger", {"run_id": run_id, "episode_id": episode_id, "category": e.category,
                                "component": e.component, "units": e.units, "unit": e.unit, "usd": e.usd})
    return episode_id, result, verdict


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
