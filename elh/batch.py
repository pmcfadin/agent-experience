"""Batch experiment runner: one short YAML file per experiment, run in sequence by `elh run-batch`.

An experiment file names a registration file (`base_config`), optional dotted `overrides`, the
arms and histories, and an ordered list of phases:

    name: exp1-memory-helps
    base_config: experiments/phase0-local.yaml
    overrides: {"memory_service.record_format": "ticket-v0"}   # optional
    family: cpu_mem
    histories: 1
    arms: [A, B]
    phases:
      - train: {tasks: {causes: [queue_pop_front], themes: [billing, logistics], seeds: [1000, 1005]}}
      - eval:  {label: after-10, tasks: {causes: [queue_pop_front], themes: [inventory], seeds: [2000, 2010]}}

Each experiment registers its own run: config `name` = `<base name>-<experiment name>`, with the
overrides applied and `arms`/`histories`/`families` taken from the experiment, so run ids and
Hindsight banks never collide with other experiments. A batch refuses to start if its run already
has episodes from this experiment (rename it to run again; there is no resume).

Phases run in order per history. `train`: learning arms (B...) only, split "train", against the live
bank. `eval`: every arm, split "checkpoint", `checkpoint` = train episodes scheduled so far in that
history; learning arms read a snapshot of their bank taken (or reused) at that point; within a phase
arms are interleaved per task so backend drift hits them equally.

Tasks expand as cause x theme x seed (seeds are a half-open range [start, end)), then a shuffle
seeded by (schedule.seed, history, phase index): reproducible, but the order differs across
histories. Seed convention: train tasks use seeds 1000+, eval tasks 2000+, so they never overlap.

Every episode carries `batch_id = <experiment>/<history>` (the analysis convention, elh/analysis.py);
the phase is recoverable from split + checkpoint.
"""

from __future__ import annotations

import random
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from elh.controller import memory_for, open_snapshot, run_id_for, run_single_episode, snapshot_memory
from elh.memory.base import NullMemory
from elh.registry import ExperimentConfig
from elh.runner.models import InfraError
from elh.tasks.base import TaskInstance, get_family

KNOWN_ARMS = ("A", "B", "C")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskSpec(_Strict):
    causes: list[str] = Field(min_length=1)
    themes: list[str] = Field(min_length=1)
    seeds: tuple[int, int]   # [start, end)

    @field_validator("seeds")
    @classmethod
    def _range(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[1] <= v[0]:
            raise ValueError(f"seed range {list(v)} is empty; use [start, end)")
        return v


class PhaseSpec(_Strict):
    label: str | None = None
    tasks: TaskSpec


class Phase(_Strict):
    train: PhaseSpec | None = None
    eval: PhaseSpec | None = None

    @model_validator(mode="after")
    def _one(self) -> "Phase":
        if (self.train is None) == (self.eval is None):
            raise ValueError("each phase is exactly one of `train` or `eval`")
        return self

    @property
    def kind(self) -> str:
        return "train" if self.train is not None else "eval"

    @property
    def spec(self) -> PhaseSpec:
        return self.train or self.eval


class Experiment(_Strict):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    base_config: str
    overrides: dict[str, Any] = Field(default_factory=dict)
    family: str = "cpu_mem"
    histories: int = Field(1, ge=1)
    arms: list[str] = Field(min_length=1)
    phases: list[Phase] = Field(min_length=1)

    @field_validator("arms")
    @classmethod
    def _arms(cls, v: list[str]) -> list[str]:
        bad = [a for a in v if a not in KNOWN_ARMS]
        if bad or len(set(v)) != len(v):
            raise ValueError(f"arms must be distinct values from {KNOWN_ARMS}, got {v}")
        return v


@dataclass(frozen=True)
class Step:
    """One scheduled episode."""

    history: int
    phase: str        # label
    kind: str         # train | eval
    arm: str
    task: TaskInstance
    after: int        # train episodes scheduled before this step in its history
    ordinal: int      # train: position in the train stream; eval: position in the phase


def _set_dotted(d: dict, key: str, value: Any) -> None:
    *path, last = key.split(".")
    for p in path:
        if not isinstance(d.get(p), dict):
            raise KeyError(f"override {key!r}: {p!r} is not a section of the base config")
        d = d[p]
    d[last] = value


def _resolve(path: str, rel_to: Path) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return rel_to / p


def load_experiment(path: str | Path) -> tuple[Experiment, ExperimentConfig]:
    """Parse an experiment file and derive its registration (validated by the registry model)."""
    path = Path(path)
    exp = Experiment.model_validate(yaml.safe_load(path.read_text()))
    base = yaml.safe_load(_resolve(exp.base_config, path.parent).read_text())
    for key, value in exp.overrides.items():
        if key.split(".")[0] in ("name", "arms", "histories", "families"):
            raise KeyError(f"override {key!r}: set it in the experiment file, not as an override")
        _set_dotted(base, key, value)
    base.update(name=f"{base['name']}-{exp.name}", arms=list(exp.arms), histories=exp.histories,
                families=[exp.family])
    cfg = ExperimentConfig.model_validate(base)
    family = get_family(exp.family)
    for ph in exp.phases:
        t = ph.spec.tasks
        unknown = [c for c in t.causes if c not in family.causes] + [x for x in t.themes if x not in family.themes]
        if unknown:
            raise ValueError(f"unknown causes/themes for family {exp.family}: {unknown}")
    labels = [label for label, _ in _labelled(exp)]
    if len(set(labels)) != len(labels):
        raise ValueError(f"phase labels must be unique: {labels}")
    return exp, cfg


def _labelled(exp: Experiment) -> list[tuple[str, Phase]]:
    """Phase labels; defaults are `train-<k>` (k-th train phase) and `after-<n>` (n = train episodes before it)."""
    out, n_train, k = [], 0, 0
    for ph in exp.phases:
        t = ph.spec.tasks
        size = len(t.causes) * len(t.themes) * (t.seeds[1] - t.seeds[0])
        k += ph.kind == "train"
        label = ph.spec.label or (f"train-{k}" if ph.kind == "train" else f"after-{n_train}")
        out.append((label, ph))
        if ph.kind == "train":
            n_train += size
    return out


def expand_tasks(family, spec: TaskSpec, *, seed: int, history: int, phase_index: int) -> list[TaskInstance]:
    tasks = [family.generate(c, th, s) for c in spec.causes for th in spec.themes for s in range(*spec.seeds)]
    random.Random(f"{seed}/h{history}/p{phase_index}").shuffle(tasks)
    return tasks


def schedule(exp: Experiment, cfg: ExperimentConfig) -> list[Step]:
    """The full ordered episode list: per history, phases in order; arms interleaved per task."""
    family = get_family(exp.family)
    learners = [a for a in exp.arms if a != "A"]
    steps: list[Step] = []
    for h in range(exp.histories):
        n_train = 0
        for i, (label, ph) in enumerate(_labelled(exp)):
            tasks = expand_tasks(family, ph.spec.tasks, seed=cfg.schedule.seed, history=h, phase_index=i)
            if ph.kind == "train":
                for task in tasks:
                    steps += [Step(h, label, "train", a, task, n_train, n_train) for a in learners]
                    n_train += 1
            else:
                for j, task in enumerate(tasks):
                    steps += [Step(h, label, "eval", a, task, n_train, j) for a in exp.arms]
    return steps


def batch_id(exp: Experiment, step: Step) -> str:
    return f"{exp.name}/{step.history}"


def print_schedule(exp: Experiment, cfg: ExperimentConfig, steps: list[Step], out=print) -> None:
    out(f"experiment {exp.name}: run {run_id_for(cfg)}, {len(steps)} episodes")
    groups: dict[tuple, list[Step]] = defaultdict(list)
    for s in steps:
        groups[(s.history, s.phase, s.kind, s.arm)].append(s)
    for (h, label, kind, arm), ss in groups.items():
        where = f"train episodes {ss[0].after + 1}-{ss[-1].after + 1}" if kind == "train" else f"after {ss[0].after}"
        out(f"h{h} {label} ({kind}, {where}) arm {arm}: {len(ss)} tasks")
        for s in ss:
            out(f"    {s.task.task_id}")


def run_batch(exp: Experiment, cfg: ExperimentConfig, store, make_client: Callable[[TaskInstance], Any], *,
              sandbox_backend: str | None = None, out=print) -> list[dict]:
    """Run the schedule in order. An InfraError skips that episode; anything else stops the batch.
    Returns one result dict per scheduled episode."""
    run_id = run_id_for(cfg)
    prior = store.db.execute("SELECT COUNT(*) FROM episodes WHERE run_id = ? AND batch_id LIKE ?",
                             (run_id, f"{exp.name}/%")).fetchone()[0]
    if prior:
        raise RuntimeError(f"run {run_id} already has {prior} episodes from experiment {exp.name}; "
                           "rename the experiment to run it again")

    steps = schedule(exp, cfg)
    live: dict[tuple[int, str], Any] = {}        # (history, arm) -> training bank
    snaps: dict[tuple[int, str, int], str] = {}  # (history, arm, after) -> snapshot id
    results: list[dict] = []
    for n, s in enumerate(steps, 1):
        if s.arm == "A":  # never scheduled for train phases
            memory, split = NullMemory(), "checkpoint"
        elif s.kind == "train":
            if (s.history, s.arm) not in live:
                live[s.history, s.arm] = memory_for(cfg, s.arm, s.history, "train", store=store)
            memory, split = live[s.history, s.arm], "train"
        else:
            key = (s.history, s.arm, s.after)
            if key not in snaps:
                snaps[key] = _snapshot(cfg, store, live, run_id, *key)
            memory, split = open_snapshot(cfg, store, snaps[key]), "checkpoint"
        res = {"history": s.history, "phase": s.phase, "kind": s.kind, "arm": s.arm,
               "task_id": s.task.task_id, "passed": None, "turns": None, "wall_s": None, "error": None}
        t0 = time.monotonic()
        try:
            _, result, verdict = run_single_episode(
                cfg, store, s.task, make_client(s.task), arm=s.arm, split=split, history=s.history,
                ordinal=s.ordinal, checkpoint=s.after if s.kind == "eval" else None, memory=memory,
                sandbox_backend=sandbox_backend, batch_id=batch_id(exp, s))
            res.update(passed=verdict.passed, turns=result.turns, wall_s=result.wall_s)
            out(f"[{n}/{len(steps)}] h{s.history} {s.phase} {s.arm} {s.task.task_id} "
                f"passed={int(verdict.passed)} turns={result.turns} wall={result.wall_s:.1f}s")
        except InfraError as exc:
            res["error"] = str(exc)
            out(f"[{n}/{len(steps)}] h{s.history} {s.phase} {s.arm} {s.task.task_id} "
                f"INFRA ERROR after {time.monotonic() - t0:.1f}s: {exc}")
        results.append(res)
    return results


def _snapshot(cfg: ExperimentConfig, store, live: dict, run_id: str, history: int, arm: str, after: int) -> str:
    """Snapshot id for (history, arm) after `after` train episodes: reused if one exists."""
    row = store.db.execute("SELECT snapshot_id FROM memory_snapshots WHERE run_id = ? AND history = ? "
                           "AND arm = ? AND after_episode = ?", (run_id, history, arm, after)).fetchone()
    if row:
        return row[0]
    if (history, arm) not in live:  # eval before any training: snapshot the (empty) bank
        live[history, arm] = memory_for(cfg, arm, history, "train", store=store)
    return snapshot_memory(cfg, store, live[history, arm], history=history, arm=arm, after_episode=after)


def summarize(results: list[dict]) -> list[str]:
    """One line per (history, phase, arm): n, pass rate, mean turns, infra errors."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        groups[(r["history"], r["phase"], r["arm"])].append(r)
    lines = [f"{'hist':4s} {'phase':16s} {'arm':3s} {'n':>3s} {'pass':>6s} {'turns':>6s} {'infra':>5s}"]
    for (h, phase, arm), rs in groups.items():
        ok = [r for r in rs if r["error"] is None]
        rate = f"{sum(r['passed'] for r in ok) / len(ok):.2f}" if ok else "-"
        turns = f"{statistics.mean(r['turns'] for r in ok):.1f}" if ok else "-"
        lines.append(f"h{h:<3d} {phase:16s} {arm:3s} {len(ok):3d} {rate:>6s} {turns:>6s} {len(rs) - len(ok):5d}")
    return lines


def report(exp: Experiment, store, results: list[dict], out=print) -> None:
    """Print the simple summary, then the analysis module's batch report if it is available."""
    for line in summarize(results):
        out(line)
    try:
        from elh.analysis import format_summary, http_trace_fetcher, summarize_batch
    except ImportError:
        return
    try:
        out(format_summary(summarize_batch(store, exp.name, fetch=http_trace_fetcher())))
    except Exception as exc:  # the batch itself is done; a report failure must not hide that
        out(f"(analysis report failed: {exc!r}; run `python -m elh.analysis {exp.name}`)")
