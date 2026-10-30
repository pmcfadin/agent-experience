"""Batch experiment runner: one short YAML file per experiment, run in sequence by `elh run-batch`.

An experiment file names a registration file (`base_config`), optional dotted `overrides`, the
groups and histories, and an ordered list of phases:

    name: exp1-memory-helps
    base_config: experiments/phase0-local.yaml
    overrides: {"memory_service.record_format": "ticket-v0"}   # optional
    family: cpu_mem
    histories: 1
    groups: [A, B]
    phases:
      - train: {tasks: {causes: [queue_pop_front], themes: [billing, logistics], seeds: [1000, 1005]}}
      - eval:  {label: after-10, tasks: {causes: [queue_pop_front], themes: [inventory], seeds: [2000, 2010]}}
    dev: {tasks: {causes: [queue_pop_front], themes: [billing], seeds: [3000, 3006]}}   # group D only

Each experiment registers its own run: config `name` = `<base name>-<experiment name>`, with the
overrides applied and `groups`/`histories`/`families` taken from the experiment, so run ids and
Hindsight banks never collide with other experiments. A batch refuses to start if its run already
has episodes from this experiment (rename it to run again; there is no resume).

Phases run in order per history. `train`: learning groups (B...) only, split "train", against the live
bank. `eval`: every group, split "checkpoint", `checkpoint` = train episodes scheduled so far in that
history; learning groups read a snapshot of their bank taken (or reused) at that point; within a phase
groups are interleaved per task so backend drift hits them equally. `retention`: like `eval` but split
"retention" and train cells only (SPEC §4.3); after group D's last episode of a retention phase the
rollback rule compares the panel with the previous retention phase (`retention_rollback`).

Tasks expand as cause x theme x seed (seeds are a half-open range [start, end)), then a shuffle
seeded by (schedule.seed, history, phase index): reproducible, but the order differs across
histories. Seed convention: train tasks use seeds 1000+, eval tasks 2000+, dev tasks 3000-3999, so
they never overlap.

Group D (SPEC §10). The `dev` panel is required with group D; its (cause, theme) cells must be train
cells and no eval or retention cell. After each D train episode, if `workflows.should_induce` (every
`induce_every` train episodes, and only when a verified success joined the pool since the last
induction), a candidate library is made: `awm` induces it from the verified train successes
(`workflows.induce`, charged as `learning`, ledger rows without an episode); `gepa` runs GEPA seeded
with the incumbent's text on the dev panel split into reflection/validation/promotion pools, with
dev episodes against a read-only view of the live bank and learner, reflection charged as
`learning`. Then the paired dev gate: incumbent and candidate each run every gate task
`workflows.repeats` times (AWM: the whole dev panel; GEPA: its promotion pool only, the tasks its
search never saw), in a seeded order per task; `workflows.promote` decides and `decide_candidate`
records it. That gate is the only promotion rule for both sources (GEPA's own paired check is
skipped). Dev episodes are charged as `learning`, carry `checkpoint` = the train episodes done at
that induction, and never write to memory or learner state.

Static-prompt groups (SPEC §6.2, H6). `P-all` shows every rule a learning group wrote in an earlier
experiment's history, frozen after `static.after` train episodes, in training order and uncapped, in
B's first-message slot; `A0` shows an empty block. Both keep the memory-group instructions (the
`search_memory` tool when `memory_service.agent_search` is on), whose searches return nothing. They
never train; history h reads the rules of `<static.from_experiment>/<h>`, and their episodes carry
`checkpoint = static.after`. `B-rules` reads the source group's own snapshot after `static.after`
train episodes (opened under the source run's registration) with every start slot and every search
hit a verbatim rule (`RulesOnlyMemory`): retrieval among rules only, no episode records.
`P-hand` (house_lib worlds only) shows the world's handbook, one hand-written `gold_text` per lesson
in catalogue order, the same for every history; it needs no `static` section and its episodes carry
the phase's own checkpoint. `P-all-ep` is P-all plus everything else the source snapshot could
recall (`record_units`: for B, the experience facts extracted from each episode record): each
rule followed by its episode's facts, in training order, so it holds B's whole retrievable pool
with no selection.

Every episode carries `batch_id = <experiment>/<history>` (the analysis convention, elh/analysis.py);
the phase is recoverable from split + checkpoint.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from elh.controller import memory_for, open_snapshot, run_id_for, run_single_episode, snapshot_memory
from elh.learner import workflows as wf
from elh.memory.base import MemoryItem, NullMemory, StaticMemory
from elh.memory.hindsight import RULE_DOC_SUFFIX, RulesOnlyMemory, render_rule_verbatim
from elh.registry import ExperimentConfig
from elh.runner.models import InfraError
from elh.tasks.base import TaskInstance, get_family

STATIC_GROUPS = ("P-all", "A0", "B-rules", "P-hand", "P-all-ep")   # fixed memory, never train
KNOWN_GROUPS = ("A", "B", "C", "D") + STATIC_GROUPS
NON_LEARNING = ("A",) + STATIC_GROUPS   # never scheduled for train phases
DEV_SEEDS = (3000, 4000)   # [start, end) every dev seed must fall in


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskSpec(_Strict):
    """Either the product causes x themes x seeds, or an explicit `items` list of [cause, theme, seed]
    (for panels that are not a product, such as a house_lib calibration panel)."""

    causes: list[str] | None = Field(None, min_length=1)
    themes: list[str] | None = Field(None, min_length=1)
    seeds: tuple[int, int] | None = None   # [start, end)
    items: list[tuple[str, str, int]] | None = Field(None, min_length=1)

    @field_validator("seeds")
    @classmethod
    def _range(cls, v: tuple[int, int] | None) -> tuple[int, int] | None:
        if v is not None and v[1] <= v[0]:
            raise ValueError(f"seed range {list(v)} is empty; use [start, end)")
        return v

    @model_validator(mode="after")
    def _one_form(self) -> "TaskSpec":
        product = (self.causes, self.themes, self.seeds)
        if self.items is None and None in product:
            raise ValueError("tasks need causes, themes and seeds, or items")
        if self.items is not None and any(x is not None for x in product):
            raise ValueError("tasks take either causes/themes/seeds or items, not both")
        if self.items is not None and len(set(self.items)) != len(self.items):
            raise ValueError("tasks items must be distinct")
        return self

    def triples(self) -> list[tuple[str, str, int]]:
        if self.items is not None:
            return [tuple(it) for it in self.items]
        return [(c, th, s) for c in self.causes for th in self.themes for s in range(*self.seeds)]

    def cells(self) -> set[tuple[str, str]]:
        return {(c, t) for c, t, _ in self.triples()}


class PhaseSpec(_Strict):
    label: str | None = None
    tasks: TaskSpec


class DevSpec(_Strict):
    """Group D's dev panel: the tasks of the promotion gate (and of GEPA's search)."""

    tasks: TaskSpec


class StaticSpec(_Strict):
    """Where P-all's rules come from: `group`'s train episodes in `from_experiment`, history by history."""

    from_experiment: str
    group: str = "B"
    after: int = Field(ge=1)


PHASE_KINDS = ("train", "eval", "retention")


class Phase(_Strict):
    train: PhaseSpec | None = None
    eval: PhaseSpec | None = None
    retention: PhaseSpec | None = None

    @model_validator(mode="after")
    def _one(self) -> "Phase":
        if sum(getattr(self, k) is not None for k in PHASE_KINDS) != 1:
            raise ValueError("each phase is exactly one of `train`, `eval` or `retention`")
        return self

    @property
    def kind(self) -> str:
        return next(k for k in PHASE_KINDS if getattr(self, k) is not None)

    @property
    def spec(self) -> PhaseSpec:
        return getattr(self, self.kind)


class Experiment(_Strict):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    base_config: str
    overrides: dict[str, Any] = Field(default_factory=dict)
    family: str = "cpu_mem"
    histories: int = Field(1, ge=1)
    groups: list[str] = Field(min_length=1, validation_alias=AliasChoices("groups", "arms"))  # "arms": legacy key
    phases: list[Phase] = Field(min_length=1)
    dev: DevSpec | None = None
    static: StaticSpec | None = None

    @model_validator(mode="after")
    def _static(self) -> "Experiment":
        needs = [g for g in ("P-all", "B-rules", "P-all-ep") if g in self.groups]
        if needs and self.static is None:
            raise ValueError(f"groups {needs} need a `static` section (the experiment and group their rules come from)")
        if "P-hand" in self.groups and not self.family.startswith("house_lib"):
            raise ValueError(f"group P-hand needs a house_lib world (a handbook of its lessons), not {self.family}")
        return self

    @field_validator("groups")
    @classmethod
    def _groups(cls, v: list[str]) -> list[str]:
        bad = [a for a in v if a not in KNOWN_GROUPS]
        if bad or len(set(v)) != len(v):
            raise ValueError(f"groups must be distinct values from {KNOWN_GROUPS}, got {v}")
        return v


@dataclass(frozen=True)
class Step:
    """One scheduled episode."""

    history: int
    phase: str        # label
    kind: str         # train | eval | retention
    group: str
    task: TaskInstance
    after: int        # train episodes scheduled before this step in its history
    ordinal: int      # train: position in the train stream; eval/retention: position in the phase


SPLIT_OF_KIND = {"train": "train", "eval": "checkpoint", "retention": "retention"}


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
        if key.split(".")[0] in ("name", "groups", "histories", "families"):
            raise KeyError(f"override {key!r}: set it in the experiment file, not as an override")
        _set_dotted(base, key, value)
    base.update(name=f"{base['name']}-{exp.name}", groups=list(exp.groups), histories=exp.histories,
                families=[exp.family])
    cfg = ExperimentConfig.model_validate(base)
    family = get_family(exp.family)
    specs = [ph.spec.tasks for ph in exp.phases] + ([exp.dev.tasks] if exp.dev else [])
    for t in specs:
        trip = t.triples()
        unknown = sorted({c for c, _, _ in trip if c not in family.causes} | {x for _, x, _ in trip if x not in family.themes})
        if unknown:
            raise ValueError(f"unknown causes/themes for family {exp.family}: {unknown}")
    labels = [label for label, _ in _labelled(exp)]
    if len(set(labels)) != len(labels):
        raise ValueError(f"phase labels must be unique: {labels}")
    _check_cells(exp)
    return exp, cfg


def _check_cells(exp: Experiment) -> None:
    """SPEC §4.3 over (cause, theme) cells: retention and dev use train cells only; dev never uses a
    cell an eval or retention phase measures; dev seeds stay in DEV_SEEDS. Group D needs a dev panel."""
    cells = {k: set() for k in PHASE_KINDS}
    for ph in exp.phases:
        cells[ph.kind] |= ph.spec.tasks.cells()
    extra = cells["retention"] - cells["train"]
    if extra:
        raise ValueError(f"retention phases may only use train cells; not trained: {sorted(extra)}")
    if exp.dev is None:
        if "D" in exp.groups:
            raise ValueError("group D needs a `dev` section (the promotion gate's task panel)")
        return
    dev = exp.dev.tasks
    if dev.seeds is None:
        raise ValueError("the dev panel takes causes/themes/seeds, not items")
    if not (DEV_SEEDS[0] <= dev.seeds[0] and dev.seeds[1] <= DEV_SEEDS[1]):
        raise ValueError(f"dev seeds {list(dev.seeds)} must lie in [{DEV_SEEDS[0]}, {DEV_SEEDS[1]})")
    not_train = dev.cells() - cells["train"]
    if not_train:
        raise ValueError(f"the dev panel may only use train cells; not trained: {sorted(not_train)}")
    measured = dev.cells() & (cells["eval"] | cells["retention"])
    if measured:
        raise ValueError(f"the dev panel uses cells an eval/retention phase measures: {sorted(measured)}")


def _labelled(exp: Experiment) -> list[tuple[str, Phase]]:
    """Phase labels; defaults are `train-<k>` (k-th train phase), `after-<n>` (eval; n = train
    episodes before it) and `retention-<n>`."""
    out, n_train, k = [], 0, 0
    for ph in exp.phases:
        t = ph.spec.tasks
        size = len(t.triples())
        k += ph.kind == "train"
        default = {"train": f"train-{k}", "eval": f"after-{n_train}", "retention": f"retention-{n_train}"}[ph.kind]
        out.append((ph.spec.label or default, ph))
        if ph.kind == "train":
            n_train += size
    return out


def expand_tasks(family, spec: TaskSpec, *, seed: int, history: int, phase_index: int | str) -> list[TaskInstance]:
    tasks = [family.generate(c, th, s) for c, th, s in spec.triples()]
    random.Random(f"{seed}/h{history}/p{phase_index}").shuffle(tasks)
    return tasks


def dev_tasks(exp: Experiment, cfg: ExperimentConfig, history: int) -> list[TaskInstance]:
    """Group D's dev panel for one history (same tasks in every history, order shuffled per history)."""
    if exp.dev is None:
        return []
    return expand_tasks(get_family(exp.family), exp.dev.tasks, seed=cfg.schedule.seed, history=history,
                        phase_index="dev")


def schedule(exp: Experiment, cfg: ExperimentConfig) -> list[Step]:
    """The full ordered episode list: per history, phases in order; groups interleaved per task.
    Group D's dev episodes are not scheduled here: they run when an induction makes a candidate."""
    family = get_family(exp.family)
    learners = [a for a in exp.groups if a not in NON_LEARNING]
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
                    steps += [Step(h, label, ph.kind, a, task, n_train, j) for a in exp.groups]
    return steps


def induction_points(exp: Experiment, cfg: ExperimentConfig) -> list[int]:
    """Train-episode counts after which group D may induce (each only if a new verified success joined
    the pool since the last induction)."""
    if "D" not in exp.groups or cfg.workflows is None:
        return []
    n = sum(1 for s in schedule(exp, cfg) if s.history == 0 and s.kind == "train" and s.group == "D")
    m = cfg.workflows.induce_every
    return list(range(m, n + 1, m))


def gate_tasks_per_induction(exp: Experiment, cfg: ExperimentConfig) -> int:
    panel = len(dev_tasks(exp, cfg, 0))
    if cfg.procedure_source == "gepa" and cfg.gepa is not None and panel:
        from elh.learner.gepa_opt import split_dev_pools

        return len(split_dev_pools([t.task_id for t in dev_tasks(exp, cfg, 0)], cfg.gepa, 0).promotion)
    return panel


def batch_id(exp: Experiment, step: Step) -> str:
    return f"{exp.name}/{step.history}"


def print_schedule(exp: Experiment, cfg: ExperimentConfig, steps: list[Step], out=print) -> None:
    out(f"experiment {exp.name}: run {run_id_for(cfg)}, {len(steps)} scheduled episodes")
    cells: dict[tuple, list[Step]] = defaultdict(list)
    for s in steps:
        cells[(s.history, s.phase, s.kind, s.group)].append(s)
    for (h, label, kind, group), ss in cells.items():
        if kind == "train":
            where = f"train episodes {ss[0].after + 1}-{ss[-1].after + 1}"
        elif group == "P-hand":
            where = f"handbook of {exp.family}"
        elif group in STATIC_GROUPS and exp.static:
            where = f"static: {exp.static.from_experiment} {exp.static.group} after {exp.static.after}"
        else:
            where = f"after {ss[0].after}"
        out(f"h{h} {label} ({kind}, {where}) group {group}: {len(ss)} tasks")
        for s in ss:
            out(f"    {s.task.task_id}")
    if exp.dev is None:
        return
    panel = dev_tasks(exp, cfg, 0)
    out(f"dev panel ({len(panel)} tasks, split dev, charged as learning; order shuffled per history):")
    for t in sorted(panel, key=lambda t: t.task_id):
        out(f"    {t.task_id}")
    points = induction_points(exp, cfg)
    if not points:
        return
    w = cfg.workflows
    gate = gate_tasks_per_induction(exp, cfg)
    per = 2 * gate * w.repeats
    out(f"group D ({cfg.procedure_source}): induction points after train episodes {points} per history "
        f"(each only if a new verified success joined the pool)")
    if cfg.procedure_source == "gepa":
        out(f"  GEPA search: up to {cfg.gepa.max_metric_calls} dev episodes per induction")
    out(f"  gate: {gate} tasks x {w.repeats} repeats x incumbent/candidate = {per} dev episodes per induction; "
        f"up to {per * len(points) * exp.histories} in all")


# --------------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------------


@dataclass
class _GroupD:
    """Group D's learning state across a batch, per history."""

    make_induction_client: Callable[[], Any]
    make_reflection_client: Callable[[], Any]
    last_pool: dict[int, list[str]] = field(default_factory=dict)
    inductions: dict[int, int] = field(default_factory=dict)
    dev_seq: dict[int, int] = field(default_factory=dict)
    _induction: Any = None
    _reflection: Any = None

    def induction_client(self):
        if self._induction is None:
            self._induction = self.make_induction_client()
        return self._induction

    def reflection_client(self):
        if self._reflection is None:
            self._reflection = self.make_reflection_client()
        return self._reflection


def run_batch(exp: Experiment, cfg: ExperimentConfig, store, make_client: Callable[[TaskInstance], Any], *,
              sandbox_backend: str | None = None, out=print,
              make_induction_client: Callable[[], Any] | None = None,
              make_reflection_client: Callable[[], Any] | None = None) -> list[dict]:
    """Run the schedule in order. An InfraError skips that episode; anything else stops the batch.
    Returns one result dict per scheduled episode, plus one per group-D dev episode (kind "dev").

    `make_induction_client()` gives AWM's induction model (default: the registered agent model,
    `workflows.make_induction_client`); `make_reflection_client()` gives GEPA's reflection model
    (default: the agent model's ModelClient). Both are built only when first needed."""
    run_id = run_id_for(cfg)
    prior = store.db.execute("SELECT COUNT(*) FROM episodes WHERE run_id = ? AND batch_id LIKE ?",
                             (run_id, f"{exp.name}/%")).fetchone()[0]
    if prior:
        raise RuntimeError(f"run {run_id} already has {prior} episodes from experiment {exp.name}; "
                           "rename the experiment to run it again")

    steps = schedule(exp, cfg)
    last_retention_d = {(s.history, s.phase): i for i, s in enumerate(steps) if s.kind == "retention" and s.group == "D"}
    group_d = _GroupD(make_induction_client or (lambda: wf.make_induction_client(cfg.model)),
                 make_reflection_client or _default_reflection_client(cfg))
    live: dict[tuple[int, str], Any] = {}        # (history, group) -> training bank
    snaps: dict[tuple[int, str, int], str] = {}  # (history, group, after) -> snapshot id
    static: dict[tuple[int, str], Any] = {}
    results: list[dict] = []
    for n, s in enumerate(steps, 1):
        split = SPLIT_OF_KIND[s.kind]
        checkpoint = s.after if s.kind != "train" else None
        if s.group == "A":  # never scheduled for train phases
            memory = NullMemory()
        elif s.group in STATIC_GROUPS:
            if (s.history, s.group) not in static:
                static[s.history, s.group] = static_memory(store, exp, s.history, s.group)
            memory = static[s.history, s.group]
            checkpoint = exp.static.after if exp.static and s.group != "P-hand" else s.after
        elif s.kind == "train":
            if (s.history, s.group) not in live:
                live[s.history, s.group] = memory_for(cfg, s.group, s.history, "train", store=store)
            memory = live[s.history, s.group]
        else:
            key = (s.history, s.group, s.after)
            if key not in snaps:
                snaps[key] = _snapshot(cfg, store, live, run_id, *key, family=exp.family)
            memory = open_snapshot(cfg, store, snaps[key])
        res = {"history": s.history, "phase": s.phase, "kind": s.kind, "group": s.group,
               "task_id": s.task.task_id, "passed": None, "turns": None, "wall_s": None, "error": None}
        t0 = time.monotonic()
        try:
            _, result, verdict = run_single_episode(
                cfg, store, s.task, make_client(s.task), group=s.group, split=split, history=s.history,
                ordinal=s.ordinal, checkpoint=checkpoint, memory=memory,
                sandbox_backend=sandbox_backend, batch_id=batch_id(exp, s))
            res.update(passed=verdict.passed, turns=result.turns, wall_s=result.wall_s)
            out(f"[{n}/{len(steps)}] h{s.history} {s.phase} {s.group} {s.task.task_id} "
                f"passed={int(verdict.passed)} turns={result.turns} wall={result.wall_s:.1f}s")
        except InfraError as exc:
            res["error"] = str(exc)
            out(f"[{n}/{len(steps)}] h{s.history} {s.phase} {s.group} {s.task.task_id} "
                f"INFRA ERROR after {time.monotonic() - t0:.1f}s: {exc}")
        results.append(res)
        if s.group == "D" and s.kind == "train":
            results += learn_procedures(exp, cfg, store, make_client, group_d, history=s.history,
                                        done=s.ordinal + 1, sandbox_backend=sandbox_backend, out=out)
        if last_retention_d.get((s.history, s.phase)) == n - 1:
            d = retention_rollback(store, cfg, run_id=run_id, history=s.history, family=exp.family,
                                   batch=batch_id(exp, s), checkpoint=s.after)
            if d is not None:
                out(f"h{s.history} {s.phase} group D retention check: "
                    f"{'ROLLED BACK' if d.promote else 'held'} ({'; '.join(d.reasons)})")
    return results


def static_rules(store, exp: Experiment, history: int) -> list[MemoryItem]:
    """P-all's block for one history: the source group's rules from its first `after` train episodes,
    in training order, rendered exactly as group B's verbatim rule items. Episodes that wrote no
    rule are skipped; a missing or ambiguous source stream is an error."""
    sp = exp.static
    rows = store.db.execute(
        "SELECT run_id, ordinal, episode_id, rule_json FROM episodes WHERE batch_id = ? AND test_group = ? "
        "AND split = 'train' AND ordinal < ? ORDER BY ordinal",
        (f"{sp.from_experiment}/{history}", sp.group, sp.after)).fetchall()
    if len({r[0] for r in rows}) > 1:
        raise RuntimeError(f"{sp.from_experiment}/{history} group {sp.group} spans several runs")
    if [r[1] for r in rows] != list(range(sp.after)):
        raise RuntimeError(f"{sp.from_experiment}/{history} group {sp.group}: need train ordinals "
                           f"0..{sp.after - 1}, found {[r[1] for r in rows]}")
    items = []
    for _, ordinal, episode_id, rule_json in rows:
        rule = (json.loads(rule_json) or {}).get("rule") if rule_json else None
        if rule:
            items.append(MemoryItem(f"rule-{ordinal:02d}", render_rule_verbatim(rule),
                                    document_id=episode_id + RULE_DOC_SUFFIX))
    return items


def static_rules_and_records(store, exp: Experiment, history: int) -> list[MemoryItem]:
    """P-all-ep's block: P-all's rules, each followed by the source snapshot's record units from the
    same episode (ordered by text, for a stable order), in training order. Every record unit must
    come from one of the first `after` train episodes."""
    sp = exp.static
    rules = {it.document_id.removesuffix(RULE_DOC_SUFFIX): it for it in static_rules(store, exp, history)}
    episodes = [r[0] for r in store.db.execute(
        "SELECT episode_id FROM episodes WHERE batch_id = ? AND test_group = ? AND split = 'train' AND ordinal < ? "
        "ORDER BY ordinal", (f"{sp.from_experiment}/{history}", sp.group, sp.after))]
    records: dict[str, list[dict]] = defaultdict(list)
    for u in source_snapshot(store, exp, history).record_units():
        records[u.get("document_id")].append(u)
    stray = sorted(set(records) - set(episodes), key=str)
    if stray:
        raise RuntimeError(f"{sp.from_experiment}/{history}: record units from unknown documents {stray}")
    items = []
    for ordinal, episode_id in enumerate(episodes):
        if episode_id in rules:
            items.append(rules[episode_id])
        for j, u in enumerate(sorted(records[episode_id], key=lambda u: u["text"]), 1):
            items.append(MemoryItem(f"record-{ordinal:02d}-{j:02d}", u["text"], document_id=episode_id))
    return items


def source_snapshot(store, exp: Experiment, history: int):
    """The source group's read-only snapshot bank for one history, opened under the source run's
    own stored registration (snapshots are tied to the run that took them)."""
    sp = exp.static
    runs = {r[0] for r in store.db.execute("SELECT DISTINCT run_id FROM episodes WHERE batch_id = ?",
                                           (f"{sp.from_experiment}/{history}",))}
    if len(runs) != 1:
        raise RuntimeError(f"{sp.from_experiment}/{history}: expected one source run, found {sorted(runs)}")
    run_id = runs.pop()
    row = store.db.execute("SELECT snapshot_id FROM memory_snapshots WHERE run_id = ? AND history = ? AND "
                           "test_group = ? AND after_episode = ?", (run_id, history, sp.group, sp.after)).fetchone()
    if row is None:
        raise RuntimeError(f"no {sp.group} snapshot after {sp.after} in {sp.from_experiment}/{history}")
    src_cfg = ExperimentConfig.model_validate_json(store.run_config_json(run_id))
    return open_snapshot(src_cfg, store, row[0])


def handbook_items(family_name: str) -> list[MemoryItem]:
    from elh.tasks.families.house_lib.calibration import handbook_lines

    return [MemoryItem(f"hand-{i:02d}", line) for i, line in enumerate(handbook_lines(get_family(family_name)), 1)]


def static_memory(store, exp: Experiment, history: int, group: str):
    """P-all: the source history's frozen rules; A0: an empty block (memory-group instructions only);
    B-rules: the source snapshot, retrieving rules only; P-hand: the world's handbook."""
    if group == "P-hand":
        return StaticMemory(handbook_items(exp.family), f"static:handbook/{exp.family}")
    if group == "A0":
        return StaticMemory([], "static:A0")
    if group == "B-rules":
        return RulesOnlyMemory(source_snapshot(store, exp, history))
    sp = exp.static
    if group == "P-all-ep":
        return StaticMemory(static_rules_and_records(store, exp, history),
                            f"static+records:{sp.from_experiment}/{history}/{sp.group}/e{sp.after}")
    items = static_rules(store, exp, history)
    return StaticMemory(items, f"static:{sp.from_experiment}/{history}/{sp.group}/e{sp.after}")


def _default_reflection_client(cfg: ExperimentConfig) -> Callable[[], Any]:
    def make():
        from elh.runner.models import make_client

        return make_client(cfg.model)
    return make


def _charge(store, cfg: ExperimentConfig, history: int, entries, purpose: str) -> None:
    """Learning spend with no episode (induction, reflection): ledger rows tagged with the stream."""
    for e in entries:
        store.insert("ledger", {"run_id": run_id_for(cfg), "episode_id": None, "category": e.category,
                                "component": e.component, "units": e.units, "unit": e.unit, "usd": e.usd,
                                "history": history, "test_group": "D", "purpose": purpose})


def learn_procedures(exp: Experiment, cfg: ExperimentConfig, store, make_client, group_d: _GroupD, *, history: int,
                     done: int, sandbox_backend: str | None = None, out=print) -> list[dict]:
    """Group D's learning step after its `done`-th train episode of a history: maybe induce a
    candidate, then run the paired dev gate and record the decision. Returns the dev results."""
    w = cfg.workflows
    run_id, fam = run_id_for(cfg), exp.family
    pool = wf.eligible_pool(store, run_id, history, "D", fam)
    pool_ids = [r["episode_id"] for r in pool]
    if not wf.should_induce(done, pool_ids, group_d.last_pool.get(history, []), w):
        return []
    group_d.last_pool[history] = pool_ids
    index = group_d.inductions[history] = group_d.inductions.get(history, 0) + 1
    inc_rev, inc_lib = wf.promoted_library(store, run_id=run_id, history=history, group="D", family=fam)
    panel = dev_tasks(exp, cfg, history)
    results: list[dict] = []
    if cfg.procedure_source == "awm":
        ind = wf.induce(store, pool, group_d.induction_client(), cfg=w,
                        seed=wf.induction_seed(run_id, history, "D", fam, index), family=fam)
        _charge(store, cfg, history, wf.learning_entries(cfg, ind), "awm_induction")
        rev = wf.record_candidate(store, ind, run_id=run_id, history=history)
        gate = panel
        out(f"h{history} group D induction {index} after {done} train episodes: {len(pool)} verified successes, "
            f"{len(ind.library)} workflows" + (f" -> candidate revision {rev}" if rev else " (empty; nothing to validate)"))
    else:
        rev, gate, gepa_results = _gepa_candidate(exp, cfg, store, make_client, group_d, history=history, done=done,
                                                  incumbent=inc_lib, sandbox_backend=sandbox_backend, out=out)
        results += gepa_results
        out(f"h{history} group D GEPA {index} after {done} train episodes"
            + (f" -> candidate revision {rev}" if rev else ": no candidate beyond the incumbent"))
    if rev is None:
        return results
    cand_lib = wf.load_library(store, run_id=run_id, history=history, group="D", family=fam, revision=rev)
    sides = {"incumbent": (inc_rev, inc_lib), "candidate": (rev, cand_lib)}
    view = memory_for(cfg, "D", history, "dev", store=store)
    rows: dict[str, list] = {"incumbent": [], "candidate": []}
    error = None
    for k in range(w.repeats):
        for task in gate:
            order = ["incumbent", "candidate"]
            random.Random(f"{cfg.schedule.seed}/h{history}/gate{index}/{k}/{task.task_id}").shuffle(order)
            for role in order:
                try:
                    eid, _, _, res = _dev_episode(exp, cfg, store, make_client, group_d, task, view, history=history,
                                            done=done, procedures=sides[role], label=role,
                                            sandbox_backend=sandbox_backend, out=out)
                except InfraError as exc:
                    error = f"{role} on {task.task_id}: {exc}"
                    break
                results.append(res)
                rows[role].append(wf.dev_result_from_row(store.episode(eid), k, cfg.budgets.max_total_tokens))
                rows.setdefault(f"{role}_ids", []).append(eid)
            if error:
                break
        if error:
            break
    if error:
        decision = wf.GateDecision(False, (f"validation incomplete: {error}",),
                                   {"gate": "paired-dev-v1", "incomplete": True})
    else:
        decision = wf.promote(rows["incumbent"], rows["candidate"], w)
    decision = wf.GateDecision(decision.promote, decision.reasons, {
        **decision.stats, "source": cfg.procedure_source, "after_train_episodes": done, "induction": index,
        "incumbent_revision": inc_rev, "gate_task_ids": [t.task_id for t in gate],
        "incumbent_episode_ids": rows.get("incumbent_ids", []), "candidate_episode_ids": rows.get("candidate_ids", [])})
    wf.decide_candidate(store, run_id=run_id, history=history, group="D", family=fam, revision=rev, decision=decision)
    st = decision.stats
    out(f"h{history} group D gate for revision {rev} vs {inc_rev if inc_rev is not None else 'none'}: "
        f"{'PROMOTED' if decision.promote else 'rejected'} "
        f"(mean reward {st.get('incumbent_mean_reward', float('nan')):.3f} -> "
        f"{st.get('candidate_mean_reward', float('nan')):.3f}, passes {st.get('incumbent_passes', '-')} -> "
        f"{st.get('candidate_passes', '-')}; {'; '.join(decision.reasons)})")
    return results


def _dev_episode(exp, cfg, store, make_client, group_d: _GroupD, task: TaskInstance, view, *, history: int, done: int,
                 procedures, label: str, sandbox_backend, out, retries: int = 1):
    """One group-D dev episode with an explicit library, charged as learning. An InfraError is retried
    `retries` times, then raised."""
    for attempt in range(retries + 1):
        seq = group_d.dev_seq[history] = group_d.dev_seq.get(history, 0) + 1
        try:
            eid, result, verdict = run_single_episode(
                cfg, store, task, make_client(task), group="D", split="dev", history=history, ordinal=seq,
                checkpoint=done, memory=view, sandbox_backend=sandbox_backend, batch_id=f"{exp.name}/{history}",
                charge_as="learning", procedures=procedures)
        except InfraError as exc:
            out(f"  h{history} dev@{done} {label} {task.task_id} INFRA ERROR: {exc}")
            if attempt == retries:
                raise
            continue
        out(f"  h{history} dev@{done} {label} {task.task_id} passed={int(verdict.passed)} turns={result.turns} "
            f"wall={result.wall_s:.1f}s")
        return eid, result, verdict, {"history": history, "phase": f"dev@{done}-{label}", "kind": "dev",
                                      "test_group": "D", "task_id": task.task_id, "passed": verdict.passed,
                                      "turns": result.turns, "wall_s": result.wall_s, "error": None}
    raise AssertionError("unreachable")


def _gepa_candidate(exp, cfg, store, make_client, group_d: _GroupD, *, history: int, done: int, incumbent,
                    sandbox_backend, out) -> tuple[int | None, list[TaskInstance], list[dict]]:
    """GEPA seeded with the incumbent's text on the dev panel's reflection/validation pools; the
    winner (if it differs from the seed) becomes a candidate revision, gated on the promotion pool."""
    from elh.learner import gepa_opt as go
    from elh.learner.utility import episode_cost_usd
    from elh.memory.hindsight import workspace_diff

    g, fam = cfg.gepa, exp.family
    panel = {t.task_id: t for t in dev_tasks(exp, cfg, history)}
    pools = go.split_dev_pools(list(panel), g, f"{cfg.schedule.seed}/h{history}/gepa-pools")
    view = memory_for(cfg, "D", history, "dev", store=store)
    results: list[dict] = []

    def run_dev(task: TaskInstance, procedure: str) -> go.DevRollout:
        lib = wf.library_from_procedure(fam, procedure)
        eid, result, verdict, res = _dev_episode(
            exp, cfg, store, make_client, group_d, task, view, history=history, done=done, procedures=(None, lib),
            label="gepa", sandbox_backend=sandbox_backend, out=out, retries=0)  # GEPA's adapter retries
        results.append(res)
        final_ws = store.get_workspace(store.episode(eid)["final_workspace_sha256"])
        return go.DevRollout(verdict, result, episode_cost_usd(store.db, eid, ("learning",)), eid,
                             workspace_diff(task.workspace, final_ws))

    seed_text = wf.procedure_text(incumbent)
    outcome = go.run_gepa(g, seed_text, pools, run_dev, group_d.reflection_client(), tasks=panel,
                          split_of=lambda _t: "dev", promotion_check=False)
    _charge(store, cfg, history, go.reflection_ledger_entries(cfg, cfg.model.id, outcome.reflection_calls),
            "gepa_reflection")
    gate = [panel[t] for t in pools.promotion]
    if go.text_sha(outcome.best_procedure.strip()) == go.text_sha(seed_text.strip()):
        return None, gate, results
    lib = wf.library_from_procedure(fam, outcome.best_procedure)
    usage: dict[str, int] = {}
    for c in outcome.reflection_calls:
        for k, v in (c.get("usage") or {}).items():
            usage[k] = usage.get(k, 0) + (v or 0)
    ind = wf.Induction(
        library=lib, raw_output=outcome.best_procedure,
        prompt_sha256=go.text_sha(go.reflection_prompt(g)), source_episode_ids=outcome.source_episode_ids(),
        usage=usage, model=cfg.model.id, seed=g.seed, dropped=[],
        extra={"source": "gepa", "gepa": {
            "best_idx": outcome.best_idx, "total_metric_calls": outcome.total_metric_calls,
            "candidates": [{k: h[k] for k in ("idx", "sha256", "parents", "val_mean")} for h in outcome.history],
            "pools": pools.model_dump(mode="json"), "gepa_version": go.GEPA_VERSION,
            "seed_procedure_sha256": go.text_sha(seed_text)}})
    return wf.record_candidate(store, ind, run_id=run_id_for(cfg), history=history), gate, results


def retention_rollback(store, cfg: ExperimentConfig, *, run_id: str, history: int, family: str, batch: str,
                       checkpoint: int) -> wf.GateDecision | None:
    """Auto-rollback hook (SPEC §10), run after group D's last episode of a retention phase.

    Compares group D's retention episodes at `checkpoint` with those of the most recent earlier
    retention checkpoint of the same batch and history, paired by (task, repeat in run order) on the
    tasks both measured, with `workflows.rollback`. Rolls back (`workflows.apply_rollback`) only when
    the rule fires, the library pinned at this checkpoint differs from the earlier one, and it is
    still the promoted revision. Returns the decision (None when there is no earlier panel)."""
    rows = store.db.execute(
        "SELECT * FROM episodes WHERE run_id = ? AND history = ? AND test_group = 'D' AND split = 'retention' "
        "AND batch_id = ? AND checkpoint <= ? ORDER BY started_at, episode_id",
        (run_id, history, batch, checkpoint)).fetchall()
    earlier = sorted({r["checkpoint"] for r in rows if r["checkpoint"] < checkpoint})
    if not earlier:
        return None
    prev_ck = earlier[-1]

    def results(ck: int) -> tuple[dict, set]:
        seen: dict[str, int] = defaultdict(int)
        out, revs = {}, set()
        for r in (r for r in rows if r["checkpoint"] == ck):
            out[(r["task_id"], seen[r["task_id"]])] = wf.dev_result_from_row(r, seen[r["task_id"]],
                                                                             cfg.budgets.max_total_tokens)
            seen[r["task_id"]] += 1
            revs.add(json.loads(r["retrieval_json"] or "{}").get("procedure_revision"))
        return out, revs

    prev, prev_revs = results(prev_ck)
    cur, cur_revs = results(checkpoint)
    common = sorted(set(prev) & set(cur))
    if not common:
        return None
    decision = wf.rollback([prev[k] for k in common], [cur[k] for k in common], cfg.workflows)
    stats = {**decision.stats, "previous_checkpoint": prev_ck, "checkpoint": checkpoint,
             "previous_revisions": sorted(prev_revs, key=str), "revisions": sorted(cur_revs, key=str)}
    if not decision.promote:
        return wf.GateDecision(False, decision.reasons, stats)
    promoted = wf.promoted_revision(store, run_id=run_id, history=history, group="D", family=family)
    reasons = list(decision.reasons)
    if len(cur_revs) != 1 or cur_revs == prev_revs:
        reasons.append("not rolled back: the library did not change between the two checkpoints")
        return wf.GateDecision(False, tuple(reasons), stats)
    (cur_rev,) = cur_revs
    if cur_rev is None or promoted != cur_rev:
        reasons.append(f"not rolled back: revision {cur_rev} is no longer the promoted one ({promoted})")
        return wf.GateDecision(False, tuple(reasons), stats)
    now = wf.apply_rollback(store, run_id=run_id, history=history, group="D", family=family,
                            evidence={**decision.to_dict(), **stats})
    return wf.GateDecision(True, tuple(reasons) + (f"revision {cur_rev} rolled back; active now: {now}",),
                           {**stats, "active_revision": now})


def _snapshot(cfg: ExperimentConfig, store, live: dict, run_id: str, history: int, group: str, after: int, *,
              family: str | None = None) -> str:
    """Snapshot id for (history, group) after `after` train episodes: reused if one exists."""
    row = store.db.execute("SELECT snapshot_id FROM memory_snapshots WHERE run_id = ? AND history = ? "
                           "AND test_group = ? AND after_episode = ?", (run_id, history, group, after)).fetchone()
    if row:
        return row[0]
    if (history, group) not in live:  # eval before any training: snapshot the (empty) bank
        live[history, group] = memory_for(cfg, group, history, "train", store=store)
    return snapshot_memory(cfg, store, live[history, group], history=history, group=group, after_episode=after,
                           family=family)


def summarize(results: list[dict]) -> list[str]:
    """One line per (history, phase, group): n, pass rate, mean turns, infra errors."""
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        cells[(r["history"], r["phase"], r["group"])].append(r)
    lines = [f"{'hist':4s} {'phase':16s} {'group':5s} {'n':>3s} {'pass':>6s} {'turns':>6s} {'infra':>5s}"]
    for (h, phase, group), rs in cells.items():
        ok = [r for r in rs if r["error"] is None]
        rate = f"{sum(r['passed'] for r in ok) / len(ok):.2f}" if ok else "-"
        turns = f"{statistics.mean(r['turns'] for r in ok):.1f}" if ok else "-"
        lines.append(f"h{h:<3d} {phase:16s} {group:5s} {len(ok):3d} {rate:>6s} {turns:>6s} {len(rs) - len(ok):5d}")
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
