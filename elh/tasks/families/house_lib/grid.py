"""Core grid task panels and scoring over kept lessons (docs/design/house-lib-core-grid.md §3-4).

- Kept lessons: the calibration filter's verdict per world, stored in
  `experiments/calibration/<world>.json` (written by `analysis/exp17_calibration.py --write`).
  A dropped lesson stays in the world (library, docs, handbook unchanged) but is never an
  evaluation ticket's primary lesson and its checks are left out of every score.
- Panels: training is N = 2L tickets, every lesson (kept or not) the primary lesson of two, on train
  seeds; evaluation is EVAL_TASKS tickets on checkpoint seeds with primary lessons spread evenly over
  the kept lessons. Themes go round-robin; the batch shuffles the order per history.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
CALIBRATION_DIR = ROOT / "experiments" / "calibration"
TRAIN_PER_LESSON = 2
EVAL_TASKS = 24
TRAIN_SEED, EVAL_SEED = 1000, 2000  # train and checkpoint seed bases (SPEC §4.3)
SCORED_CHECKS = ("integrity", "public_tests", "feature")


def kept_lessons(world: str, path: Path | None = None) -> list[str]:
    """The world's kept lessons, in catalogue order; an uncalibrated world is an error."""
    path = path or CALIBRATION_DIR / f"{world}.json"
    if not path.exists():
        raise FileNotFoundError(f"world {world} has no calibration verdict at {path}")
    doc = json.loads(path.read_text())
    if doc["world"] != world:
        raise ValueError(f"{path} is for world {doc['world']}, not {world}")
    return [r["lesson"] for r in doc["lessons"] if r["keep"]]


def grid_tasks(family, kept: list[str], seed: int = 0) -> tuple[list[tuple[str, str, int]], list[tuple[str, str, int]]]:
    """(train items, eval items) as (cause, theme, seed) triples."""
    unknown = sorted(set(kept) - set(family.causes))
    if unknown or not kept:
        raise ValueError(f"kept lessons must be a non-empty subset of the world's; unknown {unknown}")
    rng = random.Random(f"house_lib/grid/{family.name}/{seed}")
    themes = list(family.themes)
    rng.shuffle(themes)
    train = [(lesson, themes[(i * TRAIN_PER_LESSON + j) % len(themes)], TRAIN_SEED + i * TRAIN_PER_LESSON + j)
             for i, lesson in enumerate(family.causes) for j in range(TRAIN_PER_LESSON)]
    order = [k for k in family.causes if k in kept]  # catalogue order
    rng.shuffle(order)
    evals = [(order[i % len(order)], themes[i % len(themes)], EVAL_SEED + i) for i in range(EVAL_TASKS)]
    return train, evals


def scored_checks(checks: dict[str, bool], kept: list[str]) -> dict[str, bool]:
    """A verdict's checks with dropped lessons' left out."""
    keep = {f"lesson:{k}" for k in kept}
    return {k: v for k, v in checks.items() if not k.startswith("lesson:") or k in keep}


def scored_pass(checks: dict[str, bool], kept: list[str]) -> bool:
    """Task pass over kept lessons: the lesson-neutral checks and every kept lesson's check."""
    s = scored_checks(checks, kept)
    return all(s.get(k, False) for k in SCORED_CHECKS) and all(s.values())


# The trial (4.1) and full grid (4.3) settings: exp21's memory settings, the house_lib placement.
OVERRIDES = {"memory_service.record_format": "lesson-v3", "memory_service.recall_types": ["experience"],
             "memory_service.agent_search": True, "memory_service.experience_position": "after_ticket",
             "rules": {"enabled": True, "store": "verbatim", "context": True,
                       "feedback_where": True}}
HISTORIES = 3
# Full grid (4.3) recall, (memory_service.k, rules.recall_k, recall budget): chosen by exp28 (core-grid
# §9); 4.1 ran on the base config's (5, 2, 2048).
GRID_RECALL = (14, 12, 8192)


def recall_overrides(k: int, rule_k: int, budget: int) -> dict:
    """OVERRIDES with B's recall set to k items, rule_k of them rules, within budget tokens."""
    return {**OVERRIDES, "rules": {**OVERRIDES["rules"], "recall_k": rule_k}, "memory_service.k": k,
            "memory_service.recall_max_tokens": budget}


def batch_docs(family, kept: list[str], histories: int = HISTORIES,
               recall: tuple[int, int, int] | None = None) -> tuple[dict, dict]:
    """The two batches of one world: `exp22-grid-<world>` (A, B and P-hand; B trains, then all are
    evaluated) and `exp23-grid-static-<world>` (P-all, P-all-ep and P-oracle from exp22's B after
    training, same evaluation tickets). `recall` overrides B's recall (4.3: GRID_RECALL); B recalls
    while training too, so a new recall setting needs new B histories."""
    world = family.world.id
    train, evals = grid_tasks(family, kept)
    label = f"after-{len(train)}"
    common = {"base_config": "experiments/phase0-local.yaml",
              "overrides": recall_overrides(*recall) if recall else OVERRIDES,
              "family": f"house_lib/{world}", "histories": histories}
    ev = {"eval": {"label": label, "tasks": {"items": [list(t) for t in evals]}}}
    learn = {"name": f"exp22-grid-{world}", **common, "groups": ["A", "B", "P-hand"],
             "phases": [{"train": {"tasks": {"items": [list(t) for t in train]}}}, ev]}
    static = {"name": f"exp23-grid-static-{world}", **common, "groups": ["P-all", "P-all-ep", "P-oracle"],
              "static": {"from_experiment": learn["name"], "group": "B", "after": len(train)}, "phases": [ev]}
    return learn, static


# Full grid, stage 4.3 (core-grid §10): `exp29-*` learn, `exp30-*` static. 4.1's A and P-hand are reused
# at the worlds it ran (REUSED_L); a world with L >= GATE_L splits history 0 off as the gate.
WORLDS_43 = tuple(f"L{n}-c{c}-w1" for n in (4, 16, 64) for c in (0, 50))
REUSED_L, GATE_L = (16,), 64
# Replicate evaluation runs (decided 2026-11-07), at the primary cell only: B and P-all-ep run each
# (history, ticket) REPLICATES times, to split the history x ticket variance into episode noise and history effects.
REPLICATE_WORLD, REPLICATE_GROUPS, REPLICATES = "L64-c50-w1", ("B", "P-all-ep"), 3
# Parts rerun under a new name (a crashed run's bank is named after its run id, so a clean rerun needs
# a new name): (world, part) -> suffix after exp29/exp30. The L64-c50 gate crashed on 2026-11-10 after 7
# training episodes (rule-writer bug, fixed in 3807223); its rerun is exp29b/exp30b (owner, 2026-11-10).
RERUN = {("L64-c50-w1", "gate"): "b"}


def grid43_docs(family, kept: list[str], histories: int = HISTORIES) -> list[dict]:
    """Stage 4.3's batches for one world, in run order: learn then static, per part.
    A world at L >= GATE_L has two parts: `gate` (history 0) and `grid` (the remaining histories), so
    that the history-0 bank can be sized (`elh measure-block` on the gate's static batch, once its
    training is done) before the rest is run; the parts' histories, seeds and banks are those of one
    batch of all of them (`history_start`). Learn batches carry A and P-hand except where 4.1's are
    reused."""
    world, n = family.world.id, len(family.causes)
    learn, static = batch_docs(family, kept, histories, recall=GRID_RECALL)
    parts = [("gate", 0, 1), ("grid", 1, histories - 1)] if n >= GATE_L else [("grid", 0, histories)]
    out = []
    for part, start, count in parts:
        extra = {"history_start": start} if start else {}
        r = RERUN.get((world, part), "")
        lrn = {**learn, "name": f"exp29{r}-{part}-{world}", "histories": count, **extra,
               "groups": ["B"] if n in REUSED_L else learn["groups"]}
        st = {**static, "name": f"exp30{r}-{part}-static-{world}", "histories": count, **extra,
              "static": {**static["static"], "from_experiment": lrn["name"]}}
        if world == REPLICATE_WORLD:
            lrn["replicates"] = {g: REPLICATES for g in lrn["groups"] if g in REPLICATE_GROUPS}
            st["replicates"] = {g: REPLICATES for g in st["groups"] if g in REPLICATE_GROUPS}
        out += [lrn, st]
    return out


def write_43(worlds=WORLDS_43, out_dir: Path | None = None) -> list[Path]:
    """Write stage 4.3's batch files for the worlds that have a calibration verdict; say which do not."""
    from elh.tasks.base import get_family

    out_dir = out_dir or ROOT / "experiments" / "batches"
    written = []
    for world in worlds:
        try:
            kept = kept_lessons(world)
        except FileNotFoundError as exc:
            print(f"skipped {world}: {exc}")
            continue
        head = (f"# house_lib core grid stage 4.3 (estimation run), generated by grid.grid43_docs from\n"
                f"# experiments/calibration/{world}.json (docs/design/house-lib-core-grid.md §10); regenerate with\n"
                f"# `uv run python -m elh.tasks.families.house_lib.grid`.\n")
        for doc in grid43_docs(get_family(f"house_lib/{world}"), kept):
            path = out_dir / f"{doc['name']}.yaml"
            path.write_text(head + yaml.safe_dump(doc, sort_keys=False))
            written.append(path)
    return written


# Recall tuning before the full grid (core-grid §8): B's 4.1 banks replayed (`B-replay`) on tuning
# tickets (seeds 3000+, never in a reported panel) under (memory_service.k, rules.recall_k, recall budget).
TUNE_SEED, TUNE_TASKS = 3000, 12
TUNE_SETTINGS = ((5, 2, 2048), (8, 6, 4096), (14, 12, 8192))


def tune_tasks(family, kept: list[str]) -> list[tuple[str, str, int]]:
    rng = random.Random(f"house_lib/tune/{family.name}")
    themes, order = list(family.themes), [k for k in family.causes if k in kept]
    rng.shuffle(themes)
    rng.shuffle(order)
    return [(order[i % len(order)], themes[i % len(themes)], TUNE_SEED + i) for i in range(TUNE_TASKS)]


def tune_docs(family, kept: list[str], histories: int = HISTORIES) -> list[dict]:
    world, items = family.world.id, [list(t) for t in tune_tasks(family, kept)]
    after = TRAIN_PER_LESSON * len(family.causes)
    out = []
    for k, rule_k, budget in TUNE_SETTINGS:
        out.append({"name": f"exp28-tune-k{k}r{rule_k}-{world}", "base_config": "experiments/phase0-local.yaml",
                    "overrides": recall_overrides(k, rule_k, budget),
                    "family": f"house_lib/{world}", "histories": histories, "groups": ["B-replay"],
                    "static": {"from_experiment": f"exp22-grid-{world}", "group": "B", "after": after},
                    "phases": [{"eval": {"label": f"after-{after}", "tasks": {"items": items}}}]})
    out.append({"name": f"exp28-tune-pall-{world}", "base_config": "experiments/phase0-local.yaml",
                "overrides": OVERRIDES, "family": f"house_lib/{world}", "histories": histories, "groups": ["P-all"],
                "static": {"from_experiment": f"exp22-grid-{world}", "group": "B", "after": after},
                "phases": [{"eval": {"label": f"after-{after}", "tasks": {"items": items}}}]})
    return out


if __name__ == "__main__":
    for path in write_43():
        print(path.relative_to(ROOT))
