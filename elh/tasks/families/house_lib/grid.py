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
             "rules": {"enabled": True, "store": "verbatim", "context": True}}
HISTORIES = 3


def batch_docs(family, kept: list[str], histories: int = HISTORIES) -> tuple[dict, dict]:
    """The two batches of one world: `exp22-grid-<world>` (A, B and P-hand; B trains, then all are
    evaluated) and `exp23-grid-static-<world>` (P-all, P-all-ep and P-oracle from exp22's B after
    training, same evaluation tickets)."""
    world = family.world.id
    train, evals = grid_tasks(family, kept)
    label = f"after-{len(train)}"
    common = {"base_config": "experiments/phase0-local.yaml", "overrides": OVERRIDES,
              "family": f"house_lib/{world}", "histories": histories}
    ev = {"eval": {"label": label, "tasks": {"items": [list(t) for t in evals]}}}
    learn = {"name": f"exp22-grid-{world}", **common, "groups": ["A", "B", "P-hand"],
             "phases": [{"train": {"tasks": {"items": [list(t) for t in train]}}}, ev]}
    static = {"name": f"exp23-grid-static-{world}", **common, "groups": ["P-all", "P-all-ep", "P-oracle"],
              "static": {"from_experiment": learn["name"], "group": "B", "after": len(train)}, "phases": [ev]}
    return learn, static
