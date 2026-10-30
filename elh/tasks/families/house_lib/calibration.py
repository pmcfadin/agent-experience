"""Empirical calibration of a house_lib world (design §5), before it is used. Not run in CI.

Two filters on dev seeds, per lesson, from the lesson checks (`lesson:<id>`) of the verdicts:

- guessability: group A (no hints) passes the lesson's check in more than GUESS_MAX of episodes
  -> the lesson is guessable: drop or redraw it;
- learnability: P-hand (every lesson's `gold_text` in the first message) passes it in less than
  LEARN_MIN -> the model can't follow it even when told: drop it.

What exists here: the task panel (`calibration_tasks`), the P-hand block (`handbook`) and the filter
(`filter_lessons`). Not implemented: the P-hand group itself (a StaticMemory over `handbook`, wired
like P-all in elh/batch.py) and the runs, which need the GPU (~1,000 episodes for the core grid's
six worlds, design §8). A run then passes its verdicts' `checks` to `filter_lessons`.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

GUESS_MAX = 0.20
LEARN_MIN = 0.60
DEV_SEEDS = (3000, 4000)


def handbook(family) -> str:
    """P-hand / W-hand content: the gold text of every lesson in the world, in catalogue order."""
    from elh.tasks.families.house_lib.lessons import gold_text

    return "\n".join(f"- {gold_text(les)}" for les in family.world.lessons)


def calibration_tasks(family, per_lesson: int = 4, seed: int = 0) -> list[tuple[str, str, int]]:
    """(cause, theme, seed) on dev seeds: `per_lesson` tasks with each lesson as the primary one,
    themes spread round-robin. Secondary and module-wide lessons add further observations."""
    rng = random.Random(f"house_lib/calibration/{family.name}/{seed}")
    themes = list(family.themes)
    rng.shuffle(themes)
    out, s = [], DEV_SEEDS[0]
    for lesson in family.causes:
        for i in range(per_lesson):
            out.append((lesson, themes[i % len(themes)], s))
            s += 1
    if s > DEV_SEEDS[1]:
        raise ValueError(f"calibration panel needs {s - DEV_SEEDS[0]} dev seeds; only {DEV_SEEDS[1] - DEV_SEEDS[0]} exist")
    return out


@dataclass
class LessonCalibration:
    lesson: str
    a_n: int
    a_rate: float | None
    hand_n: int
    hand_rate: float | None
    keep: bool
    reason: str


def filter_lessons(episodes: list[dict], lessons: list[str]) -> list[LessonCalibration]:
    """`episodes`: one dict per graded episode, {"group": "A" | "P-hand", "checks": verdict.checks}.
    A lesson with no observations in a group is kept undecided (reason says so)."""
    seen: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for ep in episodes:
        for name, ok in ep["checks"].items():
            if name.startswith("lesson:"):
                seen[ep["group"], name.removeprefix("lesson:")].append(bool(ok))
    out = []
    for lid in lessons:
        a, h = seen.get(("A", lid), []), seen.get(("P-hand", lid), [])
        a_rate = sum(a) / len(a) if a else None
        h_rate = sum(h) / len(h) if h else None
        if a_rate is not None and a_rate > GUESS_MAX:
            keep, reason = False, f"guessable: A passes {a_rate:.0%} > {GUESS_MAX:.0%}"
        elif h_rate is not None and h_rate < LEARN_MIN:
            keep, reason = False, f"not learnable: P-hand passes {h_rate:.0%} < {LEARN_MIN:.0%}"
        elif a_rate is None or h_rate is None:
            keep, reason = True, "undecided: missing observations"
        else:
            keep, reason = True, "kept"
        out.append(LessonCalibration(lid, len(a), a_rate, len(h), h_rate, keep, reason))
    return out
