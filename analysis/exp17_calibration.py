"""house_lib calibration (design §5, phase 2): per-lesson guessability (A) and learnability (P-hand)
from the exp17-calib-<world> batches, through `calibration.filter_lessons`.

Also per group: whole-task pass rate, the feature check, and lesson-check pass rates by template, so
a world where the model cannot do the feature at all shows up before the per-lesson filter is read.
Run from the repository root:

    uv run python analysis/exp17_calibration.py [--runs-dir runs] [WORLD ...]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

from elh.tasks.base import get_family
from elh.tasks.families.house_lib.calibration import filter_lessons

WORLDS = ["L4-c0-w1", "L4-c50-w1", "L16-c0-w1", "L16-c50-w1", "L64-c0-w1", "L64-c50-w1"]


def load(db: sqlite3.Connection, world: str) -> list[dict]:
    rows = db.execute("SELECT test_group, cause, passed, verdict_json, turns, wall_s, outcome FROM episodes "
                      "WHERE batch_id = ? ORDER BY started_at", (f"exp17-calib-{world}/0",)).fetchall()
    return [{"group": g, "cause": c, "passed": bool(p), "checks": json.loads(v)["checks"], "turns": t,
             "wall_s": w, "outcome": o} for g, c, p, v, t, w, o in rows]


def report(world: str, eps: list[dict]) -> None:
    fam = get_family(f"house_lib/{world}")
    template = {les.id: les.template for les in fam.world.lessons}
    print(f"## {world}: {len(eps)} episodes\n")
    print("| Group | Episodes | Task passed | feature | public_tests | integrity | Mean turns | Mean wall (s) | Not submitted |")
    print("|---|---|---|---|---|---|---|---|---|")
    for g in ("A", "P-hand"):
        ge = [e for e in eps if e["group"] == g]
        if not ge:
            continue
        rate = lambda k: f"{sum(e['checks'].get(k, False) for e in ge)}/{len(ge)}"
        print(f"| {g} | {len(ge)} | {sum(e['passed'] for e in ge)}/{len(ge)} | {rate('feature')} | "
              f"{rate('public_tests')} | {rate('integrity')} | {statistics.mean(e['turns'] for e in ge):.1f} | "
              f"{statistics.mean(e['wall_s'] for e in ge):.0f} | {sum(e['outcome'] != 'submitted' for e in ge)} |")

    print("\n### Lesson checks by template (passed / observed)\n")
    by: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for e in eps:
        for name, ok in e["checks"].items():
            if name.startswith("lesson:"):
                by[e["group"], template[name.removeprefix("lesson:")]].append(bool(ok))
    print("| Template | A | P-hand |")
    print("|---|---|---|")
    for t in sorted({t for _, t in by}):
        cell = lambda g: f"{sum(by[g, t])}/{len(by[g, t])}" if by[g, t] else "-"
        print(f"| {t} | {cell('A')} | {cell('P-hand')} |")

    print("\n### Per-lesson filter (guessable: A > 20%; not learnable: P-hand < 60%)\n")
    print("| Lesson | A | P-hand | Keep | Reason |")
    print("|---|---|---|---|---|")
    fmt = lambda r, n: "-" if r is None else f"{r:.0%} of {n}"
    rows = filter_lessons(eps, fam.causes)
    for r in rows:
        print(f"| {r.lesson} | {fmt(r.a_rate, r.a_n)} | {fmt(r.hand_rate, r.hand_n)} | {'yes' if r.keep else 'no'} | {r.reason} |")
    print(f"\nKept {sum(r.keep for r in rows)} of {len(rows)} lessons.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("worlds", nargs="*", default=WORLDS)
    args = ap.parse_args()
    db = sqlite3.connect(Path(args.runs_dir) / "elh.sqlite3")
    for w in args.worlds:
        eps = load(db, w)
        if eps:
            report(w, eps)


if __name__ == "__main__":
    main()
