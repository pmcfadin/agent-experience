"""house_lib calibration (design §5, phase 2): per-lesson guessability (A) and learnability (P-hand)
from the exp21-calib-<world> batches, through `calibration.filter_lessons`; earlier runs on the same
tasks (exp17-19, L4-c0 only) are shown alongside but do not enter the filter.

Also per group: whole-task pass rate, the feature check, and lesson-check pass rates by template, so
a world where the model cannot do the feature at all shows up before the per-lesson filter is read.
Run from the repository root:

    uv run python analysis/exp17_calibration.py [--runs-dir runs] [--write] [WORLD ...]

Since 2026-11-02 the learnability group is P-task (exp24: only the ticket's own handbook lines);
exp21's P-hand (whole handbook) is shown beside it. `--write` stores each world's filter verdict in `experiments/calibration/<world>.json`,
which the core grid's panels and scores read (`house_lib.grid.kept_lessons`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

from elh.tasks.base import get_family
from elh.tasks.families.house_lib.calibration import calibration_tasks, filter_lessons
from elh.tasks.families.house_lib.grid import CALIBRATION_DIR

WORLDS = ["L4-c0-w1", "L4-c50-w1", "L16-c0-w1", "L16-c50-w1", "L64-c0-w1", "L64-c50-w1"]


# The calibration proper is exp21 (exp20's settings; tickets name the key and group columns, no
# `cents` unit choice, 8 tasks per lesson). exp20 and the earlier L4-c0 runs are shown as labelled
# comparisons and never enter the filter (their tasks differ from exp21's):
# exp17 (block before the ticket), exp18 (handbook inside the ticket, group A), exp19a (unhedged
# header), exp19b (block after the ticket, old wording).
BATCHES = [("exp21-calib", {"A": "A", "P-hand": "P-hand (whole handbook)"}),
           ("exp24-learn", {"P-task": "P-task"}),
           ("exp20-calib", {"A": "A (exp20)", "P-hand": "P-hand (exp20)"}),
           ("exp17-calib", {"A": "A (exp17)", "P-hand": "P-hand before (exp17)"}),
           ("exp19a-hand-plain", {"P-hand": "P-hand plain (exp19a)"}),
           ("exp19b-hand-after", {"P-hand": "P-hand after (exp19b)"}),
           ("exp18-hand-in-ticket", {"A": "A+t-hand (exp18)"})]
GROUPS = [label for _, labels in BATCHES for label in labels.values()]


def load(db: sqlite3.Connection, world: str) -> list[dict]:
    out = []
    for prefix, labels in BATCHES:
        rows = db.execute("SELECT test_group, cause, passed, verdict_json, turns, wall_s, outcome FROM episodes "
                          "WHERE batch_id = ? ORDER BY started_at", (f"{prefix}-{world}/0",)).fetchall()
        out += [{"group": labels[g], "cause": c, "passed": bool(p), "checks": json.loads(v)["checks"], "turns": t,
                 "wall_s": w, "outcome": o} for g, c, p, v, t, w, o in rows if g in labels]
    return out


def report(world: str, eps: list[dict], write: bool = False) -> None:
    fam = get_family(f"house_lib/{world}")
    template = {les.id: les.template for les in fam.world.lessons}
    print(f"## {world}: {len(eps)} episodes\n")
    print("| Group | Episodes | Task passed | feature | public_tests | integrity | Mean turns | Mean wall (s) | Not submitted |")
    print("|---|---|---|---|---|---|---|---|---|")
    for g in GROUPS:
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
    present = [g for g in GROUPS if any(e["group"] == g for e in eps)]
    print("| Template | " + " | ".join(present) + " |")
    print("|---|" + "---|" * len(present))
    for t in sorted({t for _, t in by}):
        cell = lambda g: f"{sum(by[g, t])}/{len(by[g, t])}" if by[g, t] else "-"
        print(f"| {t} | " + " | ".join(cell(g) for g in present) + " |")

    print("\n### Per-lesson filter (guessable: A > 20%; not learnable: P-task < 60%)\n")
    print("| Lesson | A | P-task | Keep | Reason |")
    print("|---|---|---|---|---|")
    fmt = lambda r, n: "-" if r is None else f"{r:.0%} of {n}"
    rows = filter_lessons([e for e in eps if e["group"] in ("A", "P-task")], fam.causes, told="P-task")
    for r in rows:
        print(f"| {r.lesson} | {fmt(r.a_rate, r.a_n)} | {fmt(r.hand_rate, r.hand_n)} | {'yes' if r.keep else 'no'} | {r.reason} |")
    print(f"\nKept {sum(r.keep for r in rows)} of {len(rows)} lessons.\n")
    if write:
        main = [e for e in eps if e["group"] in ("A", "P-task")]
        if len(main) != 2 * len(calibration_tasks(fam, per_lesson=8)):
            raise SystemExit(f"{world}: exp21 A + exp24 P-task are incomplete ({len(main)} episodes); not writing a verdict")
        out = CALIBRATION_DIR / f"{world}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"world": world, "batches": [f"exp21-calib-{world}/0 (A)", f"exp24-learn-{world}/0 (P-task)"],
                                   "episodes": len(main),
                                   "lessons": [dataclasses.asdict(r) for r in rows]}, indent=1) + "\n")
        print(f"wrote {out}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--write", action="store_true", help="store each world's verdict for the core grid")
    ap.add_argument("worlds", nargs="*", default=WORLDS)
    args = ap.parse_args()
    db = sqlite3.connect(Path(args.runs_dir) / "elh.sqlite3")
    for w in args.worlds:
        eps = load(db, w)
        if eps:
            report(w, eps, write=args.write)


if __name__ == "__main__":
    main()
