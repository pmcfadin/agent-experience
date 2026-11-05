"""house_lib recall tuning (docs/design/house-lib-core-grid.md §8): exp28-tune-<setting>-<world>, B's
exp22 banks replayed (B-replay) under (memory_service.k, rules.recall_k), and P-all, on the same 12
tuning tickets per history. Reports lesson checks and task pass over kept lessons, the shown items
and rules per ticket, and how often a shown rule came from a training ticket with the ticket's
primary lesson; each setting is paired with P-all ticket by ticket (exact McNemar). Run from the
repository root:

    uv run python analysis/exp28_tune.py [--runs-dir runs] [WORLD ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from elh.analysis import mcnemar_exact
from elh.store import Store
from elh.tasks.base import get_family
from elh.tasks.families.house_lib.grid import kept_lessons, scored_checks, scored_pass

WORLDS = ["L16-c0-w1", "L16-c50-w1"]
SETTINGS = ["k5r2", "k8r6", "k14r12", "pall"]


def train_lessons(store: Store, fam, world: str) -> dict[str, set[str]]:
    return {eid: set(fam.generate(c, th, sd).signature["lessons"]) for eid, c, th, sd in store.db.execute(
        "SELECT episode_id, cause, theme, seed FROM episodes WHERE batch_id LIKE ? AND test_group = 'B' "
        "AND split = 'train'", (f"exp22-grid-{world}/%",))}


def report(store: Store, world: str) -> None:
    fam = get_family(f"house_lib/{world}")
    kept, tl = kept_lessons(world), train_lessons(store, fam, world)
    per: dict[str, dict] = {}
    print(f"## {world}\n\n| Setting | Episodes | Task pass | Lesson checks | Items / rules shown | Primary-lesson rule shown |")
    print("|---|---|---|---|---|---|")
    for s in SETTINGS:
        rows = store.db.execute("SELECT history, task_id, cause, theme, seed, verdict_json, retrieval_json FROM episodes "
                                "WHERE batch_id LIKE ? AND split = 'checkpoint'", (f"exp28-tune-{s}-{world}/%",)).fetchall()
        if not rows:
            continue
        task, lesson, items, rules, prim = {}, {}, 0, 0, 0
        for h, tid, c, th, sd, v, rj in rows:
            checks = json.loads(v)["checks"]
            task[h, tid] = scored_pass(checks, kept)
            for name, ok in scored_checks(checks, kept).items():
                if name.startswith("lesson:"):
                    lesson[h, tid, name] = bool(ok)
            docs = [it.get("document_id") or "" for it in json.loads(rj or "{}").get("items", [])]
            shown = [d.removesuffix(":rule") for d in docs if d.endswith(":rule")]
            items, rules = items + len(docs), rules + len(shown)
            prim += any(c in tl.get(d, set()) for d in shown)
        per[s] = (task, lesson)
        n = len(rows)
        shown = f"{items / n:.1f} / {rules / n:.1f}" if s != "pall" else "all"
        print(f"| {s} | {n} | {sum(task.values())}/{n} | {sum(lesson.values())}/{len(lesson)} | {shown} | "
              f"{f'{prim}/{n}' if s != 'pall' else '-'} |")
    if "pall" in per:
        print("\n| Setting vs P-all | Outcome | Discordant (setting only vs P-all only) | p |\n|---|---|---|---|")
        for s in SETTINGS[:-1]:
            if s in per:
                for i, name in ((0, "task"), (1, "lesson")):
                    x, y = per[s][i], per["pall"][i]
                    keys = sorted(set(x) & set(y))
                    b, c = sum(x[k] and not y[k] for k in keys), sum(y[k] and not x[k] for k in keys)
                    p = mcnemar_exact(b, c)
                    print(f"| {s} | {name} | {b} vs {c} | {'-' if p is None else f'{p:.3f}'} |")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("worlds", nargs="*", default=WORLDS)
    args = ap.parse_args()
    store = Store(Path(args.runs_dir))
    for w in args.worlds:
        report(store, w)


if __name__ == "__main__":
    main()
