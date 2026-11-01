"""house_lib core grid (docs/design/house-lib-core-grid.md): exp22-grid-<world> (A, B, P-hand) and
exp23-grid-static-<world> (P-all, P-all-ep, P-oracle) on the same evaluation tickets.

Scores leave dropped lessons out (`house_lib.grid`): task pass is the lesson-neutral checks plus every
kept lesson's check; lesson pass is per kept lesson check. Groups are compared ticket by ticket within
each history (exact two-sided McNemar on discordant pairs, pooled over histories), for task pass and
for lesson checks. Also the go / no-go inputs (§6): infra errors from the batch logs, B against A, and
the largest experience block per group (≈ 4 characters per token). Run from the repository root:

    uv run python analysis/exp22_grid.py [--runs-dir runs] [WORLD ...]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from elh.analysis import mcnemar_exact
from elh.store import Store
from elh.tasks.base import get_family
from elh.tasks.families.house_lib.grid import kept_lessons, scored_checks, scored_pass

WORLDS = ["L16-c0-w1", "L16-c50-w1"]
ORDER = ["A", "B", "P-all", "P-all-ep", "P-oracle", "P-hand"]
PAIRS = [("B", "A"), ("B", "P-all"), ("B", "P-all-ep"), ("P-oracle", "B"), ("P-hand", "B"),
         ("P-all-ep", "P-all")]
CONTEXT_TOKENS = 65536


def batches(world: str) -> list[str]:
    return [f"exp22-grid-{world}", f"exp23-grid-static-{world}"]


def load(store: Store, world: str) -> list[dict]:
    out = []
    for name in batches(world):
        for g, h, task, v, sha in store.db.execute(
                "SELECT test_group, history, task_id, verdict_json, trace_sha256 FROM episodes "
                "WHERE batch_id LIKE ? AND split = 'checkpoint'", (f"{name}/%",)):
            out.append({"group": g, "history": h, "task": task, "checks": json.loads(v)["checks"], "trace": sha})
    return out


def block_tokens(store: Store, sha: str | None) -> int:
    if not sha:
        return 0
    first = json.loads(store.get_blob("traces", sha).decode().splitlines()[0]).get("first_message", "")
    i, j = first.find("<experience>"), first.find("</experience>")
    return (j - i) // 4 if i >= 0 and j > i else 0


def infra_errors(logs: Path, world: str) -> tuple[int, int]:
    done = err = 0
    for name in batches(world):
        f = logs / f"{name}.log"
        if f.exists():
            for line in f.read_text().splitlines():
                done += " passed=" in line
                err += "INFRA ERROR" in line
    return err, done + err


def pair_line(name: str, x: dict, y: dict, a: str, b: str) -> str:
    keys = sorted(set(x) & set(y))
    if not keys:
        return f"| {a} vs {b} | {name} | - | - | - | - |"
    b_only = sum(x[k] and not y[k] for k in keys)
    c_only = sum(y[k] and not x[k] for k in keys)
    p = mcnemar_exact(b_only, c_only)
    return (f"| {a} vs {b} | {name} | {sum(x[k] for k in keys)}/{len(keys)} vs {sum(y[k] for k in keys)}/{len(keys)} "
            f"| {b_only} vs {c_only} | {'-' if p is None else f'{p:.3f}'} | {len(keys)} |")


def report(store: Store, logs: Path, world: str) -> None:
    eps = load(store, world)
    if not eps:
        return
    kept = kept_lessons(world)
    template = {les.id: les.template for les in get_family(f"house_lib/{world}").world.lessons}
    task_pass: dict[str, dict] = defaultdict(dict)
    lesson_pass: dict[str, dict] = defaultdict(dict)
    by_template: dict[tuple[str, str], list[bool]] = defaultdict(list)
    biggest: dict[str, int] = defaultdict(int)
    for e in eps:
        task_pass[e["group"]][e["history"], e["task"]] = scored_pass(e["checks"], kept)
        for name, ok in scored_checks(e["checks"], kept).items():
            if name.startswith("lesson:"):
                lid = name.removeprefix("lesson:")
                lesson_pass[e["group"]][e["history"], e["task"], lid] = bool(ok)
                by_template[e["group"], template[lid]].append(bool(ok))
        biggest[e["group"]] = max(biggest[e["group"]], block_tokens(store, e["trace"]))
    groups = [g for g in ORDER if g in task_pass]
    err, total = infra_errors(logs, world)
    print(f"## {world}: {len(eps)} evaluation episodes, {len(kept)} kept lessons, "
          f"infra errors {err} of {total} logged episodes\n")
    print("| Group | Task pass (kept lessons) | Lesson checks | Largest block (≈ tokens) |")
    print("|---|---|---|---|")
    for g in groups:
        t, l = task_pass[g].values(), lesson_pass[g].values()
        print(f"| {g} | {sum(t)}/{len(t)} | {sum(l)}/{len(l)} | {biggest[g]:,} |")
    print("\n### Lesson checks by template\n")
    print("| Template | " + " | ".join(groups) + " |")
    print("|---|" + "---|" * len(groups))
    for t in sorted({t for _, t in by_template}):
        cell = lambda g: f"{sum(by_template[g, t])}/{len(by_template[g, t])}" if by_template[g, t] else "-"
        print(f"| {t} | " + " | ".join(cell(g) for g in groups) + " |")
    print("\n### Paired comparisons (same history and ticket; exact McNemar)\n")
    print("| Pair | Outcome | Passed | Discordant (first only vs second only) | p | Pairs |")
    print("|---|---|---|---|---|---|")
    for a, b in PAIRS:
        if a in task_pass and b in task_pass:
            print(pair_line("task", task_pass[a], task_pass[b], a, b))
            print(pair_line("lesson", lesson_pass[a], lesson_pass[b], a, b))
    over = [g for g in groups if biggest[g] > CONTEXT_TOKENS // 2]
    print(f"\nGo / no-go inputs: infra {err}/{total} (limit 2%); B vs A in the table above; "
          f"blocks over half the {CONTEXT_TOKENS:,}-token context: {over or 'none'}.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("worlds", nargs="*", default=WORLDS)
    args = ap.parse_args()
    store = Store(Path(args.runs_dir))
    for w in args.worlds:
        report(store, Path(args.runs_dir) / "logs", w)


if __name__ == "__main__":
    main()
