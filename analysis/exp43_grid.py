"""house_lib core grid, stage 4.3 (docs/design/house-lib-core-grid.md §10): the estimation run on six
worlds (L in 4, 16, 64 x c in 0, 0.5; w1). Not confirmatory: effect sizes and variances for the sealed
study's power analysis.

Episodes per world come from
  - learn batches `exp29<r>-<part>-<world>` (B; A and P-hand too, except at the L in `grid.REUSED_L`),
  - static batches `exp30<r>-<part>-static-<world>` (P-all, P-all-ep, P-oracle),
  - at L16 only, A and P-hand from stage 4.1's `exp22-grid-<world>` (they do not depend on recall),
where <part> is `gate` (history 0) and `grid` (histories 1-2) at L >= `grid.GATE_L` and `grid` alone
(all histories) below it, and <r> is `grid.RERUN`'s suffix (the L64-c50 gate reran as exp29b / exp30b).
`<name>-rest*` remainder batches are pooled; the history column pools gate and grid. Episodes listed in
experiments/calibration/excluded-episodes.json, and episodes the degenerate-reply checker flags
(elh/degenerate.py), are left out. Scores are over kept lessons (grid.scored_checks / scored_pass).

Per world: data completeness (scheduled vs recorded per batch and group), per-group task pass and lesson
checks, per-history breakdown, turns / wall time / tokens, largest and mean experience block, B's
retrieval (items and rules shown, how often a rule from a training ticket that shared the ticket's primary
lesson was shown), replicate counts, and paired comparisons of B against P-all, P-all-ep, P-oracle, P-hand
and A. Pairs are by (history, ticket) and exact McNemar, on one run per cell: replicate 0 (the first
replicate), so the replicate groups at L64-c50 are not counted three times. B vs P-all-ep is also tested
the way the sealed study will test it (power_sealed's primary test: the B - P-all-ep difference averaged
over a cell's runs, crossed history x ticket ANOVA, Satterthwaite t); that line uses every replicate.

`--power` runs the primary-test preview on L64-c50 (`grid.REPLICATE_WORLD`): the primary test, variance
components estimated from the replicates, and power_sealed's table (3-10 histories x 24 / 36 / 48 tickets
x 1-3 runs per cell) at the fixed 10-point effect, then again with the history SDs doubled (§10: the 4.3
estimates rest on 3 histories). Run from the repository root:

    uv run python analysis/exp43_grid.py [--runs-dir runs] [WORLD ...] [--power] [--sims 2000]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import power_sealed as ps  # noqa: E402

from elh.analysis import mcnemar_exact  # noqa: E402
from elh.batch import load_experiment, schedule  # noqa: E402
from elh.degenerate import scan  # noqa: E402
from elh.store import Store  # noqa: E402
from elh.tasks.base import get_family  # noqa: E402
from elh.tasks.families.house_lib import grid  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "experiments" / "batches"
EXCLUDED_FILE = grid.CALIBRATION_DIR / "excluded-episodes.json"
WORLDS = list(grid.WORLDS_43)
ORDER = ["A", "B", "P-all", "P-all-ep", "P-oracle", "P-hand"]
COMPARE = [("B", "P-all"), ("B", "P-all-ep"), ("B", "P-oracle"), ("B", "P-hand"), ("B", "A")]
STATIC = ("P-all", "P-all-ep", "P-oracle")
CONTEXT_TOKENS = 65536
COLUMNS = ("episode_id, batch_id, test_group, split, history, task_id, cause, theme, seed, outcome, verdict_json, "
           "turns, wall_s, input_tokens, output_tokens, trace_sha256, retrieval_json, replicate, started_at")


def world_l(world: str) -> int:
    m = re.match(r"L(\d+)-", world)
    if not m:
        raise ValueError(f"cannot read the lesson count from world id {world!r}")
    return int(m.group(1))


def sources(world: str) -> list[tuple[str, tuple[str, ...]]]:
    """(batch name, groups taken from it) for a world, in run order, as grid.grid43_docs names them."""
    n = world_l(world)
    parts = ["gate", "grid"] if n >= grid.GATE_L else ["grid"]
    out: list[tuple[str, tuple[str, ...]]] = []
    for part in parts:
        r = grid.RERUN.get((world, part), "")
        out.append((f"exp29{r}-{part}-{world}", ("B",) if n in grid.REUSED_L else ("A", "B", "P-hand")))
        out.append((f"exp30{r}-{part}-static-{world}", STATIC))
    if n in grid.REUSED_L:
        out.append((f"exp22-grid-{world}", ("A", "P-hand")))
    return out


def excluded_ids(path: Path = EXCLUDED_FILE) -> set[str]:
    return {e["episode_id"] for e in json.loads(path.read_text())["episodes"]} if path.exists() else set()


def fetch(store: Store, name: str, groups: tuple[str, ...]) -> list[dict]:
    """A batch's episodes (and its `-rest*` remainder batches') of the given groups: evaluation episodes,
    and B's training episodes."""
    rows = store.db.execute(
        f"SELECT {COLUMNS} FROM episodes WHERE (batch_id LIKE ? OR batch_id LIKE ?) AND test_group IN "
        f"({','.join('?' * len(groups))}) AND split IN ('checkpoint', 'train') ORDER BY started_at, episode_id",
        (f"{name}/%", f"{name}-rest%/%", *groups)).fetchall()
    out = []
    for r in rows:
        e = dict(zip(COLUMNS.replace(" ", "").split(","), tuple(r)))
        if e["split"] == "train" and e["test_group"] != "B":
            continue
        e["replicate"] = e["replicate"] or 0
        e["source"] = name
        out.append(e)
    return out


@dataclasses.dataclass
class WorldData:
    world: str
    kept: list[str]
    eps: list[dict]                 # evaluation episodes kept for analysis
    train: list[dict]               # B's training episodes kept
    completeness: list[dict]
    excluded: int = 0
    degenerate: int = 0
    duplicates: int = 0
    train_all: list[dict] = dataclasses.field(default_factory=list)  # every recorded B training episode (retrieval lookups)


def scheduled_counts(name: str, groups: tuple[str, ...], batch_dir: Path) -> dict[tuple[str, str], int] | None:
    """Scheduled episodes per (split, group) from the batch file, or None if it cannot be read."""
    path = batch_dir / f"{name}.yaml"
    if not path.exists():
        return None
    try:
        exp, cfg = load_experiment(path)
        steps = schedule(exp, cfg)
    except Exception:  # noqa: BLE001 - reported as "n/a"
        return None
    out: dict[tuple[str, str], int] = defaultdict(int)
    for s in steps:
        split = grid_split(s.kind)
        if s.group in groups and (split == "checkpoint" or s.group == "B") and split in ("checkpoint", "train"):
            out[split, s.group] += 1
    return dict(out)


def grid_split(kind: str) -> str:
    return {"train": "train", "eval": "checkpoint"}.get(kind, kind)


def gather(store: Store, world: str, *, kept: list[str] | None = None, batch_dir: Path = BATCH_DIR,
           excluded: set[str] | None = None, check_degenerate: bool = True) -> WorldData:
    kept = kept if kept is not None else grid.kept_lessons(world)
    excluded = excluded_ids() if excluded is None else excluded
    srcs = sources(world)
    flagged: set[str] = set()
    if check_degenerate:
        flagged = {f["episode_id"] for f in scan(store, [name for name, _ in srcs])}
    eps, train, train_all, comp = [], [], [], []
    dropped_ex = dropped_deg = 0
    for name, groups in srcs:
        rows = fetch(store, name, groups)
        sched = scheduled_counts(name, groups, batch_dir)
        by: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for e in rows:
            by[e["split"], e["test_group"]].append(e)
        for key in sorted(set(by) | set(sched or {})):
            got = by.get(key, [])
            use = [e for e in got if e["episode_id"] not in excluded and e["episode_id"] not in flagged]
            dropped_ex += sum(e["episode_id"] in excluded for e in got)
            dropped_deg += sum(e["episode_id"] in flagged and e["episode_id"] not in excluded for e in got)
            comp.append({"batch": name, "split": key[0], "group": key[1], "scheduled": None if sched is None else sched.get(key, 0),
                         "recorded": len(got), "usable": len(use)})
            (train if key[0] == "train" else eps).extend(use)
            if key[0] == "train":
                train_all.extend(got)
    for e in eps:
        e["kept_checks"] = grid.scored_checks(json.loads(e["verdict_json"])["checks"], kept)
        e["pass"] = grid.scored_pass(json.loads(e["verdict_json"])["checks"], kept)
    seen: set[tuple] = set()
    dup = 0
    for e in eps:
        key = (e["test_group"], e["history"], e["task_id"], e["replicate"])
        dup += key in seen
        seen.add(key)
    return WorldData(world, kept, eps, train, comp, dropped_ex, dropped_deg, dup, train_all)


def block_tokens(store: Store, sha: str | None) -> int:
    """Experience block size in tokens (4 characters each, as analysis/exp22_grid.py)."""
    if not sha:
        return 0
    first = json.loads(store.get_blob("traces", sha).decode().splitlines()[0]).get("first_message", "")
    i, j = first.find("<experience>"), first.find("</experience>")
    return (j - i) // 4 if i >= 0 and j > i else 0


def first_runs(eps: list[dict]) -> dict[str, dict[tuple, dict]]:
    """Per group, the replicate-0 (lowest replicate, then earliest) episode of every (history, ticket) cell."""
    out: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for e in sorted(eps, key=lambda e: (e["replicate"], e["started_at"] or "", e["episode_id"])):
        out[e["test_group"]].setdefault((e["history"], e["task_id"]), e)
    return out


def pair_counts(cells: dict[str, dict[tuple, dict]], a: str, b: str, outcome: str) -> dict:
    """Discordant counts of group a vs group b over matching (history, ticket) cells (`task`) or
    (history, ticket, lesson) checks (`lesson`)."""
    def flat(g: str) -> dict:
        if outcome == "task":
            return {k: e["pass"] for k, e in cells.get(g, {}).items()}
        return {(*k, name): bool(ok) for k, e in cells.get(g, {}).items() for name, ok in e["kept_checks"].items()
                if name.startswith("lesson:")}
    x, y = flat(a), flat(b)
    keys = sorted(set(x) & set(y))
    first = [k for k in keys if x[k] and not y[k]]
    second = [k for k in keys if y[k] and not x[k]]
    by_h: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for k in first:
        by_h[k[0]][0] += 1
    for k in second:
        by_h[k[0]][1] += 1
    return {"pairs": len(keys), "a": sum(x[k] for k in keys), "b": sum(y[k] for k in keys), "first_only": len(first),
            "second_only": len(second), "p": mcnemar_exact(len(first), len(second)), "by_history": dict(sorted(by_h.items()))}


def lesson_eps(eps: list[dict]) -> list[dict]:
    """power_sealed's episode records (group, history, ticket, lesson checks), replicates in order."""
    return [{"group": e["test_group"], "history": e["history"], "ticket": e["task_id"],
             "checks": {k: bool(v) for k, v in e["kept_checks"].items() if k.startswith("lesson:")}}
            for e in sorted(eps, key=lambda e: (e["replicate"], e["started_at"] or "", e["episode_id"]))
            if e["test_group"] in ps.GROUPS]


def primary_test(eps: list[dict]) -> tuple[ps.Design, dict] | str:
    """power_sealed's primary test on B vs P-all-ep, or why it cannot run."""
    recs = lesson_eps(eps)
    if not ({r["group"] for r in recs} >= set(ps.GROUPS)):
        return "needs both B and P-all-ep episodes"
    try:
        design = ps.build_design(recs)
        res = ps.primary_test(ps.cell_diff(design.y_new, design.y_ref), design.sizes)
    except ValueError as exc:
        return str(exc)
    return design, {k: float(v) for k, v in res.items()}


def primary_text(out: tuple[ps.Design, dict] | str) -> str:
    if isinstance(out, str):
        return f"not run ({out})"
    d, r = out
    half = ps.t_crit(r["df"]) * r["se"]
    return (f"B - P-all-ep {r['est'] * 100:+.1f} points (95% CI {(r['est'] - half) * 100:+.1f} to "
            f"{(r['est'] + half) * 100:+.1f}, df {r['df']:.1f}, p = {r['p']:.3f}); {d.y_new.shape[1]} histories x "
            f"{len(d.sizes)} tickets x {d.y_new.shape[0]} run(s) per cell, {d.dropped} episodes dropped")


def retrieval_stats(data: WorldData, family) -> dict | None:
    """B's retrieval on evaluation tickets: items and rules shown, and the share of tickets shown a rule
    from a training ticket whose lessons include the ticket's primary lesson (exp28_tune's rule)."""
    rows = [e for e in data.eps if e["test_group"] == "B" and e["retrieval_json"]]
    if not rows:
        return None
    lessons = {e["episode_id"]: set(family.generate(e["cause"], e["theme"], e["seed"]).signature["lessons"])
               for e in data.train_all}
    per: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # n, items, rules, primary shown
    for e in rows:
        docs = [it.get("document_id") or "" for it in json.loads(e["retrieval_json"]).get("items", [])]
        shown = [d.removesuffix(":rule") for d in docs if d.endswith(":rule")]
        for h in (e["history"], "all"):
            c = per[h]
            c[0] += 1
            c[1] += len(docs)
            c[2] += len(shown)
            c[3] += any(e["cause"] in lessons.get(d, set()) for d in shown)
    return dict(per)


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def pct(a: int, n: int) -> str:
    return f"{a}/{n}" if n else "-"


def report(store: Store, world: str, *, data: WorldData | None = None, family=None, **kw) -> WorldData:
    data = data or gather(store, world, **kw)
    print(f"## {world}: {len(data.eps)} evaluation episodes, {len(data.train)} B training episodes, "
          f"{len(data.kept)} kept lessons\n")
    print("### Data completeness (scheduled vs recorded)\n")
    print("| Batch | Split | Group | Scheduled | Recorded | Usable | Missing |\n|---|---|---|---|---|---|---|")
    for c in data.completeness:
        s = c["scheduled"]
        miss = "n/a" if s is None else max(0, s - c["recorded"])
        print(f"| {c['batch']} | {c['split']} | {c['group']} | {'n/a' if s is None else s} | {c['recorded']} | "
              f"{c['usable']} | {miss} |")
    print(f"\nLeft out: {data.excluded} on the exclusion list, {data.degenerate} flagged degenerate. "
          f"Duplicate (group, history, ticket, replicate) cells: {data.duplicates}.\n")
    if not data.eps:
        print("No stage 4.3 or reused 4.1 evaluation episodes recorded for this world yet.\n")
        return data
    groups = [g for g in ORDER if any(e["test_group"] == g for e in data.eps)]
    by_g = {g: [e for e in data.eps if e["test_group"] == g] for g in groups}
    histories = sorted({e["history"] for e in data.eps})
    biggest, avg = {}, {}
    for g in groups:
        sizes = [block_tokens(store, e["trace_sha256"]) for e in by_g[g]]
        biggest[g], avg[g] = max(sizes), mean(sizes)

    def lessons(es):
        ls = [bool(v) for e in es for k, v in e["kept_checks"].items() if k.startswith("lesson:")]
        return sum(ls), len(ls)

    print("### Results over kept lessons\n")
    print("| Group | Episodes | Cells | Runs per cell | Task pass | Lesson checks | Turns | Wall (s) | Tokens in / out (k) "
          "| Tokens per task pass (k) | Not submitted | Largest / mean block (≈ tokens) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for g in groups:
        es = by_g[g]
        cells: dict[tuple, int] = defaultdict(int)
        for e in es:
            cells[e["history"], e["task_id"]] += 1
        l = lessons(es)
        passes = sum(e["pass"] for e in es)
        tin, tout = mean(e["input_tokens"] or 0 for e in es), mean(e["output_tokens"] or 0 for e in es)
        per_pass = f"{sum((e['input_tokens'] or 0) + (e['output_tokens'] or 0) for e in es) / passes / 1000:,.0f}" if passes else "-"
        block = f"{biggest[g]:,} / {avg[g]:,.0f}" if biggest[g] else "-"
        reps = f"{min(cells.values())}-{max(cells.values())}" if min(cells.values()) != max(cells.values()) else str(min(cells.values()))
        print(f"| {g} | {len(es)} | {len(cells)} | {reps} | {pct(passes, len(es))} | {pct(*l)} | "
              f"{mean(e['turns'] or 0 for e in es):.1f} | {mean(e['wall_s'] or 0 for e in es):.0f} | "
              f"{tin / 1000:,.1f} / {tout / 1000:,.1f} | {per_pass} | "
              f"{sum(e['outcome'] != 'submitted' for e in es)} | {block} |")
    print("\n### By history: lesson checks (task pass)\n")
    print("| Group | " + " | ".join(f"history {h}" for h in histories) + " |\n|---|" + "---|" * len(histories))
    for g in groups:
        cols = []
        for h in histories:
            es = [e for e in by_g[g] if e["history"] == h]
            cols.append(f"{pct(*lessons(es))} ({pct(sum(e['pass'] for e in es), len(es))})" if es else "-")
        print(f"| {g} | " + " | ".join(cols) + " |")
    over = [g for g in groups if biggest[g] > CONTEXT_TOKENS // 2]
    print(f"\nBlocks over half the {CONTEXT_TOKENS:,}-token context: {over or 'none'}.\n")

    family = family or get_family(f"house_lib/{world}")
    stats = retrieval_stats(data, family)
    if stats:
        print(f"### B's retrieval (k, rules, budget = {grid.GRID_RECALL})\n")
        print("| History | Eval episodes | Items shown | Rules shown | Primary-lesson rule shown |\n|---|---|---|---|---|")
        for h, (n, items, rules, prim) in sorted(stats.items(), key=lambda kv: (kv[0] == "all", str(kv[0]))):
            print(f"| {h} | {n} | {items / n:.1f} | {rules / n:.1f} | {prim}/{n} ({prim / n:.0%}) |")
        print()

    cells = first_runs(data.eps)
    reps = {g: max(e["replicate"] for e in by_g[g]) + 1 for g in groups}
    note = ", ".join(f"{g} {reps[g]} runs" for g in groups if reps[g] > 1)
    print("### Paired comparisons (same history and ticket; exact McNemar" +
          (f"; replicate 0 only, replicates: {note}" if note else "") + ")\n")
    todo = [(a, b) for a, b in COMPARE if a in cells and b in cells]
    if not todo:
        print("No compared pair has both groups yet.")
    else:
        print("| Pair | Outcome | Passed | Discordant (first only vs second only) | p | Pairs | By history (first only / second only) |")
        print("|---|---|---|---|---|---|---|")
    for a, b in todo:
        for outcome in ("task", "lesson"):
            r = pair_counts(cells, a, b, outcome)
            if not r["pairs"]:
                print(f"| {a} vs {b} | {outcome} | - | - | - | 0 | - |")
                continue
            hist = "; ".join(f"h{h} {x}/{y}" for h, (x, y) in r["by_history"].items()) or "no discordant pairs"
            print(f"| {a} vs {b} | {outcome} | {r['a']}/{r['pairs']} vs {r['b']}/{r['pairs']} | {r['first_only']} vs "
                  f"{r['second_only']} | {'-' if r['p'] is None else f'{r['p']:.3f}'} | {r['pairs']} | {hist} |")
    print(f"\nPrimary test (power_sealed: run-averaged cell difference, crossed history x ticket ANOVA; "
          f"all replicates): {primary_text(primary_test(data.eps))}.\n")
    return data


def power_preview(data: WorldData, *, sims: int, seed: int, baseline: float | None) -> None:
    print(f"## Primary-test preview and power, {data.world}\n")
    out = primary_test(data.eps)
    print(f"Primary test on the data: {primary_text(out)}.\n")
    if isinstance(out, str):
        return
    design = out[0]
    vc = ps.estimate_components(design.y_new, design.y_ref, design.sizes)
    if baseline:
        vc = dataclasses.replace(vc, baseline=baseline)
    ps.report(vc, sims=sims, seed=seed, label=f"{data.world} data, as estimated")
    doubled = dataclasses.replace(vc, intercept={**vc.intercept, "history": 2 * vc.intercept["history"]},
                                  slope={**vc.slope, "history": 2 * vc.slope["history"]})
    print()
    ps.report(doubled, sims=sims, seed=seed, label=f"{data.world} data, history SDs doubled")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--power", action="store_true", help=f"primary-test preview and power table on {grid.REPLICATE_WORLD}")
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", type=float, help="P-all-ep pass rate to plan for (default: the data's)")
    ap.add_argument("--no-degenerate-check", action="store_true", help="skip the trace scan for degenerate replies")
    ap.add_argument("worlds", nargs="*", default=WORLDS)
    args = ap.parse_args()
    store = Store(Path(args.runs_dir))
    done: dict[str, WorldData] = {}
    for w in args.worlds:
        done[w] = report(store, w, check_degenerate=not args.no_degenerate_check)
    if args.power:
        w = grid.REPLICATE_WORLD
        data = done.get(w) or gather(store, w, check_degenerate=not args.no_degenerate_check)
        power_preview(data, sims=args.sims, seed=args.seed, baseline=args.baseline)


if __name__ == "__main__":
    main()
