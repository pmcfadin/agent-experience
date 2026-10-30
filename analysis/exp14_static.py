"""H6 step 2: static prompt (exp14 P-all, A0), rules-only retrieval (exp15 B-rules) and all rules plus
records (exp16 P-all-ep) against
retrieved memory (exp13 B) and no memory (exp13 A).

Same test tickets in every group (quadratic_membership/inventory, seeds 2024-2035), so groups are
compared ticket by ticket within each history with an exact two-sided McNemar (sign) test on the
discordant pairs, pooled over histories. Stdlib only; run from the repository root:

    python3 analysis/exp14_static.py [--runs-dir runs]
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
from math import comb
from pathlib import Path

SOURCES = {"A": "exp13-histories", "B": "exp13-histories", "C": "exp13-histories",
           "P-all": "exp14-static-prompt", "A0": "exp14-static-prompt", "B-rules": "exp15-rules-only",
           "P-all-ep": "exp16-all-rules-and-records"}
ORDER = ["A", "A0", "B", "B-rules", "C", "P-all", "P-all-ep"]
PAIRS = [("P-all", "B"), ("P-all", "A0"), ("A0", "A"), ("B", "A"), ("P-all", "A"),
         ("B-rules", "B"), ("B-rules", "P-all"), ("P-all-ep", "B"), ("P-all-ep", "P-all"),
         ("P-all-ep", "B-rules")]


def load(db: sqlite3.Connection) -> dict[tuple[str, int, str], dict]:
    eps = {}
    for group, src in SOURCES.items():
        for h, task, passed, turns, wall in db.execute(
                "SELECT history, task_id, passed, turns, wall_s FROM episodes WHERE batch_id LIKE ? "
                "AND test_group = ? AND split = 'checkpoint'", (f"{src}/%", group)):
            eps[group, h, task] = {"passed": passed, "turns": turns, "wall_s": wall}
    return eps


def sign_test(b: int, c: int) -> float:
    """Exact two-sided p for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()
    db = sqlite3.connect(Path(args.runs_dir) / "elh.sqlite3")
    eps = load(db)
    hists = sorted({h for _, h, _ in eps})
    groups = [g for g in ORDER if any(k[0] == g for k in eps)]

    print("## Pass counts per history (of 12)\n")
    print("| Group | " + " | ".join(f"h{h}" for h in hists) + " | Total | Mean turns | Mean wall (s) | Mean turns, passed |")
    print("|---|" + "---|" * (len(hists) + 4))
    for g in groups:
        rows = [v for k, v in eps.items() if k[0] == g]
        per = [sum(v["passed"] for k, v in eps.items() if k[0] == g and k[1] == h) for h in hists]
        n = [sum(1 for k in eps if k[0] == g and k[1] == h) for h in hists]
        passed_turns = [v["turns"] for v in rows if v["passed"]]
        print(f"| {g} | " + " | ".join(f"{p}/{m}" for p, m in zip(per, n)) +
              f" | {sum(per)}/{sum(n)} | {statistics.mean(v['turns'] for v in rows):.1f} | "
              f"{statistics.mean(v['wall_s'] for v in rows):.0f} | "
              f"{statistics.mean(passed_turns) if passed_turns else float('nan'):.1f} |")

    print("\n## Paired comparisons (same history, same ticket)\n")
    print("| Comparison | Pairs | Only first passed | Only second passed | p (exact, two-sided) |")
    print("|---|---|---|---|---|")
    for x, y in PAIRS:
        if x not in groups or y not in groups:
            continue
        keys = [(h, t) for (g, h, t) in eps if g == x and (y, h, t) in eps]
        b = sum(1 for h, t in keys if eps[x, h, t]["passed"] and not eps[y, h, t]["passed"])
        c = sum(1 for h, t in keys if eps[y, h, t]["passed"] and not eps[x, h, t]["passed"])
        print(f"| {x} vs {y} | {len(keys)} | {b} | {c} | {sign_test(b, c):.3f} |")


if __name__ == "__main__":
    main()
