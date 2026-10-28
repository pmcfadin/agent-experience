"""H7 go/no-go: do exp13's memory histories differ in working style beyond run-to-run noise?

Scores every exp13 episode on behavioural trace features, removes ticket and pass/fail effects,
and asks whether episodes from the same history resemble each other more than chance. Group A
carries no memory, so its three histories are the noise baseline. Stdlib only; reads
runs/elh.sqlite3 and runs/traces/ and prints a Markdown report.

    python3 analysis/exp13_traits.py [--runs-dir runs] [--perms 5000]
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sqlite3
import statistics as st
from collections import defaultdict
from pathlib import Path

BATCH = "exp13-histories/%"

WRITE = re.compile(r"sed\s+-i|cat\s*>|>\s*\S+|\btee\b|open\([^)]*['\"][wa]['\"]|\.write(_text)?\(|\bpatch\b")
SOURCE = re.compile(r"job/|\.py\b")
PROTECTED = re.compile(r"examples/|tests/|run\.py|scripts/")
READ = re.compile(r"^\s*(cat|head|tail|sed\s+-n|grep|nl|wc|ls|find|less|awk)\b")
VERIFY = re.compile(r"pytest|unittest|python3?\s+tests/test_job|\bdiff\b|\bcmp\b|run\.py\s+examples/sample_input")
PERF = re.compile(r"make_load|\btime\b|timeit|perf_counter")

# name -> (trait, description)
FEATURES = {
    "turns": ("persistence", "model turns used"),
    "budget_exhausted": ("persistence", "ran out of turns (0/1)"),
    "cmds_before_edit": ("exploration", "commands before the first source edit"),
    "reads_before_edit": ("exploration", "read-only commands before the first source edit"),
    "n_edits": ("boldness", "source-editing commands"),
    "edit_chars": ("boldness", "characters in source-editing commands"),
    "protected_writes": ("boldness", "write commands touching protected paths"),
    "verify_count": ("caution", "test runs and output comparisons"),
    "verified_at_end": ("caution", "a verification in the last 5 commands (0/1)"),
    "perf_checks": ("caution", "load generation or timing commands"),
    "error_rate": ("caution", "share of commands with a non-zero exit"),
    "text_per_turn": ("style", "assistant text characters per turn"),
    "submit_len": ("style", "characters in the submit summary"),
}


def is_source_edit(cmd: str) -> bool:
    return bool(WRITE.search(cmd)) and bool(SOURCE.search(cmd)) and not PROTECTED.search(cmd.split(">")[-1])


def is_read(cmd: str) -> bool:
    return bool(READ.search(cmd)) and not WRITE.search(cmd)


def is_verify(cmd: str) -> bool:
    return bool(VERIFY.search(cmd)) and not is_read(cmd)


def episode_features(trace: list[dict]) -> dict[str, float]:
    cmds = [e for e in trace if e["event"] == "tool_exec"]
    texts, submit = [], ""
    for e in trace:
        if e["event"] != "response":
            continue
        for b in e.get("content", []):
            if b.get("type") == "text":
                texts.append(b.get("text", ""))
            elif b.get("type") == "tool_use" and b.get("name") == "submit":
                submit = json.dumps(b.get("input", {}))
    end = next(e for e in trace if e["event"] == "episode_end")
    lines = [c.get("command", "") for c in cmds]
    first_edit = next((i for i, c in enumerate(lines) if is_source_edit(c)), len(lines))
    verify = [is_verify(c) for c in lines]
    turns = max(end.get("turns", 0), 1)
    return {
        "turns": end.get("turns", 0),
        "budget_exhausted": float(end.get("outcome") == "budget_exhausted"),
        "cmds_before_edit": first_edit,
        "reads_before_edit": sum(is_read(c) for c in lines[:first_edit]),
        "n_edits": sum(is_source_edit(c) for c in lines),
        "edit_chars": sum(len(c) for c in lines if is_source_edit(c)),
        "protected_writes": sum(bool(WRITE.search(c)) and bool(PROTECTED.search(c.split(">")[-1])) for c in lines),
        "verify_count": sum(verify),
        "verified_at_end": float(any(verify[-5:])),
        "perf_checks": sum(bool(PERF.search(c)) for c in lines),
        "error_rate": sum(c.get("exit_code", 0) != 0 for c in cmds) / max(len(cmds), 1),
        "text_per_turn": sum(len(t) for t in texts) / turns,
        "submit_len": len(submit),
    }


def load(runs: Path) -> list[dict]:
    db = sqlite3.connect(runs / "elh.sqlite3")
    rows = db.execute(
        "select episode_id, history, test_group, split, task_id, passed, trace_sha256 "
        "from episodes where batch_id like ? order by history, test_group, split, ordinal", (BATCH,)
    ).fetchall()
    out = []
    for eid, hist, group, split, task, passed, sha in rows:
        path = runs / "traces" / sha[:2] / sha
        trace = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        out.append({"episode": eid, "history": hist, "group": group, "split": split, "task": task,
                    "passed": int(passed or 0), **episode_features(trace)})
    return out


def residualise(eps: list[dict], feat: str, iters: int = 50) -> list[float]:
    """Remove ticket and pass/fail effects (two-way fixed effects by backfitting)."""
    y = [e[feat] for e in eps]
    r = [v - st.fmean(y) for v in y]
    for _ in range(iters):
        for key in ("task", "passed"):
            groups = defaultdict(list)
            for i, e in enumerate(eps):
                groups[e[key]].append(i)
            for idx in groups.values():
                m = st.fmean(r[i] for i in idx)
                for i in idx:
                    r[i] -= m
    return r


def f_stat(values: list[float], labels: list[int]) -> float:
    """One-way ANOVA F of values by label (between-history over within-history variance)."""
    groups = defaultdict(list)
    for v, lab in zip(values, labels):
        groups[lab].append(v)
    k, n = len(groups), len(values)
    grand = st.fmean(values)
    ssb = sum(len(g) * (st.fmean(g) - grand) ** 2 for g in groups.values())
    ssw = sum((v - st.fmean(g)) ** 2 for g in groups.values() for v in g)
    if ssw <= 1e-12 or k < 2:
        return 0.0
    return (ssb / (k - 1)) / (ssw / (n - k))


def within_task_perms(eps: list[dict], rng: random.Random, n: int):
    """Shuffle history labels among the episodes of each ticket, keeping the ticket structure."""
    by_task = defaultdict(list)
    for i, e in enumerate(eps):
        by_task[e["task"]].append(i)
    base = [e["history"] for e in eps]
    for _ in range(n):
        lab = base[:]
        for idx in by_task.values():
            shuffled = [base[i] for i in idx]
            rng.shuffle(shuffled)
            for i, h in zip(idx, shuffled):
                lab[i] = h
        yield lab


def zscore_matrix(eps: list[dict], feats: list[str]) -> list[list[float]]:
    cols = []
    for f in feats:
        r = residualise(eps, f)
        sd = st.pstdev(r) or 1.0
        cols.append([v / sd for v in r])
    return [list(row) for row in zip(*cols)]


def loo_accuracy(x: list[list[float]], labels: list[int], tasks: list[str]) -> float:
    """Leave-one-ticket-out nearest-centroid attribution of episodes to histories."""
    hits = 0
    for i in range(len(x)):
        cents = defaultdict(list)
        for j in range(len(x)):
            if tasks[j] != tasks[i]:
                cents[labels[j]].append(x[j])
        best, best_d = None, float("inf")
        for lab, rows in cents.items():
            c = [st.fmean(col) for col in zip(*rows)]
            d = sum((a - b) ** 2 for a, b in zip(x[i], c))
            if d < best_d:
                best, best_d = lab, d
        hits += best == labels[i]
    return hits / len(x)


def analyse(eps: list[dict], perms: int, seed: int = 13) -> str:
    rng = random.Random(seed)
    feats = [f for f in FEATURES]
    out = ["# H7 go/no-go: exp13 trace reanalysis", ""]
    out.append(f"Episodes: {len(eps)}. Features are residualised on ticket and pass/fail before every test. "
               f"p-values come from {perms} within-ticket permutations of history labels.")
    out.append("")

    # Pass rates, for the competence context.
    out.append("## Pass rate by group and history (test tickets)")
    out.append("")
    out.append("| Group | h0 | h1 | h2 |")
    out.append("|---|---|---|---|")
    for g in "ABC":
        cells = []
        for h in range(3):
            sub = [e for e in eps if e["group"] == g and e["split"] == "checkpoint" and e["history"] == h]
            cells.append(f"{sum(e['passed'] for e in sub)}/{len(sub)}")
        out.append(f"| {g} | " + " | ".join(cells) + " |")
    out.append("")

    # 1. Per-feature history effect on the shared test tickets, A as the noise baseline.
    out.append("## 1. Per-feature history effect (test tickets, 36 episodes per group)")
    out.append("")
    out.append("F = between-history variance over within-history variance after adjustment; p = permutation.")
    out.append("")
    out.append("| Feature | Trait | A: F (p) | B: F (p) | C: F (p) |")
    out.append("|---|---|---|---|---|")
    summary = {}
    for f in feats:
        row = [f, FEATURES[f][0]]
        for g in "ABC":
            sub = [e for e in eps if e["group"] == g and e["split"] == "checkpoint"]
            r = residualise(sub, f)
            obs = f_stat(r, [e["history"] for e in sub])
            null = [f_stat(r, lab) for lab in within_task_perms(sub, rng, perms)]
            p = (1 + sum(v >= obs for v in null)) / (1 + perms)
            summary[(g, f)] = p
            row.append(f"{obs:.2f} ({p:.3f})")
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    for g in "ABC":
        n_sig = sum(summary[(g, f)] < 0.05 for f in feats)
        out.append(f"- Group {g}: {n_sig} of {len(feats)} features at p < 0.05 "
                   f"(about {0.05 * len(feats):.1f} expected by chance).")
    out.append("")

    # 2. Multivariate attribution.
    out.append("## 2. Can an episode be attributed to its history? (test tickets)")
    out.append("")
    out.append("Leave-one-ticket-out nearest centroid on all standardised residual features. Chance is 1/3.")
    out.append("")
    out.append("| Group | Accuracy | Permutation p |")
    out.append("|---|---|---|")
    attr = {}
    for g in "ABC":
        sub = [e for e in eps if e["group"] == g and e["split"] == "checkpoint"]
        x = zscore_matrix(sub, feats)
        tasks = [e["task"] for e in sub]
        obs = loo_accuracy(x, [e["history"] for e in sub], tasks)
        n_perm = max(200, perms // 25)
        null = [loo_accuracy(x, lab, tasks) for lab in within_task_perms(sub, rng, n_perm)]
        p = (1 + sum(v >= obs for v in null)) / (1 + n_perm)
        attr[g] = (obs, p)
        out.append(f"| {g} | {obs:.2f} | {p:.3f} (n={n_perm}) |")
    out.append("")

    # 3. Do B and C in the same history (same training order) look alike?
    out.append("## 3. Do B and C that shared a training order look alike? (test tickets)")
    out.append("")
    test = [e for e in eps if e["group"] in "BC" and e["split"] == "checkpoint"]
    x = zscore_matrix(test, feats)
    prof = {}
    for g in "BC":
        for h in range(3):
            rows = [x[i] for i, e in enumerate(test) if e["group"] == g and e["history"] == h]
            prof[(g, h)] = [st.fmean(col) for col in zip(*rows)]

    def corr(a, b):
        ma, mb = st.fmean(a), st.fmean(b)
        num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
        den = (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** 0.5
        return num / den if den else 0.0

    same = [corr(prof[("B", h)], prof[("C", h)]) for h in range(3)]
    diff = [corr(prof[("B", h)], prof[("C", k)]) for h, k in itertools.permutations(range(3), 2)]
    out.append(f"Mean profile correlation, same history: {st.fmean(same):.2f} "
               f"({', '.join(f'{v:.2f}' for v in same)}); different histories: {st.fmean(diff):.2f}.")
    out.append("")

    # Raw means for orientation.
    out.append("## Raw feature means by group and history (test tickets)")
    out.append("")
    out.append("| Feature | " + " | ".join(f"{g}{h}" for g in "ABC" for h in range(3)) + " |")
    out.append("|---|" + "---|" * 9)
    for f in feats:
        cells = []
        for g in "ABC":
            for h in range(3):
                sub = [e[f] for e in eps if e["group"] == g and e["split"] == "checkpoint" and e["history"] == h]
                cells.append(f"{st.fmean(sub):.2f}" if sub else "–")
        out.append(f"| {f} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("Feature definitions: " + "; ".join(f"`{k}`: {v[1]}" for k, v in FEATURES.items()) + ".")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs", type=Path)
    ap.add_argument("--perms", default=5000, type=int)
    ap.add_argument("--dump", type=Path, help="also write per-episode features as JSONL")
    args = ap.parse_args()
    eps = load(args.runs_dir)
    if args.dump:
        args.dump.write_text("".join(json.dumps(e) + "\n" for e in eps))
    print(analyse(eps, args.perms))


if __name__ == "__main__":
    main()
