"""Batch analysis (SPEC §12, pilot form): turn stored episodes into an A-vs-B comparison.

Episodes are selected by batch prefix (`episodes.batch_id` is "<experiment>/<history>") or, for
older runs without a batch id, by run-id prefix. They are grouped by phase (train, or one eval
label per checkpoint) × arm, pooled over histories and per history, and reported with pass rates
(Wilson 95% interval), check failures, outcomes, turns, wall time and compute.

Compute is in GPU-seconds:
- agent: Σ over the episode's model calls of llama.cpp `timings.prompt_ms + predicted_ms`, as
  recorded by the gateway client in `llm_calls.served_json` (None when a call has no timings);
- memory LLM: Hindsight's LLM request trace for the arm's training bank (`/llm-requests`).
  Retain calls name the episode (`metadata.document_id`); consolidation calls name the facts they
  consolidated (`metadata.source_memory_ids`), which link back to the retain that produced them.
  Calls with no such link fall back to a time window (the stream's last episode that finished
  before the call), flagged as an estimate. Hindsight records request wall time (`duration_ms`),
  not server compute, so this overstates GPU time when the memory LLM serves requests in parallel.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from typing import Callable

from elh.store import Store

CHECKS = ("integrity", "public_tests", "correctness", "performance")
OUTCOMES = ("submitted", "implicit_submit", "budget_exhausted", "refused", "infra_error")
Z95 = 1.959963984540054
# Hindsight scopes that are not generative LLM calls (`call_type` in its API).
NON_LLM_PREFIXES = ("embedding_", "decision_")
NON_LLM_SCOPES = {"rerank"}

TraceFetcher = Callable[[str, str], "list[dict] | None"]


# -- statistics ------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float | None:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return None
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


# -- per-episode metrics ---------------------------------------------------------------------
def agent_compute(calls: list[dict]) -> dict:
    """GPU-seconds and llama.cpp token counts of one episode's model calls."""
    gpu_ms, missing = 0.0, 0
    toks = {"prompt_n": 0, "predicted_n": 0, "cache_n": 0}
    for c in calls:
        try:
            served = json.loads(c.get("served_json") or "{}") or {}
        except ValueError:
            served = {}
        t = served.get("timings") or {}
        if t.get("prompt_ms") is None or t.get("predicted_ms") is None:
            missing += 1
        else:
            gpu_ms += t["prompt_ms"] + t["predicted_ms"]
        for k in toks:
            toks[k] += t.get(k) or 0
    return {"agent_gpu_s": gpu_ms / 1000 if calls and not missing else None,
            "calls": len(calls), "calls_without_timings": missing, **toks}


def phase_of(split: str | None, checkpoint: int | None) -> str:
    if split == "train":
        return "train"
    label = "eval" if split == "checkpoint" else (split or "?")
    return f"{label}@{checkpoint}" if checkpoint is not None else label


def _phase_key(phase: str) -> tuple:
    if phase == "train":
        return (0, 0, "")
    name, _, ck = phase.partition("@")
    return (1 if name == "eval" else 2, int(ck) if ck.lstrip("-").isdigit() else -1, name)


def _ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def load_episodes(store: Store, batch_prefix: str | None = None, run_id: str | None = None) -> list[dict]:
    """Episodes of a batch (batch_id == prefix or starts with prefix + '/') or a run-id prefix."""
    if batch_prefix is not None:
        batch_prefix = batch_prefix.rstrip("/") or batch_prefix
        where = "(e.batch_id = ? OR substr(e.batch_id, 1, ?) = ?)"
        args: list = [batch_prefix, len(batch_prefix) + 1, batch_prefix + "/"]
    elif run_id is not None:
        where, args = "substr(e.run_id, 1, ?) = ?", [len(run_id), run_id]
    else:
        raise ValueError("give a batch prefix or a run-id prefix")
    rows = [dict(r) for r in store.db.execute(
        f"SELECT * FROM episodes e WHERE {where} ORDER BY e.started_at, e.episode_id", args)]
    calls: dict[str, list[dict]] = defaultdict(list)
    ids = [r["episode_id"] for r in rows]
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        for c in store.db.execute(f"SELECT episode_id, turn, served_json FROM llm_calls WHERE episode_id IN "
                                  f"({','.join('?' * len(chunk))}) ORDER BY turn", chunk):
            calls[c["episode_id"]].append(dict(c))
    out = []
    for r in rows:
        try:
            verdict = json.loads(r["verdict_json"]) if r["verdict_json"] else {}
        except ValueError:
            verdict = {}
        if batch_prefix is not None:
            hist = r["batch_id"][len(batch_prefix) + 1:] or r["batch_id"]
        else:
            hist = f"{r['run_id']}/h{r['history']}"
        out.append({
            "episode_id": r["episode_id"], "run_id": r["run_id"], "batch_id": r.get("batch_id"),
            "history": r["history"], "history_label": hist, "arm": r["arm"], "split": r["split"],
            "checkpoint": r["checkpoint"], "ordinal": r["ordinal"], "phase": phase_of(r["split"], r["checkpoint"]),
            "task_id": r["task_id"], "outcome": r["outcome"], "passed": bool(r["passed"]),
            "checks": verdict.get("checks") or {}, "turns": r["turns"], "wall_s": r["wall_s"],
            "started_at": r["started_at"], "finished_at": r["finished_at"],
            "memory_s": None, "memory_calls": 0, "memory_method": None,
            "memory_input_tokens": 0, "memory_output_tokens": 0,
            **agent_compute(calls[r["episode_id"]]),
        })
    return out


# -- memory LLM trace ------------------------------------------------------------------------
def http_trace_fetcher(timeout_s: float = 20) -> TraceFetcher:
    """Read-only fetch of every traced request of a bank; None when the bank does not exist."""
    def fetch(base_url: str, bank_id: str) -> list[dict] | None:
        items, offset = [], 0
        while True:
            url = (base_url.rstrip("/") + f"/v1/default/banks/{urllib.parse.quote(bank_id)}/llm-requests?"
                   + urllib.parse.urlencode({"limit": 500, "offset": offset}))
            try:
                with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                    page = json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return None
                raise
            batch = page.get("items") or []
            items += [{k: it.get(k) for k in ("id", "operation", "scope", "trace_id", "status", "started_at",
                                               "duration_ms", "input_tokens", "output_tokens", "metadata")}
                      for it in batch]
            offset += len(batch)
            if not batch or offset >= (page.get("total") or 0):
                return items
    return fetch


def is_llm_call(entry: dict) -> bool:
    scope = entry.get("scope") or ""
    return not scope.startswith(NON_LLM_PREFIXES) and scope not in NON_LLM_SCOPES


def bank_id_of(run_id: str, history: int, arm: str) -> str:
    """Training bank name; mirrors `elh.controller.bank_id_for`."""
    return f"elh-{run_id}-h{history}-{arm}"


def attribute_memory(entries: list[dict], stream: list[dict]) -> dict[str, dict]:
    """Split a bank's traced LLM calls over the train episodes of its (run, history, arm) stream.

    Returns {episode_id: {"s", "calls", "input_tokens", "output_tokens", "linked", "window"}} plus
    the key None for calls that matched no episode."""
    ids = {e["episode_id"] for e in stream}
    mem_owner: dict[str, str] = {}
    for it in entries:
        doc = (it.get("metadata") or {}).get("document_id")
        if it.get("operation") == "retain" and doc in ids:
            for m in (it.get("metadata") or {}).get("memory_ids") or []:
                mem_owner[m] = doc
    by_start = sorted((e for e in stream if e.get("finished_at")), key=lambda e: e["finished_at"])
    out: dict = defaultdict(lambda: {"s": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0,
                                     "linked": 0, "window": 0})

    def window_owner(started: str | None) -> str | None:
        t = _ts(started)
        if t is None:
            return None
        owner = None
        for e in by_start:
            if _ts(e["finished_at"]) <= t:
                owner = e["episode_id"]
        return owner

    for it in entries:
        if not is_llm_call(it) or it.get("status") not in (None, "success", "error"):
            continue
        meta = it.get("metadata") or {}
        shares: dict[str | None, float] = {}
        method = "linked"
        if meta.get("document_id") in ids:
            shares = {meta["document_id"]: 1.0}
        elif meta.get("source_memory_ids"):
            owners = [mem_owner.get(m) for m in meta["source_memory_ids"]]
            known = [o for o in owners if o]
            if known:
                for o in known:
                    shares[o] = shares.get(o, 0) + 1 / len(known)
        if not shares:
            method = "window"
            shares = {window_owner(it.get("started_at")): 1.0}
        dur = (it.get("duration_ms") or 0) / 1000
        for ep, w in shares.items():
            o = out[ep]
            o["s"] += dur * w
            o["calls"] += 1
            o["input_tokens"] += round((it.get("input_tokens") or 0) * w)
            o["output_tokens"] += round((it.get("output_tokens") or 0) * w)
            o[method] += 1
    return dict(out)


def add_memory_trace(store: Store, episodes: list[dict], fetch: TraceFetcher | None,
                     memory_url: str | None = None) -> dict:
    """Fill `memory_s` on episodes. Read-only splits and arm A spend nothing on the memory LLM;
    learning-arm train episodes get their attributed seconds, or None if the trace is unavailable."""
    status: dict = {"enabled": fetch is not None, "banks": {}, "unattributed_s": 0.0, "estimated_calls": 0}
    for e in episodes:
        if e["split"] != "train" or e["arm"] == "A":
            e["memory_s"] = 0.0
    if fetch is None:
        return status
    streams: dict[tuple, list[dict]] = defaultdict(list)
    for e in episodes:
        if e["arm"] != "A":
            streams[(e["run_id"], e["history"], e["arm"])].append(e)
    for (run_id, history, arm), stream in streams.items():
        train = [e for e in stream if e["split"] == "train"]
        if not train:
            continue
        bank = bank_id_of(run_id, history, arm)
        url = memory_url or _memory_url_for_run(store, run_id)
        if not url:
            status["banks"][bank] = "no memory service url"
            continue
        try:
            entries = fetch(url, bank)
        except (OSError, ValueError) as exc:
            status["banks"][bank] = f"unavailable: {exc}"
            continue
        if entries is None:
            status["banks"][bank] = "bank not found"
            continue
        att = attribute_memory(entries, train)
        status["banks"][bank] = f"{len(entries)} traced requests"
        for e in train:
            a = att.get(e["episode_id"])
            e["memory_s"] = a["s"] if a else 0.0
            if a:
                e["memory_calls"] = a["calls"]
                e["memory_input_tokens"], e["memory_output_tokens"] = a["input_tokens"], a["output_tokens"]
                e["memory_method"] = "window" if a["window"] else "linked"
                status["estimated_calls"] += a["window"]
        if None in att:
            status["unattributed_s"] += att[None]["s"]
    return status


def _memory_url_for_run(store: Store, run_id: str) -> str | None:
    try:
        ms = json.loads(store.run_config_json(run_id)).get("memory_service") or {}
    except (TypeError, ValueError):
        return None
    return ms.get("base_url") or (os.environ.get(ms["base_url_env"]) if ms.get("base_url_env") else None)


# -- grouping --------------------------------------------------------------------------------
def group_stats(eps: list[dict]) -> dict:
    n = len(eps)
    passes = sum(e["passed"] for e in eps)
    ci = wilson(passes, n)
    checks = Counter()
    for e in eps:
        for k, v in e["checks"].items():
            if not v:
                checks[k] += 1
    outcomes = Counter(e["outcome"] or "?" for e in eps)
    gpu = [e["agent_gpu_s"] for e in eps if e["agent_gpu_s"] is not None]
    mem = [e["memory_s"] for e in eps if e["memory_s"] is not None]
    gpu_total = sum(gpu) if len(gpu) == n and n else None
    mem_total = sum(mem) if len(mem) == n and n else None
    return {
        "n": n, "passes": passes, "pass_rate": passes / n if n else None,
        "ci95": list(ci) if ci else None,
        "check_failures": {k: checks.get(k, 0) for k in (*CHECKS, *sorted(set(checks) - set(CHECKS)))},
        "outcomes": {k: outcomes.get(k, 0) for k in (*OUTCOMES, *sorted(set(outcomes) - set(OUTCOMES)))},
        "turns": {"mean": _mean([e["turns"] for e in eps if e["turns"] is not None]),
                  "median": _median([e["turns"] for e in eps if e["turns"] is not None])},
        "wall_s": {"mean": _mean([e["wall_s"] for e in eps if e["wall_s"] is not None]),
                   "median": _median([e["wall_s"] for e in eps if e["wall_s"] is not None])},
        "agent_gpu_s": {"mean": _mean(gpu), "median": _median(gpu), "total": gpu_total, "n": len(gpu)},
        "tokens": {k: sum(e[k] for e in eps) for k in ("prompt_n", "predicted_n", "cache_n")},
        "memory_s": {"mean": _mean(mem), "total": mem_total, "n": len(mem),
                     "estimated": any(e["memory_method"] == "window" for e in eps),
                     "input_tokens": sum(e["memory_input_tokens"] for e in eps),
                     "output_tokens": sum(e["memory_output_tokens"] for e in eps)},
        "gpu_s_per_success": {
            "agent_only": gpu_total / passes if passes and gpu_total is not None else None,
            "with_memory": (gpu_total + mem_total) / passes
            if passes and gpu_total is not None and mem_total is not None else None},
    }


def _learning_to_date(train: list[dict], checkpoint: int | None) -> float | None:
    """Memory-LLM seconds spent by the first `checkpoint` train episodes of one stream."""
    if checkpoint is None:
        return None
    first = sorted(train, key=lambda e: (e["ordinal"] if e["ordinal"] is not None else 0, e["started_at"] or ""))
    first = first[:checkpoint]
    if any(e["memory_s"] is None for e in first):
        return None
    return sum(e["memory_s"] for e in first)


def grouped(eps: list[dict]) -> list[dict]:
    """Stats per phase × arm, ordered train first, then evals by checkpoint."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for e in eps:
        groups[(e["phase"], e["arm"])].append(e)
    out = []
    for (phase, arm) in sorted(groups, key=lambda k: (_phase_key(k[0]), k[1] or "")):
        g = group_stats(groups[(phase, arm)])
        g.update(phase=phase, arm=arm)
        if phase != "train":
            ck = groups[(phase, arm)][0]["checkpoint"]
            per_stream = []
            for hist in {e["history_label"] for e in groups[(phase, arm)]}:
                train = [e for e in eps if e["split"] == "train" and e["arm"] == arm and e["history_label"] == hist]
                per_stream.append(_learning_to_date(train, ck) if arm != "A" else 0.0)
            g["learning_s_to_date"] = sum(per_stream) if per_stream and None not in per_stream else None
        out.append(g)
    return out


def paired(eps: list[dict], base: str, other: str) -> list[dict]:
    """McNemar table and mean paired differences (other − base) per eval phase, matching episodes
    on (history, phase, task_id); repeats of one task are paired in run order."""
    by: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for e in eps:
        if e["phase"] != "train" and e["arm"] in (base, other):
            by[(e["phase"], e["history_label"], e["task_id"])][e["arm"]].append(e)
    phases: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for (phase, _, _), arms in by.items():
        phases[phase] += list(zip(arms[base], arms[other]))
    out = []
    for phase in sorted(phases, key=_phase_key):
        pairs = phases[phase]
        t = Counter((a["passed"], b["passed"]) for a, b in pairs)

        def diffs(key: str) -> float | None:
            d = [b[key] - a[key] for a, b in pairs if a[key] is not None and b[key] is not None]
            return _mean(d)

        out.append({"phase": phase, "base": base, "other": other, "pairs": len(pairs),
                    "both": t[(True, True)], "base_only": t[(True, False)], "other_only": t[(False, True)],
                    "neither": t[(False, False)], "mcnemar_p": mcnemar_exact(t[(True, False)], t[(False, True)]),
                    "mean_diff": {"turns": diffs("turns"), "agent_gpu_s": diffs("agent_gpu_s"),
                                  "wall_s": diffs("wall_s")}})
    return out


def _comparisons(eps: list[dict]) -> list[dict]:
    arms = sorted({e["arm"] for e in eps if e["arm"]})
    if len(arms) < 2:
        return []
    base = "A" if "A" in arms else arms[0]
    return [p for other in arms if other != base for p in paired(eps, base, other)]


def summarize_batch(store: Store, batch_prefix: str | None = None, *, run_id: str | None = None,
                    fetch: TraceFetcher | None = None, memory_url: str | None = None) -> dict:
    """Pooled and per-history comparison of a batch (or run-id prefix). `fetch` reads the memory
    trace (see `http_trace_fetcher`); None leaves memory-LLM cost of learning episodes unknown."""
    if batch_prefix is not None:
        batch_prefix = batch_prefix.rstrip("/") or batch_prefix
    eps = load_episodes(store, batch_prefix, run_id)
    trace = add_memory_trace(store, eps, fetch, memory_url)
    histories = sorted({e["history_label"] for e in eps})
    return {
        "selector": {"batch_prefix": batch_prefix, "run_id": run_id},
        "episodes": len(eps), "histories": histories,
        "memory_trace": trace,
        "pooled": {"groups": grouped(eps), "paired": _comparisons(eps)},
        "per_history": {h: {"groups": grouped([e for e in eps if e["history_label"] == h]),
                            "paired": _comparisons([e for e in eps if e["history_label"] == h])}
                        for h in histories},
        "episode_rows": [{k: e[k] for k in ("episode_id", "history_label", "arm", "phase", "task_id", "passed",
                                             "outcome", "turns", "wall_s", "agent_gpu_s", "memory_s",
                                             "memory_method", "calls_without_timings")} for e in eps],
    }


# -- text report -----------------------------------------------------------------------------
def _f(x, spec: str = ".1f", none: str = "-") -> str:
    return none if x is None else format(x, spec)


def _table(rows: list[list[str]], header: list[str]) -> list[str]:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).rjust(w) if i > 1 else str(c).ljust(w) for i, (c, w) in enumerate(zip(r, widths)))
    return [line(header), line(["-" * w for w in widths]), *map(line, rows)]


def _format_block(block: dict) -> list[str]:
    groups = block["groups"]
    out = _table([[g["phase"], g["arm"], g["n"], g["passes"], _f(g["pass_rate"], ".0%"),
                   f"{g['ci95'][0]:.0%}-{g['ci95'][1]:.0%}" if g["ci95"] else "-",
                   f"{_f(g['turns']['mean'])}/{_f(g['turns']['median'])}",
                   f"{_f(g['wall_s']['mean'])}/{_f(g['wall_s']['median'])}",
                   _f(g["agent_gpu_s"]["mean"]) + ("" if g["agent_gpu_s"]["n"] == g["n"] else f" ({g['agent_gpu_s']['n']}/{g['n']})"),
                   _f(g["memory_s"]["mean"]) + ("~" if g["memory_s"]["estimated"] else ""),
                   _f(g.get("learning_s_to_date")) if g["phase"] != "train" else "",
                   _f(g["gpu_s_per_success"]["with_memory"]), _f(g["gpu_s_per_success"]["agent_only"])]
                  for g in groups],
                 ["phase", "arm", "n", "pass", "rate", "95% CI", "turns mean/med", "wall s mean/med",
                  "agent GPU s", "mem s", "learn s to date", "GPU s/succ", "agent s/succ"])
    out.append("")
    out += _table([[g["phase"], g["arm"], *[g["check_failures"].get(k, 0) for k in CHECKS],
                    " ".join(f"{k}={v}" for k, v in g["outcomes"].items() if v),
                    f"{g['tokens']['prompt_n']}/{g['tokens']['predicted_n']}/{g['tokens']['cache_n']}"]
                   for g in groups],
                  ["phase", "arm", "fail:integrity", "public", "correct", "perf", "outcomes",
                   "agent tok prompt/pred/cache"])
    if block["paired"]:
        out.append("")
        out += _table([[p["phase"], f"{p['base']} vs {p['other']}", p["pairs"], p["both"], p["base_only"],
                        p["other_only"], p["neither"], _f(p["mcnemar_p"], ".3f"),
                        _f(p["mean_diff"]["turns"], "+.1f"), _f(p["mean_diff"]["agent_gpu_s"], "+.1f"),
                        _f(p["mean_diff"]["wall_s"], "+.1f")] for p in block["paired"]],
                      ["phase", "pair", "tasks", "both", "base only", "other only", "neither", "McNemar p",
                       "d turns", "d GPU s", "d wall s"])
    return out


def format_summary(summary: dict) -> str:
    sel = summary["selector"]
    what = f"batch {sel['batch_prefix']}" if sel["batch_prefix"] is not None else f"run-id prefix {sel['run_id']}"
    lines = [f"{what}: {summary['episodes']} episodes, {len(summary['histories'])} histories"]
    mt = summary["memory_trace"]
    if not mt["enabled"]:
        lines.append("memory-LLM trace: off (learning-arm train cost unknown)")
    else:
        for bank, st in mt["banks"].items():
            lines.append(f"memory-LLM trace {bank}: {st}")
        if mt["estimated_calls"]:
            lines.append(f"  {mt['estimated_calls']} calls attributed by time window (estimate, marked ~)")
        if mt["unattributed_s"]:
            lines.append(f"  {mt['unattributed_s']:.1f}s of traced memory-LLM time matched no episode")
    lines.append("agent GPU s = llama.cpp prompt_ms+predicted_ms per episode; mem s = Hindsight memory-LLM "
                 "request time (retain+consolidation) per episode; d = other − base on paired tasks")
    if not summary["episodes"]:
        return "\n".join(lines)
    lines += ["", "== pooled over histories =="] + _format_block(summary["pooled"])
    if len(summary["histories"]) > 1:
        for h, block in summary["per_history"].items():
            lines += ["", f"== history {h} =="] + _format_block(block)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m elh.analysis", description=__doc__.splitlines()[0])
    ap.add_argument("batch_prefix", nargs="?", help="experiment name (batch_id prefix)")
    ap.add_argument("--run-id", help="select by run_id prefix instead of batch")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    ap.add_argument("--no-memory-trace", action="store_true", help="don't query Hindsight's LLM request trace")
    ap.add_argument("--memory-url", help="Hindsight base URL (default: each run's registered memory_service)")
    args = ap.parse_args(argv)
    if (args.batch_prefix is None) == (args.run_id is None):
        ap.error("give exactly one of batch_prefix or --run-id")
    store = Store(args.runs_dir)
    fetch = None if args.no_memory_trace else http_trace_fetcher()
    summary = summarize_batch(store, args.batch_prefix, run_id=args.run_id, fetch=fetch, memory_url=args.memory_url)
    print(json.dumps(summary, indent=2, default=str) if args.json else format_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
