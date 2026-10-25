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

Continual-learning metrics (methods of the cited papers, adapted to arm A as the no-memory baseline):
- ΔRR / ΔNL (Hu, Long & Wang 2026, "When Continual Learning Moves to Memory", §2.3): each eval
  panel is split by arm A's outcome on the same task (labels fixed from A before any other arm is
  read); RR = pass rate on A-pass tasks (retention; ΔRR < 0 is harmful reuse), NL = pass rate on
  A-fail tasks (new learning). NL is suppressed when fewer than 10 tasks are A-fail.
- Paired cluster bootstrap by task (Tablan et al. 2026, "Learning on the Job", App. E): one cluster
  per task holding all its episodes over histories × repeats, the same resampled clusters for both
  arms, seeded, percentile intervals; replicates where a statistic is undefined are dropped and
  counted.
- Hold rate (ibid. §3.3.4): P(pass at checkpoint t+1 | pass at t) over tasks measured at
  successive checkpoints, per arm, against arm A's (re-draw) hold.
- FWT / BWT (Moves to Memory §2.3) on the sealed and retention splits; None while absent.
- Retrieval diagnostics (Moves to Memory App. D): coverage, top-1 concentration, and the reach of
  failed-check lessons into eval episodes (pluggable predicate).
- Steps (Wang et al., "Agent Workflow Memory"): turns, as a paired difference on all tasks and on
  tasks both arms solved.

Arm D (procedure library): `dev` episodes (the promotion gate, GEPA's search) are learning, not
evaluation, so they are left out of every comparison above and reported in their own section: per
(run, history) stream the library revisions with their status and gate decision, and the learning
spend of the procedures: dev episodes (count, agent GPU-s, ledger tokens and USD, all `learning`)
plus induction / reflection calls (ledger rows without an episode, by `purpose`).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import re
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
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
# Bootstrap defaults (Learning on the Job, App. E: B = 10,000, fixed published seed).
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261025
# Moves to Memory omits NL when the baseline-fail subset has fewer than 10 items.
MIN_NL_TASKS = 10
SEALED_SPLITS = ("sealed_familiar", "sealed_novel_combo", "sealed_novel_cause")
RETENTION_SPLIT = "retention"

TraceFetcher = Callable[[str, str], "list[dict] | None"]
# (retrieved item, its source episode or None) -> True / False / None (cannot tell).
LessonPredicate = Callable[[dict, "dict | None"], "bool | None"]


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


def _percentile(sorted_xs: list[float], q: float) -> float:
    """Linear-interpolation percentile of an already sorted list (q in [0, 1])."""
    pos = q * (len(sorted_xs) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def cluster_bootstrap(clusters: list[dict[str, float]], stats: dict[str, Callable[[dict], "float | None"]],
                      resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    """Percentile bootstrap over clusters. Each cluster is a dict of additive totals (e.g. both
    arms' episode and pass counts for one task); a replicate resamples clusters with replacement,
    sums them, and evaluates every statistic on the same draw, so contrasts between arms stay
    paired. A statistic returning None on a replicate is dropped there and counted as undefined."""
    keys = sorted({k for c in clusters for k in c})
    cols = [[c.get(k, 0.0) for c in clusters] for k in keys]

    def totals(idx: list[int] | range) -> dict:
        return {k: sum(col[i] for i in idx) for k, col in zip(keys, cols)}

    full = totals(range(len(clusters)))
    out = {name: {"est": f(full) if clusters else None, "ci95": None, "undefined": 0} for name, f in stats.items()}
    if not clusters or resamples <= 0:
        return {"clusters": len(clusters), "resamples": 0, "seed": seed, "stats": out}
    rng = random.Random(seed)
    n = len(clusters)
    draws: dict[str, list[float]] = {name: [] for name in stats}
    for _ in range(resamples):
        t = totals([rng.randrange(n) for _ in range(n)])
        for name, f in stats.items():
            v = f(t)
            if v is None:
                out[name]["undefined"] += 1
            else:
                draws[name].append(v)
    for name, xs in draws.items():
        if xs:
            xs.sort()
            out[name]["ci95"] = [_percentile(xs, 0.025), _percentile(xs, 0.975)]
    return {"clusters": n, "resamples": resamples, "seed": seed, "stats": out}


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


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
            "cause": r.get("cause"), "memory_snapshot_id": r.get("memory_snapshot_id"),
            "trace_sha256": r.get("trace_sha256"),
            "retrieved_ids": _json_ids(r.get("retrieved_memory_ids_json")),
            # Optional richer record (candidates with rank/score/source); no such column yet.
            "retrieval": _json_or_none(r.get("retrieval_json", r.get("retrieval"))),
            **agent_compute(calls[r["episode_id"]]),
        })
    return out


def _json_or_none(s):
    if s is None or isinstance(s, (list, dict)):
        return s
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return None


def _json_ids(s) -> list[str]:
    v = _json_or_none(s)
    return [str(x) for x in v] if isinstance(v, list) else []


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


def paired(eps: list[dict], base: str, other: str, *, resamples: int = BOOTSTRAP_RESAMPLES,
           seed: int = BOOTSTRAP_SEED) -> list[dict]:
    """McNemar table and mean paired differences (other − base) per eval phase, matching episodes
    on (history, phase, task_id); repeats of one task are paired in run order. `bootstrap` adds the
    paired cluster bootstrap by task (all of a task's episodes over histories × repeats form one
    cluster) for the pass-rate difference and the agent GPU-s-per-success ratio (other / base)."""
    by: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for e in eps:
        if e["phase"] != "train" and e["arm"] in (base, other):
            by[(e["phase"], e["history_label"], e["task_id"])][e["arm"]].append(e)
    phases: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    clusters: dict[str, dict[str, dict[str, list[dict]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (phase, _, task), arms in by.items():
        phases[phase] += list(zip(arms[base], arms[other]))
        if arms[base] and arms[other]:
            for arm in (base, other):
                clusters[phase][task][arm] += arms[arm]
    out = []
    for phase in sorted(phases, key=_phase_key):
        pairs = phases[phase]
        t = Counter((a["passed"], b["passed"]) for a, b in pairs)

        def diffs(key: str, only_both_pass: bool = False) -> float | None:
            d = [b[key] - a[key] for a, b in pairs if a[key] is not None and b[key] is not None
                 and (not only_both_pass or (a["passed"] and b["passed"]))]
            return _mean(d)

        out.append({"phase": phase, "base": base, "other": other, "pairs": len(pairs),
                    "both": t[(True, True)], "base_only": t[(True, False)], "other_only": t[(False, True)],
                    "neither": t[(False, False)], "mcnemar_p": mcnemar_exact(t[(True, False)], t[(False, True)]),
                    "mean_diff": {"turns": diffs("turns"), "agent_gpu_s": diffs("agent_gpu_s"),
                                  "wall_s": diffs("wall_s"),
                                  # AWM's step reduction, restricted to tasks both arms solved.
                                  "turns_both_pass": diffs("turns", only_both_pass=True)},
                    "bootstrap": paired_bootstrap(clusters[phase], base, other, resamples=resamples, seed=seed)})
    return out


def paired_bootstrap(by_task: dict[str, dict[str, list[dict]]], base: str, other: str, *,
                     resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    """Cluster bootstrap by task of (other − base) pass rate and (other / base) agent GPU-s per
    success. The GPU ratio is None when any episode lacks timings (no partial sums)."""
    units = []
    gpu_complete = True
    for task in sorted(by_task):
        u: dict[str, float] = {}
        for tag, arm in (("b", base), ("o", other)):
            es = by_task[task][arm]
            u[f"{tag}_n"] = len(es)
            u[f"{tag}_pass"] = sum(e["passed"] for e in es)
            gpu = [e["agent_gpu_s"] for e in es]
            gpu_complete &= None not in gpu
            u[f"{tag}_gpu"] = sum(g for g in gpu if g is not None)
        units.append(u)

    def pass_diff(t: dict) -> float | None:
        if not t["b_n"] or not t["o_n"]:
            return None
        return t["o_pass"] / t["o_n"] - t["b_pass"] / t["b_n"]

    def gpu_ratio(t: dict) -> float | None:
        b, o = _ratio(t["b_gpu"], t["b_pass"]), _ratio(t["o_gpu"], t["o_pass"])
        return _ratio(o, b) if b is not None and o is not None else None

    stats: dict[str, Callable] = {"pass_diff": pass_diff}
    if gpu_complete:
        stats["gpu_per_success_ratio"] = gpu_ratio
    res = cluster_bootstrap(units, stats, resamples, seed)
    if not gpu_complete:
        res["stats"]["gpu_per_success_ratio"] = {"est": None, "ci95": None, "undefined": None,
                                                 "note": "episodes without llama.cpp timings"}
    return res


def _comparisons(eps: list[dict], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> list[dict]:
    arms = sorted({e["arm"] for e in eps if e["arm"]})
    if len(arms) < 2:
        return []
    base = "A" if "A" in arms else arms[0]
    return [p for other in arms if other != base for p in paired(eps, base, other, resamples=resamples, seed=seed)]


def _base_arm(eps: list[dict]) -> str | None:
    arms = sorted({e["arm"] for e in eps if e["arm"]})
    return ("A" if "A" in arms else arms[0]) if arms else None


# -- retention / new learning (Moves to Memory ΔRR / ΔNL) ------------------------------------
def retention_new_learning(eps: list[dict], base: str = "A", control: str | None = None,
                           min_nl_tasks: int = MIN_NL_TASKS) -> list[dict]:
    """Per eval phase and non-base arm: RR on tasks the base arm passed, NL on tasks it failed.

    Labels are fixed from `base` alone, per (history, task): pass if every base episode passed,
    fail if none did, mixed otherwise (excluded and counted). Deltas are against `control` (default
    `base`; give a memory-enabled scratch arm to reproduce the paper's ∅→T control). When the
    labels come from the same base episodes, the base arm's RR is 1 and NL 0 by construction, so the
    deltas are B's absolute misses (ΔRR) and gains (ΔNL). NL and ΔNL are withheld (n still shown)
    when fewer than `min_nl_tasks` tasks are base-fail."""
    control = control or base
    by_phase: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        if e["phase"] != "train":
            by_phase[e["phase"]].append(e)
    out = []
    for phase in sorted(by_phase, key=_phase_key):
        pe = by_phase[phase]
        base_runs: dict[tuple, list[bool]] = defaultdict(list)
        for e in pe:
            if e["arm"] == base:
                base_runs[(e["history_label"], e["task_id"])].append(e["passed"])
        if not base_runs:
            continue
        label = {k: "pass" if all(v) else "fail" if not any(v) else "mixed" for k, v in base_runs.items()}
        counts = Counter(label.values())

        def rates(arm: str) -> dict:
            sub: dict[str, list[bool]] = {"pass": [], "fail": []}
            for e in pe:
                lab = label.get((e["history_label"], e["task_id"]))
                if e["arm"] == arm and lab in sub:
                    sub[lab].append(e["passed"])
            return {"rr": _mean(sub["pass"]), "rr_n": len(sub["pass"]),
                    "nl": _mean(sub["fail"]), "nl_n": len(sub["fail"])}

        ctl = rates(control)
        for arm in sorted({e["arm"] for e in pe if e["arm"] and e["arm"] not in (base, control)}):
            r = rates(arm)
            suppressed = counts["fail"] < min_nl_tasks
            d_rr = r["rr"] - ctl["rr"] if r["rr"] is not None and ctl["rr"] is not None else None
            d_nl = r["nl"] - ctl["nl"] if r["nl"] is not None and ctl["nl"] is not None else None
            out.append({"phase": phase, "arm": arm, "base": base, "control": control,
                        "tasks": {"pass": counts["pass"], "fail": counts["fail"], "mixed": counts["mixed"]},
                        "rr": r["rr"], "rr_n": r["rr_n"], "d_rr": d_rr,
                        "nl": None if suppressed else r["nl"], "nl_n": r["nl_n"],
                        "d_nl": None if suppressed else d_nl, "nl_suppressed": suppressed})
    return out


# -- hold rate (Learning on the Job) ---------------------------------------------------------
def hold_rates(eps: list[dict], base: str = "A", *, resamples: int = BOOTSTRAP_RESAMPLES,
               seed: int = BOOTSTRAP_SEED) -> list[dict]:
    """P(pass at checkpoint t+1 | pass at t) per arm, over (history, task) measured at successive
    checkpoints of one panel (phase name before '@': eval, retention, ...). Repeats at one checkpoint
    are paired in run order. Also reports conversions (fail → pass), the base arm's hold on the
    same panel (its re-draw persistence) and Δhold with a paired cluster bootstrap by task."""
    series: dict[tuple, dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for e in eps:
        if e["phase"] == "train" or e["checkpoint"] is None:
            continue
        panel = e["phase"].partition("@")[0]
        series[(panel, e["arm"], e["history_label"], e["task_id"])][e["checkpoint"]].append(e["passed"])
    # (panel, arm) -> task -> [prev_pass, held, prev_fail, converted]
    trans: dict[tuple, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    for (panel, arm, _, task), by_ck in series.items():
        cks = sorted(by_ck)
        for c0, c1 in zip(cks, cks[1:]):
            for p0, p1 in zip(by_ck[c0], by_ck[c1]):
                t = trans[(panel, arm)][task]
                if p0:
                    t[0] += 1
                    t[1] += p1
                else:
                    t[2] += 1
                    t[3] += p1
    out = []
    for (panel, arm) in sorted(trans, key=lambda k: (k[0], k[1] or "")):
        tasks = trans[(panel, arm)]
        tot = [sum(v[i] for v in tasks.values()) for i in range(4)]
        row = {"panel": panel, "arm": arm, "transitions": tot[0] + tot[2], "prev_pass": tot[0], "held": tot[1],
               "hold": _ratio(tot[1], tot[0]), "prev_fail": tot[2], "converted": tot[3],
               "conversion": _ratio(tot[3], tot[2]), "base": None, "base_hold": None, "d_hold": None,
               "bootstrap": None}
        if arm != base and (panel, base) in trans:
            btasks = trans[(panel, base)]
            row["base"] = base
            bp, bh = sum(v[0] for v in btasks.values()), sum(v[1] for v in btasks.values())
            row["base_hold"] = _ratio(bh, bp)
            if row["hold"] is not None and row["base_hold"] is not None:
                row["d_hold"] = row["hold"] - row["base_hold"]
            units = [{"o_prev": tasks[k][0], "o_held": tasks[k][1], "b_prev": btasks[k][0], "b_held": btasks[k][1]}
                     for k in sorted(set(tasks) & set(btasks))]

            def d_hold(t: dict) -> float | None:
                o, b = _ratio(t["o_held"], t["o_prev"]), _ratio(t["b_held"], t["b_prev"])
                return o - b if o is not None and b is not None else None

            row["bootstrap"] = cluster_bootstrap(units, {"d_hold": d_hold}, resamples, seed)
        out.append(row)
    return out


# -- forward / backward transfer (Moves to Memory FWT / BWT) ---------------------------------
def _acc(eps: list[dict]) -> float | None:
    return _mean([e["passed"] for e in eps])


def forward_transfer(eps: list[dict], target_split: str, arm: str, control: str | None = None) -> dict | None:
    """FWT on a target split = Acc(arm) − Acc(control) on that split's tasks.

    The paper's control is a memory-enabled scratch run on the target alone (∅→T); name its arm as
    `control`. Without one, arm A (no memory) is used, which is Learning on the Job's "consumer vs
    its own no-memory baseline" and is marked `control_kind = "no-memory"`. None when the split, the
    arm or the control has no episodes."""
    target = [e for e in eps if e["split"] == target_split]
    ctl = control or "A"
    a = [e for e in target if e["arm"] == arm]
    c = [e for e in target if e["arm"] == ctl]
    if not a or not c:
        return None
    acc_a, acc_c = _acc(a), _acc(c)
    return {"split": target_split, "arm": arm, "control": ctl,
            "control_kind": "scratch" if control else "no-memory",
            "acc": acc_a, "control_acc": acc_c, "n": len(a), "control_n": len(c), "fwt": acc_a - acc_c}


def backward_transfer(eps: list[dict], arm: str, panel_split: str = RETENTION_SPLIT,
                      reference_checkpoint: int | None = None) -> dict | None:
    """BWT on a fixed panel re-measured at checkpoints: BWT_c = Acc_panel(c) − Acc_panel(ref), with
    ref the first checkpoint after the panel's tasks were learned (default: earliest measured).
    None when the arm has no episodes on the panel split."""
    panel = [e for e in eps if e["split"] == panel_split and e["arm"] == arm and e["checkpoint"] is not None]
    if not panel:
        return None
    by_ck: dict[int, list[dict]] = defaultdict(list)
    for e in panel:
        by_ck[e["checkpoint"]].append(e)
    ref = reference_checkpoint if reference_checkpoint in by_ck else min(by_ck)
    ref_acc = _acc(by_ck[ref])
    return {"split": panel_split, "arm": arm, "reference_checkpoint": ref,
            "checkpoints": [{"checkpoint": ck, "n": len(by_ck[ck]), "acc": _acc(by_ck[ck]),
                             "bwt": _acc(by_ck[ck]) - ref_acc} for ck in sorted(by_ck)]}


def transfer_summary(eps: list[dict], scratch_controls: dict[str, str] | None = None,
                     sealed_splits: tuple[str, ...] = SEALED_SPLITS,
                     retention_split: str = RETENTION_SPLIT) -> dict:
    """FWT per sealed split and BWT on the retention panel for every learning arm; each entry is
    None when its split is absent. `scratch_controls` maps a learning arm to its scratch arm."""
    arms = sorted({e["arm"] for e in eps if e["arm"] and e["arm"] != "A"})
    scratch = set((scratch_controls or {}).values())
    learners = [a for a in arms if a not in scratch]
    return {
        "fwt": {s: {a: forward_transfer(eps, s, a, (scratch_controls or {}).get(a)) for a in learners}
                for s in sealed_splits},
        "bwt": {a: backward_transfer(eps, a, retention_split) for a in learners},
    }


# -- retrieval diagnostics -------------------------------------------------------------------
_FAILED_TAGS = ("outcome:fail", "outcome:failed", "verdict:fail", "verdict:failed", "failed-check", "failed_check")
# lesson-v2 records say "FAILED: ..." and name protected files; extracted facts paraphrase that.
_FAILED_TEXT = re.compile(r"\bfail(?:ed|s|ure|ing)?\b|protected files?|must not be (?:changed|modified)|forbid",
                          re.IGNORECASE)


def failed_check_lesson(item: dict, source: dict | None) -> bool | None:
    """Default lesson predicate: is this retrieved item a failed-check lesson?

    Evidence, in order: a failure tag on the item (`outcome:failed`, `failed-check`, ...); its
    source episode failed (needs `document_id` in the retrieval record); its text states a failure
    (regex over the injected text). None when the item carries none of these."""
    tags = item.get("tags")
    if tags is not None and any(str(t).lower().startswith(_FAILED_TAGS) for t in tags):
        return True
    if source is not None:
        return not source["passed"]
    text = item.get("text")
    if text is not None:
        return bool(_FAILED_TEXT.search(text))
    return False if tags is not None else None


def retrieval_items(e: dict) -> list[dict]:
    """The episode's retrieval as [{id, rank, injected, ...}] from the optional `retrieval` record
    (a list of candidates, or {"candidates": [...]}) or else from `retrieved_memory_ids_json`."""
    rec = e.get("retrieval")
    cands = rec.get("candidates") if isinstance(rec, dict) else rec
    injected_ids = set(e.get("retrieved_ids") or [])
    if isinstance(cands, list) and cands:
        items = []
        for i, c in enumerate(cands):
            if not isinstance(c, dict):
                c = {"id": c}
            cid = str(c.get("id", c.get("memory_id")))
            inj = c.get("injected", cid in injected_ids if injected_ids else True)
            items.append({**c, "id": cid, "rank": c.get("rank", i), "injected": bool(inj)})
        return sorted(items, key=lambda c: c["rank"])
    return [{"id": m, "rank": i, "injected": True} for i, m in enumerate(e.get("retrieved_ids") or [])]


def parse_injected_texts(first_message: str, ids: list[str]) -> dict[str, str]:
    """Texts of the memory items rendered into the first user message (`render_memory_block`:
    "[id]\\ntext" blocks inside <experience>), by item id."""
    block = first_message
    if "<experience>" in block:
        block = block.split("<experience>", 1)[1].split("</experience>", 1)[0]
    starts = sorted((block.find(f"[{i}]\n"), i) for i in ids if f"[{i}]\n" in block)
    out = {}
    for n, (pos, mid) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(block)
        out[mid] = block[pos + len(mid) + 3:end].strip()
    return out


def attach_injected_texts(store: Store, eps: list[dict]) -> int:
    """Read each retrieving episode's trace (episode_start event) and attach the injected texts to
    `e["item_texts"]`. Returns how many episodes got texts; unreadable traces are skipped."""
    done = 0
    for e in eps:
        ids = [it["id"] for it in retrieval_items(e) if it["injected"]]
        if not ids or not e.get("trace_sha256"):
            continue
        try:
            first = store.get_blob("traces", e["trace_sha256"]).split(b"\n", 1)[0]
            start = json.loads(first)
        except (OSError, ValueError):
            continue
        if start.get("event") == "episode_start" and isinstance(start.get("first_message"), str):
            e["item_texts"] = parse_injected_texts(start["first_message"], ids)
            done += 1
    return done


def snapshot_bank_sizes(store: Store, snapshot_ids) -> dict[str, int | None]:
    """Memory units (facts + observations + mental models) in each snapshot bank, from the manifest
    of its stored transfer archive; None when it cannot be read."""
    out: dict[str, int | None] = {}
    for sid in snapshot_ids:
        try:
            row = store.snapshot(sid)
            with zipfile.ZipFile(io.BytesIO(store.get_blob("snapshots", row["bank_export_sha256"]))) as z:
                m = json.loads(z.read("manifest.json"))
            out[sid] = sum(int(m.get(k) or 0) for k in ("fact_count", "observation_count", "mental_model_count"))
        except (KeyError, OSError, ValueError, zipfile.BadZipFile):
            out[sid] = None
    return out


def _bank_key(e: dict) -> str:
    return e.get("memory_snapshot_id") or f"live:{e['run_id']}/h{e['history']}/{e['arm']}"


def retrieval_diagnostics(eps: list[dict], *, bank_sizes: dict[str, int | None] | None = None,
                          predicate: LessonPredicate = failed_check_lesson,
                          sources: dict[str, dict] | None = None) -> list[dict]:
    """Per phase × arm (arms that retrieved anything):
    - coverage = unique items / retrievals (Moves to Memory App. D);
    - bank_coverage = unique items / units in the bank (snapshot banks with a known size only;
      ids are per bank, since restoring a snapshot assigns new ids);
    - top1_concentration = share of retrieving episodes whose top-1 is the most frequent top-1;
    - lesson_reach = share of episodes whose injected items include one `predicate` marks a
      lesson (default: failed-check lesson), over episodes where the predicate could decide, with
      pass rates with and without one injected; lesson_best_rank = median 0-based best rank of such
      an item among the recorded candidates (only the injected top-k unless the `retrieval` record
      lists deeper candidates);
    - same_cause_reach = share whose injected items include one from a source episode of the
      same cause (needs `document_id` on items)."""
    sources = sources or {}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for e in eps:
        groups[(e["phase"], e["arm"])].append(e)
    out = []
    for (phase, arm) in sorted(groups, key=lambda k: (_phase_key(k[0]), k[1] or "")):
        ge = groups[(phase, arm)]
        per_ep = [(e, retrieval_items(e)) for e in ge]
        if not any(items for _, items in per_ep):
            continue
        uniq: dict[str, set] = defaultdict(set)
        top1: Counter = Counter()
        retrievals = 0
        reach = {True: [], False: []}
        unknown = 0
        best_ranks: list[int] = []
        same_cause, same_cause_known = 0, 0
        for e, items in per_ep:
            texts = e.get("item_texts") or {}
            inj = [{**it, "text": it.get("text", texts.get(it["id"]))} for it in items if it["injected"]]
            retrievals += len(inj)
            uniq[_bank_key(e)].update(it["id"] for it in inj)
            if inj:
                top1[(_bank_key(e), inj[0]["id"])] += 1
            verdicts = [predicate(it, sources.get(it.get("document_id"))) for it in inj]
            if any(v is True for v in verdicts):
                reach[True].append(e["passed"])
            elif all(v is False for v in verdicts):
                reach[False].append(e["passed"])
            else:
                unknown += 1
            lesson_ranks = [it["rank"] for it in items
                            if predicate({**it, "text": it.get("text", texts.get(it["id"]))},
                                         sources.get(it.get("document_id"))) is True]
            if lesson_ranks:
                best_ranks.append(min(lesson_ranks))
            srcs = [sources.get(it.get("document_id")) for it in inj]
            if any(s is not None for s in srcs):
                same_cause_known += 1
                same_cause += any(s is not None and s.get("cause") == e.get("cause") for s in srcs)
        n_uniq = sum(len(v) for v in uniq.values())
        sizes = [(bank_sizes or {}).get(b) for b in uniq]
        bank_cov = (n_uniq / sum(sizes)) if sizes and None not in sizes and sum(sizes) else None
        retrieving = sum(top1.values())
        known = len(reach[True]) + len(reach[False])
        out.append({
            "phase": phase, "arm": arm, "episodes": len(ge), "retrieving": retrieving,
            "retrievals": retrievals, "unique": n_uniq, "coverage": _ratio(n_uniq, retrievals),
            "bank_units": sum(sizes) if sizes and None not in sizes else None, "bank_coverage": bank_cov,
            "top1_concentration": _ratio(top1.most_common(1)[0][1], retrieving) if top1 else None,
            "lesson_reach": _ratio(len(reach[True]), known), "lesson_known": known, "lesson_unknown": unknown,
            "pass_with_lesson": _mean(reach[True]), "pass_without_lesson": _mean(reach[False]),
            "lesson_best_rank": _median(best_ranks),
            "same_cause_reach": _ratio(same_cause, same_cause_known), "same_cause_known": same_cause_known,
        })
    return out


# -- arm D procedures ------------------------------------------------------------------------
def procedure_summary(store: Store, eps: list[dict]) -> list[dict]:
    """Per arm-D stream (run, history) of the selection: library revisions (status, size, source,
    gate decision) and the procedures' learning spend (dev episodes + induction + reflection)."""
    out = []
    for run_id, history in sorted({(e["run_id"], e["history"]) for e in eps if e["arm"] == "D"}):
        revs = []
        for r in store.db.execute(
                "SELECT revision, family, status, COUNT(*) AS n, MIN(created_at) AS created_at, "
                "MAX(promoted_at) AS promoted_at, MAX(rolled_back_at) AS rolled_back_at, "
                "MAX(validation_json) AS validation_json, MAX(induction_json) AS induction_json, "
                "MAX(source_episode_ids_json) AS sources FROM skills WHERE run_id = ? AND history = ? "
                "AND arm = 'D' GROUP BY revision, family, status ORDER BY revision", (run_id, history)):
            v = _json_or_none(r["validation_json"]) or {}
            ind = _json_or_none(r["induction_json"]) or {}
            revs.append({"revision": r["revision"], "family": r["family"], "status": r["status"],
                         "workflows": r["n"], "source": v.get("source") or ind.get("source") or "awm",
                         "sources": len(_json_ids(r["sources"])),
                         "after": v.get("after_train_episodes"), "incumbent": v.get("incumbent_revision"),
                         "gate": {k: v.get(k) for k in ("n", "tasks", "incumbent_mean_reward", "candidate_mean_reward",
                                                        "mean_paired_diff", "incumbent_passes", "candidate_passes",
                                                        "candidate_integrity_violations", "margin")},
                         "reasons": v.get("reasons") or [], "created_at": r["created_at"],
                         "promoted_at": r["promoted_at"], "rolled_back_at": r["rolled_back_at"]})
        dev = [e for e in eps if e["split"] == "dev" and e["run_id"] == run_id and e["history"] == history]
        ids = [e["episode_id"] for e in dev]
        dev_usd = dev_tokens = 0.0
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            row = store.db.execute(
                f"SELECT COALESCE(SUM(usd), 0), COALESCE(SUM(CASE WHEN unit = 'tokens' THEN units END), 0) "
                f"FROM ledger WHERE category = 'learning' AND episode_id IN ({','.join('?' * len(chunk))})",
                chunk).fetchone()
            dev_usd += row[0]
            dev_tokens += row[1]
        gpu = [e["agent_gpu_s"] for e in dev if e["agent_gpu_s"] is not None]
        calls = {r["purpose"]: {"rows": r["n"], "tokens": r["tokens"], "usd": r["usd"]} for r in store.db.execute(
            "SELECT purpose, COUNT(*) AS n, COALESCE(SUM(CASE WHEN unit = 'tokens' THEN units END), 0) AS tokens, "
            "COALESCE(SUM(usd), 0) AS usd FROM ledger WHERE run_id = ? AND history = ? AND arm = 'D' "
            "AND episode_id IS NULL AND category = 'learning' GROUP BY purpose", (run_id, history))}
        promoted = [r["revision"] for r in revs if r["status"] == "promoted"]
        out.append({"run_id": run_id, "history": history, "revisions": revs,
                    "active_revision": promoted[-1] if promoted else None,
                    "dev": {"episodes": len(dev), "passes": sum(e["passed"] for e in dev),
                            "agent_gpu_s": sum(gpu) if len(gpu) == len(dev) and dev else None,
                            "tokens": dev_tokens, "usd": dev_usd},
                    "calls": calls,
                    "learning_usd": dev_usd + sum(c["usd"] for c in calls.values())})
    return out


def _format_procedures(streams: list[dict]) -> list[str]:
    out: list[str] = []
    for st in streams:
        out += ["", f"== arm D procedures: {st['run_id']} h{st['history']} (active revision "
                    f"{_f(st['active_revision'], 'd')}) =="]
        if st["revisions"]:
            out += _table([[r["revision"], r["status"], r["source"], r["workflows"], r["sources"], _f(r["after"], "d"),
                            _f(r["incumbent"], "d"),
                            f"{_f(r['gate']['incumbent_mean_reward'], '.3f')}->{_f(r['gate']['candidate_mean_reward'], '.3f')}",
                            f"{_f(r['gate']['incumbent_passes'], 'd')}->{_f(r['gate']['candidate_passes'], 'd')}"
                            f"/{_f(r['gate']['n'], 'd')}", "; ".join(r["reasons"])]
                           for r in st["revisions"]],
                          ["rev", "status", "source", "workflows", "from eps", "after", "vs rev", "gate reward",
                           "gate passes", "decision"])
        else:
            out.append("no candidate library yet")
        d = st["dev"]
        rows = [["dev episodes", d["episodes"], f"{d['passes']} passed", _f(d["agent_gpu_s"]), f"{d['tokens']:.0f}",
                 f"{d['usd']:.4f}"]]
        rows += [[p or "?", c["rows"], "ledger rows", "-", f"{c['tokens']:.0f}", f"{c['usd']:.4f}"]
                 for p, c in sorted(st["calls"].items(), key=lambda kv: kv[0] or "")]
        out += [""] + _table(rows, ["learning", "n", "", "agent GPU s", "tokens", "USD"])
        out.append(f"procedure learning total: ${st['learning_usd']:.4f} (shadow prices; memory-LLM retain cost "
                   "is reported with the train phase)")
    return out


def _block(eps: list[dict], *, resamples: int, seed: int, bank_sizes: dict, predicate: LessonPredicate,
           sources: dict) -> dict:
    base = _base_arm(eps)
    return {"groups": grouped(eps), "paired": _comparisons(eps, resamples, seed),
            "rr_nl": retention_new_learning(eps, base) if base else [],
            "hold": hold_rates(eps, base, resamples=resamples, seed=seed) if base else [],
            "retrieval": retrieval_diagnostics(eps, bank_sizes=bank_sizes, predicate=predicate, sources=sources)}


def summarize_batch(store: Store, batch_prefix: str | None = None, *, run_id: str | None = None,
                    fetch: TraceFetcher | None = None, memory_url: str | None = None,
                    resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
                    lesson_predicate: LessonPredicate = failed_check_lesson, read_traces: bool = True,
                    scratch_controls: dict[str, str] | None = None) -> dict:
    """Pooled and per-history comparison of a batch (or run-id prefix). `fetch` reads the memory
    trace (see `http_trace_fetcher`); None leaves memory-LLM cost of learning episodes unknown.
    `read_traces` recovers injected memory texts from episode traces for the lesson predicate."""
    if batch_prefix is not None:
        batch_prefix = batch_prefix.rstrip("/") or batch_prefix
    everything = load_episodes(store, batch_prefix, run_id)
    eps = [e for e in everything if e["split"] != "dev"]  # arm D's gate/search episodes: learning only
    trace = add_memory_trace(store, eps, fetch, memory_url)
    texts = attach_injected_texts(store, eps) if read_traces else 0
    bank_sizes = snapshot_bank_sizes(store, sorted({e["memory_snapshot_id"] for e in eps if e["memory_snapshot_id"]}))
    sources = {e["episode_id"]: {"passed": e["passed"], "cause": e["cause"], "checks": e["checks"]} for e in eps}
    histories = sorted({e["history_label"] for e in eps})
    kw = dict(resamples=resamples, seed=seed, bank_sizes=bank_sizes, predicate=lesson_predicate, sources=sources)
    return {
        "selector": {"batch_prefix": batch_prefix, "run_id": run_id},
        "episodes": len(eps), "histories": histories,
        "dev_episodes": len(everything) - len(eps),
        "procedures": procedure_summary(store, everything),
        "memory_trace": trace,
        "bootstrap": {"resamples": resamples, "seed": seed, "cluster": "task"},
        "retrieval_source": {"episodes_with_injected_texts": texts, "bank_sizes": bank_sizes,
                             "rich_records": sum(e["retrieval"] is not None for e in eps)},
        "transfer": transfer_summary(eps, scratch_controls),
        "pooled": _block(eps, **kw),
        "per_history": {h: _block([e for e in eps if e["history_label"] == h], **kw) for h in histories},
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
                        _f(p["mean_diff"]["wall_s"], "+.1f"), _f(p["mean_diff"].get("turns_both_pass"), "+.1f")]
                       for p in block["paired"]],
                      ["phase", "pair", "tasks", "both", "base only", "other only", "neither", "McNemar p",
                       "d turns", "d GPU s", "d wall s", "d turns both-pass"])
        boot = [p for p in block["paired"] if p.get("bootstrap")]
        if boot:
            out.append("")
            out += _table([[p["phase"], f"{p['base']} vs {p['other']}", p["bootstrap"]["clusters"],
                            _est_ci(p["bootstrap"]["stats"]["pass_diff"], "+.1%"),
                            _est_ci(p["bootstrap"]["stats"]["gpu_per_success_ratio"], ".2f", "x")]
                           for p in boot],
                          ["phase", "pair", "task clusters", "d pass rate [95% CI]",
                           "agent GPU s/succ ratio [95% CI]"])
    if block.get("rr_nl"):
        out.append("")
        out += _table([[r["phase"], f"{r['arm']} vs {r['control']}",
                        f"{r['tasks']['pass']}/{r['tasks']['fail']}/{r['tasks']['mixed']}",
                        _f(r["rr"], ".0%"), _f(r["d_rr"], "+.0%"),
                        f"n<{MIN_NL_TASKS} ({r['tasks']['fail']})" if r["nl_suppressed"] else _f(r["nl"], ".0%"),
                        "-" if r["nl_suppressed"] else _f(r["d_nl"], "+.0%")] for r in block["rr_nl"]],
                      ["phase", "arm", "A-labelled tasks pass/fail/mixed", "RR", "dRR", "NL", "dNL"])
    if block.get("hold"):
        out.append("")
        out += _table([[h["panel"], h["arm"], h["transitions"], f"{h['held']}/{h['prev_pass']}", _f(h["hold"], ".0%"),
                        f"{h['converted']}/{h['prev_fail']}", _f(h["base_hold"], ".0%"),
                        _est_ci(h["bootstrap"]["stats"]["d_hold"], "+.0%") if h["bootstrap"] else "-"]
                       for h in block["hold"]],
                      ["panel", "arm", "transitions", "held", "hold", "converted", "A hold", "d hold [95% CI]"])
    if block.get("retrieval"):
        out.append("")
        out += _table([[r["phase"], r["arm"], f"{r['retrieving']}/{r['episodes']}", r["retrievals"], r["unique"],
                        _f(r["coverage"], ".0%"),
                        f"{_f(r['bank_coverage'], '.0%')} of {r['bank_units']}" if r["bank_units"] else "-",
                        _f(r["top1_concentration"], ".0%"),
                        f"{_f(r['lesson_reach'], '.0%')} ({r['lesson_known']})" if r["lesson_known"] else "-",
                        f"{_f(r['pass_with_lesson'], '.0%')}/{_f(r['pass_without_lesson'], '.0%')}",
                        _f(r["lesson_best_rank"], ".0f"),
                        _f(r["same_cause_reach"], ".0%")] for r in block["retrieval"]],
                      ["phase", "arm", "retrieving", "items", "unique", "uniq/items", "bank coverage",
                       "top-1 conc", "lesson reach (n)", "pass w/ / w/o lesson", "lesson rank", "same-cause"])
    return out


def _est_ci(s: dict | None, spec: str, suffix: str = "") -> str:
    if not s or s.get("est") is None:
        return "-" if not s or not s.get("note") else f"- ({s['note']})"
    txt = f"{format(s['est'], spec)}{suffix}"
    if s.get("ci95"):
        txt += f" [{format(s['ci95'][0], spec)}, {format(s['ci95'][1], spec)}]"
    if s.get("undefined"):
        txt += f" ({s['undefined']} undef)"
    return txt


def _format_transfer(tr: dict) -> list[str]:
    rows = [[f"FWT {s}", a, f"{_f(v['acc'], '.0%')} vs {v['control']} {_f(v['control_acc'], '.0%')} ({v['control_kind']})",
             _f(v["fwt"], "+.0%")]
            for s, by_arm in tr["fwt"].items() for a, v in by_arm.items() if v]
    rows += [[f"BWT {v['split']}@{c['checkpoint']}", a, f"{_f(c['acc'], '.0%')} (n={c['n']}) vs @{v['reference_checkpoint']}",
              _f(c["bwt"], "+.0%")]
             for a, v in tr["bwt"].items() if v for c in v["checkpoints"]]
    if not rows:
        return ["FWT/BWT: no sealed or retention episodes in this selection"]
    return _table(rows, ["metric", "arm", "accuracy", "value"])


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
    if "bootstrap" in summary:
        bs, rs = summary["bootstrap"], summary["retrieval_source"]
        lines.append(f"bootstrap: paired by task cluster (histories x repeats pooled), {bs['resamples']} resamples, "
                     f"seed {bs['seed']}; RR/NL: tasks labelled by arm A, NL withheld below {MIN_NL_TASKS} A-fail "
                     f"tasks; retrieval: ids from the episode row, texts from {rs['episodes_with_injected_texts']} "
                     f"traces, {rs['rich_records']} rich retrieval records")
        lines += _format_transfer(summary["transfer"])
    if summary.get("dev_episodes"):
        lines.append(f"{summary['dev_episodes']} dev episodes (arm D gate / GEPA, charged as learning) are left out "
                     "of the comparisons; see the arm D procedures section")
    lines += ["", "== pooled over histories =="] + _format_block(summary["pooled"])
    if len(summary["histories"]) > 1:
        for h, block in summary["per_history"].items():
            lines += ["", f"== history {h} =="] + _format_block(block)
    if summary.get("procedures"):
        lines += _format_procedures(summary["procedures"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m elh.analysis", description=__doc__.splitlines()[0])
    ap.add_argument("batch_prefix", nargs="?", help="experiment name (batch_id prefix)")
    ap.add_argument("--run-id", help="select by run_id prefix instead of batch")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    ap.add_argument("--no-memory-trace", action="store_true", help="don't query Hindsight's LLM request trace")
    ap.add_argument("--memory-url", help="Hindsight base URL (default: each run's registered memory_service)")
    ap.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES, help="bootstrap resamples (0: off)")
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED, help="bootstrap seed")
    ap.add_argument("--no-traces", action="store_true", help="don't read traces for injected memory texts")
    args = ap.parse_args(argv)
    if (args.batch_prefix is None) == (args.run_id is None):
        ap.error("give exactly one of batch_prefix or --run-id")
    store = Store(args.runs_dir)
    fetch = None if args.no_memory_trace else http_trace_fetcher()
    summary = summarize_batch(store, args.batch_prefix, run_id=args.run_id, fetch=fetch, memory_url=args.memory_url,
                              resamples=args.resamples, seed=args.seed, read_traces=not args.no_traces)
    print(json.dumps(summary, indent=2, default=str) if args.json else format_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
