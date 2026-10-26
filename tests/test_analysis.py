import io
import json
import zipfile

import pytest

from elh.analysis import (attribute_memory, backward_transfer, cluster_bootstrap, failed_check_lesson, format_summary,
                          forward_transfer, hold_rates, load_episodes, main, mcnemar_exact, parse_injected_texts,
                          retention_new_learning, retrieval_diagnostics, snapshot_bank_sizes, summarize_batch,
                          transfer_summary, wilson)
from elh.memory.base import MemoryItem, render_memory_block
from elh.store import Store

RUN = "pilot-v1-abc"


def _served(prompt_ms, predicted_ms, prompt_n=100, predicted_n=10, cache_n=50):
    return json.dumps({"server": "llama.cpp", "timings": {"prompt_ms": prompt_ms, "predicted_ms": predicted_ms,
                                                          "prompt_n": prompt_n, "predicted_n": predicted_n,
                                                          "cache_n": cache_n}})


def _ep(store, eid, *, hist, group, split, task, passed, checkpoint=None, ordinal=0, turns=10, wall=20.0,
        outcome="submitted", calls=((1000, 2000),), t0="2026-10-25T01:00:00+00:00",
        t1="2026-10-25T01:01:00+00:00", batch="pilot", retrieved=None, snapshot=None, trace=None, cause=None,
        **extra):
    checks = {"integrity": True, "public_tests": True, "correctness": passed, "performance": passed}
    store.insert("episodes", {
        **extra, "retrieved_memory_ids_json": json.dumps(retrieved or []), "memory_snapshot_id": snapshot,
        "trace_sha256": store.put_trace(trace) if trace else None, "cause": cause, "episode_id": eid, "run_id": RUN, "history": hist, "test_group": group, "split": split, "checkpoint": checkpoint,
        "ordinal": ordinal, "task_id": task, "outcome": outcome, "passed": int(passed),
        "verdict_json": json.dumps({"passed": passed, "checks": checks}), "turns": turns, "wall_s": wall,
        "batch_id": f"{batch}/{hist}", "started_at": t0, "finished_at": t1})
    for i, c in enumerate(calls):
        store.insert("llm_calls", {"episode_id": eid, "turn": i,
                                   "served_json": _served(*c) if c is not None else json.dumps({})})


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path)
    s.ensure_run(RUN, "pilot", 1, "abc", json.dumps({"memory_service": {"base_url": "http://mem"}}), None, "t")
    for h in (0, 1):
        # Group B trains on two tasks; both groups then run the same eval panel at checkpoint 2.
        _ep(s, f"tr{h}a", hist=h, group="B", split="train", task=f"t{h}x", passed=False, ordinal=0,
            t0="2026-10-25T01:00:00+00:00", t1="2026-10-25T01:01:00+00:00")
        _ep(s, f"tr{h}b", hist=h, group="B", split="train", task=f"t{h}y", passed=True, ordinal=1,
            t0="2026-10-25T01:02:00+00:00", t1="2026-10-25T01:03:00+00:00")
        for task, a_pass, b_pass in (("e1", True, True), ("e2", False, True), ("e3", False, False)):
            _ep(s, f"ev{h}A{task}", hist=h, group="A", split="checkpoint", checkpoint=2, task=task, passed=a_pass,
                turns=20, calls=((2000, 2000),))
            _ep(s, f"ev{h}B{task}", hist=h, group="B", split="checkpoint", checkpoint=2, task=task, passed=b_pass,
                turns=12, calls=((1000, 1000),))
    # Another experiment sharing the prefix's first letters must not be picked up.
    _ep(s, "other", hist=0, group="A", split="checkpoint", checkpoint=2, task="e1", passed=True, batch="pilot2")
    return s


def fake_trace(calls_seen=None):
    """Bank trace: retain (linked by document_id), consolidation (linked via source memory ids),
    an embedding (ignored) and an unlinked LLM call (time-window estimate)."""
    def fetch(url, bank):
        if calls_seen is not None:
            calls_seen.append((url, bank))
        if not bank.endswith("-B"):
            return None
        h = bank.split("-h")[-1].split("-")[0]
        return [
            {"operation": "retain", "scope": "retain_extract_facts", "status": "success", "duration_ms": 10000,
             "started_at": "2026-10-25T01:01:01+00:00", "input_tokens": 100, "output_tokens": 50,
             "metadata": {"document_id": f"tr{h}a", "memory_ids": ["m1", "m2"]}},
            {"operation": "retain", "scope": "embedding_document", "status": "success", "duration_ms": 30,
             "started_at": "2026-10-25T01:01:11+00:00", "metadata": {"document_id": f"tr{h}a", "memory_ids": ["m1"]}},
            {"operation": "consolidation", "scope": "consolidation", "status": "success", "duration_ms": 5000,
             "started_at": "2026-10-25T01:01:12+00:00", "metadata": {"source_memory_ids": ["m1", "m2"]}},
            {"operation": "retain", "scope": "retain_extract_facts", "status": "success", "duration_ms": 12000,
             "started_at": "2026-10-25T01:03:01+00:00", "metadata": {"document_id": f"tr{h}b", "memory_ids": ["m3"]}},
            {"operation": "consolidation", "scope": "consolidation", "status": "success", "duration_ms": 3000,
             "started_at": "2026-10-25T01:03:20+00:00", "metadata": {}},
        ]
    return fetch


def _group(block, phase, group):
    return next(g for g in block["cells"] if g["phase"] == phase and g["group"] == group)


def test_wilson_and_mcnemar():
    lo, hi = wilson(1, 2)
    assert lo == pytest.approx(0.0945, abs=1e-3) and hi == pytest.approx(0.9055, abs=1e-3)
    assert wilson(0, 0) is None
    lo, hi = wilson(10, 10)
    assert hi == pytest.approx(1.0) and lo == pytest.approx(0.7225, abs=1e-3)
    assert mcnemar_exact(0, 0) is None
    assert mcnemar_exact(0, 2) == pytest.approx(0.5)
    assert mcnemar_exact(3, 3) == 1.0


def test_grouping_and_gpu_seconds(store):
    s = summarize_batch(store, "pilot", fetch=None)
    assert s["episodes"] == 16 and s["histories"] == ["0", "1"]
    pooled = s["pooled"]
    assert [(g["phase"], g["group"]) for g in pooled["cells"]] == [("train", "B"), ("eval@2", "A"), ("eval@2", "B")]
    train = _group(pooled, "train", "B")
    assert train["n"] == 4 and train["passes"] == 2
    assert train["agent_gpu_s"]["total"] == pytest.approx(12.0)  # 4 episodes × 3 s
    assert train["tokens"] == {"prompt_n": 400, "predicted_n": 40, "cache_n": 200}
    assert train["memory_s"]["total"] is None  # trace off: learning cost unknown
    assert train["gpu_s_per_success"]["with_memory"] is None
    assert train["gpu_s_per_success"]["agent_only"] == pytest.approx(6.0)
    a = _group(pooled, "eval@2", "A")
    assert a["n"] == 6 and a["passes"] == 2 and a["check_failures"]["performance"] == 4
    assert a["outcomes"]["submitted"] == 6
    assert a["agent_gpu_s"]["mean"] == pytest.approx(4.0) and a["turns"]["median"] == 20
    assert a["memory_s"]["total"] == 0.0 and a["gpu_s_per_success"]["with_memory"] == pytest.approx(12.0)
    assert a["ci95"] == pytest.approx(list(wilson(2, 6)))
    b = _group(s["per_history"]["1"], "eval@2", "B")
    assert b["n"] == 3 and b["passes"] == 2


def test_paired_table(store):
    s = summarize_batch(store, "pilot")
    (p,) = s["pooled"]["paired"]
    assert (p["phase"], p["base"], p["other"]) == ("eval@2", "A", "B")
    assert (p["pairs"], p["both"], p["base_only"], p["other_only"], p["neither"]) == (6, 2, 0, 2, 2)
    assert p["mean_diff"]["turns"] == pytest.approx(-8)
    assert p["mean_diff"]["agent_gpu_s"] == pytest.approx(-2.0)
    assert p["mcnemar_p"] == pytest.approx(0.5)
    (ph,) = s["per_history"]["0"]["paired"]
    assert (ph["both"], ph["other_only"], ph["neither"]) == (1, 1, 1)


def test_memory_trace_attribution(store):
    seen = []
    s = summarize_batch(store, "pilot", fetch=fake_trace(seen))
    assert sorted(b for _, b in seen) == [f"elh-{RUN}-h0-B", f"elh-{RUN}-h1-B"]
    assert {u for u, _ in seen} == {"http://mem"}
    rows = {r["episode_id"]: r for r in s["episode_rows"]}
    assert rows["tr0a"]["memory_s"] == pytest.approx(15.0) and rows["tr0a"]["memory_method"] == "linked"
    assert rows["tr0b"]["memory_s"] == pytest.approx(15.0) and rows["tr0b"]["memory_method"] == "window"
    assert rows["ev0Be1"]["memory_s"] == 0.0
    train = _group(s["pooled"], "train", "B")
    assert train["memory_s"]["total"] == pytest.approx(60.0) and train["memory_s"]["estimated"]
    assert train["gpu_s_per_success"]["with_memory"] == pytest.approx((12 + 60) / 2)
    assert _group(s["pooled"], "eval@2", "B")["learning_s_to_date"] == pytest.approx(60.0)
    assert _group(s["pooled"], "eval@2", "A")["learning_s_to_date"] == 0.0
    assert s["memory_trace"]["estimated_calls"] == 2
    text = format_summary(s)
    assert "estimate" in text and "eval@2" in text and "history 1" in text


def test_consolidation_split_across_episodes():
    stream = [{"episode_id": "a", "finished_at": "2026-10-25T01:00:00+00:00"},
              {"episode_id": "b", "finished_at": "2026-10-25T01:05:00+00:00"}]
    entries = [
        {"operation": "retain", "scope": "retain_extract_facts", "duration_ms": 0, "status": "success",
         "metadata": {"document_id": "a", "memory_ids": ["x"]}},
        {"operation": "retain", "scope": "retain_extract_facts", "duration_ms": 0, "status": "success",
         "metadata": {"document_id": "b", "memory_ids": ["y", "z"]}},
        {"operation": "consolidation", "scope": "consolidation", "duration_ms": 9000, "status": "success",
         "started_at": "2026-10-25T01:06:00+00:00", "metadata": {"source_memory_ids": ["x", "y", "z", "gone"]}},
        {"operation": "consolidation", "scope": "consolidation", "duration_ms": 4000, "status": "success",
         "started_at": "2026-10-25T00:00:00+00:00", "metadata": {}},
    ]
    att = attribute_memory(entries, stream)
    assert att["a"]["s"] == pytest.approx(3.0) and att["b"]["s"] == pytest.approx(6.0)
    assert att[None]["s"] == pytest.approx(4.0)  # before any episode: unattributed


def test_missing_timings_and_unreachable_trace(store):
    _ep(store, "nt", hist=0, group="A", split="checkpoint", checkpoint=5, task="e9", passed=True,
        calls=((1000, 1000), None))
    _ep(store, "nocalls", hist=0, group="A", split="checkpoint", checkpoint=5, task="e8", passed=False, calls=())

    def down(url, bank):
        raise OSError("connection refused")

    s = summarize_batch(store, "pilot", fetch=down)
    g = _group(s["pooled"], "eval@5", "A")
    assert g["agent_gpu_s"]["n"] == 0 and g["agent_gpu_s"]["total"] is None
    assert g["gpu_s_per_success"]["agent_only"] is None
    rows = {r["episode_id"]: r for r in s["episode_rows"]}
    assert rows["nt"]["agent_gpu_s"] is None and rows["nt"]["calls_without_timings"] == 1
    assert all(v.startswith("unavailable") for v in s["memory_trace"]["banks"].values())
    assert _group(s["pooled"], "train", "B")["memory_s"]["total"] is None
    format_summary(s)


def test_run_id_selector_and_cli(store, tmp_path, capsys):
    s = summarize_batch(store, run_id="pilot-v1")
    assert s["episodes"] == 17 and s["histories"] == [f"{RUN}/h0", f"{RUN}/h1"]
    assert summarize_batch(store, "pilot/")["episodes"] == 16
    assert main(["pilot", "--runs-dir", str(tmp_path), "--no-memory-trace", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["episodes"] == 16
    assert main(["pilot", "--runs-dir", str(tmp_path), "--no-memory-trace"]) == 0
    assert "pooled over histories" in capsys.readouterr().out


# -- continual-learning metrics -----------------------------------------------------------------
def test_cluster_bootstrap_is_seeded_and_paired():
    units = [{"b_n": 1, "b_pass": b, "o_n": 1, "o_pass": o} for b, o in ((1, 1), (0, 1), (0, 0), (1, 1), (0, 1))]
    stat = {"d": lambda t: t["o_pass"] / t["o_n"] - t["b_pass"] / t["b_n"]}
    r1 = cluster_bootstrap(units, stat, 2000, seed=7)
    assert r1 == cluster_bootstrap(units, stat, 2000, seed=7)
    d = r1["stats"]["d"]
    assert d["est"] == pytest.approx(0.4)
    # Paired resampling: every cluster has o >= b, so no replicate can go negative.
    assert 0.0 <= d["ci95"][0] <= 0.4 <= d["ci95"][1] <= 1.0 and d["undefined"] == 0
    undef = cluster_bootstrap([{"x": 1, "y": 0}, {"x": 1, "y": 1}],
                              {"r": lambda t: t["x"] / t["y"] if t["y"] else None}, 500, seed=1)["stats"]["r"]
    assert undef["undefined"] > 0 and undef["est"] == 2.0
    assert cluster_bootstrap([], stat)["stats"]["d"]["est"] is None


def test_paired_bootstrap_and_steps(store):
    s = summarize_batch(store, "pilot", resamples=500)
    (p,) = s["pooled"]["paired"]
    b = p["bootstrap"]
    assert b["clusters"] == 3 and b["resamples"] == 500   # tasks e1..e3, each over both histories
    assert b["stats"]["pass_diff"]["est"] == pytest.approx(4 / 6 - 2 / 6)
    lo, hi = b["stats"]["pass_diff"]["ci95"]
    assert lo <= b["stats"]["pass_diff"]["est"] <= hi
    # GPU s/success: B 2 s × 6 / 4 passes = 3; A 4 s × 6 / 2 = 12.
    assert b["stats"]["gpu_per_success_ratio"]["est"] == pytest.approx(3 / 12)
    assert p["mean_diff"]["turns_both_pass"] == pytest.approx(-8)
    assert s["bootstrap"] == {"resamples": 500, "seed": 20261025, "cluster": "task"}


def test_gpu_ratio_withheld_without_timings(store):
    _ep(store, "nt", hist=0, group="A", split="checkpoint", checkpoint=2, task="e4", passed=True, calls=(None,))
    _ep(store, "nt2", hist=0, group="B", split="checkpoint", checkpoint=2, task="e4", passed=True)
    (p,) = summarize_batch(store, "pilot", resamples=100)["pooled"]["paired"]
    g = p["bootstrap"]["stats"]["gpu_per_success_ratio"]
    assert g["est"] is None and "timings" in g["note"]


def test_retention_new_learning(store):
    s = summarize_batch(store, "pilot", resamples=0)
    (r,) = s["pooled"]["rr_nl"]
    # Per history A passes e1 and fails e2, e3: pooled 2 A-pass and 4 A-fail tasks.
    assert r["tasks"] == {"pass": 2, "fail": 4, "mixed": 0}
    assert r["rr"] == 1.0 and r["d_rr"] == 0.0 and r["rr_n"] == 2
    assert r["nl_suppressed"] and r["nl"] is None and r["d_nl"] is None and r["nl_n"] == 4
    eps = load_episodes(store, "pilot")
    (r,) = retention_new_learning(eps, "A", min_nl_tasks=4)
    assert r["nl"] == pytest.approx(0.5) and r["d_nl"] == pytest.approx(0.5)
    # Harmful reuse: B breaks an A-pass task.
    _ep(store, "x1", hist=0, group="A", split="checkpoint", checkpoint=2, task="e5", passed=True)
    _ep(store, "x2", hist=0, group="B", split="checkpoint", checkpoint=2, task="e5", passed=False)
    (r,) = retention_new_learning(load_episodes(store, "pilot"), "A")
    assert r["rr"] == pytest.approx(2 / 3) and r["d_rr"] == pytest.approx(-1 / 3)
    text = format_summary(summarize_batch(store, "pilot", resamples=0))
    assert "dRR" in text and "n<10 (4)" in text


def test_retention_labels_mixed_repeats():
    eps = [dict(phase="eval@1", group=a, history_label="0", task_id="t", passed=p, checkpoint=1)
           for a, p in (("A", True), ("A", False), ("B", True))]
    (r,) = retention_new_learning(eps, "A")
    assert r["tasks"] == {"pass": 0, "fail": 0, "mixed": 1} and r["rr"] is None


def _ck(group, ck, task, passed, hist="0", split="checkpoint"):
    return dict(phase=f"{'eval' if split == 'checkpoint' else split}@{ck}", split=split, group=group, checkpoint=ck,
                history_label=hist, task_id=task, passed=passed)


def test_hold_rate():
    eps = []
    # Group A re-draws: t1 P,F,P  t2 P,P,F ; group B holds: t1 P,P,P  t2 F,P,P
    for task, a, b in (("t1", (1, 0, 1), (1, 1, 1)), ("t2", (1, 1, 0), (0, 1, 1))):
        for i, ck in enumerate((5, 10, 15)):
            eps += [_ck("A", ck, task, bool(a[i])), _ck("B", ck, task, bool(b[i]))]
    rows = {h["group"]: h for h in hold_rates(eps, "A", resamples=300)}
    a, b = rows["A"], rows["B"]
    assert (a["prev_pass"], a["held"]) == (3, 1) and a["hold"] == pytest.approx(1 / 3)
    assert (b["prev_pass"], b["held"], b["prev_fail"], b["converted"]) == (3, 3, 1, 1)
    assert b["hold"] == 1.0 and b["base_hold"] == pytest.approx(1 / 3) and b["d_hold"] == pytest.approx(2 / 3)
    assert b["bootstrap"]["stats"]["d_hold"]["est"] == pytest.approx(2 / 3)
    assert a["bootstrap"] is None
    # A single checkpoint has no transitions.
    assert hold_rates([_ck("A", 5, "t1", True), _ck("B", 5, "t1", True)], "A") == []


def test_fwt_bwt_absent_and_present(store):
    eps = load_episodes(store, "pilot")
    tr = transfer_summary(eps)
    assert all(v is None for by in tr["fwt"].values() for v in by.values())
    assert tr["bwt"] == {"B": None}
    assert "FWT/BWT: no sealed or retention" in format_summary(summarize_batch(store, "pilot", resamples=0))
    syn = [dict(split="sealed_novel_cause", group=a, passed=p, checkpoint=None)
           for a, p in (("A", False), ("A", True), ("B", True), ("B", True), ("Bs", False), ("Bs", True))]
    f = forward_transfer(syn, "sealed_novel_cause", "B")
    assert f["fwt"] == pytest.approx(0.5) and f["control_kind"] == "no-memory"
    f = forward_transfer(syn, "sealed_novel_cause", "B", control="Bs")
    assert f["fwt"] == pytest.approx(0.5) and f["control_kind"] == "scratch"
    assert forward_transfer(syn, "sealed_familiar", "B") is None
    assert transfer_summary(syn, {"B": "Bs"})["fwt"]["sealed_novel_cause"]["B"]["control"] == "Bs"
    ret = [_ck("B", ck, t, p, split="retention") for ck, t, p in
           ((5, "r1", True), (5, "r2", True), (10, "r1", True), (10, "r2", False))]
    bw = backward_transfer(ret, "B")
    assert bw["reference_checkpoint"] == 5 and [c["bwt"] for c in bw["checkpoints"]] == [0.0, -0.5]
    assert backward_transfer(ret, "C") is None


def test_failed_check_predicate():
    assert failed_check_lesson({"tags": ["outcome:failed"]}, None) is True
    assert failed_check_lesson({"id": "x"}, {"passed": False}) is True
    assert failed_check_lesson({"id": "x"}, {"passed": True}) is False
    assert failed_check_lesson({"text": "The attempt failed because it modified protected files."}, None) is True
    assert failed_check_lesson({"text": "Replaced list.pop(0) with deque.popleft()."}, None) is False
    assert failed_check_lesson({"id": "x"}, None) is None


def test_parse_injected_texts_matches_renderer():
    items = [MemoryItem("m1", "first fact\nsecond line"), MemoryItem("m2", "FAILED: protected file changed")]
    msg = "<experience>\nNotes retrieved...\n\n" + render_memory_block(items) + "\n</experience>\n\n<ticket>\n[m9]\nx"
    assert parse_injected_texts(msg, ["m1", "m2", "m9"]) == {"m1": "first fact\nsecond line",
                                                              "m2": "FAILED: protected file changed"}


def _start(ids_texts):
    block = render_memory_block([MemoryItem(i, t) for i, t in ids_texts])
    return [{"event": "episode_start", "first_message": f"<experience>\nNotes\n\n{block}\n</experience>\n\nticket"}]


def test_retrieval_diagnostics_from_store(tmp_path):
    s = Store(tmp_path)
    s.ensure_run(RUN, "pilot", 1, "abc", "{}", None, "t")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("manifest.json", json.dumps({"fact_count": 6, "observation_count": 2}))
    s.insert("memory_snapshots", {"snapshot_id": "snap", "run_id": RUN, "history": 0, "test_group": "B", "after_episode": 2,
                                  "source_bank_id": "b", "bank_id": "b-e2",
                                  "bank_export_sha256": s.put_blob("snapshots", buf.getvalue()),
                                  "content_sha256": "c", "created_at": "t"})
    texts = {"m1": "Fixed by using a deque.", "m2": "The fix failed: it changed protected files.", "m3": "Ticket restated."}
    panel = [("e1", ["m1", "m3"], True), ("e2", ["m1", "m2"], False), ("e3", ["m1", "m3"], True), ("e4", [], False)]
    for task, ids, passed in panel:
        _ep(s, f"evB{task}", hist=0, group="B", split="checkpoint", checkpoint=2, task=task, passed=passed,
            retrieved=ids, snapshot="snap", trace=_start([(i, texts[i]) for i in ids]) if ids else None)
        _ep(s, f"evA{task}", hist=0, group="A", split="checkpoint", checkpoint=2, task=task, passed=True)
    sm = summarize_batch(s, "pilot", resamples=0)
    assert sm["retrieval_source"]["bank_sizes"] == {"snap": 8}
    assert sm["retrieval_source"]["episodes_with_injected_texts"] == 3
    (r,) = sm["pooled"]["retrieval"]   # group A retrieves nothing and gets no row
    assert (r["group"], r["episodes"], r["retrieving"], r["retrievals"], r["unique"]) == ("B", 4, 3, 6, 3)
    assert r["coverage"] == pytest.approx(0.5) and r["bank_coverage"] == pytest.approx(3 / 8) and r["bank_units"] == 8
    assert r["top1_concentration"] == 1.0
    assert r["lesson_reach"] == pytest.approx(1 / 4) and r["lesson_known"] == 4
    assert r["pass_with_lesson"] == 0.0 and r["pass_without_lesson"] == pytest.approx(2 / 3)
    assert r["lesson_best_rank"] == 1 and r["same_cause_reach"] is None
    assert snapshot_bank_sizes(s, ["missing"]) == {"missing": None}
    # Without traces nothing can be said about lessons, but counts still work.
    (r,) = summarize_batch(s, "pilot", resamples=0, read_traces=False)["pooled"]["retrieval"]
    assert r["lesson_reach"] == 0.0 and r["lesson_unknown"] == 3 and r["lesson_known"] == 1
    text = format_summary(sm)
    assert "top-1 conc" in text and "38% of 8" in text and "25% (4)" in text


def test_rich_retrieval_records(tmp_path):
    """An optional `retrieval_json` column (candidates with rank, score, source document) is used
    when present: deeper candidates give the lesson's rank, document ids give same-cause reach."""
    s = Store(tmp_path)
    s._add_columns("episodes", {"retrieval_json": "TEXT"})
    s.ensure_run(RUN, "pilot", 1, "abc", "{}", None, "t")
    _ep(s, "tr1", hist=0, group="B", split="train", task="tq", passed=False, cause="quadratic")
    _ep(s, "tr2", hist=0, group="B", split="train", task="tq2", passed=True, cause="queue")
    cands = [{"id": "m1", "rank": 0, "score": 0.9, "document_id": "tr2"},
             {"id": "m2", "rank": 1, "score": 0.8, "document_id": "tr2"},
             {"id": "m3", "rank": 7, "score": 0.1, "document_id": "tr1", "injected": False}]
    _ep(s, "ev1", hist=0, group="B", split="checkpoint", checkpoint=2, task="e1", passed=True, cause="quadratic",
        retrieved=["m1", "m2"], retrieval_json=json.dumps({"query": "q", "candidates": cands}))
    eps = [e for e in load_episodes(s, "pilot") if e["phase"] != "train"]
    assert eps[0]["retrieval"]["query"] == "q"
    sources = {"tr1": {"passed": False, "cause": "quadratic"}, "tr2": {"passed": True, "cause": "queue"}}
    (r,) = retrieval_diagnostics(eps, sources=sources)
    assert r["retrievals"] == 2 and r["lesson_reach"] == 0.0 and r["lesson_best_rank"] == 7
    assert r["same_cause_reach"] == 0.0 and r["same_cause_known"] == 1
    # A pluggable predicate: "lesson" = any item from a same-cause source.
    (r,) = retrieval_diagnostics(eps, sources=sources, predicate=lambda it, src: bool(src) and src["cause"] == "queue")
    assert r["lesson_reach"] == 1.0
