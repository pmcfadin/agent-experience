import json

import pytest

from elh.analysis import (attribute_memory, format_summary, main, mcnemar_exact, summarize_batch, wilson)
from elh.store import Store

RUN = "pilot-v1-abc"


def _served(prompt_ms, predicted_ms, prompt_n=100, predicted_n=10, cache_n=50):
    return json.dumps({"server": "llama.cpp", "timings": {"prompt_ms": prompt_ms, "predicted_ms": predicted_ms,
                                                          "prompt_n": prompt_n, "predicted_n": predicted_n,
                                                          "cache_n": cache_n}})


def _ep(store, eid, *, hist, arm, split, task, passed, checkpoint=None, ordinal=0, turns=10, wall=20.0,
        outcome="submitted", calls=((1000, 2000),), t0="2026-10-25T01:00:00+00:00",
        t1="2026-10-25T01:01:00+00:00", batch="pilot"):
    checks = {"integrity": True, "public_tests": True, "correctness": passed, "performance": passed}
    store.insert("episodes", {
        "episode_id": eid, "run_id": RUN, "history": hist, "arm": arm, "split": split, "checkpoint": checkpoint,
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
        # Arm B trains on two tasks; both arms then run the same eval panel at checkpoint 2.
        _ep(s, f"tr{h}a", hist=h, arm="B", split="train", task=f"t{h}x", passed=False, ordinal=0,
            t0="2026-10-25T01:00:00+00:00", t1="2026-10-25T01:01:00+00:00")
        _ep(s, f"tr{h}b", hist=h, arm="B", split="train", task=f"t{h}y", passed=True, ordinal=1,
            t0="2026-10-25T01:02:00+00:00", t1="2026-10-25T01:03:00+00:00")
        for task, a_pass, b_pass in (("e1", True, True), ("e2", False, True), ("e3", False, False)):
            _ep(s, f"ev{h}A{task}", hist=h, arm="A", split="checkpoint", checkpoint=2, task=task, passed=a_pass,
                turns=20, calls=((2000, 2000),))
            _ep(s, f"ev{h}B{task}", hist=h, arm="B", split="checkpoint", checkpoint=2, task=task, passed=b_pass,
                turns=12, calls=((1000, 1000),))
    # Another experiment sharing the prefix's first letters must not be picked up.
    _ep(s, "other", hist=0, arm="A", split="checkpoint", checkpoint=2, task="e1", passed=True, batch="pilot2")
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


def _group(block, phase, arm):
    return next(g for g in block["groups"] if g["phase"] == phase and g["arm"] == arm)


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
    assert [(g["phase"], g["arm"]) for g in pooled["groups"]] == [("train", "B"), ("eval@2", "A"), ("eval@2", "B")]
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
    _ep(store, "nt", hist=0, arm="A", split="checkpoint", checkpoint=5, task="e9", passed=True,
        calls=((1000, 1000), None))
    _ep(store, "nocalls", hist=0, arm="A", split="checkpoint", checkpoint=5, task="e8", passed=False, calls=())

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
