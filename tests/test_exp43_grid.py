"""Stage 4.3 analysis (analysis/exp43_grid.py) on a small synthetic store: which batches feed which group
(L16 reuse of 4.1's A and P-hand, gate + grid pooling by history, the exp29b rerun name, `-rest*`
batches), replicates, the exclusion list and the degenerate-reply filter."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from elh.store import Store
from elh.tasks.base import get_family
from elh.tasks.families.house_lib import grid

_spec = importlib.util.spec_from_file_location("exp43_grid", Path(__file__).parent.parent / "analysis" / "exp43_grid.py")
x = importlib.util.module_from_spec(_spec)
sys.modules["exp43_grid"] = x
_spec.loader.exec_module(x)

RUN = "run1"
L16, L64 = "L16-c0-w1", "L64-c50-w1"
T0 = "2026-11-20T00:00:00+00:00"
COUNTER = [0]


def add(store, batch, history, group, split="checkpoint", task="t0", passes=(True,), replicate=None, cause="c",
        theme="inventory", seed=1, retrieval=None, trace=None, eid=None):
    """One episode; `passes` are its lesson checks (the world's first kept lessons, in order)."""
    kept = grid.kept_lessons(L64 if "L64" in batch else L16)
    checks = {"integrity": True, "public_tests": True, "feature": True,
              **{f"lesson:{kept[i]}": ok for i, ok in enumerate(passes)}}
    COUNTER[0] += 1
    eid = eid or f"ep{COUNTER[0]}"
    store.insert("episodes", {
        "episode_id": eid, "run_id": RUN, "history": history, "test_group": group, "split": split, "task_id": task,
        "cause": cause, "theme": theme, "seed": seed, "outcome": "submitted", "passed": int(all(checks.values())),
        "verdict_json": json.dumps({"passed": all(checks.values()), "checks": checks}), "turns": 10, "wall_s": 20.0,
        "input_tokens": 1000, "output_tokens": 100, "batch_id": f"{batch}/{history}", "started_at": T0,
        "retrieval_json": json.dumps(retrieval) if retrieval is not None else None,
        "trace_sha256": store.put_trace(trace) if trace else None, "replicate": replicate})
    return eid


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path)
    s.ensure_run(RUN, "r", 1, "h", "{}", None, "t")
    return s


def gather(store, world, **kw):
    kw.setdefault("excluded", set())
    return x.gather(store, world, **kw)


def test_sources_follow_grid_names():
    assert x.sources(L16) == [("exp29-grid-L16-c0-w1", ("B",)), ("exp30-grid-static-L16-c0-w1", x.STATIC),
                              ("exp22-grid-L16-c0-w1", ("A", "P-hand"))]
    names = [n for n, _ in x.sources(L64)]
    assert names == ["exp29b-gate-L64-c50-w1", "exp30b-gate-static-L64-c50-w1", "exp29-grid-L64-c50-w1",
                     "exp30-grid-static-L64-c50-w1"]  # the gate reran as exp29b / exp30b (grid.RERUN)
    assert x.sources("L64-c0-w1")[0] == ("exp29-gate-L64-c0-w1", ("A", "B", "P-hand"))
    assert x.sources("L4-c0-w1") == [("exp29-grid-L4-c0-w1", ("A", "B", "P-hand")),
                                     ("exp30-grid-static-L4-c0-w1", x.STATIC)]


def test_l16_takes_a_and_phand_from_41_and_the_rest_from_43(store):
    for h in (0, 1):
        for t in ("t0", "t1"):
            add(store, "exp22-grid-L16-c0-w1", h, "A", task=t, passes=(False,))
            add(store, "exp22-grid-L16-c0-w1", h, "P-hand", task=t, passes=(True,))
            add(store, "exp22-grid-L16-c0-w1", h, "B", task=t, passes=(False,))      # 4.1's B: another recall, not used
            add(store, "exp22-grid-L16-c0-w1", h, "P-all", task=t, passes=(False,))  # nor its static groups
            add(store, "exp29-grid-L16-c0-w1", h, "B", task=t, passes=(True,))
            add(store, "exp29-grid-L16-c0-w1", h, "A", task=t, passes=(True,))       # the L16 learn batch has no A: ignored
            add(store, "exp30-grid-static-L16-c0-w1", h, "P-all-ep", task=t, passes=(False,))
    d = gather(store, L16, check_degenerate=False)
    by_group = {}
    for e in d.eps:
        by_group.setdefault(e["test_group"], []).append((e["history"], e["task_id"]))
    assert sorted(by_group) == ["A", "B", "P-all-ep", "P-hand"] and all(len(v) == 4 for v in by_group.values())
    assert {e["source"] for e in d.eps if e["test_group"] == "A"} == {"exp22-grid-L16-c0-w1"}
    assert {e["source"] for e in d.eps if e["test_group"] == "B"} == {"exp29-grid-L16-c0-w1"}
    comp = {(c["batch"], c["group"], c["split"]): c for c in d.completeness}
    a = comp["exp22-grid-L16-c0-w1", "A", "checkpoint"]
    assert (a["scheduled"], a["recorded"]) == (72, 4)  # schedule() vs the store
    b = comp["exp29-grid-L16-c0-w1", "B", "train"]
    assert (b["scheduled"], b["recorded"]) == (96, 0)
    cells = x.first_runs(d.eps)
    r = x.pair_counts(cells, "B", "A", "lesson")
    assert (r["pairs"], r["first_only"], r["second_only"]) == (4, 4, 0) and r["p"] == pytest.approx(0.125)
    assert x.pair_counts(cells, "B", "P-hand", "task")["first_only"] == 0


def test_gate_and_grid_pool_by_history_with_rerun_name(store):
    for name, hs in (("exp29b-gate-L64-c50-w1", [0]), ("exp29-grid-L64-c50-w1", [1, 2])):
        for h in hs:
            for g in ("A", "B", "P-hand"):
                add(store, name, h, g)
    for name, hs in (("exp30b-gate-static-L64-c50-w1", [0]), ("exp30-grid-static-L64-c50-w1", [1, 2])):
        for h in hs:
            add(store, name, h, "P-all-ep")
    # the crashed first gate attempt stays in the database but is not part of the analysis
    add(store, "exp29-gate-L64-c50-w1", 0, "B", split="train")
    add(store, "exp29-gate-L64-c50-w1", 0, "B")
    d = gather(store, L64, check_degenerate=False)
    assert sorted({e["history"] for e in d.eps if e["test_group"] == "B"}) == [0, 1, 2]
    assert all(e["source"] != "exp29-gate-L64-c50-w1" for e in d.eps + d.train)
    assert len(d.eps) == 12 and d.train == []
    comp = {(c["batch"], c["group"], c["split"]): c for c in d.completeness}
    g = comp["exp29b-gate-L64-c50-w1", "B", "checkpoint"]
    assert (g["scheduled"], g["recorded"]) == (72, 1)  # B and its 2 replicates, one history
    assert comp["exp29-grid-L64-c50-w1", "B", "train"]["scheduled"] == 256  # histories 1-2
    assert {e["history"] for e in d.eps if e["test_group"] == "P-all-ep"} == {0, 1, 2}


def test_replicates_count_once_in_pairs_and_all_in_the_primary_test(store):
    for h in (0, 1, 2):
        for t in ("t0", "t1", "t2"):
            for r in range(3):
                add(store, "exp29-grid-L64-c50-w1", h, "B", task=t, replicate=r, passes=(r == 0, True))
                add(store, "exp30-grid-static-L64-c50-w1", h, "P-all-ep", task=t, replicate=r, passes=(False, r == 0))
    add(store, "exp29-grid-L64-c50-w1", 0, "A")  # a NULL replicate (older row) is replicate 0
    d = gather(store, L64, check_degenerate=False)
    assert len(d.eps) == 55
    assert {e["replicate"] for e in d.eps if e["test_group"] == "A"} == {0}
    cells = x.first_runs(d.eps)
    assert len(cells["B"]) == 9 and all(e["replicate"] == 0 for e in cells["B"].values())
    r = x.pair_counts(cells, "B", "P-all-ep", "lesson")
    assert r["pairs"] == 18 and (r["first_only"], r["second_only"]) == (9, 0)  # replicate 0 only (all runs would give 9 + 18 B-only checks)
    design, res = x.primary_test(d.eps)
    assert design.y_new.shape[:2] == (3, 3)  # 3 runs per cell, 3 histories
    assert len(x.lesson_eps(d.eps)) == 54


def test_exclusion_list_and_degenerate_filter(store, tmp_path):
    keep = add(store, "exp30-grid-static-L16-c0-w1", 0, "P-all", task="t0")
    gone = add(store, "exp30-grid-static-L16-c0-w1", 0, "P-all", task="t1")
    add(store, "exp30-grid-static-L16-c0-w1", 0, "P-all", task="t2",
        trace=[{"event": "response", "turn": 3, "stop_reason": "max_tokens",
                "content": [{"type": "text", "text": "?" * 8192}]}])
    excl = tmp_path / "excluded.json"
    excl.write_text(json.dumps({"episodes": [{"episode_id": gone}]}))
    assert x.excluded_ids(excl) == {gone} and x.excluded_ids(tmp_path / "none.json") == set()
    d = x.gather(store, L16, excluded=x.excluded_ids(excl))
    assert [e["episode_id"] for e in d.eps] == [keep]
    assert (d.excluded, d.degenerate) == (1, 1)
    c = next(c for c in d.completeness if c["group"] == "P-all")
    assert (c["recorded"], c["usable"]) == (3, 1)


def test_rest_batches_are_pooled(store):
    add(store, "exp29-grid-L16-c0-w1", 0, "B", task="t0")
    add(store, "exp29-grid-L16-c0-w1-rest", 0, "B", task="t1")
    add(store, "exp29-grid-L16-c0-w1-rest2-b", 1, "B", task="t2")
    add(store, "exp29-grid-L16-c50-w1", 0, "B", task="t3")  # another world
    d = gather(store, L16, check_degenerate=False)
    assert sorted(e["task_id"] for e in d.eps) == ["t0", "t1", "t2"]
    assert next(c for c in d.completeness if c["group"] == "B" and c["split"] == "checkpoint")["recorded"] == 3


def test_retrieval_stats_primary_rule_shown(store):
    fam = get_family(f"house_lib/{L16}")
    cause, theme = fam.causes[0], fam.themes[0]
    tr = add(store, "exp29-grid-L16-c0-w1", 0, "B", split="train", cause=cause, theme=theme, seed=1000)
    other = add(store, "exp29-grid-L16-c0-w1", 0, "B", split="train", cause=fam.causes[5], theme=theme, seed=1001)
    assert cause in fam.generate(cause, theme, 1000).signature["lessons"]
    assert cause not in fam.generate(fam.causes[5], theme, 1001).signature["lessons"]
    add(store, "exp29-grid-L16-c0-w1", 0, "B", task="t0", cause=cause,
        retrieval={"items": [{"document_id": f"{tr}:rule"}, {"document_id": "memo-1"}]})
    add(store, "exp29-grid-L16-c0-w1", 0, "B", task="t1", cause=cause,
        retrieval={"items": [{"document_id": f"{other}:rule"}]})
    d = gather(store, L16, check_degenerate=False)
    assert x.retrieval_stats(d, fam)["all"] == [2, 3, 2, 1]  # tickets, items, rules, primary-lesson rule shown


def test_report_prints_markdown_and_degrades_without_43_batches(store, capsys):
    for g in ("A", "P-hand"):
        add(store, "exp22-grid-L16-c0-w1", 0, g, task="t0", passes=(g == "P-hand",))
    x.report(store, L16, excluded=set(), check_degenerate=False)
    out = capsys.readouterr().out
    assert "Data completeness" in out and "| P-hand | 1 | 1 | 1 | 1/1 |" in out
    assert "No compared pair has both groups yet." in out and "needs both B and P-all-ep" in out
    x.report(store, "L4-c0-w1", excluded=set(), check_degenerate=False)
    assert "No stage 4.3 or reused 4.1 evaluation episodes recorded" in capsys.readouterr().out


def test_power_preview_prints_the_table(store, capsys):
    import random
    rng = random.Random(0)
    for h in (0, 1, 2):
        for t in range(6):
            for r in range(3):
                add(store, "exp29-grid-L64-c50-w1", h, "B", task=f"t{t}", replicate=r,
                    passes=(rng.random() < 0.6, rng.random() < 0.6))
                add(store, "exp30-grid-static-L64-c50-w1", h, "P-all-ep", task=f"t{t}", replicate=r,
                    passes=(rng.random() < 0.4, rng.random() < 0.4))
    d = gather(store, L64, check_degenerate=False)
    x.power_preview(d, sims=20, seed=0, baseline=None)
    out = capsys.readouterr().out
    assert "Primary test on the data: B - P-all-ep" in out and "3 run(s)" not in out.split("Primary")[0]
    assert "| Runs per cell | Eval tickets | 3 hist. | 4 hist. |" in out and "10 hist. |" in out
    assert "history SDs doubled" in out
