"""Group D workflow memory (AWM online + dev gate): inputs, prompt, library, versioning, gate, guards."""

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

from elh.learner import workflows as wf
from elh.learner.workflows import (DevResult, Induction, LeakageError, WorkflowConfig, WorkflowLibrary, apply_rollback,
                                   assemble_library, build_prompt, decide_candidate, eligible_pool, example_from_trace,
                                   experience_block, induce, learner_state_sha256, learning_entries, load_library,
                                   procedure_ids, promote, promoted_library, record_candidate, render_procedure_block,
                                   revision_status, rollback, sample_per_cell, should_induce)
from elh.registry import Budgets, load_config
from elh.runner.agent import run_episode
from elh.runner.models import ModelResponse
from elh.runner.scripts import HEREDOC_TAG
from elh.sandbox.local import LocalSandbox
from elh.store import Store
from elh.tasks.base import get_family

CONFIG = Path(__file__).parent.parent / "experiments" / "phase0-local.yaml"
RUN = "run-x"

LIBRARY_OUTPUT = """\
## Find the slow loop in {module_path}
This workflow is used to locate the code that makes the job slow on large inputs.
I first read the job's core module.
cat {module_path}
Then I time the job on a generated input of production size.
python scripts/make_load.py {large_n} /tmp/big.json && time timeout 100 python run.py /tmp/big.json /tmp/out.json

## Verify the fix
This workflow is used to check behaviour is unchanged after a change.
Run the public tests.
python -m unittest discover -s tests -q
"""


class _Agent:
    """Scripted agent with visible reasoning text before each tool call (like the real model)."""

    model_id = "scripted"

    def __init__(self, turns):
        self.turns, self.i = turns, 0

    def create(self, **_):
        if self.i >= len(self.turns):
            return ModelResponse([{"type": "text", "text": "Done."}], "end_turn", {}, "scripted")
        think, calls = self.turns[self.i]
        content = [{"type": "text", "text": think}] + [
            {"type": "tool_use", "id": f"t{self.i}_{j}", "name": n, "input": inp} for j, (n, inp) in enumerate(calls)]
        self.i += 1
        return ModelResponse(content, "tool_use", {"input_tokens": 10, "output_tokens": 5}, "scripted", f"r{self.i}")


def _reference_turns(task, family):
    core = family.solution_core(task)
    return [
        ("I'll look at the workspace first.", [("bash", {"command": "ls -R && cat job/core.py"})]),
        ("Restart for no reason.", [("bash", {"restart": True})]),
        ("Echo something.", [("bash", {"command": "echo hello"})]),
        ("Bad call.", [("frobnicate", {"x": 1})]),
        ("Write the fix.", [("bash", {"command": f"cat > job/core.py <<'{HEREDOC_TAG}'\n{core}{HEREDOC_TAG}"})]),
        ("Run the tests.", [("bash", {"command": "python -m unittest discover -s tests -q"})]),
        ("Run the tests again.", [("bash", {"command": "python -m unittest discover -s tests -q"})]),
        ("Submit.", [("submit", {"summary": "fixed"})]),
    ]


def _add_episode(store, task, *, split="train", passed=True, outcome="submitted", integrity=True,
                 group="D", history=0, ordinal=0, family=None, memory_block=""):
    family = family or get_family(task.family)
    sbx = LocalSandbox()
    try:
        sbx.start(task.workspace)
        res = run_episode(task, sbx, _Agent(_reference_turns(task, family)), Budgets(max_turns=20), 1000,
                          memory_block=memory_block)
    finally:
        sbx.close()
    verdict = {"passed": passed, "checks": {"integrity": integrity, "correctness": passed, "performance": passed},
               "detail": {"quality": {"reference_s": 0.123456, "agent_s": 0.2, "budget_s": 1.0}}}
    trace = store.put_trace(res.trace + [{"event": "verdict", **verdict}])
    eid = uuid.uuid4().hex
    store.insert("episodes", {
        "episode_id": eid, "run_id": RUN, "history": history, "test_group": group, "split": split, "ordinal": ordinal,
        "task_id": task.task_id, "family": task.family, "cause": task.cause, "theme": task.theme, "seed": task.seed,
        "task_hash": task.content_hash, "outcome": outcome, "passed": int(passed), "verdict_json": json.dumps(verdict),
        "turns": res.turns, "input_tokens": 1000, "output_tokens": 100, "cache_read_tokens": 0,
        "cache_write_tokens": 0, "trace_sha256": trace, "started_at": f"2026-10-25T00:00:{ordinal:02d}"})
    return eid


@dataclass
class _Echo:
    """Fake induction model: returns a fixed text, or (adversarially) the prompt itself."""

    text: str = LIBRARY_OUTPUT
    echo: bool = False
    model_id: str = "study-qwen3-coder-30b-a3b"

    def __post_init__(self):
        self.prompts, self.kwargs = [], []

    def complete(self, prompt, *, max_tokens, temperature):
        self.prompts.append(prompt)
        self.kwargs.append({"max_tokens": max_tokens, "temperature": temperature})
        out = prompt.split("## Summary Workflows")[-1] + "\n\n" + prompt if self.echo else self.text
        return wf.Completion(out, {"input_tokens": 3000, "output_tokens": 400, "cache_read_input_tokens": 0,
                                   "cache_creation_input_tokens": 0}, self.model_id, "req-1")


@pytest.fixture(scope="module")
def family():
    return get_family("cpu_mem")


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "runs")


@pytest.fixture
def pool(store, family):
    """Three train successes in two (cause, theme) cells, plus episodes induction must never see."""
    ids = {}
    ids["a1"] = _add_episode(store, family.generate("queue_pop_front", "billing", 1000), ordinal=0)
    ids["a2"] = _add_episode(store, family.generate("queue_pop_front", "billing", 1001), ordinal=1)
    ids["b1"] = _add_episode(store, family.generate("quadratic_membership", "telemetry", 1002), ordinal=2,
                             memory_block="[m1]\nrecalled note text XYZZY")
    ids["fail"] = _add_episode(store, family.generate("queue_pop_front", "logistics", 1003), ordinal=3, passed=False)
    ids["budget"] = _add_episode(store, family.generate("queue_pop_front", "logistics", 1004), ordinal=4,
                                 outcome="budget_exhausted")
    for i, split in enumerate(["dev", "checkpoint", "sealed_familiar", "sealed_novel_combo", "sealed_novel_cause",
                               "retention"]):
        ids[split] = _add_episode(store, family.generate("nested_scan_join", "inventory", 2000 + i), split=split,
                                  ordinal=5 + i)
    ids["other_group"] = _add_episode(store, family.generate("repeated_sort_min", "billing", 1009), group="C", ordinal=0)
    return ids


# -- inputs ------------------------------------------------------------------------------------


def test_pool_holds_only_verified_train_successes_of_the_stream(store, pool):
    rows = eligible_pool(store, RUN, 0, "D", "cpu_mem")
    assert [r["episode_id"] for r in rows] == [pool["a1"], pool["a2"], pool["b1"]]
    assert [r["episode_id"] for r in eligible_pool(store, RUN, 0, "D", "cpu_mem", max_ordinal=1)] == \
        [pool["a1"], pool["a2"]]


@pytest.mark.parametrize("key", ["dev", "checkpoint", "sealed_familiar", "sealed_novel_combo", "sealed_novel_cause",
                                 "retention", "fail", "budget"])
def test_induction_refuses_any_episode_outside_the_verified_train_pool(store, pool, key):
    client = _Echo()
    with pytest.raises(LeakageError):
        induce(store, [pool["a1"], pool[key]], client)
    assert client.prompts == []  # nothing was sent


def test_integrity_failure_is_not_inducible(store, family):
    eid = _add_episode(store, family.generate("queue_pop_front", "billing", 1000), integrity=False)
    with pytest.raises(LeakageError, match="integrity"):
        induce(store, [eid], _Echo())


def test_held_out_episode_text_never_reaches_the_prompt(store, pool):
    client = _Echo()
    induce(store, eligible_pool(store, RUN, 0, "D", "cpu_mem"), client)
    prompt = client.prompts[0]
    for key in ("dev", "checkpoint", "sealed_familiar", "sealed_novel_combo", "sealed_novel_cause", "retention",
                "fail", "budget"):
        task_id = store.episode(pool[key])["task_id"]
        ticket_head = get_family("cpu_mem").generate(*task_id.split("/")[1:3], int(task_id.split("/")[3])).prompt
        assert ticket_head.strip() not in prompt


def test_one_sample_per_cell_is_seeded(store, pool):
    rows = eligible_pool(store, RUN, 0, "D", "cpu_mem")
    picks = {tuple(r["episode_id"] for r in sample_per_cell(rows, s)) for s in range(20)}
    assert all(len(p) == 2 for p in picks)  # two cells
    assert len(picks) == 2  # the billing cell's two episodes both get chosen across seeds
    assert sample_per_cell(rows, 7) == sample_per_cell(rows, 7)


def test_example_is_think_action_observation_without_invalid_steps(store, pool, family):
    row = store.episode(pool["b1"])
    events = wf._load_trace(store, row)
    ex = example_from_trace(row["episode_id"], (row["cause"], row["theme"]), events)
    task = family.generate(row["cause"], row["theme"], row["seed"])
    assert ex.query == task.prompt.strip()  # the ticket only, never the <experience> block
    thinks = [s.think for s in ex.steps]
    assert thinks == ["I'll look at the workspace first.", "Write the fix.", "Run the tests."]
    assert ex.steps[0].actions == ("ls -R && cat job/core.py",) and "README.md" in ex.steps[0].observation
    assert "lines omitted" in ex.steps[0].observation
    assert "[exit code" not in ex.steps[2].observation
    text = wf.format_example(ex)
    assert "XYZZY" not in text and "Echo something" not in text and "frobnicate" not in text
    assert "Restart" not in text and "Submit" not in text and "Run the tests again" not in text


def test_prompt_is_awm_instruction_one_shot_then_examples(store, pool):
    client = _Echo()
    ind = induce(store, eligible_pool(store, RUN, 0, "D", "cpu_mem"), client, seed=3)
    prompt = client.prompts[0]
    assert prompt.startswith(wf.INDUCTION_INSTRUCTION + "\n\n" + wf.ONE_SHOT + "\n\n## Concrete Examples\n\nQuery: ")
    assert prompt.endswith("## Summary Workflows")
    assert prompt.count("Query: ") == 2 + 2  # two one-shot queries, one per cell
    assert client.kwargs == [{"max_tokens": 2048, "temperature": 0.0}]
    assert ind.source_episode_ids == [r["episode_id"] for r in
                                      sample_per_cell(eligible_pool(store, RUN, 0, "D", "cpu_mem"), 3)]
    assert "reference_s" not in prompt and "0.123456" not in prompt  # verdict never read


def test_instruction_differs_from_awm_only_in_domain_words():
    awm = ("Given a list of web navigation tasks, your task is to extract the common workflows to solve these tasks.\n"
           "Each given task contains a natural language instruction, and a series of actions to solve the task. You need "
           "to find the repetitive subset of actions across multiple tasks, and extract each of them out as a workflow.\n"
           "Each workflow should be a commonly-reused sub-routine of the tasks. Do not generate similar or overlapping "
           "workflows. Each workflow should have at least two steps. Represent the non-fixed elements (input text, "
           "button strings) with descriptive variable names as shown in the example.\n"
           "Keep the values of invariant elements, e.g., id of \"Search\" or \"Customers\", as they will share and stay "
           "invariant across tasks.\n"
           "Try to generate as many workflows that can cover all the tasks in the input list.")
    ours = wf.INDUCTION_INSTRUCTION
    for a, b in [("web navigation tasks", "terminal-based software performance tickets"),
                 ("(input text, button strings)",
                  "(file paths, module/function/variable names, domain vocabulary, data sizes)"),
                 ('id of "Search" or "Customers"',
                  "`python run.py INPUT.json OUTPUT.json` or `python -m unittest discover -s tests`")]:
        awm = awm.replace(a, b)
    assert ours == awm


# -- reference / oracle guard ------------------------------------------------------------------


def test_reference_fix_never_reaches_prompt_or_library(store, pool, family):
    client = _Echo(echo=True)  # adversarial model: copies its whole input back
    rows = eligible_pool(store, RUN, 0, "D", "cpu_mem")
    ind = induce(store, rows, client)
    refs = [family.solution_core(family.generate(r["cause"], r["theme"], r["seed"])) for r in rows]
    for t in (client.prompts[0], ind.library.text, render_procedure_block(ind.library)):
        for ref in refs:
            assert ref not in t and wf._norm(ref) not in wf._norm(t)
        assert "def reference(" not in t
    # the agent's write of the reference file reaches the prompt only as its diff against the original
    assert "# (file write of job/core.py, shown as its diff)" in client.prompts[0]
    assert "+++ b/job/core.py" in client.prompts[0] and "[file edit withheld]" not in client.prompts[0]


def test_reference_text_outside_a_file_write_is_withheld(store, pool, family):
    row = store.episode(pool["a1"])
    ref = family.solution_core(family.generate(row["cause"], row["theme"], row["seed"]))
    events = wf._load_trace(store, row)
    ex = example_from_trace(row["episode_id"], ("c", "t"), events, forbidden=[ref])  # no originals
    text = wf.format_example(ex)
    assert "[file edit withheld]" in text and wf._norm(ref) not in wf._norm(text)


def test_output_blocks_with_reference_or_one_shot_text_are_dropped(family):
    task = family.generate("queue_pop_front", "billing", 1)
    ref = family.solution_core(task)
    leaked = "## Apply the fix\nThis workflow writes the fix.\nWrite the file.\ncat > job/core.py <<EOF\n" + ref + "EOF"
    copied = "## Profile the greenhouse job\nThis workflow profiles.\nstep one\nstep two"
    lib, dropped = assemble_library("cpu_mem", LIBRARY_OUTPUT + "\n\n" + leaked + "\n\n" + copied,
                                    max_library_tokens=10_000, forbidden=[ref])
    assert [w.title for w in lib.workflows] == ["Find the slow loop in {module_path}", "Verify the fix"]
    assert {d["reason"] for d in dropped} == {"contains reference or oracle text", "copies the one-shot example"}
    assert ref not in lib.text


def test_verbatim_copies_of_the_one_shot_summary_workflows_are_dropped():
    summary = wf.ONE_SHOT.split("## Summary Workflows", 1)[1]
    lib, dropped = assemble_library("cpu_mem", summary + "\n\n" + LIBRARY_OUTPUT, max_library_tokens=10_000)
    assert len(dropped) == 2 and {d["reason"] for d in dropped} == {"copies the one-shot example"}
    assert len(lib) == 2  # the task-derived workflows survive (they share only a test command with it)


# -- library ------------------------------------------------------------------------------------


def test_library_parse_render_and_cap():
    lib, _ = assemble_library("cpu_mem", LIBRARY_OUTPUT, max_library_tokens=10_000)
    assert len(lib) == 2
    w = lib.workflows[0]
    assert w.description.startswith("This workflow is used to locate")
    assert w.placeholders == ["{large_n}", "{module_path}"] and len(w.steps) == 4
    block = render_procedure_block(lib)
    assert block.startswith(wf.PROCEDURE_HEADER) and block.endswith("python -m unittest discover -s tests -q")
    assert render_procedure_block(WorkflowLibrary("cpu_mem")) == ""
    assert experience_block(lib, "[m1]\nnote") == block + "\n\n[m1]\nnote"
    assert experience_block(None, "[m1]\nnote") == "[m1]\nnote"
    small, dropped = assemble_library("cpu_mem", LIBRARY_OUTPUT, max_library_tokens=wf.approx_tokens(
        wf.PROCEDURE_HEADER) + wf.approx_tokens(lib.workflows[0].body) + 1)
    assert len(small) == 1 and dropped[0]["reason"] == "over max_library_tokens"
    # AWM-style "Workflow N:" headers and blank lines inside a workflow parse too; `#` comments do not split
    awm = ("Workflow 1: Time the job\nUsed to time it.\n# generate input\npython scripts/make_load.py {n} /tmp/b.json\n"
           "\ntime python run.py /tmp/b.json /tmp/o.json\n\nWorkflow 2: too short\nx\n")
    parsed = wf.parse_workflows(awm)
    assert [p.title for p in parsed] == ["Time the job"] and len(parsed[0].steps) == 3


def test_procedures_enter_the_first_message_not_the_system_prompt(family):
    from elh.runner.prompts import SYSTEM_PROMPT, first_user_message

    lib, _ = assemble_library("cpu_mem", LIBRARY_OUTPUT, max_library_tokens=10_000)
    task = family.generate("queue_pop_front", "billing", 1)
    msg = first_user_message(task, "/w", experience_block(lib))
    assert msg.startswith("<experience>") and render_procedure_block(lib) in msg
    assert "Workflows" not in SYSTEM_PROMPT


def test_induction_returns_learning_usage(store, pool):
    ind = induce(store, eligible_pool(store, RUN, 0, "D", "cpu_mem"), _Echo())
    entries = learning_entries(load_config(CONFIG), ind)
    assert {e.category for e in entries} == {"learning"}
    assert sum(e.units for e in entries) == 3400
    assert ind.provenance()["usage"]["output_tokens"] == 400


def test_should_induce_every_m_with_new_successes():
    cfg = WorkflowConfig(induce_every=5)
    assert should_induce(5, ["a", "b"], [], cfg)
    assert not should_induce(4, ["a", "b"], [], cfg)
    assert not should_induce(10, ["a", "b"], ["a", "b"], cfg)
    assert not should_induce(0, ["a"], [], cfg)


# -- versioning ----------------------------------------------------------------------------------


def _ind(text, sources=("e1",)):
    lib, dropped = assemble_library("cpu_mem", text, max_library_tokens=10_000)
    return Induction(lib, text, "p", list(sources), {"input_tokens": 1}, "m", 0, dropped)


def test_revisions_provenance_promotion_and_rollback(store):
    scope = dict(run_id=RUN, history=0, group="D", family="cpu_mem")
    assert promoted_library(store, **scope) == (None, WorkflowLibrary("cpu_mem"))
    empty_state = learner_state_sha256(store, **scope)
    r1 = record_candidate(store, _ind(LIBRARY_OUTPUT, ["e1", "e2"]), run_id=RUN, history=0)
    assert r1 == 1 and revision_status(store, revision=1, **scope) == "candidate"
    lib1 = load_library(store, revision=1, **scope)
    assert lib1 == _ind(LIBRARY_OUTPUT).library
    row = store.db.execute("SELECT * FROM skills WHERE revision = 1 ORDER BY position").fetchone()
    assert json.loads(row["source_episode_ids_json"]) == ["e1", "e2"]
    assert row["body_sha256"] == lib1.workflows[0].body_sha256 and row["library_sha256"] == lib1.library_sha256
    assert procedure_ids(lib1, 1) == [f"{w.skill_id}@1" for w in lib1.workflows]

    ok = promote([DevResult("t", 0, True, 0.5)], [DevResult("t", 0, True, 0.6)], WorkflowConfig(repeats=1))
    decide_candidate(store, revision=1, decision=ok, **scope)
    assert promoted_library(store, **scope) == (1, lib1)
    state1 = learner_state_sha256(store, **scope)
    assert state1 != empty_state

    text2 = LIBRARY_OUTPUT.replace("Verify the fix", "Check the fix")
    r2 = record_candidate(store, _ind(text2), run_id=RUN, history=0)
    bad = promote([DevResult("t", 0, True, 0.6)], [DevResult("t", 0, True, 0.5)], WorkflowConfig(repeats=1))
    decide_candidate(store, revision=r2, decision=bad, **scope)
    assert revision_status(store, revision=2, **scope) == "rejected"
    assert promoted_library(store, **scope)[0] == 1

    r3 = record_candidate(store, _ind(text2), run_id=RUN, history=0)
    decide_candidate(store, revision=r3, decision=ok, **scope)
    assert revision_status(store, revision=1, **scope) == "superseded" and promoted_library(store, **scope)[0] == 3
    assert apply_rollback(store, evidence={"why": "retention"}, **scope) == 1
    assert revision_status(store, revision=3, **scope) == "rolled_back"
    assert promoted_library(store, **scope)[0] == 1 and learner_state_sha256(store, **scope) == state1
    with pytest.raises(ValueError, match="not allowed"):
        decide_candidate(store, revision=3, decision=ok, **scope)
    assert record_candidate(store, _ind("nothing parseable"), run_id=RUN, history=0) is None


def test_tampered_library_body_is_detected(store):
    record_candidate(store, _ind(LIBRARY_OUTPUT), run_id=RUN, history=0)
    store.db.execute("UPDATE skills SET body = body || 'x' WHERE position = 0")
    with pytest.raises(ValueError, match="hash"):
        load_library(store, run_id=RUN, history=0, group="D", family="cpu_mem", revision=1)


# -- gate ----------------------------------------------------------------------------------------


def _results(rewards, passes=None, viol=None, repeats=2):
    out = []
    for i, r in enumerate(rewards):
        t, k = divmod(i, repeats)
        out.append(DevResult(f"task{t}", k, (passes[i] if passes else r > 0), r, bool(viol and viol[i])))
    return out


def test_gate_promotes_only_on_paired_improvement():
    cfg = WorkflowConfig(repeats=2, margin=0.05)
    inc = _results([0.5, 0.5, 0.4, 0.4])
    assert promote(inc, _results([0.6, 0.6, 0.5, 0.5]), cfg).promote
    d = promote(inc, _results([0.52, 0.52, 0.42, 0.42]), cfg)
    assert not d.promote and "margin" in d.reasons[0]
    d = promote(inc, _results([0.9, 0.9, 0.0, 0.0]), cfg)  # higher mean reward but fewer passes
    assert not d.promote and any("pass count" in r for r in d.reasons)
    d = promote(inc, _results([0.6, 0.6, 0.5, 0.5], viol=[0, 1, 0, 0]), cfg)
    assert not d.promote and any("integrity" in r for r in d.reasons)
    assert d.to_dict()["candidate_integrity_violations"] == 1


def test_gate_requires_pairing_and_repeats():
    cfg = WorkflowConfig(repeats=2)
    with pytest.raises(ValueError, match="paired"):
        promote(_results([0.5, 0.5]), [DevResult("other", 0, True, 1.0), DevResult("other", 1, True, 1.0)], cfg)
    with pytest.raises(ValueError, match="repeats"):
        promote(_results([0.5], repeats=1), _results([0.6], repeats=1), cfg)


def test_rollback_on_retention_drop():
    cfg = WorkflowConfig(rollback_margin=0.1)
    prev = _results([0.5, 0.5, 0.5, 0.5])
    assert not rollback(prev, _results([0.45, 0.5, 0.5, 0.5]), cfg).promote
    d = rollback(prev, _results([0.5, 0.0, 0.5, 0.5]), cfg)
    assert d.promote and any("pass rate" in r for r in d.reasons)


def test_reward_and_dev_result_from_row(store, pool):
    assert wf.episode_reward(True, 100, 400) == 0.75 and wf.episode_reward(False, 1, 400) == 0.0
    assert wf.episode_reward(True, 1000, 400) == 0.0
    r = wf.dev_result_from_row(store.episode(pool["dev"]), 0, 4400)
    assert r.passed and r.reward == pytest.approx(0.75) and not r.integrity_violation
