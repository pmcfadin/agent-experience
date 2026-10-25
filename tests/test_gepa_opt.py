"""GEPA procedure optimiser (arm D): scoring, feedback hygiene, dev-only pools, reflection LM, and
a full GEPA run with a fake agent and a fake reflector."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from elh.judge import Verdict
from elh.learner.gepa_opt import (COMPONENT, DevPools, DevRollout, GepaConfig, ProcedureAdapter, ReflectionLM,
                                  ReflectionTruncated, check_dev_only, procedure_block, reflection_ledger_entries,
                                  reflection_prompt, run_gepa, score_rollout)
from elh.memory.hindsight import CHECK_FAILED, CHECK_PASSED
from elh.registry import load_config
from elh.runner.agent import EpisodeResult
from elh.runner.models import InfraError, ModelResponse
from elh.tasks.base import get_family

CONFIG = Path(__file__).parent.parent / "experiments" / "phase0-local.yaml"
KEYWORD = "profile"
HIDDEN = "HIDDEN_CASE_INPUT_7731"


def _tasks(n, base=5000):
    fam = get_family("cpu_mem")
    causes = ["queue_pop_front", "quadratic_membership", "repeated_sort_min"]
    tasks = [fam.generate(causes[i % 3], "billing", base + i) for i in range(n)]
    return {t.task_id: t for t in tasks}


def _result(summary="used a deque"):
    r = EpisodeResult(outcome="submitted", turns=6, tool_calls=6, submit_summary=summary)
    r.trace = [
        {"event": "episode_start", "first_message": "<procedure>SECRET-PROCEDURE</procedure> ticket"},
        {"event": "response", "turn": 0, "content": [{"type": "text", "text": "Let me time the job."}]},
        {"event": "tool_exec", "turn": 0, "command": "python run.py in.json out.json", "exit_code": 0,
         "timed_out": False, "wall_s": 4.2, "stdout": "", "stderr": ""},
    ]
    return r


def _verdict(passed, *, integrity=True, perf=True, correct=True, violations=()):
    detail = {"quality": {"reference_s": 0.123456, "budget_s": 3.0, "agent_s": 1.0 if perf else 9.0,
                          "agent_over_reference": 8.1, "large_status": "ok" if perf else "timeout",
                          "large_correct": True},
              "cases": [{"input": HIDDEN, "expected": HIDDEN}], "stderr_tail": HIDDEN}
    if violations:
        detail["integrity_violations"] = list(violations)
    return Verdict(passed, {"integrity": integrity, "public_tests": True, "correctness": correct,
                            "performance": perf}, detail)


def fake_episode(task, procedure):
    """Deterministic agent: the keyword in the procedure makes it fast enough; otherwise too slow."""
    good = KEYWORD in procedure.lower()
    return DevRollout(_verdict(good, perf=good), _result(), cost=0.01, episode_id=f"ep-{task.task_id}-{good}",
                      diff="--- a/job/core.py\n+++ b/job/core.py\n-x\n+y")


class FakeReflector:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return "Here is the procedure:\n```\n1. Reproduce the slowness.\n2. Profile before you change code.\n```"


def test_score_partial_credit_cost_and_integrity_zero():
    cfg = GepaConfig()
    cheap, dear = score_rollout(_verdict(True), 0.0, cfg), score_rollout(_verdict(True), 1.0, cfg)
    assert cheap == 1.0 and dear == pytest.approx(0.6)
    perf = score_rollout(_verdict(False, perf=False), 0.0, cfg)
    assert perf == pytest.approx(0.3 + 0.2 * 3.0 / 9.0)
    assert score_rollout(_verdict(False, correct=False), 0.0, cfg) == pytest.approx(0.1)
    assert dear > perf > score_rollout(_verdict(False, correct=False), 0.0, cfg) > 0
    # integrity broken => 0, even when everything else passed
    assert score_rollout(_verdict(True, integrity=False), 0.0, cfg) == 0.0
    assert score_rollout(_verdict(True, violations=["tests/test_job.py"]), 0.0, cfg) == 0.0


def test_integrity_violation_scores_zero_through_the_adapter():
    tasks = _tasks(1)
    tid = next(iter(tasks))

    def cheat(task, procedure):
        return _verdict(True, integrity=False, violations=["tests/test_job.py"]), _result(), 0.0  # plain tuple

    ad = ProcedureAdapter(GepaConfig(), tasks, cheat)
    batch = ad.evaluate([tid], {COMPONENT: "x"}, capture_traces=True)
    assert batch.scores == [0.0] and ad.rollouts[0]["integrity"] is False
    fb = ad.make_reflective_dataset({COMPONENT: "x"}, batch, [COMPONENT])[COMPONENT][0]["Feedback"]
    assert CHECK_FAILED["integrity"] in fb and "tests/test_job.py" in fb and "Score: 0 (integrity)" in fb


def test_reflective_dataset_has_lesson_v2_feedback_and_no_leaks():
    family = get_family("cpu_mem")
    tasks = _tasks(2)
    ids = list(tasks)
    ad = ProcedureAdapter(GepaConfig(), tasks, fake_episode)
    batch = ad.evaluate(ids, {COMPONENT: "be careful"}, capture_traces=True)
    ds = ad.make_reflective_dataset({COMPONENT: "be careful"}, batch, [COMPONENT])
    recs = ds[COMPONENT]
    assert len(recs) == 2 and set(recs[0]) == {"Inputs", "Generated Outputs", "Feedback"}
    text = json.dumps(ds)
    fb = recs[0]["Feedback"]
    # lesson-v2 wording, imported from hindsight: failure wording for failed checks
    assert f"FAILED: {CHECK_FAILED['performance']}" in fb and CHECK_PASSED["correctness"] in fb
    assert "9.00s against a budget of 3.00s" in fb and "used a deque" in fb and "a/job/core.py" in fb
    assert "time the job" in text and "python run.py" in text
    for tid in ids:
        task = tasks[tid]
        assert family.solution_core(task) not in text and "def reference(" not in text
        for content in task.oracle.files.values():
            assert content.strip() not in text
    for leak in ("reference_s", "0.123", "agent_over_reference", HIDDEN, "SECRET-PROCEDURE", "hidden_seed"):
        assert leak not in text


def test_pools_disjoint_and_dev_only():
    with pytest.raises(ValidationError, match="share tasks"):
        DevPools(reflection=["a", "b"], validation=["b"], promotion=["c"])
    pools = DevPools(reflection=["a", "b"], validation=["c"], promotion=["d"])
    check_dev_only(pools, lambda t: "dev")
    with pytest.raises(ValueError, match="only use dev tasks"):
        check_dev_only(pools, lambda t: "sealed_familiar" if t == "c" else "dev")
    tasks = _tasks(4)
    ids = list(tasks)
    pools = DevPools(reflection=ids[:3], validation=[ids[3]], promotion=[ids[0] + "x"])
    with pytest.raises(ValueError, match="only use dev tasks"):
        run_gepa(GepaConfig(max_metric_calls=5), "", pools, fake_episode, FakeReflector(), tasks=tasks,
                 split_of=lambda t: "checkpoint")


def test_reflection_prompt_generalises_and_keeps_gepa_placeholders():
    p = reflection_prompt(GepaConfig())
    assert "<curr_param>" in p and "<side_info>" in p and "400 words" in p
    assert "niche" not in p and "OTHER tickets" in p
    assert procedure_block("") == "" and procedure_block(" 1. x ") == "<procedure>\n1. x\n</procedure>"


class _Client:
    model_id = "study-qwen3-coder-30b-a3b"

    def __init__(self, text, stop="end_turn"):
        self.text, self.stop, self.seen = text, stop, []

    def create(self, *, system, messages, tools, max_tokens):
        self.seen.append((system, messages, tools, max_tokens))
        return ModelResponse([{"type": "text", "text": self.text}], self.stop,
                             {"input_tokens": 900, "output_tokens": 120, "cache_read_input_tokens": 100,
                              "cache_creation_input_tokens": 0}, self.model_id, "req1")


def test_reflection_lm_charges_learning():
    seen = []
    client = _Client("```\n1. a\n```")
    lm = ReflectionLM(client, max_tokens=500, on_usage=lambda u, c: seen.append(u))
    assert lm("hello") == "```\n1. a\n```"
    assert client.seen[0][1] == [{"role": "user", "content": "hello"}] and client.seen[0][2] == []
    assert seen[0]["output_tokens"] == 120 and lm.usage_total()["input_tokens"] == 900
    entries = reflection_ledger_entries(load_config(CONFIG), client.model_id, lm.calls)
    assert entries and {e.category for e in entries} == {"learning"}
    with pytest.raises(ReflectionTruncated):
        ReflectionLM(_Client("```\n1. a", stop="max_tokens"))("hi")


def test_gepa_improves_the_procedure_within_a_tiny_budget():
    tasks = _tasks(7)
    ids = list(tasks)
    pools = DevPools(reflection=ids[:3], validation=ids[3:5], promotion=ids[5:])
    reflector = FakeReflector()
    cfg = GepaConfig(max_metric_calls=12)
    out = run_gepa(cfg, "1. Read the ticket.", pools, fake_episode, reflector, tasks=tasks, split_of=lambda t: "dev")
    assert KEYWORD in out.best_procedure.lower() and out.best_idx == 1
    assert [h["parents"] for h in out.history] == [[None], [0]]
    assert out.history[1]["val_mean"] > out.history[0]["val_mean"]
    assert set(out.history[0]["val_scores"]) == set(pools.validation)
    # reflection saw our template and lesson-v2 feedback, only from reflection-pool tasks
    assert len(reflector.prompts) >= 1
    prompt = reflector.prompts[0]
    assert "OTHER tickets" in prompt and "niche" not in prompt and "1. Read the ticket." in prompt
    assert f"FAILED: {CHECK_FAILED['performance']}" in prompt
    assert not any(tasks[t].prompt.strip()[:200] in prompt for t in pools.validation + pools.promotion)
    # GEPA spent only reflection/validation rollouts; the promotion check used the third pool
    gepa_tasks = {r["task_id"] for r in out.rollouts if r["stage"] == "gepa"}
    assert gepa_tasks <= set(pools.reflection) | set(pools.validation)
    promo = [r for r in out.rollouts if r["stage"] == "promotion"]
    assert {r["task_id"] for r in promo} == set(pools.promotion) and len(promo) == 4
    assert out.promote and out.promotion.mean_delta > 0
    assert out.total_metric_calls <= cfg.max_metric_calls + 2 * cfg.reflection_minibatch_size + len(pools.validation)
    assert out.provenance["gepa_version"] == "0.1.4" and out.source_episode_ids()


def test_no_promotion_when_nothing_beats_the_seed():
    tasks = _tasks(7)
    ids = list(tasks)
    pools = DevPools(reflection=ids[:3], validation=ids[3:5], promotion=ids[5:])

    class Useless(FakeReflector):
        def __call__(self, prompt):
            self.prompts.append(prompt)
            return "```\n1. Work harder.\n```"

    out = run_gepa(GepaConfig(max_metric_calls=10), "1. Read the ticket.", pools, fake_episode, Useless(),
                   tasks=tasks, split_of=lambda t: "dev")
    assert out.best_procedure == "1. Read the ticket." and not out.promote and out.promotion is None


def test_infra_errors_retry_then_raise():
    tasks = _tasks(1)
    tid = next(iter(tasks))
    calls = []

    def flaky(task, procedure):
        calls.append(1)
        if len(calls) == 1:
            raise InfraError("backend changed")
        return fake_episode(task, procedure)

    ProcedureAdapter(GepaConfig(), tasks, flaky).evaluate([tid], {COMPONENT: ""})
    assert len(calls) == 2

    def broken(task, procedure):
        raise InfraError("down")

    with pytest.raises(InfraError):
        ProcedureAdapter(GepaConfig(), tasks, broken).evaluate([tid], {COMPONENT: ""})
