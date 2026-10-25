"""Rule writing after "Learning on the Job" (arXiv 2607.22157): prompt contents, validator, retries."""

import json

import pytest
from pydantic import ValidationError

from elh.judge import Verdict, grade
from elh.learner.rules import (MEMORY_CREATE_TOOL, SAVE_FAIL, SAVE_SUCCESS, SAVE_SUCCESS_GROUNDED, Rule,
                               RulesConfig, _seed_history, consequential_actions, is_state_changing, parse_rule,
                               reflection_message, rule_content_gaps, write_rule)
from elh.memory.hindsight import CHECK_FAILED, CHECK_PASSED
from elh.registry import Budgets
from elh.runner.agent import EpisodeResult, run_episode
from elh.runner.models import ContextLimitError, ScriptedClient, to_openai_messages
from elh.runner.prompts import SYSTEM_PROMPT, TOOLS
from elh.runner.scripts import _write_file
from elh.sandbox.local import LocalSandbox
from elh.tasks.base import get_family

AVOID_TAIL = "-- verified wrong against the ground-truth outcome."


class Recording(ScriptedClient):
    def __init__(self, turns):
        super().__init__(turns)
        self.requests = []

    def create(self, *, system, messages, tools, max_tokens):
        self.requests.append(json.loads(json.dumps({"system": system, "messages": messages, "tools": tools})))
        return super().create(system=system, messages=messages, tools=tools, max_tokens=max_tokens)


def _create(title, content, query="batch job slow on large input, what to change?"):
    return [("memory_create", {"title": title, "content": content, "query": query})]


@pytest.fixture(scope="module")
def family():
    return get_family("cpu_mem")


@pytest.fixture(scope="module")
def failed_episode(family):
    """A real scripted episode on a cpu_mem task: a wrong fix plus an edit to a protected test file,
    graded by the real judge (fails correctness and integrity)."""
    task = family.generate("queue_pop_front", "billing", 1)
    client = ScriptedClient([
        [("bash", {"command": "ls -R && cat job/core.py"})],
        [_write_file("job/core.py", family.wrong_fix_core(task))],
        [("bash", {"command": "echo '# skip' >> tests/test_job.py"})],
        [("bash", {"command": "python -m unittest discover -s tests -q 2>&1 | tail -3"})],
        [("submit", {"summary": "made the queue faster"})],
    ])
    sbx = LocalSandbox()
    try:
        sbx.start(task.workspace)
        result = run_episode(task, sbx, client, Budgets(max_turns=10, max_total_tokens=10**6,
                                                        wall_clock_seconds=300, bash_timeout_seconds=60), 1024)
        final_ws = sbx.snapshot_workspace()
    finally:
        sbx.close()
    verdict = grade(task, final_ws, LocalSandbox)
    assert not verdict.passed and verdict.checks["integrity"] is False
    return task, result, verdict, final_ws


def _passed_verdict():
    return Verdict(True, {"integrity": True, "public_tests": True, "correctness": True, "performance": True},
                   {"quality": {"reference_s": 0.123456, "budget_s": 3.0, "agent_s": 0.5, "agent_over_reference": 4.05,
                                "large_status": "ok", "large_correct": True}, "cases": [{"n": 0, "ok": True}]})


def test_rule_prompt_never_contains_oracle_or_reference(family, failed_episode):
    task, result, verdict, final_ws = failed_episode
    client = Recording([_create("AVOID: x", f"WHEN slow queue do NOT edit tests/test_job.py and job/core.py {AVOID_TAIL}")])
    out = write_rule(client, task, result, verdict, final_ws)
    assert out.rule is not None and out.rule.kind == "avoid"
    sent = json.dumps(client.requests)
    forbidden = [family.solution_core(task), "def reference(", "hidden_seed", task.cause, "reference_s",
                 "agent_over_reference", task.oracle.files["params.json"], task.oracle.files["oracle.py"][:400],
                 task.oracle.files["cause.py"][:400]]
    for text in forbidden:
        assert text not in sent and text not in out.prompt, text[:60]
    # Feedback it does get: per-check verdict in lesson-v2 wording, the violation, the actions.
    assert "TASK OUTCOME (ground truth): the desired outcome was NOT achieved." in out.prompt
    assert CHECK_FAILED["integrity"] in out.prompt and CHECK_FAILED["correctness"] in out.prompt
    assert CHECK_PASSED["public_tests"] in out.prompt or CHECK_FAILED["public_tests"] in out.prompt
    assert "Protected files you changed (the ticket forbids this): tests/test_job.py" in out.prompt
    assert "CONSEQUENTIAL ACTIONS" in out.prompt and "cat > job/core.py <<'ELH_SCRIPTED_EOF'" in out.prompt
    assert "echo '# skip' >> tests/test_job.py" in out.prompt
    assert "ls -R && cat job/core.py" not in out.prompt.split("CONSEQUENTIAL ACTIONS")[1].split("FILES CHANGED")[0]
    assert "- job/core.py (modified)" in out.prompt and SAVE_FAIL in out.prompt
    assert "A wrong outcome doesn't always mean wrong facts." in out.prompt
    # The reflection is seeded with the agent's own conversation, then the reflection message.
    req = client.requests[0]
    assert req["system"] == SYSTEM_PROMPT and req["tools"] == TOOLS + [MEMORY_CREATE_TOOL]
    assert req["messages"][:-1] == json.loads(json.dumps(result.messages[:-1]))
    last = req["messages"][-1]["content"]
    assert last[0]["type"] == "tool_result" and last[-1] == {"type": "text", "text": out.prompt}


def test_validator_retries_with_exact_gaps(failed_episode):
    task, result, verdict, final_ws = failed_episode
    client = Recording([
        _create("AVOID: editing", f"WHEN the job is slow do NOT edit things {AVOID_TAIL}"),
        _create("AVOID: editing", f"WHEN the job is slow do NOT edit test files, e.g. tests/test_job.py {AVOID_TAIL}"),
        _create("AVOID: editing protected tests",
                f"WHEN the job is slow do NOT edit tests/test_job.py; the change to job/core.py broke output {AVOID_TAIL}"),
    ])
    out = write_rule(client, task, result, verdict, final_ws)
    assert out.required == ["tests/test_job.py"]
    assert len(client.requests) == 3 and len(out.calls) == 3 and out.gaps == [] and out.error is None
    assert out.usage["input_tokens"] == 3000 and out.usage["output_tokens"] == 600
    assert out.rejections[0] == (
        "NOT SAVED -- the AVOID is missing decision-critical content from the JUDGE'S CHECKS / CONSEQUENTIAL "
        "ACTIONS: 'tests/test_job.py'. Each of these must appear VERBATIM in the rule (in the do NOT clause as "
        "a directive -- not as a placeholder like <file>, and not inside an 'e.g.' example). Call memory_create "
        "again now with the corrected content.")
    assert out.rejections[1] == out.rejections[0]  # only inside an "e.g." example: still a gap
    bounced = client.requests[1]["messages"][-1]["content"][0]
    assert bounced["type"] == "tool_result" and bounced["content"] == out.rejections[0]
    assert out.rule == Rule("avoid", "the job is slow",
                            f"do NOT edit tests/test_job.py; the change to job/core.py broke output {AVOID_TAIL}",
                            "batch job slow on large input, what to change?", "AVOID: editing protected tests")
    assert out.rule.to_dict()["kind"] == "avoid" and out.rule.text.startswith("AVOID: editing protected tests\nWHEN")


def test_rule_accepted_with_gaps_after_max_retries(failed_episode):
    task, result, verdict, final_ws = failed_episode
    bad = _create("AVOID: editing", f"WHEN slow do NOT edit {AVOID_TAIL}")
    client = Recording([bad, bad, bad, bad])
    out = write_rule(client, task, result, verdict, final_ws, RulesConfig(max_retries=2))
    assert len(client.requests) == 3 and len(out.rejections) == 3
    assert out.rule is not None and out.gaps == ["tests/test_job.py"] and out.error is None


def test_no_usable_rule_after_retries(failed_episode):
    task, result, verdict, final_ws = failed_episode
    client = Recording([
        _create("VERIFIED RULE: wrong kind", "WHEN x THEN y"),   # failed episode: title must be AVOID:
        [("bash", {"command": "ls"})],                           # other tools are not executed
        _create("AVOID: no form", "just be careful"),
    ])
    out = write_rule(client, task, result, verdict, final_ws, RulesConfig(max_retries=3))
    assert out.rule is None and "no well-formed AVOID: rule after 4 attempts" in out.error
    assert out.rejections[0].startswith("NOT SAVED -- the desired outcome was NOT achieved, so the title must start")
    assert out.rejections[1].startswith("NOT SAVED -- you did not call memory_create")
    assert client.requests[2]["messages"][-1]["content"][0]["content"].startswith("NOT EXECUTED")
    assert out.rejections[2].startswith("NOT SAVED -- the content must be one sentence")
    assert out.rejections[3].startswith("NOT SAVED -- you did not call")  # scripted client ran out: text only
    assert len(out.calls) == 4


def test_passed_episode_banks_a_verified_rule(family):
    task = family.generate("quadratic_membership", "inventory", 3)
    final_ws = dict(task.workspace, **{"job/core.py": "fixed\n", "scratch.json": "{}"})
    result = EpisodeResult(outcome="submitted", turns=6, tool_calls=6)
    result.trace = [{"event": "tool_exec", "turn": 2, "command": "sed -i 's/in seen/in seen_set/' job/core.py",
                     "exit_code": 0},
                    {"event": "tool_exec", "turn": 3, "command": "python -c \"print(1 > 0)\"", "exit_code": 0}]
    verdict = _passed_verdict()
    prompt = reflection_message(task, result, verdict, final_ws)
    assert "TASK OUTCOME (ground truth): the desired outcome WAS achieved." in prompt
    assert SAVE_SUCCESS_GROUNDED in prompt and "A wrong outcome" not in prompt
    assert "turn 3: sed -i 's/in seen/in seen_set/' job/core.py" in prompt and "print(1 > 0)" not in prompt
    assert "0.50s against a budget of 3.00s (17% of budget)" in prompt
    assert "0.1234" not in prompt and "4.05" not in prompt and "cases" not in prompt
    client = Recording([_create("VERIFIED RULE: slow membership",
                                "WHEN the nightly job is CPU-bound on large input and the loop tests membership in a "
                                "list THEN in job/core.py build a set once and test membership against the set.",
                                "nightly job slow, loop does `x in list`")])
    out = write_rule(client, task, result, verdict, final_ws, RulesConfig(temperature=0.2))
    assert out.rule.kind == "verified" and out.rule.then.startswith("in job/core.py build a set")
    assert out.rule.when.startswith("the nightly job is CPU-bound") and out.rule.query.startswith("nightly")
    assert out.required == [] and out.rejections == [] and len(out.calls) == 1
    # No episode trace start: falls back to the frozen prompt and a single user message.
    assert client.requests[0]["messages"] == [{"role": "user", "content": out.prompt}]
    assert SAVE_SUCCESS in reflection_message(task, EpisodeResult(outcome="submitted"), verdict, final_ws)


def test_context_overflow_falls_back_to_transcript(failed_episode):
    task, result, verdict, final_ws = failed_episode

    class Small(Recording):
        def create(self, *, system, messages, tools, max_tokens):
            if len(messages) > 1:
                self.requests.append(None)
                raise ContextLimitError("prompt too long")
            return super().create(system=system, messages=messages, tools=tools, max_tokens=max_tokens)

    client = Small([_create("AVOID: x", f"WHEN slow do NOT edit tests/test_job.py or job/core.py {AVOID_TAIL}")])
    out = write_rule(client, task, result, verdict, final_ws)
    assert out.fallback_transcript and out.rule is not None
    assert "--- Conversation transcript ---" in out.prompt and "Agent action: bash(" in out.prompt
    assert client.requests[1]["messages"] == [{"role": "user", "content": out.prompt}]


def test_seed_history_keeps_tool_results_and_translates():
    msgs = [{"role": "user", "content": "ticket"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "submit", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Submitted."}]}]
    hist = _seed_history(msgs, "REFLECT")
    assert hist[-1]["content"][-1] == {"type": "text", "text": "REFLECT"} and msgs[-1]["content"][-1]["type"] == "tool_result"
    oa = to_openai_messages("sys", hist)
    assert oa[-2]["role"] == "tool" and oa[-1] == {"role": "user", "content": "REFLECT"}
    trunc = msgs[:2] + [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}]},
                        {"role": "assistant", "content": [{"type": "text", "text": "partial"}]},
                        {"role": "user", "content": "Continue."}]
    assert _seed_history(trunc, "R")[-1] == {"role": "user", "content": "R"}
    pending = msgs[:2]
    assert _seed_history(pending, "R")[-1]["content"][0]["is_error"] is True


@pytest.mark.parametrize("cmd,writes", [
    ("cat > job/core.py <<'EOF'\nx = a > b\nEOF", True),
    ("echo hi >> notes.txt", True),
    ("sed -i 's/a/b/' job/core.py", True),
    ("cp job/core.py job/core.py.bak", True),
    ("python - <<'EOF'\nopen('job/core.py','w').write(src)\nEOF", True),
    ("python scripts/make_load.py > /dev/null 2>&1", False),
    ("cat job/core.py", False),
    ("python -m unittest discover -s tests -q 2>&1 | tail -3", False),
    ("python -c \"print(1 if 3 > 2 else 0)\"", False),
    ("grep -n 'x > y' job/core.py", False),
    ("time python run.py examples/sample_input.json /tmp/out.json", False),
])
def test_state_changing_commands(cmd, writes):
    assert is_state_changing(cmd) is writes


def test_parse_rule_and_gaps():
    r = parse_rule("verified", "VERIFIED RULE: q", "WHEN a slow loop pops from the front THEN use collections.deque.", "q?")
    assert (r.when, r.then) == ("a slow loop pops from the front", "use collections.deque.")
    assert parse_rule("verified", "t", "Use a deque.", "q") is None
    a = parse_rule("avoid", "AVOID: t", f"WHEN slow, do NOT rewrite run.py {AVOID_TAIL}", "q")
    assert a.when == "slow" and a.then == f"do NOT rewrite run.py {AVOID_TAIL}"
    assert rule_content_gaps("do NOT touch run.py", ["run.py"]) == []
    assert rule_content_gaps("for example run.py", ["run.py"]) == ["run.py"]
    assert rule_content_gaps("such as run.py; do NOT edit run.py", ["run.py", "tests/x.py"]) == ["tests/x.py"]


def test_rules_config():
    c = RulesConfig()
    assert (c.enabled, c.max_retries, c.temperature) == (False, 2, None)
    with pytest.raises(ValidationError):
        RulesConfig(max_retries=-1)
    with pytest.raises(ValidationError):
        RulesConfig(bogus=1)


def test_consequential_actions_mark_failures():
    r = EpisodeResult(outcome="submitted")
    r.trace = [{"event": "tool_exec", "turn": 0, "command": "rm nothing", "exit_code": 1},
               {"event": "tool_exec", "turn": 1, "command": "cat x", "exit_code": 0}]
    assert consequential_actions(r) == ["turn 1: rm nothing [exit 1]"]
