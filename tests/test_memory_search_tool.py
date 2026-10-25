"""Optional agent-issued memory search (paper 2607.22157 §3.3.3 read path, App. A.1-A.3).

Off (the default) must leave every byte the model sees exactly as it was before the tool existed;
on, the agent's own queries go to a read-only search callable and every call is recorded."""

import hashlib
import json

from elh.memory.base import MemoryItem
from elh.registry import Budgets
from elh.runner.agent import run_episode
from elh.runner.models import ModelResponse, ScriptedClient, to_openai_messages, to_openai_tools
from elh.runner.prompts import (MEMORY_CHECK_REMINDER, SEARCH_MEMORY_TOOL, SYSTEM_PROMPT, TOOLS,
                                first_user_message, system_prompt)
from elh.sandbox.base import ExecResult
from elh.tasks.base import get_family

# sha256 of the prompts as they were at commit 2f86d23, before the search tool existed.
GOLDEN_SYSTEM = "01fcdee0b8a9cf595deb0a22ac6b134a38529191468c43e76a4106fb97f2b151"
GOLDEN_TOOLS = "ee1d55b4fb885ed6bbf85aadd94671bd580c3e8b28f89c07296be2d9fab98c1f"
GOLDEN_FIRST = "802e88e6e679c1c97c6c34a95699de9e824bd5d7e7deac7aba4a4b3cf5c64e5d"
GOLDEN_FIRST_WITH_MEMORY = "320370b258647ee909d8be618c58c11305b0fd43a01b43aa9b584f68d8ed217f"
GOLDEN_OPENAI_TOOLS = "d4857ea44e04410533ad4dba256aa681fba11a4015648e6aef940013d6f17af7"
# sha256 over every request (system, messages, tools, max_tokens) of the scripted episode below,
# recorded with the agent loop at 2f86d23.
GOLDEN_REQUESTS = "fe5bc44fec04708a27b39bd94796b858da172b23b3721ab5bd5b9957b1927e37"


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


class _FakeSandbox:
    workdir = "/workspace"

    def exec(self, cmd, timeout_s):
        return ExecResult(stdout=f"ran: {cmd}\n", stderr="", exit_code=0, timed_out=False, wall_s=0.01)


class _Recording(ScriptedClient):
    def __init__(self, turns):
        super().__init__(turns)
        self.requests = []

    def create(self, *, system, messages, tools, max_tokens):
        self.requests.append(json.loads(json.dumps({"system": system, "messages": messages, "tools": tools,
                                                    "max_tokens": max_tokens})))
        return super().create(system=system, messages=messages, tools=tools, max_tokens=max_tokens)


def _task():
    return get_family("cpu_mem").generate("queue_pop_front", "billing", 1)


def _budgets():
    return Budgets(max_turns=10, max_total_tokens=100000, wall_clock_seconds=600, bash_timeout_seconds=30)


def _script():
    return [[("bash", {"command": "ls"})], [("submit", {"summary": "done"})]]


def test_prompts_unchanged_when_search_is_off():
    task = _task()
    assert _h(SYSTEM_PROMPT) == GOLDEN_SYSTEM and system_prompt(False) == SYSTEM_PROMPT
    assert _h(json.dumps(TOOLS, sort_keys=True)) == GOLDEN_TOOLS
    assert _h(json.dumps(to_openai_tools(TOOLS), sort_keys=True)) == GOLDEN_OPENAI_TOOLS
    assert _h(first_user_message(task, "/workspace")) == GOLDEN_FIRST
    assert _h(first_user_message(task, "/workspace", "[m1]\nfact")) == GOLDEN_FIRST_WITH_MEMORY


def test_every_request_byte_identical_when_search_is_off():
    client = _Recording(_script())
    result = run_episode(_task(), _FakeSandbox(), client, _budgets(), 1024, memory_block="[m1]\nfact")
    assert result.outcome == "submitted" and result.memory_searches == []
    assert _h(json.dumps(client.requests, sort_keys=True)) == GOLDEN_REQUESTS
    for req in client.requests:
        assert req["system"] == SYSTEM_PROMPT and req["tools"] == TOOLS
        assert "search_memory" not in json.dumps(req)
    start = result.trace[0]
    assert start["system"] == SYSTEM_PROMPT and start["tools"] == TOOLS and "search_memory" not in start


def test_search_on_sends_results_as_tool_result_and_records_each_call():
    calls = []

    def search(query, k):
        calls.append((query, k))
        return [MemoryItem("r7", "VERIFIED RULE: slow queue\nWHEN x THEN use collections.deque", 0.9, "ep1",
                           {"rerank": 0.9}),
                MemoryItem("r2", "AVOID: list.pop(0)\nWHEN x do NOT y", 0.4, "ep2")]

    client = _Recording([
        [("bash", {"command": "cat job/core.py"})],
        [("search_memory", {"query": "batch job slow on large input; plan: replace list queue"})],
        [("search_memory", {"query": 7})],
        [("submit", {"summary": "done"})],
    ])
    task = _task()
    result = run_episode(task, _FakeSandbox(), client, _budgets(), 1024, search=search, search_k=3)
    assert result.outcome == "submitted" and result.tool_calls == 4
    assert calls == [("batch job slow on large input; plan: replace list queue", 3)]

    first = client.requests[0]
    assert first["system"] == system_prompt(True) != SYSTEM_PROMPT
    assert "## Experience memory" in first["system"] and "`search_memory`" in first["system"]
    assert first["tools"] == TOOLS + [SEARCH_MEMORY_TOOL]
    assert first["messages"][0]["content"] == first_user_message(task, "/workspace", memory_check=True)
    assert first["messages"][0]["content"].endswith(MEMORY_CHECK_REMINDER)
    assert all(r["system"] == first["system"] and r["tools"] == first["tools"] for r in client.requests)

    # The results are a tool_result in the transcript the model sees on its next turn.
    tool_result = client.requests[2]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "toolu_scripted_1_0"
    assert tool_result["content"] == ("[r7]\nVERIFIED RULE: slow queue\nWHEN x THEN use collections.deque\n\n"
                                      "[r2]\nAVOID: list.pop(0)\nWHEN x do NOT y")
    bad = client.requests[3]["messages"][-1]["content"][0]
    assert bad.get("is_error") is True and "query" in bad["content"]
    # The OpenAI-compatible translation carries the same result as a tool message.
    oa = to_openai_messages(client.requests[2]["system"], client.requests[2]["messages"])
    assert oa[-1] == {"role": "tool", "tool_call_id": "toolu_scripted_1_0", "content": tool_result["content"]}
    assert to_openai_tools(first["tools"])[-1]["function"]["name"] == "search_memory"

    assert result.memory_searches == [{
        "turn": 1, "tool_use_id": "toolu_scripted_1_0", "k": 3,
        "query": "batch job slow on large input; plan: replace list queue",
        "ids": ["r7", "r2"], "ranks": [1, 2], "document_ids": ["ep1", "ep2"],
        "scores": [0.9, 0.4], "stage_scores": [{"rerank": 0.9}, {}]}]
    assert [e for e in result.trace if e["event"] == "memory_search"][0]["ids"] == ["r7", "r2"]
    assert result.trace[0]["system"] == system_prompt(True) and result.trace[0]["search_memory"] is True


def test_search_with_no_results_says_so():
    client = _Recording([[("search_memory", {"query": "anything"})], [("submit", {"summary": "s"})]])
    result = run_episode(_task(), _FakeSandbox(), client, _budgets(), 1024, search=lambda q, k: [])
    assert client.requests[1]["messages"][-1]["content"][0]["content"] == "No matching memories found."
    assert result.memory_searches[0]["ids"] == [] and result.memory_searches[0]["ranks"] == []


def test_search_tool_is_invalid_when_off():
    client = _Recording([[("search_memory", {"query": "x"})], [("submit", {"summary": "s"})]])
    result = run_episode(_task(), _FakeSandbox(), client, _budgets(), 1024)
    assert client.requests[1]["messages"][-1]["content"][0]["is_error"] is True
    assert result.memory_searches == []


def test_response_type_unchanged():
    # Guard against accidental API drift in the fake used above.
    assert ModelResponse([], "end_turn", {}, "m").content == []
