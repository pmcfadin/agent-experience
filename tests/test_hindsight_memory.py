"""HindsightMemory against a stub server: request shapes, read-only banks, ledger mapping."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from pydantic import ValidationError

from elh.memory.hindsight import (HindsightMemory, render_context, render_episode, render_episode_ticket_v0,
                                  usage_to_ledger, workspace_diff)
from elh.registry import MemoryServiceConfig
from elh.tasks.base import get_family


class _Stub:
    def __init__(self, op_states):
        self.requests, self.op_states = [], list(op_states)
        stub = self

        class H(BaseHTTPRequestHandler):
            def _reply(self, body):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self):
                n = int(self.headers.get("content-length") or 0)
                return json.loads(self.rfile.read(n)) if n else None

            def do_PUT(self):
                stub.requests.append(("PUT", self.path, self._body()))
                self._reply({})

            def do_POST(self):
                stub.requests.append(("POST", self.path, self._body()))
                if self.path.endswith("/recall"):
                    self._reply({"results": [{"id": f"m{i}", "text": f"fact {i}"} for i in range(7)]})
                else:
                    self._reply({"success": True, "usage": {"input_tokens": 1000, "output_tokens": 50,
                                                            "cached_tokens": 600, "thoughts_tokens": 0}})

            def do_GET(self):
                state = stub.op_states.pop(0) if stub.op_states else "completed"
                self._reply({"operations": [{"status": state}]})

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"


def _episode(task):
    return {"episode_id": "e1", "family": task.family, "prompt": task.prompt, "title": "Ticket X: slow",
            "outcome": "submitted", "submit_summary": "used a deque", "diff": "--- a/job/core.py\n+++ b/job/core.py",
            "verdict": {"passed": False, "checks": {"integrity": True, "performance": False},
                        "detail": {"quality": {"agent_s": 3.0, "budget_s": 3.0, "large_status": "timeout"}}},
            "turns": 12, "tool_calls": 12}


def test_recall_retain_and_idle_wait():
    task = get_family("cpu_mem").generate("queue_pop_front", "billing", 1)
    stub = _Stub(["processing", "pending", "completed"])
    mem = HindsightMemory(stub.url, "b1")
    mem.ensure_bank()
    items = mem.recall(task, 5)
    assert [it.memory_id for it in items] == ["m0", "m1", "m2", "m3", "m4"]
    assert mem.retain(_episode(task)) == ["e1"]
    mem.wait_idle(poll_s=0.01)
    method, path, body = stub.requests[2]
    assert path == "/v1/default/banks/b1/memories" and body["items"][0]["document_id"] == "e1"
    assert body["items"][0]["context"].endswith(task.prompt.strip())
    assert stub.requests[1][2]["query"] == task.prompt
    assert usage_to_ledger(mem.last_usage) == {"input_tokens": 400, "output_tokens": 50,
                                               "cache_read_input_tokens": 600, "cache_creation_input_tokens": 0}
    assert not stub.op_states  # polled until no operation was active


def test_readonly_bank_refuses_retain():
    task = get_family("cpu_mem").generate("queue_pop_front", "billing", 1)
    with pytest.raises(RuntimeError, match="read-only"):
        HindsightMemory("http://127.0.0.1:9", "b1", readonly=True).retain(_episode(task))


def test_episode_record_has_feedback_not_oracle():
    family = get_family("cpu_mem")
    task = family.generate("queue_pop_front", "billing", 1)
    text = render_episode(_episode(task))
    assert text.startswith("Outcome of an attempt") and "FAILED: too slow on the large input" in text
    assert "3.00s against a budget of 3.00s" in text and "used a deque" in text and "a/job/core.py" in text
    assert task.prompt.strip() not in text and task.prompt.strip() in render_context(_episode(task))
    for t in (text, render_context(_episode(task))):
        assert family.solution_core(task) not in t and "def reference(" not in t


def test_workspace_diff_keeps_source_and_names_scratch_files():
    before = {"job/core.py": "a = 1\n", "tests/t.py": "x\n"}
    after = {"job/core.py": "a = 2\n", "tests/t.py": "x\n", "big.json": "[" + "1," * 300000 + "1]",
             "job/core.py.bak": "a = 1\n"}
    d = workspace_diff(before, after)
    assert "-a = 1" in d and "+a = 2" in d and len(d) < 1000
    assert "big.json (created, 600003 bytes)" in d and "job/core.py.bak (created" in d
    assert workspace_diff(before, {**before, "out.json": "{}"}).startswith("No source file was changed.")


@pytest.mark.parametrize("fmt", ["lesson-v2", "lesson-v1", "ticket-v0"])
def test_retain_sends_the_configured_record_format(fmt):
    task = get_family("cpu_mem").generate("queue_pop_front", "billing", 1)
    ep = _episode(task)
    stub = _Stub([])
    HindsightMemory(stub.url, "b1", record_format=fmt).retain(ep)
    item = stub.requests[-1][2]["items"][0]
    if fmt == "lesson-v2":
        assert (item["content"], item["context"]) == (render_episode(ep), render_context(ep))
    elif fmt == "lesson-v1":
        assert (item["content"], item["context"]) == (render_episode(ep, version=1), render_context(ep))
    else:
        assert item["content"] == render_episode_ticket_v0(ep)
        assert item["context"] == "a coding-agent episode and the judge's verified result"
    assert item["document_id"] == "e1" and item["tags"] == ["family:cpu_mem"]


def test_ticket_v0_record_is_the_original_ticket_first_format():
    family = get_family("cpu_mem")
    task = family.generate("queue_pop_front", "billing", 1)
    text = render_episode_ticket_v0(_episode(task))
    assert text == (
        "Task family cpu_mem. Outcome: FAILED (integrity=pass, performance=FAIL).\n\n"
        f"Ticket:\n{task.prompt.strip()}\n\n"
        "Agent's own summary of its change:\nused a deque\n\n"
        "Judge timing on the large input: agent 3.0s vs budget 3.0s (status timeout).\n\n"
        "Episode used 12 turns and 12 tool calls.")
    assert family.solution_core(task) not in text and "def reference(" not in text


def test_unknown_record_format_is_rejected():
    with pytest.raises(ValueError, match="record format"):
        HindsightMemory("http://127.0.0.1:9", "b1", record_format="lesson-v9")


def test_registry_record_format_values():
    assert MemoryServiceConfig().record_format == "lesson-v2"
    assert MemoryServiceConfig(record_format="ticket-v0").record_format == "ticket-v0"
    with pytest.raises(ValidationError):
        MemoryServiceConfig(record_format="lesson-v9")


def test_failed_integrity_is_stated_as_a_violation():
    task = get_family("cpu_mem").generate("queue_pop_front", "billing", 1)
    ep = _episode(task)
    ep["verdict"] = {"passed": False, "checks": {"integrity": False, "public_tests": True, "correctness": True,
                                                 "performance": True},
                     "detail": {"integrity_violations": ["examples/sample_output.json"]}}
    text = render_episode(ep)
    assert "FAILED: modified files the ticket says must not be changed." in text
    assert "examples/sample_output.json" in text and "untouched" not in text
    assert "fast enough on the large input" in text  # passed checks keep their pass wording
    # v1 is kept only to reproduce old runs; it states the failure with the pass wording.
    assert "FAILED: left the tests and harness untouched" in render_episode(ep, version=1)
