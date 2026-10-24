"""The self-hosted model path: format translation and a full episode against a stub gateway."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from elh.controller import run_single_episode
from elh.registry import load_config
from elh.runner.models import OpenAICompatClient, to_openai_messages, to_openai_tools
from elh.runner.prompts import TOOLS
from elh.store import Store
from elh.tasks.base import get_family

CONFIG = Path(__file__).parent.parent / "experiments" / "phase0-local.yaml"


def test_tools_translate_to_functions():
    tools = to_openai_tools(TOOLS)
    assert [t["function"]["name"] for t in tools] == ["bash", "submit"]
    assert tools[0]["function"]["parameters"]["required"] == ["command"]


def test_history_translates_to_chat_messages():
    history = [
        {"role": "user", "content": "ticket"},
        {"role": "assistant", "content": [
            {"type": "reasoning", "text": "hidden"},
            {"type": "text", "text": "Looking."},
            {"type": "tool_use", "id": "c1", "name": "bash", "input": {"command": "ls"}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "run.py"}]},
        {"role": "user", "content": "continue"},
    ]
    msgs = to_openai_messages("sys", history)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "user"]
    assert msgs[2]["content"] == "Looking."
    assert json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"]) == {"command": "ls"}
    assert msgs[3] == {"role": "tool", "tool_call_id": "c1", "content": "run.py"}
    assert "hidden" not in json.dumps(msgs)


class _Gateway:
    """Minimal OpenAI-compatible server that replays scripted assistant turns."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []
        gw = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                gw.requests.append(body)
                calls = gw.turns.pop(0) if gw.turns else []
                msg = {"role": "assistant", "content": None if calls else "done",
                       "tool_calls": [{"id": f"call_{len(gw.requests)}_{i}", "type": "function",
                                       "function": {"name": n, "arguments": a if isinstance(a, str) else json.dumps(a)}}
                                      for i, (n, a) in enumerate(calls)] or None}
                resp = {"id": f"chatcmpl-{len(gw.requests)}", "object": "chat.completion", "created": 0,
                        "model": "qwen3-coder-30b-a3b",
                        "choices": [{"index": 0, "message": msg,
                                     "finish_reason": "tool_calls" if calls else "stop"}],
                        "usage": {"prompt_tokens": 1200, "completion_tokens": 80, "total_tokens": 1280,
                                  "prompt_tokens_details": {"cached_tokens": 1000}}}
                data = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"


@pytest.fixture
def family():
    return get_family("cpu_mem")


def test_episode_through_openai_compat_client(tmp_path, family):
    cfg = load_config(CONFIG)
    task = family.generate("queue_pop_front", "network_flows", 3)
    fix = family.solution_core(task)
    gw = _Gateway([
        [("bash", {"command": "cat job/core.py"})],
        [("bash", "{not json")],  # malformed arguments are returned to the model as an error
        [("bash", {"command": f"cat > job/core.py <<'EOF'\n{fix}EOF"})],
        [("submit", {"summary": "deque"})],
    ])
    client = OpenAICompatClient(cfg.model.id, gw.url, "test-key", cfg.model.temperature, cfg.model.top_p)
    store = Store(tmp_path)
    episode_id, result, verdict = run_single_episode(cfg, store, task, client, sandbox_backend="local")

    assert result.outcome == "submitted" and verdict.passed
    assert gw.requests[0]["temperature"] == 0.7 and gw.requests[0]["top_p"] == 0.8
    assert gw.requests[0]["tools"][0]["function"]["name"] == "bash"
    tool_msgs = [m for m in gw.requests[2]["messages"] if m["role"] == "tool"]
    assert "Invalid tool call" in tool_msgs[-1]["content"]
    assert result.usage["cache_read_input_tokens"] == 4000 and result.usage["input_tokens"] == 800
    ledger = store.db.execute("SELECT component, usd FROM ledger WHERE episode_id = ? AND category = 'inference'",
                              (episode_id,)).fetchall()
    assert {r["component"] for r in ledger} == {"pilot-coder:input", "pilot-coder:output", "pilot-coder:cache_read"}


def test_unpinned_local_model_refused_outside_dev(tmp_path, family):
    cfg = load_config(CONFIG)
    client = OpenAICompatClient(cfg.model.id, "http://127.0.0.1:9/v1", "k")
    with pytest.raises(ValueError, match="not pinned"):
        run_single_episode(cfg, Store(tmp_path), family.generate("queue_pop_front", "billing", 1), client,
                           split="sealed_familiar", sandbox_backend="local")
