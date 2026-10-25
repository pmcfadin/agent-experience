"""The self-hosted model path: format translation and a full episode against a stub gateway."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from elh.controller import run_single_episode
from elh.registry import ModelPin, load_config
from elh.runner.models import InfraError, OpenAICompatClient, to_openai_messages, to_openai_tools
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

    def __init__(self, turns, headers=None):
        self.turns = list(turns)
        self.headers = headers or (lambda n: {})
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
                for k, v in gw.headers(len(gw.requests)).items():
                    self.send_header(k, v)
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
    assert {r["component"] for r in ledger} == {f"{cfg.model.id}:{k}" for k in ("input", "output", "cache_read")}


def test_unpinned_local_model_refused_outside_dev(tmp_path, family):
    cfg = load_config(CONFIG)
    cfg = cfg.model_copy(update={"model": cfg.model.model_copy(update={"pin": ModelPin()})})
    client = OpenAICompatClient(cfg.model.id, "http://127.0.0.1:9/v1", "k")
    with pytest.raises(ValueError, match="not pinned"):
        run_single_episode(cfg, Store(tmp_path), family.generate("queue_pop_front", "billing", 1), client,
                           split="sealed_familiar", sandbox_backend="local")


def _served(alias, deployment_id="dep-1"):
    return lambda n: {"x-litellm-model-group": alias, "x-litellm-model-name": "openai/wintermute-coder",
                      "x-litellm-model-id": deployment_id, "llm_provider-server": "llama.cpp"}


def test_served_identity_recorded_and_alias_checked(tmp_path, family, monkeypatch):
    monkeypatch.setenv("ELH_LOCK_DIR", str(tmp_path / "locks"))
    cfg = load_config(CONFIG)
    task = family.generate("queue_pop_front", "billing", 1)
    gw = _Gateway([[("submit", {"summary": "x"})]], headers=_served(cfg.model.id))
    store = Store(tmp_path)
    episode_id, _, _ = run_single_episode(cfg, store, task, OpenAICompatClient(cfg.model.id, gw.url, "k"),
                                          sandbox_backend="local")
    served = json.loads(store.db.execute("SELECT served_json FROM llm_calls WHERE episode_id = ?",
                                         (episode_id,)).fetchone()[0])
    assert served["alias"] == cfg.model.id and served["deployment_id"] == "dep-1"

    wrong = _Gateway([[("submit", {"summary": "x"})]], headers=_served("pilot-coder"))
    with pytest.raises(InfraError, match="served alias"):
        run_single_episode(cfg, store, task, OpenAICompatClient(cfg.model.id, wrong.url, "k"),
                           sandbox_backend="local")


def test_backend_swap_mid_episode_is_infra_error(tmp_path, family, monkeypatch):
    monkeypatch.setenv("ELH_LOCK_DIR", str(tmp_path / "locks"))
    cfg = load_config(CONFIG)
    gw = _Gateway([[("bash", {"command": "ls"})], [("submit", {"summary": "x"})]],
                  headers=lambda n: _served(cfg.model.id, f"dep-{n}")(n))
    with pytest.raises(InfraError, match="backend changed"):
        run_single_episode(cfg, Store(tmp_path), family.generate("queue_pop_front", "billing", 1),
                           OpenAICompatClient(cfg.model.id, gw.url, "k"), sandbox_backend="local")


def test_props_required_outside_dev(tmp_path, family, monkeypatch):
    cfg = load_config(CONFIG)
    pinned = cfg.model_copy(update={"model": cfg.model.model_copy(update={"props_url": "", "pin": cfg.model.pin.model_copy(update={
        "weights_sha256": "ab", "quantization": "q", "server": "s", "context_tokens": 1})})})
    monkeypatch.delenv("ELH_BACKEND_PROPS_URL", raising=False)
    with pytest.raises(ValueError, match="/props is required"):
        run_single_episode(pinned, Store(tmp_path), family.generate("queue_pop_front", "billing", 1),
                           OpenAICompatClient(cfg.model.id, "http://127.0.0.1:9/v1", "k"),
                           split="sealed_familiar", sandbox_backend="local")


def test_props_checked_against_pin():
    pin = load_config(CONFIG).model.pin
    good = {"model_path": "/srv/models/" + pin.weights_file, "build_info": "b1-4da6337",
            "default_generation_settings": {"n_ctx": pin.context_tokens}}
    assert pin.props_mismatches(good) == []
    swapped = {**good, "model_path": "/srv/models/other.gguf", "build_info": "b2-ffffff",
               "default_generation_settings": {"n_ctx": 32768}}
    assert len(pin.props_mismatches(swapped)) == 3
