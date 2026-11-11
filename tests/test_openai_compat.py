"""The self-hosted model path: format translation and a full episode against a stub gateway."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from elh.controller import run_single_episode
from elh.registry import ModelPin, load_config
from elh.runner.models import (DegenerateOutputError, InfraError, OpenAICompatClient, degenerate_reason, to_openai_messages, to_openai_tools)
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
                text_reply = calls if isinstance(calls, str) else None  # a str turn: a text-only reply that hits the limit
                calls = [] if text_reply else calls
                msg = {"role": "assistant", "content": text_reply or (None if calls else "done"),
                       "tool_calls": [{"id": f"call_{len(gw.requests)}_{i}", "type": "function",
                                       "function": {"name": n, "arguments": a if isinstance(a, str) else json.dumps(a)}}
                                      for i, (n, a) in enumerate(calls)] or None}
                resp = {"id": f"chatcmpl-{len(gw.requests)}", "object": "chat.completion", "created": 0,
                        "model": "qwen3-coder-30b-a3b",
                        "choices": [{"index": 0, "message": msg,
                                     "finish_reason": "tool_calls" if calls else "length" if text_reply else "stop"}],
                        "usage": {"prompt_tokens": 1200, "completion_tokens": 80, "total_tokens": 1280,
                                  "prompt_tokens_details": {"cached_tokens": 0 if body.get("cache_prompt") is False else 1000}}}
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


REAL_CODE = "".join(f"def f{i}(x):\n    return [y * {i} for y in x if y % 3 == 0]\n\n" for i in range(40))
BIG_TABLE = "| col_a | col_b | col_c |\n|-------|-------|-------|\n" + "".join(
    f"| {i:5d} | {'x' * 5} | {i * 7:5d} |\n" for i in range(60))


def test_degenerate_detector():
    assert degenerate_reason("?" * 8192, "max_tokens")
    assert degenerate_reason("?" * 8192)  # one character is enough, whatever the stop reason
    assert degenerate_reason("?" * 7600 + "ok go" * 20)  # 98% '?'
    assert degenerate_reason("ab" * 4000, "max_tokens")  # a short repeated pattern at the token limit
    assert degenerate_reason("ab" * 4000, "end_turn") is None  # ... only judged at the limit
    # Never trips: short replies, real code, a long table, plain prose, separators inside real text.
    assert degenerate_reason("?" * 150, "max_tokens") is None
    assert degenerate_reason("Done.") is None
    assert degenerate_reason(REAL_CODE, "max_tokens") is None
    assert len(BIG_TABLE) > 1000 and degenerate_reason(BIG_TABLE, "max_tokens") is None
    assert degenerate_reason("The quick brown fox jumps over the lazy dog. " * 100, "max_tokens") is None
    assert degenerate_reason("x = 1\n" + "#" * 100 + "\n" + REAL_CODE) is None


def test_client_raises_on_a_degenerate_reply(tmp_path, family, monkeypatch):
    monkeypatch.setenv("ELH_LOCK_DIR", str(tmp_path / "locks"))
    cfg = load_config(CONFIG)
    gw = _Gateway(["?" * 8192, "?" * 8192])
    client = OpenAICompatClient(cfg.model.id, gw.url, "k")
    with pytest.raises(DegenerateOutputError, match="degenerate reply") as exc:
        client.create(system="s", messages=[{"role": "user", "content": "hi"}], tools=TOOLS, max_tokens=8192)
    assert isinstance(exc.value, InfraError)
    # A legitimate long reply at the limit is returned as is.
    ok = OpenAICompatClient(cfg.model.id, _Gateway([REAL_CODE]).url, "k").create(
        system="s", messages=[{"role": "user", "content": "hi"}], tools=TOOLS, max_tokens=8192)
    assert ok.stop_reason == "max_tokens" and ok.content[0]["text"] == REAL_CODE


def _create(client):
    return client.create(system="s", messages=[{"role": "user", "content": "hi"}], tools=TOOLS, max_tokens=8192)


def test_degenerate_reply_is_retried_once_without_prompt_cache(capsys):
    gw = _Gateway(["?" * 8192, [("bash", {"command": "ls"})]])
    resp = _create(OpenAICompatClient(load_config(CONFIG).model.id, gw.url, "k"))
    assert resp.stop_reason == "tool_use" and resp.content[0]["name"] == "bash"
    assert "cache_prompt" not in gw.requests[0] and gw.requests[1]["cache_prompt"] is False
    assert {k: v for k, v in gw.requests[0].items() if k != "cache_prompt"} == \
        {k: v for k, v in gw.requests[1].items() if k != "cache_prompt"}
    cr = resp.cache_retry
    assert cr["retry_clean"] is True and cr["retry_cached_tokens"] == 0
    assert cr["first_output_tokens"] == 80 and cr["first_finish"] == "max_tokens" and cr["first_request_id"]
    assert "DEGENERATE reply" in capsys.readouterr().err


def test_degenerate_twice_raises_and_logs_the_retry(capsys):
    gw = _Gateway(["?" * 8192, "?" * 8192])
    with pytest.raises(DegenerateOutputError) as exc:
        _create(OpenAICompatClient(load_config(CONFIG).model.id, gw.url, "k"))
    assert len(gw.requests) == 2 and gw.requests[1]["cache_prompt"] is False
    assert exc.value.cache_retry["retry_clean"] is False
    assert "-> degenerate" in capsys.readouterr().err


def test_normal_reply_has_no_retry_and_no_extra_body():
    gw = _Gateway([[("bash", {"command": "ls"})]])
    resp = _create(OpenAICompatClient(load_config(CONFIG).model.id, gw.url, "k"))
    assert len(gw.requests) == 1 and "cache_prompt" not in gw.requests[0] and resp.cache_retry is None


def test_trace_response_event_carries_cache_retry(tmp_path, family, monkeypatch):
    monkeypatch.setenv("ELH_LOCK_DIR", str(tmp_path / "locks"))
    cfg = load_config(CONFIG)
    gw = _Gateway(["?" * 8192, [("submit", {"summary": "x"})]])
    _, result, _ = run_single_episode(cfg, Store(tmp_path), family.generate("queue_pop_front", "billing", 1),
                                      OpenAICompatClient(cfg.model.id, gw.url, "k"), sandbox_backend="local")
    resp = [e for e in result.trace if e["event"] == "response"]
    assert resp[0]["cache_retry"]["retry_clean"] is True
    assert result.calls[0]["cache_retry"] == resp[0]["cache_retry"]


def test_degenerate_reply_mid_episode_records_no_episode(tmp_path, family, monkeypatch):
    monkeypatch.setenv("ELH_LOCK_DIR", str(tmp_path / "locks"))
    cfg = load_config(CONFIG)
    gw = _Gateway([[("bash", {"command": "ls"})], "?" * 8192, "?" * 8192])
    store = Store(tmp_path)
    with pytest.raises(DegenerateOutputError):
        run_single_episode(cfg, store, family.generate("queue_pop_front", "billing", 1),
                           OpenAICompatClient(cfg.model.id, gw.url, "k"), sandbox_backend="local")
    assert store.db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 0
