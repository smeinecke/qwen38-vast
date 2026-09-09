"""Unit tests for proxy helpers and TokenizedProxy internals."""

import asyncio
import json
from pathlib import Path
from unittest import mock

import aiohttp

from hostai import proxy, tls
from hostai.state import State

START_MARKER = proxy._THINK_START_MARKERS[0]
END_MARKER = proxy._THINK_END_MARKERS[0]


def test_find_marker():
    assert proxy._find_marker(f"foo{START_MARKER}bar", (START_MARKER,)) == 3
    assert proxy._find_marker("foo", (START_MARKER,)) == -1


def test_split_reasoning():
    r, c = proxy._split_reasoning(f"reason{END_MARKER}answer")
    assert r == "reason"
    assert c == "answer"
    r, c = proxy._split_reasoning("no marker")
    assert r == ""
    assert c == "no marker"


def test_find_stop():
    assert proxy._find_stop("hello world", ["world"]) == 6
    assert proxy._find_stop("hello", ["world"]) == -1
    assert proxy._find_stop("", ["x"]) == -1


def test_split_delta_with_reasoning():
    text = f"{START_MARKER}reason{END_MARKER}content"
    delta = proxy._split_delta(text, 0, len(text), expect_reasoning=True)
    assert delta["reasoning_content"] == "reason"
    assert delta["content"] == "content"


def test_split_delta_without_reasoning():
    delta = proxy._split_delta("hello world", 0, 5, expect_reasoning=False)
    assert delta["content"] == "hello"


def test_parse_tool_calls():
    content = '<tool_call>{"name": "function1", "arguments": {"x": 1}}</tool_call>'
    calls = proxy._parse_tool_calls(content)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "function1"


def test_parse_tool_calls_returns_empty():
    assert proxy._parse_tool_calls("no tools") == []


def test_map_finish_reason():
    assert proxy.TokenizedProxy._map_finish_reason({"stopped_limit": True}) == "length"
    assert proxy.TokenizedProxy._map_finish_reason({"stopped_eos": True}) == "stop"
    assert proxy.TokenizedProxy._map_finish_reason({"stopped_word": True}) == "stop"
    assert proxy.TokenizedProxy._map_finish_reason({}) is None


def test_build_completion_payload():
    body = {
        "top_p": 0.9,
        "min_p": 0.05,
        "n_probs": 10,
    }
    payload = proxy.TokenizedProxy.build_completion_payload([1, 2, 3], 10, 0.7, False, body)
    assert payload["prompt"] == [1, 2, 3]
    assert payload["n_predict"] == 10
    assert payload["return_tokens"] is True
    assert payload["top_p"] == 0.9


def test_resolve_upstream_tcp(project_dir):
    state = State(project_dir / ".hostai-vast" / "state.json")
    state.local_port = 18080
    state.unsecure = True
    up, sock = proxy._resolve_upstream(state)
    assert up == "http://127.0.0.1:18080"
    assert sock is None


def test_resolve_upstream_secure(project_dir):
    state = State(project_dir / ".hostai-vast" / "state.json")
    state.local_port = 18080
    state.unsecure = False
    up, sock = proxy._resolve_upstream(state)
    assert up == "https://127.0.0.1:18080"


def test_resolve_upstream_socket(project_dir):
    state = State(project_dir / ".hostai-vast" / "state.json")
    state.unsecure = True
    state.data["upstream_socket"] = "/tmp/s.sock"
    up, sock = proxy._resolve_upstream(state)
    assert up == "http://localhost"
    assert sock == "/tmp/s.sock"


def test_default_socket_path(project_dir):
    state = State(project_dir / ".hostai-vast" / "state.json")
    assert proxy._default_socket_path(state) == project_dir / ".hostai-vast" / "proxy.sock"


def test_ssl_context_unsecure(project_dir):
    state = State(project_dir / ".hostai-vast" / "state.json")
    state.unsecure = True
    state.tls_ca = None
    ctx = proxy._ssl_context(state)
    assert ctx.verify_mode == 0


def test_ssl_context_secure_no_ca(project_dir):
    state = State(project_dir / ".hostai-vast" / "state.json")
    state.unsecure = False
    state.tls_ca = None
    ctx = proxy._ssl_context(state)
    assert ctx is not None
    assert ctx.verify_mode == 0


def test_ssl_context_secure_with_ca(project_dir):
    state = State(project_dir / ".hostai-vast" / "state.json")
    tls_dir = project_dir / "tls"
    crt, _ = tls.generate_cert(tls_dir, common_name="test")
    state.unsecure = False
    state.tls_ca = tls_dir / "ca.crt"
    ctx = proxy._ssl_context(state)
    assert ctx is not None


def test_token_detokenizer_stops():
    tokenizer = mock.Mock()
    tokenizer.decode = mock.Mock(return_value="a stop")
    detok = proxy._TokenDetokenizer(tokenizer, stop_strings=["stop"], expect_reasoning=False)
    detok.add([1, 2, 3])
    assert detok.stopped is True


async def _run_complete_chat(response_data, decode_text="hello world"):
    state = State.__new__(State)
    state.__dict__.update({
        "local_port": 12345,
        "unsecure": True,
        "upstream_socket": None,
        "tls_ca": None,
        "api_key": "",
        "_data": {},
    })
    config = mock.Mock()
    config.proxy.tokenized_only = True
    config.model.model = "qwen"
    config.bench.max_tokens = 512
    config.bench.temperature = 0.6

    tokenizer = mock.Mock()
    tokenizer.decode = mock.Mock(return_value=decode_text)
    tokenizer.apply_chat_template = mock.Mock(return_value=[1, 2, 3])

    inst = proxy.TokenizedProxy(config, state, tokenizer, Path("/tmp/proxy.sock"), port=0)
    inst.ready = True

    response = mock.Mock(spec=aiohttp.ClientResponse)
    response.json = mock.AsyncMock(return_value=response_data)

    result = await inst._complete_chat(response, 3, [])
    return json.loads(result.text)


def test_complete_chat_tokens():
    data = asyncio.run(_run_complete_chat({
        "tokens": [10, 11, 12],
        "tokens_predicted": 3,
        "stop": True,
    }))
    assert data["object"] == "chat.completion"


def test_complete_chat_text_fallback():
    data = asyncio.run(_run_complete_chat({
        "content": "hello",
        "stop": True,
    }, decode_text=""))
    assert data["choices"][0]["message"]["content"] == "hello"
