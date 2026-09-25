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
    content = (
        "<tool_call>\n<function=function1>\n"
        "<parameter=x>\n1\n</parameter>\n"
        "<parameter=y>\nhello\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls, rest = proxy._parse_tool_calls(content)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "function1"
    assert calls[0]["type"] == "function"
    assert calls[0]["index"] == 0
    assert json.loads(calls[0]["function"]["arguments"]) == {"x": "1", "y": "hello"}
    assert rest == ""


def test_parse_tool_calls_coerces_schema_types():
    tools = [{
        "type": "function",
        "function": {
            "name": "f",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "ratio": {"type": "number"},
                    "flag": {"type": "boolean"},
                    "items": {"type": "array"},
                },
            },
        },
    }]
    content = (
        "<tool_call>\n<function=f>\n"
        "<parameter=count>\n5\n</parameter>\n"
        "<parameter=ratio>\n0.5\n</parameter>\n"
        "<parameter=flag>\ntrue\n</parameter>\n"
        "<parameter=items>\n[1, 2]\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls, _ = proxy._parse_tool_calls(content, tools)
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "count": 5,
        "ratio": 0.5,
        "flag": True,
        "items": [1, 2],
    }


def test_parse_tool_calls_json_fallback():
    content = '<tool_call>{"name": "function1", "arguments": {"x": 1}}</tool_call>'
    calls, rest = proxy._parse_tool_calls(content)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "function1"
    assert json.loads(calls[0]["function"]["arguments"]) == {"x": 1}
    assert rest == ""


def test_parse_tool_calls_keeps_preamble_and_unparseable():
    good = "<tool_call>\n<function=f>\n</function>\n</tool_call>"
    bad = "<tool_call>not json</tool_call>"
    calls, rest = proxy._parse_tool_calls(f"let me help{good}trailing{bad}")
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "f"
    assert calls[0]["function"]["arguments"] == "{}"
    assert rest == f"let me helptrailing{bad}"


def test_parse_tool_calls_returns_empty():
    assert proxy._parse_tool_calls("no tools") == ([], "no tools")


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


def test_token_detokenizer_tool_calls():
    tokenizer = mock.Mock()
    parts = iter([
        "run this<tool_call>\n<function=bash>\n<parameter=command>\nls",
        "run this<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n"
        "</parameter>\n</function>\n</tool_call>",
    ])
    tokenizer.decode = mock.Mock(side_effect=lambda ids, **kw: next(parts))
    detok = proxy._TokenDetokenizer(tokenizer, expect_reasoning=False, tools=[{"function": {"name": "bash"}}])

    first = detok.add([1])
    # The opening tag boundary is held back; only the preamble is emitted.
    assert first == [{"content": "run this"}]

    second = detok.add([2])
    assert detok.saw_tool_calls is True
    assert len(second) == 1
    calls = second[0]["tool_calls"]
    assert calls[0]["index"] == 0
    assert calls[0]["function"]["name"] == "bash"
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls -la"}


def test_token_detokenizer_tool_calls_multiple_and_suffix():
    tokenizer = mock.Mock()
    block = (
        "<tool_call>\n<function=a>\n</function>\n</tool_call>"
        "<tool_call>\n<function=b>\n</function>\n</tool_call>"
    )
    tokenizer.decode = mock.Mock(return_value=block + "done")
    detok = proxy._TokenDetokenizer(
        tokenizer, expect_reasoning=False, tools=[{"function": {"name": "a"}}]
    )
    deltas = detok.add([1]) + detok.finish()
    names = [
        c["function"]["name"] for d in deltas for c in d.get("tool_calls", [])
    ]
    assert names == ["a", "b"]
    assert deltas[-1] == {"content": "done"}


def test_token_detokenizer_unterminated_tool_call_is_content():
    tokenizer = mock.Mock()
    tokenizer.decode = mock.Mock(return_value="text<tool_call>\n<function=x>")
    detok = proxy._TokenDetokenizer(
        tokenizer, expect_reasoning=False, tools=[{"function": {"name": "x"}}]
    )
    detok.add([1])
    tail = detok.finish()
    assert detok.saw_tool_calls is False
    assert tail == [{"content": "<tool_call>\n<function=x>"}]


def test_token_detokenizer_partial_open_tag_held_back():
    tokenizer = mock.Mock()
    parts = iter(["abc<tool_", "abc<tool_call>\n<function=f>\n</function>\n</tool_call>"])
    tokenizer.decode = mock.Mock(side_effect=lambda ids, **kw: next(parts))
    detok = proxy._TokenDetokenizer(
        tokenizer, expect_reasoning=False, tools=[{"function": {"name": "f"}}]
    )
    # The "<tool_" prefix is inside the holdback window, so nothing leaks.
    assert detok.add([1]) == []
    tail = detok.finish()
    assert tail[0] == {"content": "abc"}
    assert tail[1]["tool_calls"][0]["function"]["name"] == "f"


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
    config.proxy.log_content = False
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
