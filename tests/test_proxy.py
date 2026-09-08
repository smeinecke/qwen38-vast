"""Tests for the local tokenizing proxy."""

import json

import pytest

from hostai.proxy import (
    TokenizedProxy,
    _find_stop,
    _parse_tool_calls,
    _split_reasoning,
    _TokenDetokenizer,
)
from hostai.tokenize import default_reasoning_kwargs


@pytest.mark.parametrize(
    ("upstream", "expected"),
    [
        ({"stopped_eos": True}, "stop"),
        ({"stopped_word": True}, "stop"),
        ({"stopped_limit": True}, "length"),
        ({"stop": True}, "stop"),
        ({}, None),
    ],
)
def test_map_finish_reason(upstream, expected):
    assert TokenizedProxy._map_finish_reason(upstream) == expected


def test_default_reasoning_kwargs(config):
    kwargs = default_reasoning_kwargs(config)
    assert kwargs["enable_thinking"] is True
    assert kwargs["preserve_thinking"] is True
    assert kwargs["reasoning_effort"] == config.model.reasoning_effort


def test_build_completion_payload():
    body = {
        "temperature": 0.5,
        "top_p": 0.9,
        "frequency_penalty": 0.2,
        "seed": 42,
        "stop": ["\n"],
        "extra_ignored": "value",
    }
    payload = TokenizedProxy.build_completion_payload([1, 2, 3], 64, 0.7, False, body)
    assert payload["prompt"] == [1, 2, 3]
    assert payload["n_predict"] == 64
    assert payload["temperature"] == 0.7
    assert payload["stream"] is False
    assert payload["return_tokens"] is True
    assert payload["token_only"] is True
    assert payload["top_p"] == 0.9
    assert payload["frequency_penalty"] == 0.2
    assert payload["seed"] == 42
    assert payload["stop"] == ["\n"]
    assert "extra_ignored" not in payload


def test_parse_tool_calls_extracts_openai_format():
    content = '<tool_call>{"name": "get_weather", "arguments": {"city": "Berlin"}}</tool_call>'
    calls = _parse_tool_calls(content)
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Berlin"}


def test_parse_tool_calls_with_string_arguments():
    content = '<tool_call>{"name": "search", "arguments": "{\\"q\\":\\"cats\\"}"}</tool_call>'
    calls = _parse_tool_calls(content)
    assert calls[0]["function"]["arguments"] == '{"q":"cats"}'


def test_parse_tool_calls_returns_empty_for_plain_text():
    assert _parse_tool_calls("hello world") == []


def test_parse_tool_calls_with_multiple_calls():
    content = (
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Berlin"}}</tool_call>'
        '<tool_call>{"name": "get_time", "arguments": {"timezone": "CET"}}</tool_call>'
    )
    calls = _parse_tool_calls(content)
    assert len(calls) == 2
    assert calls[0]["function"]["name"] == "get_weather"
    assert calls[1]["function"]["name"] == "get_time"


def test_parse_tool_calls_ignores_invalid_json():
    content = '<tool_call>not json</tool_call><tool_call>{"name":"x"}</tool_call>'
    calls = _parse_tool_calls(content)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "x"


def test_split_reasoning_strips_orphan_closing_tag():
    reasoning, answer = _split_reasoning("We need answer simple arithmetic.</think>\n\n4")
    assert reasoning == "We need answer simple arithmetic."
    assert answer == "4"


def test_split_reasoning_strips_both_markers():
    reasoning, answer = _split_reasoning("<think>thinking</think>\n\nanswer")
    assert reasoning == "thinking"
    assert answer == "answer"


def test_split_reasoning_accepts_legacy_arrow_marker():
    reasoning, answer = _split_reasoning("\u25b6thinking\u25c0\n\nanswer")
    assert reasoning == "thinking"
    assert answer == "answer"


def test_split_reasoning_no_marker_returns_content():
    reasoning, answer = _split_reasoning("plain answer")
    assert reasoning == ""
    assert answer == "plain answer"


class _FakeTokenizer:
    """Decode stub mapping each token id to a fixed piece."""

    def __init__(self, pieces):
        self._pieces = pieces

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(self._pieces.get(i, "") for i in token_ids)


def test_find_stop_returns_earliest_match():
    assert _find_stop("hello STOP world END", ["END", "STOP"]) == 6
    assert _find_stop("nothing here", ["x"]) == -1
    assert _find_stop("abc", []) == -1


def test_detokenizer_emits_content_incrementally():
    tok = _FakeTokenizer({1: "Hello", 2: " ", 3: "world"})
    detok = _TokenDetokenizer(tok, expect_reasoning=False)
    assert detok.add([1]) == {"content": "Hello"}
    assert detok.add([2, 3]) == {"content": " world"}
    assert detok.finish() == {}


def test_detokenizer_splits_reasoning_marker():
    tok = _FakeTokenizer({1: "think ", 2: "hard", 3: "</think>", 4: "answer"})
    detok = _TokenDetokenizer(tok, expect_reasoning=True)
    # The marker holdback keeps the last len("</think>")-1 chars pending.
    assert detok.add([1, 2]) == {"reasoning_content": "thi"}
    assert detok.add([3, 4]) == {"reasoning_content": "nk hard"}
    assert detok.finish() == {"content": "answer"}


def test_detokenizer_straddling_delta_splits_both_keys():
    tok = _FakeTokenizer({1: "think", 2: "</think>ans", 3: "wer"})
    detok = _TokenDetokenizer(tok, expect_reasoning=True)
    assert detok.add([1, 2, 3]) == {"reasoning_content": "think"}
    assert detok.finish() == {"content": "answer"}


def test_detokenizer_marker_straddling_decode_boundary():
    # "…</thi" then "nk>ans" - the 7-char marker holdback keeps the partial
    # marker from being emitted as reasoning.
    tok = _FakeTokenizer({1: "wor", 2: "ds</thi", 3: "nk>", 4: "ans"})
    detok = _TokenDetokenizer(tok, expect_reasoning=True)
    assert detok.add([1, 2]) == {"reasoning_content": "wor"}
    assert detok.add([3, 4]) == {"reasoning_content": "ds"}
    assert detok.finish() == {"content": "ans"}


def test_detokenizer_no_marker_all_reasoning_when_expected():
    tok = _FakeTokenizer({1: "think", 2: "ing"})
    detok = _TokenDetokenizer(tok, expect_reasoning=True)
    # "thinking" is 7 chars; the marker holdback keeps it pending until finish.
    assert detok.add([1, 2]) == {"reasoning_content": "t"}
    assert detok.finish() == {"reasoning_content": "hinking"}


def test_detokenizer_stop_string_truncates():
    tok = _FakeTokenizer({1: "alpha ", 2: "STOP", 3: " beta"})
    detok = _TokenDetokenizer(tok, stop_strings=["STOP"], expect_reasoning=False)
    delta = detok.add([1, 2, 3])
    assert delta == {"content": "alpha "}
    assert detok.stopped is True
    # further tokens emit nothing
    assert detok.add([4]) == {}


def test_detokenizer_stop_holds_back_boundary_prefix():
    # "STOP" split across decode boundary: "ST" then "OP more" - with a
    # 4-char stop the last 3 decoded chars are held back until the match
    # resolves.
    tok = _FakeTokenizer({1: "say ST", 2: "OP more", 3: " tail"})
    detok = _TokenDetokenizer(tok, stop_strings=["STOP"], expect_reasoning=False)
    # len("say ST")=6, holdback=3 -> emits "say"; " ST" is held back.
    assert detok.add([1]) == {"content": "say"}
    # Decoded "say STOP more": stop matches at 4, text truncates to "say ".
    delta = detok.add([2])
    assert delta == {"content": " "}
    assert detok.stopped is True


def test_detokenizer_finish_flushes_holdback():
    tok = _FakeTokenizer({1: "abc", 2: "def"})
    detok = _TokenDetokenizer(tok, stop_strings=["ZZZZ"], expect_reasoning=False)
    assert detok.add([1, 2]) == {"content": "abc"}
    assert detok.finish() == {"content": "def"}


def test_detokenizer_strips_leading_open_marker():
    tok = _FakeTokenizer({1: "<think>think", 2: "</think>", 3: "ok"})
    detok = _TokenDetokenizer(tok, expect_reasoning=True)
    assert detok.add([1, 2, 3]) == {"reasoning_content": "think"}
    assert detok.finish() == {"content": "ok"}

