"""Tests for the local tokenizing proxy."""

import json

import pytest

from hostai.proxy import TokenizedProxy, _parse_tool_calls
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
