"""Tests for the local tokenizing proxy."""

import pytest

from hostai.proxy import TokenizedProxy
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
