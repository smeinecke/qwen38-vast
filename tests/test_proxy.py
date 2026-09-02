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
