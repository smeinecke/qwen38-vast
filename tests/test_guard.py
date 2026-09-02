"""Tests for the remote token-only guard."""

import pytest

from hostai.remote_guard import _is_prompt_tokenized


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ([1, 2, 3], True),
        ([], True),
        ("hello", False),
        (["hello"], False),
        ([1, "two"], False),
        ([1.0, 2.0], False),
        ([True, False], False),
        ([[1, 2], [3, 4]], False),
    ],
)
def test_is_prompt_tokenized(prompt, expected):
    assert _is_prompt_tokenized(prompt) is expected
