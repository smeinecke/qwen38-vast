"""Tests for hostai.tokenize with real tokenizer fixtures."""

import json
from pathlib import Path
from unittest import mock

import pytest

from hostai.config import ProxySection
from hostai.tokenize import (
    Tokenizer,
    TokenizerError,
    _normalize_tool_call_arguments,
    default_reasoning_kwargs,
)


def _fixture() -> dict:
    return json.loads((Path(__file__).parent / "fixtures" / "tokenizer_golden.json").read_text())


def _skip_if_no_transformers() -> bool:
    try:
        from transformers import AutoTokenizer  # noqa: F401

        return False
    except Exception:  # pragma: no cover
        return True


@pytest.fixture
def tokenizer(config, project_dir):
    # Keep tokenizer cache inside the test project so it is isolated.
    config.proxy = ProxySection(
        tokenized_only=False,
        socket_path="",
        port=0,
        tokenizer_model="Qwen/Qwen3.8-27B",
        tokenizer_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        cache_dir=str(project_dir / ".hostai-cache" / "tokenizer"),
    )
    return Tokenizer(config)


@pytest.mark.skipif(_skip_if_no_transformers(), reason="transformers not available")
@pytest.mark.slow(reason="loads Qwen/Qwen3.8-27B tokenizer from Hugging Face cache")
def test_tokenizer_uses_pinned_revision(tokenizer, project_dir):
    """The tokenizer must be loaded at the configured pinned revision."""
    fixture = _fixture()
    with mock.patch("transformers.AutoTokenizer") as AutoTokenizer:
        tok = Tokenizer(tokenizer._config)
        tok._load()
    assert AutoTokenizer.from_pretrained.call_args.kwargs.get("revision") == fixture["revision"]


@pytest.mark.skipif(_skip_if_no_transformers(), reason="transformers not available")
@pytest.mark.slow(reason="loads Qwen/Qwen3.8-27B tokenizer from Hugging Face cache")
def test_tokenizer_special_tokens(tokenizer):
    """The tokenizer must expose the expected Qwen3.8 special token IDs."""
    fixture = _fixture()
    specials = tokenizer.special_tokens()
    expected = fixture["special_tokens"]
    assert specials["eos_token_id"] == expected["eos_token_id"]
    assert specials["pad_token_id"] == expected["pad_token_id"]
    assert specials["im_start"] == expected["im_start"]
    assert specials["im_end"] == expected["im_end"]
    assert specials["tool_call"] == expected["tool_call"]
    assert specials["end_tool"] == expected["end_tool"]
    assert specials["tool_response"] == expected["tool_response"]
    assert specials["end_tool_response"] == expected["end_tool_response"]
    assert specials["sep"] == expected["sep"]


@pytest.mark.skipif(_skip_if_no_transformers(), reason="transformers not available")
@pytest.mark.slow(reason="loads Qwen/Qwen3.8-27B tokenizer from Hugging Face cache")
def test_tokenizer_golden_xhigh(tokenizer):
    """Tokenized output must match the pinned-revision golden fixtures."""
    fixture = _fixture()
    for name, messages in [
        ("simple_user", [{"role": "user", "content": "hello"}]),
        (
            "system_user",
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
            ],
        ),
        (
            "multi_turn",
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "how are you"},
            ],
        ),
    ]:
        expected = fixture["effort_xhigh"][name]
        encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        assert encoded == expected, f"{name} tokenization mismatch"


@pytest.mark.skipif(_skip_if_no_transformers(), reason="transformers not available")
@pytest.mark.slow(reason="loads Qwen/Qwen3.8-27B tokenizer from Hugging Face cache")
def test_tokenizer_reasoning_effort_changes_length(tokenizer):
    """Different reasoning_effort values must produce different token counts."""
    messages = [{"role": "user", "content": "explain"}]
    low = tokenizer.apply_chat_template(messages, add_generation_prompt=True, reasoning_effort="low")
    medium = tokenizer.apply_chat_template(messages, add_generation_prompt=True, reasoning_effort="medium")
    xhigh = tokenizer.apply_chat_template(messages, add_generation_prompt=True, reasoning_effort="xhigh")
    assert len({len(low), len(medium), len(xhigh)}) == 3


def test_normalize_tool_call_arguments_parses_json_strings():
    """OpenAI string-encoded arguments must become mappings for |items."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                },
                {
                    "id": "2",
                    "type": "function",
                    "function": {"name": "noop", "arguments": "not json"},
                },
                "weird",
            ],
        },
        {"role": "user", "content": "next"},
    ]
    out = _normalize_tool_call_arguments(messages)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
    # Invalid JSON and non-dict calls are left untouched.
    assert out[0]["tool_calls"][1]["function"]["arguments"] == "not json"
    assert out[0]["tool_calls"][2] == "weird"
    # The caller's messages are not mutated.
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"command": "ls"}'
    # Messages without tool_calls keep their identity.
    assert out[1] is messages[1]


def test_normalize_tool_call_arguments_top_level():
    """Templates without a function wrapper read arguments at top level."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "bash", "arguments": '{"command": "ls"}'}],
        }
    ]
    out = _normalize_tool_call_arguments(messages)
    assert out[0]["tool_calls"][0]["arguments"] == {"command": "ls"}


@pytest.mark.skipif(_skip_if_no_transformers(), reason="transformers not available")
@pytest.mark.slow(reason="loads Qwen/Qwen3.8-27B tokenizer from Hugging Face cache")
def test_tokenizer_tool_calls_with_string_arguments(tokenizer):
    """OpenAI-style string arguments must not fail the chat template."""
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "x",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "x", "content": "out"},
        {"role": "user", "content": "next"},
    ]
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    assert encoded


def test_default_reasoning_kwargs_validates_effort(config):
    config.model.reasoning_effort = "low"
    assert default_reasoning_kwargs(config)["reasoning_effort"] == "low"

    config.model.reasoning_effort = "medium"
    assert default_reasoning_kwargs(config)["reasoning_effort"] == "medium"

    config.model.reasoning_effort = "xhigh"
    assert default_reasoning_kwargs(config)["reasoning_effort"] == "xhigh"

    config.model.reasoning_effort = "high"
    with pytest.raises(TokenizerError, match="invalid reasoning_effort"):
        default_reasoning_kwargs(config)


def test_tokenizer_error_propagates_chat_template_failure(config, project_dir):
    """A failing chat template must raise TokenizerError."""
    config.proxy = ProxySection(
        tokenized_only=False,
        socket_path="",
        port=0,
        tokenizer_model="Qwen/Qwen3.8-27B",
        tokenizer_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        cache_dir=str(project_dir / ".hostai-cache" / "tokenizer"),
    )
    tokenizer = Tokenizer(config)
    # The model needs at least one user query.
    with pytest.raises(TokenizerError):
        tokenizer.apply_chat_template([{"role": "assistant", "content": "no user"}])


def test_tokenizer_rejects_image_url_input(config, project_dir):
    """image_url content must be rejected before tokenization."""
    config.proxy = ProxySection(
        tokenized_only=False,
        socket_path="",
        port=0,
        tokenizer_model="Qwen/Qwen3.8-27B",
        tokenizer_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        cache_dir=str(project_dir / ".hostai-cache" / "tokenizer"),
    )
    tokenizer = Tokenizer(config)
    with pytest.raises(TokenizerError, match="multimodal input not supported"):
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "http://example.com/x.png"}},
                    ],
                }
            ]
        )


def test_tokenizer_rejects_audio_input(config, project_dir):
    config.proxy = ProxySection(
        tokenized_only=False,
        socket_path="",
        port=0,
        tokenizer_model="Qwen/Qwen3.8-27B",
        tokenizer_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        cache_dir=str(project_dir / ".hostai-cache" / "tokenizer"),
    )
    tokenizer = Tokenizer(config)
    with pytest.raises(TokenizerError, match="multimodal input not supported"):
        tokenizer.apply_chat_template([{"role": "user", "content": [{"type": "audio", "audio_url": "x.mp3"}]}])
