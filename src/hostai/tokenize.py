"""Client-side tokenizer and chat-template handler for the proxy."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from hostai.config import Config
from hostai.state import cache_dir

_logger = logging.getLogger(__name__)


def _tokenizer_cache(config: Config) -> Path:
    """Return the local directory where the tokenizer files are cached."""
    if config.proxy.cache_dir:
        return Path(config.proxy.cache_dir)
    return cache_dir(config.root_dir) / "tokenizer"


class TokenizerError(RuntimeError):
    """The local tokenizer cannot be loaded or the chat template cannot be applied."""


# Content types that the tokenized proxy does not support.  These map to the
# common OpenAI message keys for multimodal payloads.
_MULTIMODAL_KEYS = {
    "image_url",
    "image",
    "video",
    "video_url",
    "audio",
    "audio_url",
    "file",
    "file_url",
    "document",
    "document_url",
}


def _content_is_multimodal(content: Any) -> Optional[str]:
    """Return the first multimodal content key found, or None."""
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                # The OpenAI message format sets "type" to "image_url" etc.
                item_type = item.get("type")
                if item_type in _MULTIMODAL_KEYS:
                    return item_type
                for key in _MULTIMODAL_KEYS:
                    if key in item:
                        return key
    return None


def _has_multimodal_input(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Detect unsupported image/video/audio/file inputs in a conversation."""
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        multimodal = _content_is_multimodal(content)
        if multimodal:
            return multimodal
    return None


class Tokenizer:
    """Lazy-loaded Qwen tokenizer with an optional remote chat template."""

    def __init__(
        self,
        config: Config,
        chat_template: Optional[str] = None,
    ) -> None:
        self._config = config
        self._chat_template = chat_template
        self._tokenizer: Optional[Any] = None

    def _load(self) -> Any:
        """Load and cache the tokenizer from the configured model ID."""
        if self._tokenizer is not None:
            return self._tokenizer

        cache_path = _tokenizer_cache(self._config)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_path.chmod(0o700)

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise TokenizerError(
                "transformers is required for client-side tokenization. "
                "Install it with 'uv sync' or 'pip install transformers'."
            ) from exc

        model_id = self._config.proxy.tokenizer_model or "Qwen/Qwen3.8-27B"
        revision = self._config.proxy.tokenizer_revision
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=str(cache_path),
                local_files_only=False,
                trust_remote_code=False,
                revision=revision or None,
            )
        except Exception as exc:
            raise TokenizerError(f"failed to load tokenizer for {model_id}@{revision}: {exc}") from exc

        if self._chat_template:
            tokenizer.chat_template = self._chat_template

        self._tokenizer = tokenizer
        return tokenizer

    def apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> List[int]:
        """Apply the model's chat template and return token IDs."""
        multimodal = _has_multimodal_input(messages, tools)
        if multimodal:
            raise TokenizerError(f"multimodal input not supported: {multimodal}")

        tokenizer = self._load()

        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                return_dict=False,
                return_tensors=None,
                **kwargs,
            )
        except Exception as exc:
            raise TokenizerError(f"chat template failed: {exc}") from exc

        if not isinstance(encoded, list):
            # Some tokenizers return a torch/NumPy tensor depending on options.
            try:
                encoded = encoded.tolist()
            except AttributeError as exc:
                raise TokenizerError(f"unexpected tokenizer output type: {type(encoded)}") from exc

        return [int(t) for t in encoded]

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode a list of token IDs back to a string."""
        tokenizer = self._load()
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def special_tokens(self) -> Dict[str, Optional[int]]:
        """Return a map of important special token IDs.

        Unknown or missing special tokens are returned as ``None`` so callers
        can decide whether to fail or fall back.
        """
        tokenizer = self._load()
        out: Dict[str, Optional[int]] = {
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "im_start": tokenizer.convert_tokens_to_ids("<|im_start|>"),
            "im_end": tokenizer.convert_tokens_to_ids("<|im_end|>"),
            "tool_call": tokenizer.convert_tokens_to_ids("<tool_call>"),
            "end_tool": tokenizer.convert_tokens_to_ids("</tool_call>"),
            "tool_response": tokenizer.convert_tokens_to_ids("<tool_response>"),
            "end_tool_response": tokenizer.convert_tokens_to_ids("</tool_response>"),
            "sep": tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        }
        return out


def default_reasoning_kwargs(config: Config) -> Dict[str, Any]:
    """Return Qwen3.8 reasoning kwargs that mirror the server defaults."""
    effort = (config.model.reasoning_effort or "xhigh").strip().lower()
    valid = {"low", "medium", "xhigh"}
    if effort not in valid:
        raise TokenizerError(
            f"invalid reasoning_effort {effort!r}; must be one of {sorted(valid)}"
        )
    return {
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_effort": effort,
    }
