"""Local OpenAI-compatible proxy that tokenizes prompts before forwarding."""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import re
import ssl
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import web

from hostai import ssh
from hostai.commands import _common
from hostai.config import Config
from hostai.state import State
from hostai.tokenize import Tokenizer, TokenizerError, default_reasoning_kwargs

_logger = logging.getLogger(__name__)


# Patterns for tool-call output produced by the Qwen tool-use template.  It
# wraps calls between <tool_call> and </tool_call> tags, each containing a
# <function=name> block with <parameter=key>value</parameter> entries.
_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"
_TOOL_CALL_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>",
    re.DOTALL,
)
_TOOL_FUNCTION_RE = re.compile(
    r"<function=(?P<name>[^>\n]+)>(?P<body>.*?)</function>",
    re.DOTALL,
)
_TOOL_PARAM_RE = re.compile(
    r"<parameter=(?P<name>[^>\n]+)>(?P<value>.*?)</parameter>",
    re.DOTALL,
)

# The Qwen chat template puts the opening thinking marker into the prompt, so
# generated output normally contains only the closing tag. Split on it. Both
# the canonical ``</think>`` and the ``◀`` variant are accepted.
_THINK_START_MARKERS = ("<think>", "▶")
_THINK_END_MARKERS = ("</think>", "◀")


def _find_marker(text: str, markers: Tuple[str, ...]) -> int:
    """Return the index of the earliest marker occurrence, or -1."""
    pos = -1
    for marker in markers:
        i = text.find(marker)
        if i >= 0 and (pos < 0 or i < pos):
            pos = i
    return pos


def _split_reasoning(content: str) -> Tuple[str, str]:
    """Split raw completion output into (reasoning, answer).

    The opening thinking marker lives in the prompt, so the model output
    typically contains only the closing tag. If the model also emitted the
    opening tag, it is stripped from the reasoning text.
    """
    end = _find_marker(content, _THINK_END_MARKERS)
    if end < 0:
        return "", content
    reasoning = content[:end]
    answer = content[end:]
    for marker in _THINK_END_MARKERS:
        if answer.startswith(marker):
            answer = answer[len(marker) :]
            break
    start = _find_marker(reasoning, _THINK_START_MARKERS)
    if start >= 0:
        marker = next(m for m in _THINK_START_MARKERS if reasoning.startswith(m, start))
        reasoning = reasoning[start + len(marker) :]
    return reasoning.strip(), answer.lstrip()


def _tool_param_types(tools: Optional[List[Dict[str, Any]]], name: str) -> Dict[str, str]:
    """Return the declared parameter types for tool ``name``."""
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict) or fn.get("name") != name:
            continue
        params = fn.get("parameters")
        props = params.get("properties") if isinstance(params, dict) else None
        if not isinstance(props, dict):
            return {}
        return {
            key: value["type"]
            for key, value in props.items()
            if isinstance(value, dict) and isinstance(value.get("type"), str)
        }
    return {}


def _coerce_arg(value: str, schema_type: Optional[str]) -> Any:
    """Coerce a raw ``<parameter>`` string to the declared schema type.

    The XML tool-call format carries only strings; llama.cpp converts them
    back to JSON values using the tool schema, so the proxy does the same.
    Without a declared type only JSON containers (``{``/``[``) are decoded —
    scalars stay strings to avoid mangling text like ``"true"``.
    """
    text = value.strip()
    if schema_type == "integer":
        try:
            return int(text)
        except ValueError:
            return text
    if schema_type == "number":
        try:
            return float(text)
        except ValueError:
            return text
    if schema_type == "boolean":
        lowered = text.lower()
        if lowered in ("true", "1"):
            return True
        if lowered in ("false", "0"):
            return False
        return text
    if schema_type in ("array", "object") or (schema_type is None and text[:1] in ("{", "[")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return text


def _parse_tool_call_block(
    block: str,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Parse a single ``<tool_call>...</tool_call>`` block into an OpenAI call.

    Handles the Hermes-style ``<function=name><parameter=key>`` markup the
    chat template emits, with a fallback for JSON-in-tags output. Returns
    ``None`` when the block cannot be understood.
    """
    match = _TOOL_CALL_RE.search(block)
    if not match:
        return None
    inner = match.group(1).strip()
    if not inner:
        return None

    function = _TOOL_FUNCTION_RE.search(inner)
    if function:
        name = function.group("name").strip()
        if not name:
            return None
        types = _tool_param_types(tools, name)
        arguments = {
            param.group("name").strip(): _coerce_arg(param.group("value"), types.get(param.group("name").strip()))
            for param in _TOOL_PARAM_RE.finditer(function.group("body"))
        }
    else:
        try:
            parsed = json.loads(inner)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        name = parsed.get("name")
        if not name:
            return None
        arguments = parsed.get("arguments", {})

    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)

    return {
        "id": f"call_{os.urandom(8).hex()}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def _parse_tool_calls(
    content: str,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Parse ``<tool_call>...</tool_call>`` output into OpenAI tool_calls.

    Returns ``(tool_calls, remainder)`` where ``remainder`` is the content
    with every successfully parsed block removed (the template instructs the
    model to put optional natural-language reasoning before the call).
    Unparseable blocks stay in the remainder.
    """
    tool_calls: List[Dict[str, Any]] = []
    spans: List[Tuple[int, int]] = []
    for match in _TOOL_CALL_RE.finditer(content):
        call = _parse_tool_call_block(match.group(0), tools)
        if call is None:
            continue
        call["index"] = len(tool_calls)
        tool_calls.append(call)
        spans.append(match.span())
    remainder = content
    for start, end in reversed(spans):
        remainder = remainder[:start] + remainder[end:]
    return tool_calls, remainder.strip()


def _find_stop(text: str, stops: List[str]) -> int:
    """Return the index of the earliest stop-string match, or -1."""
    cut = -1
    for word in stops:
        pos = text.find(word)
        if pos >= 0 and (cut < 0 or pos < cut):
            cut = pos
    return cut


def _split_delta(text: str, start: int, end: int, expect_reasoning: bool) -> Dict[str, str]:
    """Split the new range ``text[start:end]`` into reasoning/content delta keys.

    The closing thinking marker (when ``expect_reasoning`` is set) separates
    reasoning from the final answer: everything before it is reasoning,
    everything after is content. Without an expected marker, or before it
    appears, text is attributed to the current phase.
    """
    if start >= end:
        return {}
    marker = _find_marker(text, _THINK_END_MARKERS) if expect_reasoning else -1
    marker_len = 0
    if marker >= 0:
        marker_len = len(next(m for m in _THINK_END_MARKERS if text.startswith(m, marker)))
    delta: Dict[str, str] = {}
    if marker < 0:
        delta["reasoning_content" if expect_reasoning else "content"] = text[start:end]
        return delta
    reasoning = text[start : min(end, marker)]
    content = text[max(start, marker + marker_len) : end]
    if reasoning:
        if start == 0:
            for m in _THINK_START_MARKERS:
                if reasoning.startswith(m):
                    reasoning = reasoning[len(m) :]
                    break
        if reasoning:
            delta["reasoning_content"] = reasoning
    if content:
        delta["content"] = content
    return delta


class _TokenDetokenizer:
    """Incrementally detokenize generated token ids into response deltas.

    With ``token_only`` upstream requests, the server emits raw token ids
    instead of text. The proxy decodes them locally, splits reasoning from
    content on ``◀``, parses ``<tool_call>`` markup into OpenAI tool_calls
    deltas, and applies ``stop`` strings itself since server-side stop
    matching is text-based and cannot run without detokenization.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        stop_strings: Optional[List[str]] = None,
        expect_reasoning: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._stops = [s for s in stop_strings or [] if s]
        self._expect_reasoning = expect_reasoning
        # Tool-call markup is only parsed when the request declared tools;
        # without them the model has nothing to call and the tags stay text.
        self._tools = tools
        self._parse_tools = bool(tools)
        self._ids: List[int] = []
        self._emitted = 0
        self._cut = -1
        self.stopped = False
        self.saw_tool_calls = False
        self._tool_index = 0
        # Hold back trailing chars so a stop string, reasoning marker, or
        # <tool_call> opening tag that straddles a decode boundary can still
        # be detected before its prefix is emitted as content.
        boundary_lengths = [len(s) for s in self._stops]
        if expect_reasoning:
            boundary_lengths += [len(m) for m in _THINK_END_MARKERS]
        if self._parse_tools:
            boundary_lengths.append(len(_TOOL_OPEN))
        self._holdback = max(0, max(boundary_lengths, default=0) - 1)

    def add(self, token_ids: List[int]) -> List[Dict[str, Any]]:
        """Append generated token ids and return the new delta payloads."""
        self._ids.extend(token_ids)
        return self._drain(final=False)

    def finish(self) -> List[Dict[str, Any]]:
        """Flush any held-back text at the end of the stream."""
        return self._drain(final=True)

    @property
    def token_count(self) -> int:
        return len(self._ids)

    def _drain(self, final: bool) -> List[Dict[str, Any]]:
        text = self._tokenizer.decode(self._ids, skip_special_tokens=True)
        if self._cut < 0:
            self._cut = _find_stop(text, self._stops)
            if self._cut >= 0:
                self.stopped = True
        if self._cut >= 0:
            text = text[: self._cut]
        limit = len(text)
        if not final and self._cut < 0:
            limit = max(0, limit - self._holdback)

        deltas: List[Dict[str, Any]] = []
        while self._emitted < len(text):
            open_pos = text.find(_TOOL_OPEN, self._emitted) if self._parse_tools else -1
            seg_end = open_pos if open_pos >= 0 else len(text)
            seg_end = min(seg_end, limit)
            if seg_end > self._emitted:
                delta = _split_delta(text, self._emitted, seg_end, self._expect_reasoning)
                if delta:
                    deltas.append(delta)
                self._emitted = seg_end
            if open_pos < 0 or self._emitted != open_pos:
                break
            close = text.find(_TOOL_CLOSE, open_pos)
            if close < 0:
                # Incomplete block: wait for more tokens, or emit the markup
                # verbatim at end of stream.
                if final:
                    delta = _split_delta(text, self._emitted, len(text), self._expect_reasoning)
                    if delta:
                        deltas.append(delta)
                    self._emitted = len(text)
                break
            end = close + len(_TOOL_CLOSE)
            call = _parse_tool_call_block(text[open_pos:end], self._tools)
            if call is None:
                deltas.append({"content": text[open_pos:end]})
            else:
                call["index"] = self._tool_index
                self._tool_index += 1
                self.saw_tool_calls = True
                deltas.append({"tool_calls": [call]})
            self._emitted = end
        return deltas


def _one_line(exc: BaseException) -> str:
    """Flatten an exception to a single line safe for an HTTP reason phrase."""
    return " ".join(str(exc).split())[:200] or type(exc).__name__


class ProxyError(Exception):
    """The tokenized proxy cannot handle a request."""


def _default_socket_path(state: State) -> Path:
    return state.state_file.parent / "proxy.sock"


def _resolve_upstream(state: State) -> Tuple[str, Optional[str]]:
    """Return (upstream_base_url, upstream_unix_socket_path).

    The base URL is used as the host/authority in HTTP requests; when the
    upstream is a Unix domain socket the connector sends the request there
    instead of resolving the hostname.
    """
    unix_socket = state.data.get("upstream_socket") or ""
    if state.unsecure:
        if unix_socket:
            return "http://localhost", unix_socket
        return f"http://127.0.0.1:{state.local_port}", None
    if unix_socket:
        return "https://localhost", unix_socket
    return f"https://127.0.0.1:{state.local_port}", None


def _ssl_context(state: State) -> ssl.SSLContext:
    if state.unsecure or not state.tls_ca or not state.tls_ca.exists():
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.load_verify_locations(str(state.tls_ca))
    return ctx


class UnixTLSConnector(aiohttp.UnixConnector):
    """Unix domain socket connector that supports TLS over the socket.

    ``aiohttp.UnixConnector`` ignores the ``ssl`` argument. This subclass
    creates the Unix connection with ``ssl=`` so ``https`` requests can be
    forwarded to a TLS-speaking backend over a local Unix socket.
    """

    def __init__(
        self,
        path: str,
        *,
        ssl: ssl.SSLContext | bool | None = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(path=path, **kwargs)
        self._ssl = ssl

    def _get_ssl_context(self, req: Any) -> Optional[ssl.SSLContext]:
        """Return the SSL context to use for a request, mirroring TCPConnector."""
        if not req.is_ssl():
            return None

        # Request-level SSL override takes precedence.
        ctx = req.ssl
        if isinstance(ctx, ssl.SSLContext):
            return ctx
        if ctx is False:
            unverified = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            unverified.check_hostname = False
            unverified.verify_mode = ssl.CERT_NONE
            return unverified

        # Connector-level SSL context.
        ctx = self._ssl
        if isinstance(ctx, ssl.SSLContext):
            return ctx
        if ctx is False or ctx is None:
            unverified = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            unverified.check_hostname = False
            unverified.verify_mode = ssl.CERT_NONE
            return unverified

        # Default: verified, system CA store.
        return ssl.create_default_context()

    async def _create_connection(
        self,
        req: Any,
        traces: List[Any],
        timeout: Any,
    ) -> Any:
        _ = traces
        from aiohttp.client_exceptions import UnixClientConnectorError
        from aiohttp.helpers import ceil_timeout

        ssl_context = self._get_ssl_context(req)
        server_hostname = req.host if ssl_context else None

        try:
            async with ceil_timeout(timeout.sock_connect, ceil_threshold=timeout.ceil_threshold):
                _, proto = await self._loop.create_unix_connection(
                    self._factory,
                    self._path,
                    ssl=ssl_context,
                    server_hostname=server_hostname,
                )
        except OSError as exc:
            raise UnixClientConnectorError(self.path, req.connection_key, exc) from exc

        return proto


def _instance_tag(instance: Optional[str]) -> str:
    """Filename suffix for per-instance proxy artifacts ('' for default)."""
    from hostai import state as state_mod

    resolved = state_mod.normalize_instance_name(instance)
    return "" if resolved == state_mod.DEFAULT_INSTANCE else f"-{resolved}"


def _content_log_path(config: Config, instance: Optional[str] = None) -> Path:
    return config.root_dir / ".hostai-cache" / f"proxy-content{_instance_tag(instance)}.jsonl"


class _ContentLog:
    """Opt-in JSON-lines log of prompts and responses.

    Each record is flushed immediately so ``tail -f`` shows live traffic.
    Kept separate from proxy.log, which only ever carries operational
    metadata — this file contains full prompt and output content.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        self._file = os.fdopen(fd, "a", encoding="utf-8", buffering=1)
        self.path = path

    def write(self, event: str, request_id: str = "", **fields: Any) -> None:
        record: Dict[str, Any] = {"ts": round(time.time(), 3), "event": event}
        if request_id:
            record["id"] = request_id
        record.update(fields)
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.close()
        except OSError:
            pass


class TokenizedProxy:
    """An OpenAI-compatible proxy that tokenizes prompts client-side."""

    def __init__(
        self,
        config: Config,
        state: State,
        tokenizer: Tokenizer,
        socket_path: Path,
        port: int = 0,
    ) -> None:
        self.config = config
        self.state = state
        self.tokenizer = tokenizer
        self.socket_path = socket_path
        self.port = port
        self.upstream, self.upstream_socket = _resolve_upstream(state)
        self.api_key = state.api_key or ""
        self.ssl_ctx = _ssl_context(state)
        self.session: Optional[aiohttp.ClientSession] = None
        self.ready = False
        # Set by retarget(); cleared once upstream /health succeeds again and
        # /props has been re-fetched for the new backend.
        self.retarget_pending = False
        self.app = web.Application(client_max_size=64 * 1024 * 1024)
        self.app.router.add_post("/v1/chat/completions", self._chat)
        self.app.router.add_get("/v1/models", self._models)
        self.app.router.add_get("/health", self._health)
        self.app.router.add_get("/_hostai/backend", self._backend)
        # Generic pass-through for all other endpoints (slots, metrics, props, ...).
        self.app.router.add_route("*", "/{path:.*}", self._generic)
        self.app.on_startup.append(self._on_startup)
        self.app.on_cleanup.append(self._on_cleanup)
        self.content_log: Optional[_ContentLog] = None
        if config.proxy.log_content:
            self.content_log = _ContentLog(_content_log_path(config, _proxy_instance_name(state)))
            _logger.warning(
                "content logging enabled; prompts and responses are written to %s",
                self.content_log.path,
            )

    def _log_content(self, event: str, request_id: str = "", **fields: Any) -> None:
        if not self.content_log:
            return
        try:
            self.content_log.write(event, request_id, **fields)
        except OSError as exc:
            _logger.warning("content log write failed: %s", exc)

    def _build_upstream_session(self) -> aiohttp.ClientSession:
        """Create the upstream-facing session for the current state.

        The Authorization header and TLS/Unix connector are baked into the
        session at construction, so retarget() rebuilds it from scratch.
        """
        if self.upstream_socket:
            connector: aiohttp.BaseConnector = UnixTLSConnector(
                path=self.upstream_socket, ssl=self.ssl_ctx, limit=20, force_close=True
            )
        else:
            connector = aiohttp.TCPConnector(ssl=self.ssl_ctx, limit=20, force_close=True)
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return aiohttp.ClientSession(
            connector=connector,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=None, connect=30, sock_read=900),
        )

    async def _on_startup(self, app: web.Application) -> None:
        self.session = self._build_upstream_session()

    async def retarget(self, fresh: State) -> None:
        """Point the proxy at a new upstream instance without dropping listeners.

        Called by the upstream supervisor when ``state.json`` flips under the
        running proxy (``hostai replace`` writes it atomically at cutover).
        Client-facing sockets stay bound; only the upstream session and the
        identity fields move.  In-flight upstream requests are cancelled.
        """
        old_id = self.state.instance_id
        self.state = fresh
        self.upstream, self.upstream_socket = _resolve_upstream(fresh)
        self.api_key = fresh.api_key or ""
        self.ssl_ctx = _ssl_context(fresh)
        self.ready = False
        self.retarget_pending = True
        if self.session is not None:
            await self.session.close()
        self.session = self._build_upstream_session()
        _logger.warning("retargeted upstream to instance %s (was %s)", fresh.instance_id, old_id)

    async def _on_cleanup(self, app: web.Application) -> None:
        if self.session:
            await self.session.close()
        if self.content_log:
            self.content_log.close()

    async def _chat(self, request: web.Request) -> web.StreamResponse:
        if not self.ready or self.session is None:
            raise web.HTTPServiceUnavailable(reason="proxy not ready")

        req_id = f"req-{os.urandom(8).hex()}"

        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise web.HTTPBadRequest(reason=f"invalid JSON: {exc}") from exc

        if not self.config.proxy.tokenized_only:
            # In non-tokenized mode pass the OpenAI request through as-is.
            self._log_content("request", req_id, passthrough=True, body=body)
            return await self._forward_to_upstream(request, "/v1/chat/completions", log_id=req_id)

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            raise web.HTTPBadRequest(reason="request must contain a non-empty messages list")

        max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
        if max_tokens is None:
            # No explicit limit: use the configured default (-1 = request-
            # defined, run to EOS / context end).
            default = self.config.proxy.default_max_tokens
            if not isinstance(default, int) or isinstance(default, bool) or default == 0:
                default = -1
            n_predict = default
        elif not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise web.HTTPBadRequest(reason="max_tokens must be a positive integer")
        else:
            n_predict = max_tokens

        temperature = body.get("temperature", self.config.bench.temperature)
        if not isinstance(temperature, (int, float)):
            raise web.HTTPBadRequest(reason="temperature must be a number")

        tools = body.get("tools")
        stream = bool(body.get("stream", False))
        stream_options = body.get("stream_options")
        include_usage = bool(isinstance(stream_options, dict) and stream_options.get("include_usage"))

        stop_param = body.get("stop")
        if isinstance(stop_param, str):
            stop_strings = [stop_param]
        elif isinstance(stop_param, list):
            stop_strings = [s for s in stop_param if isinstance(s, str) and s]
        else:
            stop_strings = []

        reasoning_kwargs = default_reasoning_kwargs(self.config)
        expect_reasoning = bool(reasoning_kwargs.get("enable_thinking", True))

        try:
            token_ids = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                **reasoning_kwargs,
            )
        except TokenizerError as exc:
            self._log_content("error", req_id, error=f"tokenization failed: {exc}", messages=messages)
            raise web.HTTPBadRequest(reason=f"tokenization failed: {exc}") from exc

        self._log_content(
            "request",
            req_id,
            messages=messages,
            tools=tools,
            stream=stream,
            max_tokens=n_predict,
            temperature=temperature,
            stop=stop_strings,
            prompt_tokens=len(token_ids),
        )

        payload = self.build_completion_payload(token_ids, n_predict, temperature, stream, body)

        try:
            upstream_response = await self.session.post(
                f"{self.upstream}/completion",
                json=payload,
            )
        except aiohttp.ClientError as exc:
            self._log_content("error", req_id, error=f"upstream connect: {_one_line(exc)}")
            raise web.HTTPBadGateway(reason=f"upstream unavailable: {_one_line(exc)}") from exc

        if upstream_response.status != 200:
            text = await upstream_response.text()
            self._log_content("error", req_id, status=upstream_response.status, error=" ".join(text.split())[:500])
            raise web.HTTPInternalServerError(
                reason=f"upstream returned {upstream_response.status}: {' '.join(text.split())[:200]}"
            )

        _logger.info(
            "chat request: %d prompt tokens, stream=%s, stops=%d",
            len(token_ids),
            stream,
            len(stop_strings),
        )
        try:
            if stream:
                return await self._stream_chat(
                    request,
                    upstream_response,
                    stop_strings,
                    expect_reasoning,
                    tools,
                    req_id,
                    prompt_tokens=len(token_ids),
                    include_usage=include_usage,
                )
            return await self._complete_chat(upstream_response, len(token_ids), stop_strings, tools, req_id)
        except (asyncio.CancelledError, ConnectionError):
            # Client went away (handler_cancellation) or the downstream write
            # failed — dropping the upstream connection aborts generation.
            upstream_response.close()
            self._log_content("error", req_id, error="client disconnected")
            _logger.info("client disconnected; aborted upstream request")
            raise

    @staticmethod
    def build_completion_payload(
        token_ids: List[int],
        n_predict: int,
        temperature: float,
        stream: bool,
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a native /completion payload from an OpenAI chat request."""
        payload: Dict[str, Any] = {
            "prompt": token_ids,
            "n_predict": n_predict,
            "temperature": temperature,
            "stream": stream,
            # Ask the upstream to return raw token ids (and, on patched images,
            # to skip detokenization entirely). The proxy decodes them locally
            # so generated text never exists on the remote host.
            "return_tokens": True,
            "token_only": True,
        }

        # Forward common OpenAI/llama-server sampling parameters.
        optional_params = {
            "top_p",
            "top_k",
            "min_p",
            "stop",
            "frequency_penalty",
            "presence_penalty",
            "repeat_penalty",
            "seed",
            "logit_bias",
            "dynatemp_range",
            "dynatemp_exponent",
            "typical_p",
            "tfs_z",
            "mirostat",
            "mirostat_tau",
            "mirostat_eta",
            "n_probs",
            "grammar",
            "json_schema",
        }
        for key in optional_params:
            if key in body and body[key] is not None:
                payload[key] = body[key]

        return payload

    async def _complete_chat(
        self,
        response: aiohttp.ClientResponse,
        prompt_tokens: int,
        stop_strings: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        req_id: str = "",
    ) -> web.Response:
        try:
            data = await response.json()
        except json.JSONDecodeError as exc:
            raise web.HTTPInternalServerError(reason=f"invalid upstream JSON: {exc}") from exc
        except (asyncio.CancelledError, Exception):
            # Cancelled = client disconnected; closing the upstream response
            # drops the connection so llama-server aborts the generation.
            response.close()
            raise

        finish_reason = self._map_finish_reason(data)
        completion_tokens = data.get("tokens_predicted", 0) or 0
        evaluated = data.get("tokens_evaluated") or prompt_tokens
        timings = data.get("timings")
        timings = timings if isinstance(timings, dict) else None

        tokens = data.get("tokens") or []
        if tokens:
            # token_only path: detokenize locally, then apply stop strings
            # client-side since upstream cannot match them without text.
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            cut = _find_stop(text, stop_strings or [])
            if cut >= 0:
                text = text[:cut]
                finish_reason = "stop"
                _logger.warning("stop string matched client-side; truncated decoded output")
            reasoning, content = _split_reasoning(text)
            completion_tokens = completion_tokens or len(tokens)
        else:
            _logger.debug("upstream returned no token ids; using text fallback")
            content = data.get("content", "")
            reasoning = data.get("reasoning_content") or ""
            if not reasoning:
                reasoning, content = _split_reasoning(content)

        tool_calls, remainder = _parse_tool_calls(content, tools) if tools else ([], content)
        if tool_calls:
            message: Dict[str, Any] = {
                "role": "assistant",
                "content": remainder or None,
                "tool_calls": tool_calls,
            }
            if finish_reason != "length":
                finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning

        output = {
            "id": f"chatcmpl-{os.urandom(12).hex()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.state.model or self.config.model.model or "local",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": self._usage_dict(evaluated, completion_tokens, (timings or {}).get("cache_n")),
        }
        if timings:
            output["timings"] = timings
        self._log_content(
            "response",
            req_id,
            message=message,
            finish_reason=finish_reason,
            usage=output["usage"],
        )
        return web.json_response(output)

    @staticmethod
    def _usage_dict(prompt_tokens: int, completion_tokens: int, cached_tokens: Any = None) -> Dict[str, Any]:
        """OpenAI ``usage`` object, with a cached-token detail when known."""
        usage: Dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        # Emit the detail whenever upstream reported the counter (0 is real
        # data: the prompt was not cached) — absent only when it is unknown.
        if isinstance(cached_tokens, int) and not isinstance(cached_tokens, bool) and cached_tokens >= 0:
            usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
        return usage

    @staticmethod
    def _stream_usage(
        last_obj: Optional[Dict[str, Any]],
        prompt_tokens: int,
        detok: _TokenDetokenizer,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Usage for a finished stream: upstream counters win, detok is the fallback.

        Upstream chunks carry cumulative ``tokens_predicted``/``tokens_evaluated``
        and the final one adds ``timings`` (with ``cache_n`` = per-request
        cached prompt tokens) — all of which the proxy would otherwise drop.
        Note ``tokens_cached`` is deliberately unused: it counts every token
        resident in the slot's KV cache, not this request's cache hits.
        ``timings`` is returned separately so the caller can attach it next to
        ``usage`` like llama.cpp does.
        """
        stats = last_obj or {}
        predicted = stats.get("tokens_predicted") or detok.token_count
        evaluated = stats.get("tokens_evaluated") or prompt_tokens
        timings = stats.get("timings")
        timings = timings if isinstance(timings, dict) else None
        usage = TokenizedProxy._usage_dict(evaluated, predicted, (timings or {}).get("cache_n"))
        return usage, timings

    def _build_sse_chunk(
        self,
        completion_id: str,
        created: int,
        model: str,
        delta: Dict[str, Any],
        finish_reason: Optional[str],
    ) -> bytes:
        chunk: Dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(chunk)}\n\n".encode("utf-8")

    @staticmethod
    def _build_usage_chunk(
        completion_id: str,
        created: int,
        model: str,
        usage: Dict[str, Any],
        timings: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """Terminal ``choices: []`` chunk carrying ``usage`` (stream_options)."""
        chunk: Dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": usage,
        }
        if timings:
            chunk["timings"] = timings
        return f"data: {json.dumps(chunk)}\n\n".encode("utf-8")

    def _detect_token_mode(
        self,
        obj: Dict[str, Any],
        token_mode: Optional[bool],
        stop: bool,
    ) -> Optional[bool]:
        if token_mode is not None or stop:
            return token_mode
        if obj.get("tokens"):
            return True
        if obj.get("content") or obj.get("reasoning_content"):
            return False
        return None

    def _process_stream_chunk(
        self,
        obj: Dict[str, Any],
        detok: _TokenDetokenizer,
        token_mode: Optional[bool],
        sent_reasoning: bool,
        sent_content: bool,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[bool], bool, bool]:
        stop = obj.get("stop", False)
        token_mode = self._detect_token_mode(obj, token_mode, stop)
        deltas: List[Dict[str, Any]] = []
        finish: Optional[str] = None

        if token_mode:
            deltas = detok.add(obj.get("tokens") or [])
            if stop:
                deltas += detok.finish()
            if detok.stopped:
                finish = "stop"
            elif stop:
                finish = self._map_finish_reason(obj)
            if finish == "stop" and detok.saw_tool_calls:
                finish = "tool_calls"
            sent_content = sent_content or any("content" in d for d in deltas)
        # The final chunk carries the full content/reasoning_content,
        # not deltas. Its content was already streamed, so only the
        # reasoning blob is forwarded, and only when no deltas were sent.
        elif stop:
            reasoning = (obj.get("reasoning_content") or "") if not sent_reasoning else ""
            if reasoning:
                deltas.append({"reasoning_content": reasoning})
            tool_calls, _ = _parse_tool_calls(obj.get("content") or "", tools) if tools else ([], "")
            if tool_calls and not sent_content:
                deltas.append({"tool_calls": tool_calls})
                finish = "tool_calls"
            else:
                finish = self._map_finish_reason(obj)
        else:
            delta_payload: Dict[str, Any] = {}
            delta = obj.get("content", "")
            if delta:
                delta_payload["content"] = delta
                sent_content = True
            reasoning_delta = obj.get("reasoning_content", "")
            if reasoning_delta:
                delta_payload["reasoning_content"] = reasoning_delta
                sent_reasoning = True
            if delta_payload:
                deltas.append(delta_payload)

        return deltas, finish, token_mode, sent_reasoning, sent_content

    async def _stream_end(
        self,
        stream: web.StreamResponse,
        response: aiohttp.ClientResponse,
        detok: _TokenDetokenizer,
        token_mode: Optional[bool],
        stop: bool,
        finish: Optional[str],
        req_id: str = "",
        usage_chunk: Optional[bytes] = None,
    ) -> web.StreamResponse:
        if usage_chunk:
            await stream.write(usage_chunk)
        await stream.write(b"data: [DONE]\n\n")
        self._log_content(
            "done",
            req_id,
            finish_reason=finish,
            completion_tokens=detok.token_count,
            tool_calls=detok.saw_tool_calls,
        )
        if token_mode and detok.stopped and not stop:
            # Cancel upstream generation; the server stops the task
            # when the client connection closes.
            response.close()
            _logger.warning(
                "stop string matched client-side; aborted upstream stream at %d tokens",
                detok.token_count,
            )
        _logger.info(
            "stream finished: %d tokens, finish=%s",
            detok.token_count,
            finish,
        )
        return stream

    async def _stream_chat(
        self,
        request: web.Request,
        response: aiohttp.ClientResponse,
        stop_strings: Optional[List[str]] = None,
        expect_reasoning: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
        req_id: str = "",
        prompt_tokens: int = 0,
        include_usage: bool = False,
    ) -> web.StreamResponse:
        stream = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await stream.prepare(request)

        completion_id = f"chatcmpl-{os.urandom(12).hex()}"
        model = self.state.model or self.config.model.model or "local"
        created = int(time.time())
        sent_reasoning = False
        sent_content = False
        detok = _TokenDetokenizer(self.tokenizer, stop_strings, expect_reasoning, tools)
        # Decided by the first chunk carrying data: upstream partials always
        # populate `tokens`, so the detokenize path is used whenever ids are
        # present; upstream text deltas are ignored in that case.
        token_mode: Optional[bool] = None
        # SSE data: lines can be split across TCP chunks; buffer between
        # reads so a partial line is never parsed as JSON and dropped.
        buffer = ""
        # Upstream counters (tokens_predicted/evaluated/cached, timings) ride
        # along on every chunk; the last one seen feeds the usage chunk.
        last_obj: Optional[Dict[str, Any]] = None

        try:
            while True:
                # Any await here (read or write) raises CancelledError the
                # moment the client connection dies (handler_cancellation);
                # closing the upstream response aborts remote generation.
                raw = await response.content.readany()
                if not raw:
                    break
                buffer += raw.decode("utf-8", errors="replace")
                *lines, buffer = buffer.split("\n")
                for line in lines:
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    last_obj = obj

                    deltas, finish, token_mode, sent_reasoning, sent_content = self._process_stream_chunk(
                        obj, detok, token_mode, sent_reasoning, sent_content, tools
                    )
                    if deltas:
                        for i, delta_payload in enumerate(deltas):
                            self._log_content("delta", req_id, **delta_payload)
                            chunk_finish = finish if i == len(deltas) - 1 else None
                            await stream.write(
                                self._build_sse_chunk(completion_id, created, model, delta_payload, chunk_finish)
                            )
                    elif finish:
                        await stream.write(self._build_sse_chunk(completion_id, created, model, {}, finish))

                    stop = obj.get("stop", False)
                    if stop or (token_mode and detok.stopped):
                        usage_chunk = self._usage_chunk(
                            completion_id, created, model, include_usage, last_obj, prompt_tokens, detok
                        )
                        return await self._stream_end(
                            stream, response, detok, token_mode, stop, finish, req_id, usage_chunk
                        )
        except ConnectionError:
            # A write to a dead client socket lands here.
            response.close()
            self._log_content("error", req_id, error="client connection reset")
            _logger.info("client connection reset; aborted upstream stream at %d tokens", detok.token_count)
            return stream
        except asyncio.CancelledError:
            # handler_cancellation: client disconnected mid-stream.
            response.close()
            self._log_content("error", req_id, error="client disconnected")
            _logger.info("client disconnected; aborted upstream stream at %d tokens", detok.token_count)
            raise
        except Exception:
            response.close()
            raise

        if token_mode:
            for tail in detok.finish():
                self._log_content("delta", req_id, **tail)
                await stream.write(self._build_sse_chunk(completion_id, created, model, tail, None))
        usage_chunk = self._usage_chunk(completion_id, created, model, include_usage, last_obj, prompt_tokens, detok)
        if usage_chunk:
            await stream.write(usage_chunk)
        await stream.write(b"data: [DONE]\n\n")
        self._log_content(
            "done",
            req_id,
            finish_reason=None,
            completion_tokens=detok.token_count,
            tool_calls=detok.saw_tool_calls,
        )
        return stream

    def _usage_chunk(
        self,
        completion_id: str,
        created: int,
        model: str,
        include_usage: bool,
        last_obj: Optional[Dict[str, Any]],
        prompt_tokens: int,
        detok: _TokenDetokenizer,
    ) -> Optional[bytes]:
        """Terminal usage chunk, emitted only when stream_options asked for it."""
        if not include_usage:
            return None
        usage, timings = self._stream_usage(last_obj, prompt_tokens, detok)
        return self._build_usage_chunk(completion_id, created, model, usage, timings)

    @staticmethod
    def _map_finish_reason(data: Dict[str, Any]) -> Optional[str]:
        if data.get("stopped_limit"):
            return "length"
        if data.get("stopped_eos") or data.get("stopped_word") or data.get("stop"):
            return "stop"
        return None

    async def _forward_to_upstream(
        self,
        request: web.Request,
        upstream_path: str,
        log_id: str = "",
    ) -> web.StreamResponse:
        """Pass an arbitrary request through to the upstream server."""
        if not self.ready or self.session is None:
            raise web.HTTPServiceUnavailable(reason="proxy not ready")

        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
        body = await request.read()

        query = request.query_string
        url = f"{self.upstream}{upstream_path}"
        if query:
            url = f"{url}?{query}"

        try:
            async with self.session.request(
                request.method,
                url,
                headers=headers,
                data=body,
            ) as upstream_response:
                self._log_content("upstream_response", log_id, status=upstream_response.status)
                response = web.StreamResponse(status=upstream_response.status)
                # aiohttp auto-decompresses the body, so hop-by-hop framing
                # and Content-Encoding headers must not be forwarded — the
                # bytes we write are already decoded.
                for k, v in upstream_response.headers.items():
                    if k.lower() in {
                        "content-encoding",
                        "content-length",
                        "transfer-encoding",
                        "connection",
                    }:
                        continue
                    response.headers[k] = v
                await response.prepare(request)
                try:
                    while True:
                        chunk = await upstream_response.content.readany()
                        if not chunk:
                            break
                        if log_id:
                            self._log_content("upstream_chunk", log_id, data=chunk.decode("utf-8", "replace"))
                        await response.write(chunk)
                    await response.write_eof()
                except ConnectionError:
                    upstream_response.close()
                    self._log_content("error", log_id, error="client disconnected")
                except (asyncio.CancelledError, Exception):
                    # Cancelled = client disconnected (handler_cancellation).
                    upstream_response.close()
                    raise
                return response
        except aiohttp.ClientError as exc:
            self._log_content("error", log_id, error=_one_line(exc))
            raise web.HTTPBadGateway(reason=f"upstream request failed: {_one_line(exc)}") from exc

    async def _generic(self, request: web.Request) -> web.StreamResponse:
        """Catch-all reverse proxy for any endpoint not handled above."""
        return await self._forward_to_upstream(request, request.path)

    async def _models(self, request: web.Request) -> web.Response:
        if not self.ready:
            raise web.HTTPServiceUnavailable(reason="proxy not ready")
        model_id = self.state.model or self.config.model.model or "local"
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": model_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "hostai",
                    }
                ],
            }
        )

    async def _backend(self, request: web.Request) -> web.Response:
        """Report which upstream instance the proxy is currently bound to.

        ``hostai replace`` polls this to confirm the hot-retarget landed.
        """
        return web.json_response(
            {
                "instance_id": self.state.instance_id,
                "upstream": self.upstream,
                "upstream_socket": self.upstream_socket,
                "ready": self.ready,
            }
        )

    async def _health(self, request: web.Request) -> web.Response:
        if not self.ready or self.session is None:
            raise web.HTTPServiceUnavailable(reason="proxy not ready")
        try:
            async with self.session.get(f"{self.upstream}/health") as response:
                text = await response.text()
                return web.Response(text=text, status=response.status)
        except aiohttp.ClientError as exc:
            raise web.HTTPBadGateway(reason=f"upstream health failed: {_one_line(exc)}") from exc

    async def run(self) -> None:
        # handler_cancellation: aiohttp cancels the request handler task as
        # soon as the client connection dies — without it a closed client is
        # only noticed when a write fails, and a non-streaming request would
        # let llama-server finish a generation nobody is listening to.
        runner = web.AppRunner(self.app, handler_cancellation=True)
        await runner.setup()

        # Pid file: lets `down`/`up` find this daemon even when state.json was
        # reset or proxy_pid was never persisted.
        pid_file = self.state.state_file.parent / "proxy.pid"
        pid_file.write_text(str(os.getpid()))

        sites: List[web.BaseSite] = []
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        try:
            unix_site = web.UnixSite(runner, str(self.socket_path))
            await unix_site.start()
            sites.append(unix_site)
            os.chmod(self.socket_path, 0o600)
            print(f"proxy listening on unix socket {self.socket_path}")

            if self.port:
                tcp_site = web.TCPSite(
                    runner,
                    "127.0.0.1",
                    self.port,
                    reuse_address=True,
                )
                for attempt in range(10):
                    try:
                        await tcp_site.start()
                        break
                    except OSError as exc:
                        if exc.errno == errno.EADDRINUSE and attempt < 9:
                            print(f"port {self.port} in use, retrying in 0.5s (attempt {attempt + 1}/10)")
                            await asyncio.sleep(0.5)
                            continue
                        raise
                sites.append(tcp_site)
                print(f"proxy listening on tcp 127.0.0.1:{self.port}")
                # Reload before writing so a concurrent down/up isn't
                # clobbered by this stale copy.
                fresh = State.load(self.state.state_file)
                fresh.local_port = self.port
                fresh.save()

            while True:
                await asyncio.sleep(3600)
        finally:
            for site in sites:
                await site.stop()
            await runner.cleanup()
            self.socket_path.unlink(missing_ok=True)
            pid_file.unlink(missing_ok=True)


async def _fetch_props_once(config: Config, state: State) -> Optional[Dict[str, Any]]:
    """Fetch /props from the upstream to obtain the remote chat template."""
    upstream, upstream_socket = _resolve_upstream(state)
    ssl_ctx = _ssl_context(state)
    api_key = state.api_key or ""
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        connector: aiohttp.BaseConnector
        if upstream_socket:
            connector = UnixTLSConnector(path=upstream_socket, ssl=ssl_ctx)
        else:
            connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        async with aiohttp.ClientSession(
            connector=connector,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as session:
            async with session.get(f"{upstream}/props") as response:
                if response.status == 200:
                    return await response.json()
    except aiohttp.ClientError:
        pass
    return None


async def _wait_for_upstream_health(
    proxy: TokenizedProxy,
    server_task: asyncio.Task,
    interval: float = 2.0,
) -> bool:
    """Poll upstream /health until the model is ready to serve requests.

    Logs the failure reason periodically and re-establishes the SSH unix
    tunnel when it dies (the local socket path is removed by the tunnel
    worker on connection loss).  If the provider confirms the instance is
    gone, gives up so the proxy exits instead of polling a dead host forever.
    """
    if proxy.session is None:
        return False
    last_log = 0.0
    last_err = ""
    last_dead_check = 0.0
    while not server_task.done():
        err = ""
        try:
            async with proxy.session.get(f"{proxy.upstream}/health") as response:
                if response.status == 200:
                    return True
                err = f"upstream /health status {response.status}"
        except aiohttp.ClientError as exc:
            err = str(exc) or type(exc).__name__

        now = time.monotonic()
        if err != last_err or now - last_log >= 60:
            last_log = now
            last_err = err
            _logger.info("waiting for upstream /health: %s", err)

        # A preempted/deleted instance never becomes healthy — confirm via the
        # provider (double-checked, fail-open) and stop waiting.
        if now - last_dead_check >= 60 and proxy.state.instance_id:
            last_dead_check = now
            dead = await asyncio.to_thread(_common.confirm_instance_dead, proxy.config, proxy.state.instance_id)
            if dead:
                _logger.error("upstream instance %s; giving up", dead)
                return False

        if proxy.upstream_socket and not Path(proxy.upstream_socket).exists():
            _logger.warning("upstream socket %s gone; restarting SSH unix tunnel", proxy.upstream_socket)
            try:
                local_path = await asyncio.to_thread(ssh.ensure_unix_tunnel, proxy.config, proxy.state)
                _logger.info("SSH unix tunnel re-established on %s", local_path)
            except Exception as exc:
                _logger.error("SSH unix tunnel restart failed: %s", exc)

        await asyncio.sleep(interval)
    # The server task died (likely a startup failure); re-raise its exception.
    await server_task
    return False


async def _probe_upstream_health(proxy: TokenizedProxy, timeout: float = 10.0) -> bool:
    """Return True when upstream ``/health`` answers 200 within *timeout*."""
    if proxy.session is None:
        return False
    try:
        async with proxy.session.get(
            f"{proxy.upstream}/health", timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            return response.status == 200
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return False


def _retarget_tunnel(proxy: TokenizedProxy, old_socket: Optional[str]) -> None:
    """Drop the SSH forward to the previous host and re-create it for the new one.

    The local path/port stays identical (``hostai replace`` preserves it), so
    the proxy's connector and client-facing listeners need no changes.  For
    the TCP case ``stop_tunnel`` must run on the *fresh* state — it is keyed
    on the same local_port, and saving the stale pre-flip state would race
    the ``state.json`` cutover.
    """
    if old_socket:
        ssh.stop_unix_tunnel(old_socket)
    else:
        ssh.stop_tunnel(proxy.state)
    if proxy.upstream_socket:
        local_path = ssh.ensure_unix_tunnel(proxy.config, proxy.state)
        _logger.info("SSH unix tunnel re-established on %s", local_path)
    else:
        ssh.ensure_tunnel(proxy.config, proxy.state)
        _logger.info("SSH TCP tunnel re-established on :%s", proxy.state.local_port)


async def _maybe_retarget(proxy: TokenizedProxy) -> bool:
    """Reload ``state.json`` and retarget when the instance id changed.

    ``hostai replace`` provisions under a sidecar state file and flips
    ``state.json`` atomically at cutover — the instance_id swap is the
    trigger.  Returns True when a retarget happened this cycle.
    """
    try:
        fresh = await asyncio.to_thread(State.load, proxy.state.state_file)
    except Exception:
        return False
    if not fresh.instance_id or fresh.instance_id == proxy.state.instance_id:
        return False
    old_socket = proxy.upstream_socket
    await proxy.retarget(fresh)
    try:
        await asyncio.to_thread(_retarget_tunnel, proxy, old_socket)
    except Exception as exc:
        _logger.error("upstream tunnel retarget failed: %s", exc)
    return True


async def _upstream_supervisor(
    proxy: TokenizedProxy,
    server_task: asyncio.Task,
    interval: float = 5.0,
) -> None:
    """Keep the upstream reachable after bootstrap marked the proxy ready.

    The SSH tunnel worker removes the local socket when the remote sshd
    restarts, and post-bootstrap nothing recreated it — every request then
    failed with a bare 500 forever.  This loop re-establishes the tunnel,
    gates ``proxy.ready`` on upstream /health (clients see a clean 503
    while the model restarts), follows ``state.json`` instance swaps from
    ``hostai replace``, and stops the proxy once the provider
    confirms the instance is gone.
    """
    consecutive_failures = 0
    last_dead_check = 0.0
    while not server_task.done():
        try:
            await _maybe_retarget(proxy)
        except Exception as exc:
            _logger.error("upstream retarget failed: %s", exc)

        healthy = await _probe_upstream_health(proxy)
        if healthy:
            if proxy.retarget_pending:
                proxy.retarget_pending = False
                try:
                    props = await _fetch_props_once(proxy.config, proxy.state)
                    if props and props.get("chat_template"):
                        proxy.tokenizer = Tokenizer(proxy.config, chat_template=props["chat_template"])
                        _logger.info("tokenizer reloaded from remote /props after retarget")
                except Exception as exc:
                    _logger.warning("/props refresh after retarget failed: %s", exc)
            if not proxy.ready:
                proxy.ready = True
                _logger.info("upstream /health recovered; marking proxy ready")
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= 2 and proxy.ready:
                proxy.ready = False
                _logger.warning("upstream /health failing; marking proxy not ready")

        if proxy.upstream_socket:
            socket_alive = Path(proxy.upstream_socket).exists() and await asyncio.to_thread(
                ssh._unix_socket_is_open, proxy.upstream_socket, 3
            )
            if not socket_alive:
                _logger.warning("upstream socket %s gone; restarting SSH unix tunnel", proxy.upstream_socket)
                try:
                    await asyncio.to_thread(ssh.ensure_unix_tunnel, proxy.config, proxy.state)
                    _logger.info("SSH unix tunnel re-established on %s", proxy.upstream_socket)
                except Exception as exc:
                    _logger.error("SSH unix tunnel restart failed: %s", exc)
        elif proxy.state.local_port and not ssh._tunnel_is_running(proxy.state):
            _logger.warning("SSH TCP tunnel on :%s down; restarting", proxy.state.local_port)
            try:
                await asyncio.to_thread(ssh.ensure_tunnel, proxy.config, proxy.state)
                _logger.info("SSH TCP tunnel re-established on :%s", proxy.state.local_port)
            except Exception as exc:
                _logger.error("SSH TCP tunnel restart failed: %s", exc)

        # A preempted/deleted instance never recovers — stop the proxy so it
        # exits instead of serving 503s forever.  Only poll the provider while
        # the upstream is actually failing.
        now = time.monotonic()
        if not healthy and now - last_dead_check >= 60 and proxy.state.instance_id:
            last_dead_check = now
            dead = await asyncio.to_thread(_common.confirm_instance_dead, proxy.config, proxy.state.instance_id)
            if dead:
                _logger.error("upstream instance %s; stopping proxy", dead)
                server_task.cancel()
                return

        await asyncio.sleep(interval)


async def _bootstrap_proxy(
    proxy: TokenizedProxy,
    config: Config,
    state: State,
    server_task: asyncio.Task,
) -> None:
    """Wait for the upstream, fetch /props, and configure the tokenizer."""
    # Wait for the web server to finish startup (which creates self.session).
    for _ in range(200):
        if proxy.session is not None:
            break
        if server_task.done():
            await server_task  # re-raise the server failure
        await asyncio.sleep(0.05)
    else:
        raise RuntimeError("proxy server did not start")

    if not await _wait_for_upstream_health(proxy, server_task):
        raise ProxyError("upstream /health did not become ready")

    props = await _fetch_props_once(config, state)
    chat_template: Optional[str] = None
    if props and props.get("chat_template"):
        chat_template = props["chat_template"]
        _logger.info("using chat template from remote /props")

    proxy.tokenizer = Tokenizer(config, chat_template=chat_template)
    proxy.ready = True
    _logger.info("proxy ready")


def _proxy_instance_name(state: State) -> str:
    """Instance name owning this state ('default' for the legacy layout)."""
    from hostai import state as state_mod

    return state.data.get("instance_name") or state_mod.instance_name_for_state_file(state.state_file)


def _proxy_log_file(config: Config, instance: Optional[str] = None) -> Path:
    return config.root_dir / ".hostai-cache" / f"proxy{_instance_tag(instance)}.log"


def _configure_proxy_logging(config: Config, instance: Optional[str] = None) -> Path:
    """Attach a file handler so proxy activity is logged locally.

    Only operational metadata is logged (token counts, timings, warnings) —
    never prompt or generated content.
    """
    log_file = _proxy_log_file(config, instance)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("hostai")
    existing = [
        h
        for h in logger.handlers
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(log_file)
    ]
    if not existing:
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        if logger.level > logging.INFO or logger.level == logging.NOTSET:
            logger.setLevel(logging.INFO)
    return log_file


async def run_proxy(config: Config, state: State) -> None:
    """Start the local proxy for the active state.

    The proxy owns its own SSH tunnel to the remote Unix socket and keeps the
    connection alive as long as it is running. When tokenized-only is enabled it
    tokenizes /v1/chat/completions; otherwise it passes traffic through.
    """
    instance_name = _proxy_instance_name(state)
    log_file = _configure_proxy_logging(config, instance_name)
    _logger.info("proxy starting (instance %s), logging to %s", instance_name, log_file)

    # A second `hostai proxy` would unlink the running proxy's socket file and
    # overwrite proxy_pid in state, orphaning the first daemon — refuse early.
    existing_pid = _common.running_proxy_pid(state)
    if existing_pid:
        raise ProxyError(f"proxy already running (pid {existing_pid})")

    socket_path = Path(config.proxy.socket_path) if config.proxy.socket_path else _default_socket_path(state)
    port = config.proxy.port or config.ssh.local_port or 0

    # Record the proxy pid so hostai down can stop it, and pre-register the
    # upstream Unix socket path so the proxy can create its aiohttp app
    # immediately (the SSH tunnel and remote socket are set up concurrently
    # below; the proxy returns 503 until the model is ready).  Reload state
    # first — the copy loaded at CLI entry may be stale by the time we write.
    upstream_socket = state.data.get("upstream_socket") or str(state.state_file.parent / "upstream.sock")
    state = State.load(state.state_file)
    state.data["proxy_pid"] = os.getpid()
    state.data["upstream_socket"] = upstream_socket
    # The client-facing TCP port, distinct from state.local_port which the
    # unsecure-mode SSH tunnel claims after the proxy already bound its own.
    state.data["proxy_port"] = port
    state.save()

    tokenizer = Tokenizer(config)
    proxy = TokenizedProxy(config, state, tokenizer, socket_path, port)
    if proxy.content_log:
        print(f"content logging enabled -> {proxy.content_log.path} (contains prompts and outputs)")

    # Start the client-facing web server immediately so `hostai up` can see the
    # port and move on to its own long `wait_for_api`. The upstream model load
    # happens in the background and the proxy returns 503 until it is ready.
    server_task = asyncio.create_task(proxy.run())

    try:
        if not state.unsecure:
            # The tunnel setup can take minutes while the remote model loads, so
            # run it in a thread so the web server can bind its local port now.
            local_socket = await asyncio.to_thread(ssh.ensure_unix_tunnel, config, state)
            _logger.info("proxy SSH unix tunnel on %s", local_socket)
        else:
            local_port = await asyncio.to_thread(ssh.ensure_tunnel, config, state)
            _logger.info("proxy SSH TCP tunnel on localhost:%d", local_port)

        await _bootstrap_proxy(proxy, config, state, server_task)
    except Exception as exc:
        _logger.error("proxy bootstrap failed: %s", exc)
        if not server_task.done():
            server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        raise

    supervisor = asyncio.create_task(_upstream_supervisor(proxy, server_task))
    try:
        await server_task
    except asyncio.CancelledError:
        pass  # supervisor stopped the server (upstream instance confirmed gone)
    finally:
        supervisor.cancel()
        try:
            await supervisor
        except asyncio.CancelledError:
            pass
