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
from hostai.config import Config
from hostai.state import State
from hostai.tokenize import Tokenizer, TokenizerError, default_reasoning_kwargs

_logger = logging.getLogger(__name__)


# Pattern for tool-call output produced by the Qwen tool-use template.  It
# wraps JSON tool calls between <tool_call> and </tool_call> tags.
_TOOL_CALL_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>",
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


def _parse_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Parse Qwen-style <tool_call>...</tool_call> output into OpenAI tool_calls.

    Returns an empty list when the content contains no tool-call tags or the
    JSON cannot be parsed.
    """
    tool_calls: List[Dict[str, Any]] = []
    for match in _TOOL_CALL_RE.finditer(content):
        raw = match.group(1).strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue

        name = parsed.get("name")
        arguments = parsed.get("arguments", {})
        if not name:
            continue

        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        else:
            arguments = str(arguments)

        tool_calls.append(
            {
                "id": f"call_{os.urandom(8).hex()}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )
    return tool_calls


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
    content on ``◀``, and applies ``stop`` strings itself since server-side
    stop matching is text-based and cannot run without detokenization.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        stop_strings: Optional[List[str]] = None,
        expect_reasoning: bool = True,
    ) -> None:
        self._tokenizer = tokenizer
        self._stops = [s for s in stop_strings or [] if s]
        self._expect_reasoning = expect_reasoning
        self._ids: List[int] = []
        self._emitted = 0
        self._cut = -1
        self.stopped = False
        # Hold back trailing chars so a stop string or reasoning marker that
        # straddles a decode boundary can still be detected before its prefix
        # is emitted.
        boundary_lengths = [len(s) for s in self._stops]
        if expect_reasoning:
            boundary_lengths += [len(m) for m in _THINK_END_MARKERS]
        self._holdback = max(0, max(boundary_lengths, default=0) - 1)

    def add(self, token_ids: List[int]) -> Dict[str, str]:
        """Append generated token ids and return the new delta text."""
        self._ids.extend(token_ids)
        return self._drain(final=False)

    def finish(self) -> Dict[str, str]:
        """Flush any held-back text at the end of the stream."""
        return self._drain(final=True)

    @property
    def token_count(self) -> int:
        return len(self._ids)

    def _drain(self, final: bool) -> Dict[str, str]:
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
        limit = min(limit, len(text))
        start = min(self._emitted, limit)
        self._emitted = limit
        return _split_delta(text, start, limit, self._expect_reasoning)


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
        self.app = web.Application(client_max_size=64 * 1024 * 1024)
        self.app.router.add_post("/v1/chat/completions", self._chat)
        self.app.router.add_get("/v1/models", self._models)
        self.app.router.add_get("/health", self._health)
        # Generic pass-through for all other endpoints (slots, metrics, props, ...).
        self.app.router.add_route("*", "/{path:.*}", self._generic)
        self.app.on_startup.append(self._on_startup)
        self.app.on_cleanup.append(self._on_cleanup)

    async def _on_startup(self, app: web.Application) -> None:
        if self.upstream_socket:
            connector: aiohttp.BaseConnector = UnixTLSConnector(
                path=self.upstream_socket, ssl=self.ssl_ctx, limit=20, force_close=True
            )
        else:
            connector = aiohttp.TCPConnector(ssl=self.ssl_ctx, limit=20, force_close=True)
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self.session = aiohttp.ClientSession(
            connector=connector,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=None, connect=30, sock_read=900),
        )

    async def _on_cleanup(self, app: web.Application) -> None:
        if self.session:
            await self.session.close()

    async def _chat(self, request: web.Request) -> web.StreamResponse:
        if not self.ready or self.session is None:
            raise web.HTTPServiceUnavailable(reason="proxy not ready")

        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise web.HTTPBadRequest(reason=f"invalid JSON: {exc}") from exc

        if not self.config.proxy.tokenized_only:
            # In non-tokenized mode pass the OpenAI request through as-is.
            return await self._forward_to_upstream(request, "/v1/chat/completions")

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            raise web.HTTPBadRequest(reason="request must contain a non-empty messages list")

        max_tokens = body.get("max_tokens", self.config.bench.max_tokens)
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise web.HTTPBadRequest(reason="max_tokens must be a positive integer")

        temperature = body.get("temperature", self.config.bench.temperature)
        if not isinstance(temperature, (int, float)):
            raise web.HTTPBadRequest(reason="temperature must be a number")

        tools = body.get("tools")
        stream = bool(body.get("stream", False))

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
            raise web.HTTPBadRequest(reason=f"tokenization failed: {exc}") from exc

        payload = self.build_completion_payload(token_ids, max_tokens, temperature, stream, body)

        upstream_response = await self.session.post(
            f"{self.upstream}/completion",
            json=payload,
        )

        if upstream_response.status != 200:
            text = await upstream_response.text()
            raise web.HTTPInternalServerError(reason=f"upstream returned {upstream_response.status}: {text[:200]}")

        _logger.info(
            "chat request: %d prompt tokens, stream=%s, stops=%d",
            len(token_ids),
            stream,
            len(stop_strings),
        )
        if stream:
            return await self._stream_chat(request, upstream_response, stop_strings, expect_reasoning)
        return await self._complete_chat(upstream_response, len(token_ids), stop_strings)

    @staticmethod
    def build_completion_payload(
        token_ids: List[int],
        max_tokens: int,
        temperature: float,
        stream: bool,
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a native /completion payload from an OpenAI chat request."""
        payload: Dict[str, Any] = {
            "prompt": token_ids,
            "n_predict": max_tokens,
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
    ) -> web.Response:
        try:
            data = await response.json()
        except json.JSONDecodeError as exc:
            raise web.HTTPInternalServerError(reason=f"invalid upstream JSON: {exc}") from exc

        finish_reason = self._map_finish_reason(data)
        completion_tokens = data.get("tokens_predicted", 0) or 0

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

        tool_calls = _parse_tool_calls(content)
        if tool_calls:
            message: Dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": tool_calls}
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
            "model": self.config.model.model or "local",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        return web.json_response(output)

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
    ) -> Tuple[Dict[str, Any], Optional[str], Optional[bool], bool]:
        stop = obj.get("stop", False)
        token_mode = self._detect_token_mode(obj, token_mode, stop)

        if token_mode:
            delta_payload = detok.finish() if stop else detok.add(obj.get("tokens") or [])
            finish = "stop" if detok.stopped else (self._map_finish_reason(obj) if stop else None)
        # The final chunk carries the full content/reasoning_content,
        # not deltas. Its content was already streamed, so only the
        # reasoning blob is forwarded, and only when no deltas were sent.
        elif stop:
            delta_payload = {}
            reasoning = (obj.get("reasoning_content") or "") if not sent_reasoning else ""
            if reasoning:
                delta_payload["reasoning_content"] = reasoning
            finish = self._map_finish_reason(obj)
        else:
            delta_payload = {}
            delta = obj.get("content", "")
            if delta:
                delta_payload["content"] = delta
            reasoning_delta = obj.get("reasoning_content", "")
            if reasoning_delta:
                delta_payload["reasoning_content"] = reasoning_delta
                sent_reasoning = True
            finish = None

        return delta_payload, finish, token_mode, sent_reasoning

    async def _stream_end(
        self,
        stream: web.StreamResponse,
        response: aiohttp.ClientResponse,
        detok: _TokenDetokenizer,
        token_mode: Optional[bool],
        stop: bool,
        finish: Optional[str],
    ) -> web.StreamResponse:
        await stream.write(b"data: [DONE]\n\n")
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
        model = self.config.model.model or "local"
        created = int(time.time())
        sent_reasoning = False
        detok = _TokenDetokenizer(self.tokenizer, stop_strings, expect_reasoning)
        # Decided by the first chunk carrying data: upstream partials always
        # populate `tokens`, so the detokenize path is used whenever ids are
        # present; upstream text deltas are ignored in that case.
        token_mode: Optional[bool] = None

        async for raw in response.content:
            for line in raw.decode("utf-8", errors="replace").splitlines():
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

                delta_payload, finish, token_mode, sent_reasoning = self._process_stream_chunk(
                    obj, detok, token_mode, sent_reasoning
                )
                await stream.write(self._build_sse_chunk(completion_id, created, model, delta_payload, finish))

                stop = obj.get("stop", False)
                if stop or (token_mode and detok.stopped):
                    return await self._stream_end(stream, response, detok, token_mode, stop, finish)

        if token_mode:
            tail = detok.finish()
            if tail:
                await stream.write(self._build_sse_chunk(completion_id, created, model, tail, None))
        await stream.write(b"data: [DONE]\n\n")
        return stream

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
                response = web.StreamResponse(status=upstream_response.status)
                response.headers.update(upstream_response.headers)
                await response.prepare(request)
                async for chunk in upstream_response.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                return response
        except aiohttp.ClientError as exc:
            raise web.HTTPBadGateway(reason=f"upstream request failed: {exc}") from exc

    async def _generic(self, request: web.Request) -> web.StreamResponse:
        """Catch-all reverse proxy for any endpoint not handled above."""
        return await self._forward_to_upstream(request, request.path)

    async def _models(self, request: web.Request) -> web.Response:
        if not self.ready:
            raise web.HTTPServiceUnavailable(reason="proxy not ready")
        model_id = self.config.model.model or "local"
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

    async def _health(self, request: web.Request) -> web.Response:
        if not self.ready or self.session is None:
            raise web.HTTPServiceUnavailable(reason="proxy not ready")
        try:
            async with self.session.get(f"{self.upstream}/health") as response:
                text = await response.text()
                return web.Response(text=text, status=response.status)
        except aiohttp.ClientError as exc:
            raise web.HTTPBadGateway(reason=f"upstream health failed: {exc}") from exc

    async def run(self) -> None:
        runner = web.AppRunner(self.app)
        await runner.setup()

        sites: List[web.BaseSite] = []
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
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
            self.state.local_port = self.port
            self.state.save()

        try:
            while True:
                await asyncio.sleep(3600)
        finally:
            for site in sites:
                await site.stop()
            await runner.cleanup()
            self.socket_path.unlink(missing_ok=True)


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
    """Poll upstream /health until the model is ready to serve requests."""
    if proxy.session is None:
        return False
    while not server_task.done():
        try:
            async with proxy.session.get(f"{proxy.upstream}/health") as response:
                if response.status == 200:
                    return True
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(interval)
    # The server task died (likely a startup failure); re-raise its exception.
    await server_task
    return False


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
        raise RuntimeError("upstream /health did not become ready")

    props = await _fetch_props_once(config, state)
    chat_template: Optional[str] = None
    if props and props.get("chat_template"):
        chat_template = props["chat_template"]
        _logger.info("using chat template from remote /props")

    proxy.tokenizer = Tokenizer(config, chat_template=chat_template)
    proxy.ready = True
    _logger.info("proxy ready")


def _proxy_log_file(config: Config) -> Path:
    return config.root_dir / ".hostai-cache" / "proxy.log"


def _configure_proxy_logging(config: Config) -> Path:
    """Attach a file handler so proxy activity is logged locally.

    Only operational metadata is logged (token counts, timings, warnings) —
    never prompt or generated content.
    """
    log_file = _proxy_log_file(config)
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
    log_file = _configure_proxy_logging(config)
    _logger.info("proxy starting, logging to %s", log_file)

    # Record the proxy pid so hostai down can stop it.
    state.data["proxy_pid"] = os.getpid()
    state.save()

    socket_path = Path(config.proxy.socket_path) if config.proxy.socket_path else _default_socket_path(state)
    port = config.proxy.port or config.ssh.local_port or 0

    # Pre-register the upstream Unix socket path so the proxy can create its
    # aiohttp app immediately.  The SSH tunnel (and remote socket) is set up
    # concurrently below; the proxy returns 503 until the model is ready.
    upstream_socket = state.data.get("upstream_socket") or str(state.state_file.parent / "upstream.sock")
    state.data["upstream_socket"] = upstream_socket
    state.save()

    tokenizer = Tokenizer(config)
    proxy = TokenizedProxy(config, state, tokenizer, socket_path, port)

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

    await server_task
