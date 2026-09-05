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
# generated output normally contains only the closing marker. Split on it.
_THINK_START = "▶"
_THINK_END = "◀"


def _split_reasoning(content: str) -> Tuple[str, str]:
    """Split raw completion output into (reasoning, answer).

    The opening thinking marker lives in the prompt, so the model output
    typically contains only the closing tag. If the model also emitted the
    opening tag, it is stripped from the reasoning text.
    """
    if _THINK_END not in content:
        return "", content
    reasoning, answer = content.split(_THINK_END, 1)
    if _THINK_START in reasoning:
        reasoning = reasoning.split(_THINK_START, 1)[1]
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


def _strip_tool_call_tags(content: str) -> str:
    """Return content with tool-call tags removed, preserving surrounding text."""
    return _TOOL_CALL_RE.sub("", content).strip()


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
            connector: aiohttp.BaseConnector = UnixTLSConnector(path=self.upstream_socket, ssl=self.ssl_ctx, limit=20)
        else:
            connector = aiohttp.TCPConnector(ssl=self.ssl_ctx, limit=20)
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

        try:
            token_ids = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                **default_reasoning_kwargs(self.config),
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

        if stream:
            return await self._stream_chat(request, upstream_response)
        return await self._complete_chat(upstream_response, len(token_ids))

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
    ) -> web.Response:
        try:
            data = await response.json()
        except json.JSONDecodeError as exc:
            raise web.HTTPInternalServerError(reason=f"invalid upstream JSON: {exc}") from exc

        content = data.get("content", "")
        reasoning = data.get("reasoning_content") or ""
        if not reasoning:
            reasoning, content = _split_reasoning(content)
        finish_reason = self._map_finish_reason(data)
        completion_tokens = data.get("tokens_predicted", 0) or 0

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

    async def _stream_chat(
        self,
        request: web.Request,
        response: aiohttp.ClientResponse,
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

                # The final chunk carries the full content/reasoning_content,
                # not deltas. Its content was already streamed, so only the
                # reasoning blob is forwarded, and only when no deltas were sent.
                stop = obj.get("stop", False)
                if stop:
                    delta_payload = {}
                    reasoning = (obj.get("reasoning_content") or "") if not sent_reasoning else ""
                    if reasoning:
                        delta_payload["reasoning_content"] = reasoning
                else:
                    delta_payload = {}
                    delta = obj.get("content", "")
                    if delta:
                        delta_payload["content"] = delta
                    reasoning_delta = obj.get("reasoning_content", "")
                    if reasoning_delta:
                        delta_payload["reasoning_content"] = reasoning_delta
                        sent_reasoning = True

                chunk: Dict[str, Any] = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta_payload,
                            "finish_reason": None,
                        }
                    ],
                }
                if stop:
                    chunk["choices"][0]["finish_reason"] = self._map_finish_reason(obj)

                await stream.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                if stop:
                    await stream.write(b"data: [DONE]\n\n")
                    return stream

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


async def run_proxy(config: Config, state: State) -> None:
    """Start the local proxy for the active state.

    The proxy owns its own SSH tunnel to the remote Unix socket and keeps the
    connection alive as long as it is running. When tokenized-only is enabled it
    tokenizes /v1/chat/completions; otherwise it passes traffic through.
    """
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
