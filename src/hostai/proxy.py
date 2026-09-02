"""Local OpenAI-compatible proxy that tokenizes prompts before forwarding."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import web

from hostai.config import Config
from hostai.state import State
from hostai.tokenize import Tokenizer, TokenizerError, default_reasoning_kwargs

_logger = logging.getLogger(__name__)


class ProxyError(Exception):
    """The tokenized proxy cannot handle a request."""


def _default_socket_path(state: State) -> Path:
    return state.state_file.parent / "proxy.sock"


def _upstream_url(config: Config, state: State) -> str:
    if state.unsecure:
        return f"http://127.0.0.1:{state.local_port}"
    return f"https://127.0.0.1:{state.local_port}"


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
        self.upstream = _upstream_url(config, state)
        self.api_key = state.api_key or ""
        self.ssl_ctx = _ssl_context(state)
        self.session: Optional[aiohttp.ClientSession] = None
        self.app = web.Application(client_max_size=64 * 1024 * 1024)
        self.app.router.add_post("/v1/chat/completions", self._chat)
        self.app.router.add_get("/v1/models", self._models)
        self.app.router.add_get("/health", self._health)
        self.app.on_startup.append(self._on_startup)
        self.app.on_cleanup.append(self._on_cleanup)

    async def _on_startup(self, app: web.Application) -> None:
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
        if self.session is None:
            raise web.HTTPServiceUnavailable(reason="proxy client not initialized")

        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise web.HTTPBadRequest(reason=f"invalid JSON: {exc}") from exc

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

        payload: Dict[str, Any] = {
            "prompt": token_ids,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if "top_p" in body:
            payload["top_p"] = body["top_p"]
        if "top_k" in body:
            payload["top_k"] = body["top_k"]
        if "stop" in body:
            payload["stop"] = body["stop"]

        upstream_response = await self.session.post(
            f"{self.upstream}/completion",
            json=payload,
        )

        if upstream_response.status != 200:
            text = await upstream_response.text()
            raise web.HTTPInternalServerError(
                reason=f"upstream returned {upstream_response.status}: {text[:200]}"
            )

        if stream:
            return await self._stream_chat(request, upstream_response)
        return await self._complete_chat(upstream_response, len(token_ids))

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
        finish_reason = self._map_finish_reason(data)
        completion_tokens = data.get("tokens_predicted", 0) or 0

        output = {
            "id": f"chatcmpl-{os.urandom(12).hex()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.config.model.model or "local",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
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

                delta = obj.get("content", "")
                stop = obj.get("stop", False)

                chunk: Dict[str, Any] = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta},
                            "finish_reason": None,
                        }
                    ],
                }
                if stop:
                    chunk["choices"][0]["finish_reason"] = self._map_finish_reason(obj)
                    chunk["choices"][0]["delta"] = {}

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

    async def _models(self, request: web.Request) -> web.Response:
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
        if self.session is None:
            raise web.HTTPServiceUnavailable(reason="proxy client not initialized")
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
        _logger.info("proxy listening on unix socket %s", self.socket_path)

        if self.port:
            tcp_site = web.TCPSite(runner, "127.0.0.1", self.port)
            await tcp_site.start()
            sites.append(tcp_site)
            _logger.info("proxy listening on tcp 127.0.0.1:%d", self.port)

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
    upstream = _upstream_url(config, state)
    ssl_ctx = _ssl_context(state)
    api_key = state.api_key or ""
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=ssl_ctx),
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as session:
            async with session.get(f"{upstream}/props") as response:
                if response.status == 200:
                    return await response.json()
    except aiohttp.ClientError:
        pass
    return None


async def run_proxy(config: Config, state: State) -> None:
    """Start the local tokenized proxy for the active state."""
    if not state.local_port:
        raise ProxyError("no active SSH tunnel; run 'hostai up' first")

    socket_path = Path(config.proxy.socket_path) if config.proxy.socket_path else _default_socket_path(state)
    port = config.proxy.port or 0

    chat_template: Optional[str] = None
    props = await _fetch_props_once(config, state)
    if props and props.get("chat_template"):
        chat_template = props["chat_template"]
        _logger.info("using chat template from remote /props")

    tokenizer = Tokenizer(config, chat_template=chat_template)
    proxy = TokenizedProxy(config, state, tokenizer, socket_path, port)
    await proxy.run()
