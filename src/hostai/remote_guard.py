#!/usr/bin/env python3
"""Remote token-only guard that sits in front of llama-server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
import sys
from pathlib import Path
from typing import Any, Optional

import aiohttp
from aiohttp import web

_logger = logging.getLogger(__name__)

PUBLIC_SOCKET = os.environ.get("HOSTAI_GUARD_LISTEN", "/dev/shm/qwen38/llama.sock")
BACKEND_SOCKET = os.environ.get(
    "HOSTAI_GUARD_BACKEND", "/dev/shm/qwen38/llama-internal.sock"
)
CERT_FILE = os.environ.get("HOSTAI_GUARD_CERT", "/dev/shm/qwen38/certs/server.crt")
KEY_FILE = os.environ.get("HOSTAI_GUARD_KEY", "/dev/shm/qwen38/certs/server.key")

BLOCKED_PATHS: set = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/rerank",
    "/tokenize",
    "/detokenize",
    "/apply-template",
    "/chat/format",
    "/infill",
}

ALLOWED_METHODS = {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"}


def _is_prompt_tokenized(prompt: Any) -> bool:
    """Return True only if prompt is a list of integers."""
    if not isinstance(prompt, list):
        return False
    for token in prompt:
        if isinstance(token, bool):
            return False
        if not isinstance(token, int):
            return False
    return True


class TokenOnlyGuard:
    """Reverse proxy that rejects any request carrying plain text."""

    def __init__(
        self,
        public_socket: str,
        backend_socket: str,
        cert_file: Optional[str] = None,
        key_file: Optional[str] = None,
    ) -> None:
        self.public_socket = public_socket
        self.backend_socket = backend_socket
        self.cert_file = cert_file
        self.key_file = key_file
        self.connector = aiohttp.UnixConnector(path=backend_socket)
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=aiohttp.ClientTimeout(total=None, connect=10, sock_read=900),
        )

    async def _forward(self, request: web.Request) -> web.StreamResponse:
        path = request.path_qs
        method = request.method

        if path in BLOCKED_PATHS:
            raise web.HTTPForbidden(
                reason="endpoint disabled in tokenized-only mode"
            )

        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower()
            not in {
                "host",
                "content-length",
                "transfer-encoding",
                "connection",
            }
        }
        headers["Host"] = "localhost"

        body = await request.read()

        if method == "POST" and path == "/completion" and body:
            try:
                data = json.loads(body)
            except json.JSONDecodeError as exc:
                raise web.HTTPBadRequest(reason=f"invalid JSON: {exc}") from exc

            prompt = data.get("prompt")
            if not _is_prompt_tokenized(prompt):
                raise web.HTTPBadRequest(
                    reason="tokenized-only: prompt must be an array of token IDs"
                )

        try:
            async with self.session.request(
                method,
                f"http://localhost{path}",
                headers=headers,
                data=body or None,
            ) as response:

                resp_headers = {
                    k: v
                    for k, v in response.headers.items()
                    if k.lower()
                    not in {
                        "transfer-encoding",
                        "content-length",
                        "connection",
                    }
                }

                proxy = web.StreamResponse(
                    status=response.status,
                    reason=response.reason,
                    headers=resp_headers,
                )
                await proxy.prepare(request)

                async for chunk in response.content.iter_chunked(8192):
                    await proxy.write(chunk)

                await proxy.write_eof()
                return proxy
        except aiohttp.ClientError as exc:
            _logger.error("backend request failed: %s", exc)
            raise web.HTTPBadGateway(reason=f"backend unreachable: {exc}") from exc

    async def _handler(self, request: web.Request) -> web.StreamResponse:
        if request.method not in ALLOWED_METHODS:
            raise web.HTTPMethodNotAllowed(request.method, ALLOWED_METHODS)
        return await self._forward(request)

    async def run(self) -> None:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_route("*", "/{path:.*}", self._handler)

        runner = web.AppRunner(app)
        await runner.setup()

        public_path = Path(self.public_socket)
        public_path.parent.mkdir(parents=True, exist_ok=True)
        public_path.unlink(missing_ok=True)

        ssl_ctx: Optional[ssl.SSLContext] = None
        if self.cert_file and self.key_file and Path(self.cert_file).exists() and Path(self.key_file).exists():
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(self.cert_file, self.key_file)

        site = web.UnixSite(runner, self.public_socket, ssl_context=ssl_ctx)
        await site.start()
        if ssl_ctx:
            _logger.info("token-only guard listening on %s (TLS)", self.public_socket)
        else:
            _logger.info("token-only guard listening on %s (plain)", self.public_socket)

        try:
            while True:
                await asyncio.sleep(3600)
        finally:
            await runner.cleanup()
            public_path.unlink(missing_ok=True)
            await self.session.close()


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    guard = TokenOnlyGuard(
        public_socket=PUBLIC_SOCKET,
        backend_socket=BACKEND_SOCKET,
        cert_file=CERT_FILE,
        key_file=KEY_FILE,
    )
    await guard.run()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
