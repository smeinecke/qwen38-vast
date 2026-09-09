"""Integration-style tests for the tokenizing proxy using aiohttp TestServer."""

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from hostai.proxy import TokenizedProxy


class _FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]

    def decode(self, token_ids, **kwargs):
        mapping = {1: "hi ", 2: "there", 3: "!"}
        return "".join(mapping.get(i, "") for i in token_ids)

    @property
    def eos_token_id(self):
        return 3


async def upstream_app():
    app = web.Application()

    async def health(request):
        return web.json_response({"status": "ok"})

    async def completion(request):
        body = await request.json()
        if body.get("stream"):
            response = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            chunk = json.dumps({"tokens": [1, 2], "stop": False})
            await response.write(f"data: {chunk}\n\n".encode())
            chunk = json.dumps({"tokens": [3], "stop": True, "stopped_eos": True})
            await response.write(f"data: {chunk}\n\n".encode())
            return response
        return web.json_response({
            "tokens": [1, 2, 3],
            "tokens_predicted": 3,
            "stop": True,
            "stopped_eos": True,
        })

    async def echo(request):
        return web.json_response({"path": request.path})

    app.router.add_get("/health", health)
    app.router.add_post("/completion", completion)
    app.router.add_route("*", "/{path:.*}", echo)
    return app


@pytest.fixture
def fake_tokenizer():
    return _FakeTokenizer()


async def run_proxy(config, running_state, fake_tokenizer, tmp_path, requests):
    async with TestServer(await upstream_app(), host="127.0.0.1") as upstream:
        running_state.local_port = upstream.port
        socket_path = tmp_path / "proxy.sock"
        proxy = TokenizedProxy(config, running_state, fake_tokenizer, socket_path, port=0)
        proxy.ready = True
        async with TestServer(proxy.app, host="127.0.0.1") as proxy_server:
            async with TestClient(proxy_server) as client:
                await requests(client)


def test_proxy_health_and_models(config, running_state, fake_tokenizer, tmp_path):
    running_state.unsecure = True
    config.proxy.tokenized_only = True
    config.model.model = "qwen-test"

    async def requests(client):
        resp = await client.get("/health")
        assert resp.status == 200

        resp = await client.get("/v1/models")
        assert resp.status == 200
        data = await resp.json()
        assert data["data"][0]["id"] == config.model.model

        resp = await client.get("/metrics")
        assert resp.status == 200

        resp = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "stream": False,
        })
        assert resp.status == 200
        data = await resp.json()
        assert data["object"] == "chat.completion"

    asyncio.run(run_proxy(config, running_state, fake_tokenizer, tmp_path, requests))


def test_proxy_stream_chat(config, running_state, fake_tokenizer, tmp_path):
    running_state.unsecure = True
    config.proxy.tokenized_only = True
    config.model.model = "qwen-test"

    async def requests(client):
        resp = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "stream": True,
        })
        assert resp.status == 200
        text = await resp.text()
        assert text.startswith("data:")

    asyncio.run(run_proxy(config, running_state, fake_tokenizer, tmp_path, requests))


def test_proxy_completions_forwarded(config, running_state, fake_tokenizer, tmp_path):
    running_state.unsecure = True
    config.proxy.tokenized_only = True
    config.model.model = "qwen-test"

    async def requests(client):
        resp = await client.post("/v1/completions", json={
            "prompt": "hi",
            "max_tokens": 10,
            "stream": False,
        })
        assert resp.status == 200
        data = await resp.json()
        assert data["path"] == "/v1/completions"

    asyncio.run(run_proxy(config, running_state, fake_tokenizer, tmp_path, requests))


def test_proxy_generic_route(config, running_state, fake_tokenizer, tmp_path):
    running_state.unsecure = True
    config.proxy.tokenized_only = True
    config.model.model = "qwen-test"

    async def requests(client):
        resp = await client.get("/unknown/path")
        assert resp.status == 200
        data = await resp.json()
        assert data["path"] == "/unknown/path"

    asyncio.run(run_proxy(config, running_state, fake_tokenizer, tmp_path, requests))
