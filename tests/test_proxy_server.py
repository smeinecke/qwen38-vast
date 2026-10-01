"""Integration-style tests for the tokenizing proxy using aiohttp TestServer."""

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from hostai.proxy import TokenizedProxy


class _FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        # Requests carrying tools use the tool-call response fixture.
        if kwargs.get("tools"):
            return [7]
        return [1, 2, 3]

    def decode(self, token_ids, **kwargs):
        mapping = {
            1: "hi ",
            2: "there",
            3: "!",
            7: "<tool_call>\n<function=bash>\n<parameter=command>\nls",
            8: " -la\n</parameter>\n</function>\n</tool_call>",
        }
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
        tool_prompt = body.get("prompt") == [7]
        if body.get("stream"):
            response = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            if tool_prompt:
                chunks = [
                    {"tokens": [7], "stop": False},
                    {"tokens": [8], "stop": True, "stopped_eos": True},
                ]
            else:
                chunks = [
                    {"tokens": [1, 2], "stop": False},
                    {"tokens": [3], "stop": True, "stopped_eos": True},
                ]
            for chunk in chunks:
                await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
            return response
        tokens = [7, 8] if tool_prompt else [1, 2, 3]
        return web.json_response({
            "tokens": tokens,
            "tokens_predicted": len(tokens),
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


async def run_proxy(config, running_state, fake_tokenizer, tmp_path, requests, app_factory=upstream_app):
    async with TestServer(await app_factory(), host="127.0.0.1") as upstream:
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


_TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
        },
    },
}]


def test_proxy_complete_tool_calls(config, running_state, fake_tokenizer, tmp_path):
    running_state.unsecure = True
    config.proxy.tokenized_only = True
    config.model.model = "qwen-test"

    async def requests(client):
        resp = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "list files"}],
            "tools": _TOOLS,
            "stream": False,
        })
        assert resp.status == 200
        data = await resp.json()
        choice = data["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        message = choice["message"]
        assert message["content"] is None
        calls = message["tool_calls"]
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "bash"
        assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls -la"}

    asyncio.run(run_proxy(config, running_state, fake_tokenizer, tmp_path, requests))


def test_proxy_stream_tool_calls(config, running_state, fake_tokenizer, tmp_path):
    running_state.unsecure = True
    config.proxy.tokenized_only = True
    config.model.model = "qwen-test"

    async def requests(client):
        resp = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "list files"}],
            "tools": _TOOLS,
            "stream": True,
        })
        assert resp.status == 200
        text = await resp.text()
        chunks = [
            json.loads(line[5:].strip())
            for line in text.splitlines()
            if line.startswith("data:") and line[5:].strip() != "[DONE]"
        ]
        deltas = [c["choices"][0]["delta"] for c in chunks]
        tool_deltas = [d for d in deltas if d.get("tool_calls")]
        assert len(tool_deltas) == 1
        call = tool_deltas[0]["tool_calls"][0]
        assert call["index"] == 0
        assert call["function"]["name"] == "bash"
        assert json.loads(call["function"]["arguments"]) == {"command": "ls -la"}
        # No raw markup may leak into content deltas.
        assert not any("<tool_call>" in (d.get("content") or "") for d in deltas)
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"

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


def test_proxy_content_log(config, running_state, fake_tokenizer, tmp_path):
    running_state.unsecure = True
    config.proxy.tokenized_only = True
    config.proxy.log_content = True
    config.model.model = "qwen-test"
    log_file = config.root_dir / ".hostai-cache" / "proxy-content.jsonl"

    async def requests(client):
        resp = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "stream": True,
        })
        assert resp.status == 200
        await resp.text()

        resp = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "stream": False,
        })
        assert resp.status == 200
        await resp.json()

    asyncio.run(run_proxy(config, running_state, fake_tokenizer, tmp_path, requests))

    records = [json.loads(line) for line in log_file.read_text().splitlines()]

    reqs = [r for r in records if r["event"] == "request"]
    assert len(reqs) == 2
    assert reqs[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert reqs[0]["prompt_tokens"] == 3
    assert reqs[0]["stream"] is True

    deltas = [r for r in records if r["event"] == "delta"]
    assert deltas
    assert all(d["id"] == reqs[0]["id"] for d in deltas)

    done = [r for r in records if r["event"] == "done"]
    assert done and done[0]["completion_tokens"] == 3

    responses = [r for r in records if r["event"] == "response"]
    assert len(responses) == 1
    assert responses[0]["id"] == reqs[1]["id"]
    assert responses[0]["message"]["content"] == "hi there!"
    assert responses[0]["usage"]["prompt_tokens"] == 3


def test_proxy_content_log_passthrough(config, running_state, fake_tokenizer, tmp_path):
    running_state.unsecure = True
    config.proxy.tokenized_only = False
    config.proxy.log_content = True
    log_file = config.root_dir / ".hostai-cache" / "proxy-content.jsonl"

    async def requests(client):
        resp = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
        assert resp.status == 200
        await resp.json()

    asyncio.run(run_proxy(config, running_state, fake_tokenizer, tmp_path, requests))

    records = [json.loads(line) for line in log_file.read_text().splitlines()]
    req = next(r for r in records if r["event"] == "request")
    assert req["passthrough"] is True
    assert req["body"]["messages"] == [{"role": "user", "content": "hi"}]

    upstream_resp = next(r for r in records if r["event"] == "upstream_response")
    assert upstream_resp["id"] == req["id"]
    assert upstream_resp["status"] == 200

    chunks = [r for r in records if r["event"] == "upstream_chunk"]
    assert chunks and "v1/chat/completions" in chunks[0]["data"]


def test_proxy_no_content_log_by_default(config, running_state, fake_tokenizer, tmp_path):
    running_state.unsecure = True
    config.proxy.tokenized_only = True
    log_file = config.root_dir / ".hostai-cache" / "proxy-content.jsonl"

    async def requests(client):
        resp = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "stream": False,
        })
        assert resp.status == 200

    asyncio.run(run_proxy(config, running_state, fake_tokenizer, tmp_path, requests))
    assert not log_file.exists()


def test_proxy_client_disconnect_aborts_upstream_stream(config, running_state, fake_tokenizer, tmp_path):
    """A client that hangs up mid-SSE stream must abort upstream generation —
    closing the upstream response is what tells llama-server to stop."""
    running_state.unsecure = True
    config.proxy.tokenized_only = True
    config.model.model = "qwen-test"
    flags = {"aborted": False}

    async def endless_app():
        app = web.Application()

        async def completion(request):
            response = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            try:
                while True:
                    await response.write(b'data: {"tokens": [1], "stop": false}\n\n')
                    await asyncio.sleep(0.02)
            except (ConnectionError, asyncio.CancelledError):
                flags["aborted"] = True
                raise

        app.router.add_post("/completion", completion)
        return app

    async def requests(client):
        resp = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "stream": True,
        })
        assert resp.status == 200
        first = await resp.content.readany()
        assert b"data:" in first
        resp.close()
        for _ in range(100):
            if flags["aborted"]:
                break
            await asyncio.sleep(0.05)
        assert flags["aborted"], "proxy did not abort upstream generation on client disconnect"

    asyncio.run(run_proxy(config, running_state, fake_tokenizer, tmp_path, requests, app_factory=endless_app))


def test_proxy_client_disconnect_aborts_nonstream(config, running_state, fake_tokenizer, tmp_path):
    """Non-streaming: a client disconnect while llama-server is still
    generating must close the upstream request too."""
    running_state.unsecure = True
    config.proxy.tokenized_only = True
    config.model.model = "qwen-test"
    flags = {"aborted": False}

    async def slow_app():
        app = web.Application()

        async def completion(request):
            try:
                for _ in range(200):
                    transport = request.transport
                    if transport is None or transport.is_closing():
                        flags["aborted"] = True
                        return web.Response(status=200)
                    await asyncio.sleep(0.05)
                return web.json_response({"tokens": [1], "stop": True})
            except (asyncio.CancelledError, ConnectionError):
                # Upstream connection dropped = generation aborted.
                flags["aborted"] = True
                raise

        app.router.add_post("/completion", completion)
        return app

    async def requests(client):
        # The upstream only responds after ~10s, so drive the request on a
        # dedicated force_close session and drop the socket while the proxy
        # is still waiting for the response body.
        import aiohttp as _aiohttp

        session = _aiohttp.ClientSession(connector=_aiohttp.TCPConnector(force_close=True))
        try:
            task = asyncio.ensure_future(
                session.post(str(client.make_url("/v1/chat/completions")), json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 10,
                    "stream": False,
                })
            )
            await asyncio.sleep(0.2)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        finally:
            await session.close()
        for _ in range(100):
            if flags["aborted"]:
                break
            await asyncio.sleep(0.05)
        assert flags["aborted"], "proxy did not abort non-stream upstream request on client disconnect"

    asyncio.run(run_proxy(config, running_state, fake_tokenizer, tmp_path, requests, app_factory=slow_app))
