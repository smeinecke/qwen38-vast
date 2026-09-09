"""Tests for hostai.remote_guard."""

import asyncio
from unittest import mock

import pytest
from aiohttp import web

from hostai.remote_guard import TokenOnlyGuard, _is_prompt_tokenized, main


def test_is_prompt_tokenized_accepts_integers():
    assert _is_prompt_tokenized([1, 2, 3]) is True


def test_is_prompt_tokenized_rejects_bool_in_list():
    assert _is_prompt_tokenized([1, True, 3]) is False


def test_is_prompt_tokenized_rejects_strings():
    assert _is_prompt_tokenized([1, "hello", 3]) is False


def test_is_prompt_tokenized_rejects_non_list():
    assert _is_prompt_tokenized("hello") is False


def _make_request(method, path_qs, body=b""):
    return mock.Mock(method=method, path_qs=path_qs, headers={}, read=mock.AsyncMock(return_value=body))


def test_handler_allows_get_request():
    async def _test():
        guard = TokenOnlyGuard("/tmp/test-public.sock", "/tmp/test-backend.sock")
        with mock.patch.object(guard, "_forward", return_value=web.Response(text="ok")) as fwd:
            response = await guard._handler(_make_request("GET", "/v1/models"))
        assert response.text == "ok"
        fwd.assert_awaited_once()
        await guard.session.close()

    asyncio.run(_test())


def test_handler_rejects_unknown_method():
    async def _test():
        guard = TokenOnlyGuard("/tmp/test-public.sock", "/tmp/test-backend.sock")
        with pytest.raises(web.HTTPMethodNotAllowed):
            await guard._handler(_make_request("PATCH", "/v1/models"))
        await guard.session.close()

    asyncio.run(_test())


def test_forward_blocks_disabled_endpoints():
    async def _test():
        guard = TokenOnlyGuard("/tmp/test-public.sock", "/tmp/test-backend.sock")
        with pytest.raises(web.HTTPForbidden):
            await guard._forward(_make_request("POST", "/v1/chat/completions"))
        await guard.session.close()

    asyncio.run(_test())


def test_forward_rejects_plain_text_prompt(tmp_path):
    async def _test():
        guard = TokenOnlyGuard(str(tmp_path / "public.sock"), str(tmp_path / "backend.sock"))
        with pytest.raises(web.HTTPBadRequest):
            await guard._forward(_make_request("POST", "/completion", b'{"prompt": "hello"}'))
        await guard.session.close()

    asyncio.run(_test())


def test_forward_accepts_tokenized_prompt(tmp_path):
    async def _test():
        guard = TokenOnlyGuard(str(tmp_path / "public.sock"), str(tmp_path / "backend.sock"))
        # No backend server, so it will fail with 502, but the prompt validation passes.
        with pytest.raises(web.HTTPBadGateway):
            await guard._forward(_make_request("POST", "/completion", b'{"prompt": [1, 2, 3]}'))
        await guard.session.close()

    asyncio.run(_test())


def test_forward_rejects_invalid_json(tmp_path):
    async def _test():
        guard = TokenOnlyGuard(str(tmp_path / "public.sock"), str(tmp_path / "backend.sock"))
        with pytest.raises(web.HTTPBadRequest):
            await guard._forward(_make_request("POST", "/completion", b"not json"))
        await guard.session.close()

    asyncio.run(_test())


def test_main_starts_guard():
    with mock.patch("hostai.remote_guard.TokenOnlyGuard") as guard_cls:
        instance = guard_cls.return_value
        instance.run = mock.AsyncMock(side_effect=asyncio.CancelledError)
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(main())
        guard_cls.assert_called_once()
