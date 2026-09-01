"""Tests for hostai.api with mocked HTTP responses."""

import json
from unittest import mock

import pytest
import requests
import responses

from hostai.api import LlamaClient, _parse_prom, is_api_ready, wait_for_api


@pytest.fixture
def client(config, state):
    state.local_port = 18080
    state.api_key = "test-key"
    state.unsecure = True
    return LlamaClient(config, state)


def test_parse_prom_basic():
    text = """# HELP foo
foo 1.0
bar 2.5
"""
    assert _parse_prom(text) == {"foo": 1.0, "bar": 2.5}


def test_parse_prom_empty():
    assert _parse_prom("") == {}
    assert _parse_prom(None) == {}


def test_parse_prom_json_wrapped_string():
    text = '"foo 1.0\\nbar 2.5"'
    assert _parse_prom(text) == {"foo": 1.0, "bar": 2.5}


@responses.activate
def test_health_ok_text(client):
    responses.add(responses.GET, "http://127.0.0.1:18080/health", body="ok", status=200)
    assert client.health() is True


@responses.activate
def test_health_ok_json(client):
    responses.add(
        responses.GET,
        "http://127.0.0.1:18080/health",
        json={"status": "ok"},
        status=200,
    )
    assert client.health() is True


@responses.activate
def test_health_not_ok(client):
    responses.add(responses.GET, "http://127.0.0.1:18080/health", body="loading", status=200)
    assert client.health() is False


@responses.activate
def test_health_connection_error(client):
    responses.add(responses.GET, "http://127.0.0.1:18080/health", body=requests.ConnectionError("Connection refused"))
    assert client.health() is False


@responses.activate
def test_wait_for_health_succeeds(client):
    responses.add(responses.GET, "http://127.0.0.1:18080/health", body="ok", status=200)
    assert client.wait_for_health(timeout=1.0, quiet=True) is True


@responses.activate
def test_wait_for_health_times_out(client):
    responses.add(responses.GET, "http://127.0.0.1:18080/health", body="loading", status=503)
    assert client.wait_for_health(timeout=0.1, quiet=True) is False


@responses.activate
def test_wait_for_health_zero_timeout(client):
    responses.add(responses.GET, "http://127.0.0.1:18080/health", body="ok", status=200)
    assert client.wait_for_health(timeout=0.0, quiet=True) is True


@responses.activate
def test_get_metrics_text(client):
    prom = "llamacpp:prompt_tokens_total 42\n"
    responses.add(responses.GET, "http://127.0.0.1:18080/metrics", body=prom, status=200)
    assert client.get_metrics_text() == prom


@responses.activate
def test_get_metrics_parsed(client):
    prom = "llamacpp:prompt_tokens_total 42\n"
    responses.add(responses.GET, "http://127.0.0.1:18080/metrics", body=prom, status=200)
    assert client.get_metrics() == {"llamacpp:prompt_tokens_total": 42.0}


@responses.activate
def test_get_metrics_non_200(client):
    responses.add(responses.GET, "http://127.0.0.1:18080/metrics", body="error", status=500)
    assert client.get_metrics() == {}


@responses.activate
def test_slots_returns_list(client):
    responses.add(
        responses.GET,
        "http://127.0.0.1:18080/slots",
        json=[{"id": 0, "n_ctx": 4096}],
        status=200,
    )
    assert client.slots() == [{"id": 0, "n_ctx": 4096}]


@responses.activate
def test_slots_returns_empty_on_error(client):
    responses.add(responses.GET, "http://127.0.0.1:18080/slots", body="error", status=500)
    assert client.slots() == []


@responses.activate
def test_slot_save_accepts_200_and_202(client):
    for status in (200, 202):
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://127.0.0.1:18080/slots/0?action=save",
                status=status,
            )
            assert client.slot_save(0) is True


@responses.activate
def test_slot_save_false_on_error(client):
    responses.add(
        responses.POST,
        "http://127.0.0.1:18080/slots/0?action=save",
        status=500,
    )
    assert client.slot_save(0) is False


@responses.activate
def test_chat_non_stream(client):
    responses.add(
        responses.POST,
        "http://127.0.0.1:18080/v1/chat/completions",
        json={"choices": [{"message": {"content": "hi"}}]},
        status=200,
    )
    result = client.chat([{"role": "user", "content": "hello"}], max_tokens=10, temperature=0.7)
    assert result == {"choices": [{"message": {"content": "hi"}}]}

    request = json.loads(responses.calls[0].request.body)
    assert request["model"] == "model.gguf"
    assert request["messages"] == [{"role": "user", "content": "hello"}]
    assert request["max_tokens"] == 10
    assert request["temperature"] == 0.7
    assert request["stream"] is False


def test_chat_validation(client):
    with pytest.raises(ValueError, match="messages must not be empty"):
        client.chat([], max_tokens=10, temperature=0.7)
    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        client.chat([{"role": "user", "content": "hi"}], max_tokens=0, temperature=0.7)
    with pytest.raises(ValueError, match="temperature must be between"):
        client.chat([{"role": "user", "content": "hi"}], max_tokens=10, temperature=2.5)
    with pytest.raises(ValueError, match="timeout must be a positive number"):
        client.chat([{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.7, timeout=0)


def test_chat_streaming(client):
    response = mock.Mock()
    response.status_code = 200
    response.raise_for_status = mock.Mock()
    response.iter_lines.return_value = [
        'data: {"choices":[{"delta":{"content":"h"}}]}',
        'data: {"choices":[{"delta":{"content":"i"}}]}',
        "data: [DONE]",
    ]

    with mock.patch("requests.post", return_value=response) as post:
        stream = client.chat([{"role": "user", "content": "hello"}], max_tokens=10, temperature=0.7, stream=True)
        chunks = list(stream)

    assert len(chunks) == 2
    request = post.call_args.kwargs["json"]
    assert request["stream"] is True
    assert request["stream_options"] == {"include_usage": True}
    assert request["max_tokens"] == 10
    assert request["temperature"] == 0.7
    assert chunks[0]["choices"][0]["delta"]["content"] == "h"
    assert chunks[1]["choices"][0]["delta"]["content"] == "i"


def test_chat_streaming_ignores_malformed_lines(client):
    response = mock.Mock()
    response.status_code = 200
    response.raise_for_status = mock.Mock()
    response.iter_lines.return_value = [
        "not a data line",
        "data: not json",
        'data: {"choices":[{"delta":{"content":"ok"}}]}',
    ]

    with mock.patch("requests.post", return_value=response):
        stream = client.chat([{"role": "user", "content": "hello"}], max_tokens=10, temperature=0.7, stream=True)
        chunks = list(stream)

    assert len(chunks) == 1


@responses.activate
def test_is_api_ready(config, state):
    state.local_port = 18080
    state.unsecure = True
    responses.add(responses.GET, "http://127.0.0.1:18080/health", body="ok", status=200)
    assert is_api_ready(config, state) is True


@responses.activate
def test_wait_for_api(config, state):
    state.local_port = 18080
    state.unsecure = True
    responses.add(responses.GET, "http://127.0.0.1:18080/health", body="ok", status=200)
    assert wait_for_api(config, state, timeout=1.0) is True
