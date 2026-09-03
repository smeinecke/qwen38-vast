"""Tests for hostai.vast with mocked vastai SDK calls."""

from unittest import mock

import pytest

from hostai import vast


def config_with_key(config):
    config.secrets["VAST_API_KEY"] = "test-key"
    return config


def test_api_key_missing(config):
    config.secrets = {}
    with pytest.raises(vast.VastError, match="VAST_API_KEY"):
        vast._api_key(config)


def test_api_key_prefers_vast_api_key(config):
    config.secrets["VAST_API_KEY"] = "first"
    config.secrets["VASTAI_API_KEY"] = "second"
    assert vast._api_key(config) == "first"


def test_api_key_fallback(config):
    config.secrets["VASTAI_API_KEY"] = "fallback"
    assert vast._api_key(config) == "fallback"


def test_search_instance_offers(config):
    config = config_with_key(config)
    with mock.patch("hostai.providers.vast.search_offers", return_value=[{"id": 1, "dph_total": 0.5}]) as search:
        offers = vast.search_instance_offers(
            config,
            "dph_total <= 1.0",
            limit=10,
            order="dph_total",
            storage=100,
        )
    assert len(offers) == 1
    assert offers[0]["dph_total"] == 0.5
    assert search.call_count == 1
    kwargs = search.call_args.kwargs
    assert kwargs["limit"] == 10
    assert kwargs["storage"] == 100


def test_get_instance(config):
    config = config_with_key(config)
    with mock.patch("hostai.providers.vast.show_instance", return_value={"id": 123, "actual_status": "running"}) as show:
        inst = vast.get_instance(config, 123)
    assert inst == {"id": 123, "actual_status": "running"}
    show.assert_called_once()


def test_get_instance_not_found(config):
    config = config_with_key(config)
    with mock.patch("hostai.providers.vast.show_instance", return_value=None):
        assert vast.get_instance(config, 123) is None


def test_destroy(config):
    config = config_with_key(config)
    with mock.patch("hostai.providers.vast.destroy_instance", return_value={"success": True}) as destroy:
        result = vast.destroy(config, 123, timeout=30)
    assert result["success"] is True
    destroy.assert_called_once()


def test_pause(config):
    config = config_with_key(config)
    with mock.patch("hostai.providers.vast.stop_instance", return_value={"success": True}) as stop:
        result = vast.pause(config, 123)
    assert result["success"] is True
    stop.assert_called_once()


def test_start(config):
    config = config_with_key(config)
    with mock.patch("hostai.providers.vast.start_instance", return_value={"success": True}) as start:
        result = vast.start(config, 123)
    assert result["success"] is True
    start.assert_called_once()


def test_create_instance_from_offer(config):
    config = config_with_key(config)
    with mock.patch("hostai.providers.vast.create_instance", return_value={"id": 42}) as create:
        result = vast.create_instance_from_offer(
            config,
            1,
            image="test-image",
            disk=100,
            env={"FOO": "bar"},
            price=0.5,
        )
    assert result["id"] == 42
    assert create.call_args.kwargs["image"] == "test-image"
    assert create.call_args.kwargs["disk"] == 100
    assert create.call_args.kwargs["env"] == {"FOO": "bar"}
    assert create.call_args.kwargs["price"] == 0.5


def test_get_instance_logs_returns_none_for_dict(config):
    """SDK may return a dict when logs are not yet ready; we coerce to None."""
    config = config_with_key(config)
    with mock.patch("hostai.providers.vast.fetch_logs", return_value={"not_ready": True}):
        logs = vast.get_instance_logs(config, 123)
    assert logs is None


def test_get_instance_logs_returns_text(config):
    config = config_with_key(config)
    with mock.patch("hostai.providers.vast.fetch_logs", return_value="line1\nline2"):
        logs = vast.get_instance_logs(config, 123)
    assert logs == "line1\nline2"
