"""Tests for hostai.commands.up helpers and CLI validation."""

from unittest import mock

import click
import pytest
from click.testing import CliRunner

from hostai.commands.up import _build_query, _select_offer, cmd_up


def make_offer(**overrides):
    base = {
        "id": 42,
        "ask_contract_id": 42,
        "gpu_name": "RTX 4090",
        "dph_total": 0.5,
        "dph_base": 0.5,
        "inet_down": 1000,
        "inet_up": 1000,
    }
    base.update(overrides)
    return base


def test_build_query_basic(config):
    profile = mock.Mock()
    profiles = mock.Mock()
    profiles.market_policy.require_free_traffic = False
    query, max_dph = _build_query(config, profiles, profile, "gpu_name == RTX 4090", None, False, False, None)
    assert "RTX 4090" in query
    assert "dph_total" in query
    assert max_dph == config.market.max_dph


def test_build_query_with_offer_id_skips_dph(config):
    profile = mock.Mock()
    profiles = mock.Mock()
    profiles.market_policy.require_free_traffic = False
    query, max_dph = _build_query(config, profiles, profile, "gpu_name == RTX 4090", None, False, False, 123)
    assert "dph_total" not in query


def test_build_query_negative_max_price(config):
    profile = mock.Mock()
    profiles = mock.Mock()
    with pytest.raises(click.ClickException, match="non-negative"):
        _build_query(config, profiles, profile, "gpu_name == RTX 4090", -1.0, False, False, None)


def test_select_offer_chooses_cheapest(config):
    offers = [make_offer(dph_total=0.6), make_offer(dph_total=0.4)]
    with mock.patch("hostai.commands.up.search_instance_offers", return_value=offers) as search:
        selected = _select_offer(config, "gpu_name == RTX 4090", 1.0, False, False, None)
    assert selected["dph_total"] == 0.4
    assert search.call_args.args[1] == "gpu_name == RTX 4090"
    assert search.call_args.kwargs == {"limit": 25, "order": "dph_total", "storage": 100}


def test_select_offer_specific_id(config):
    offers = [make_offer(id=1), make_offer(id=2)]
    with mock.patch("hostai.commands.up.search_instance_offers", return_value=offers):
        selected = _select_offer(config, "gpu_name == RTX 4090", 1.0, False, False, 2)
    assert selected["id"] == 2


def test_select_offer_no_matches(config):
    with mock.patch("hostai.commands.up.search_instance_offers", return_value=[]):
        with pytest.raises(click.ClickException, match="no matching offer"):
            _select_offer(config, "gpu_name == RTX 4090", 1.0, False, False, None)


def test_cmd_up_rejects_invalid_port(config):
    runner = CliRunner()
    result = runner.invoke(cmd_up, ["--local-port", "70000"], obj=config)
    assert result.exit_code != 0
    assert "65535" in result.output


def test_cmd_up_rejects_negative_max_price(config):
    runner = CliRunner()
    result = runner.invoke(cmd_up, ["--max-price", "-1"], obj=config)
    assert result.exit_code != 0
    assert "non-negative" in result.output
