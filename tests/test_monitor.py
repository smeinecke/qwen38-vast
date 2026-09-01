"""Tests for hostai.commands.monitor with mocked Vast search."""

from unittest import mock

import click
import pytest
from click.testing import CliRunner

from hostai.commands.monitor import _resolve_monitor_targets, _search_targets, cmd_monitor_once


def test_resolve_monitor_targets_from_profile(config):
    profiles = mock.Mock()
    profiles.profiles = []
    assert _resolve_monitor_targets(config, profiles, "test", None) == ["test"]


def test_resolve_monitor_targets_from_group(config):
    profiles = mock.Mock()
    p = mock.Mock()
    p.name = "test"
    p.monitor_group = "cheap"
    profiles.profiles = [p]
    config.monitor.group = "cheap"
    assert _resolve_monitor_targets(config, profiles, None, None) == ["test"]


def test_resolve_monitor_targets_unknown_group(config):
    profiles = mock.Mock()
    profiles.profiles = []
    config.monitor.group = "missing"
    with pytest.raises(click.ClickException, match="no profiles"):
        _resolve_monitor_targets(config, profiles, None, None)


def test_search_targets_merges_and_sorts(config):
    offers_a = [{"id": 1, "dph_total": 0.6}]
    offers_b = [{"id": 2, "dph_total": 0.4}]

    def fake_search(cfg, pr, name):
        if name == "a":
            return offers_a
        if name == "b":
            return offers_b
        return None

    with mock.patch("hostai.commands.monitor._search_profile", side_effect=lambda c, p, n: fake_search(c, p, n)):
        result = _search_targets(config, mock.Mock(), ["a", "b"])

    assert [o["id"] for o in result] == [2, 1]


def test_cmd_monitor_once_no_offers(config):
    with mock.patch("hostai.commands.monitor.Profiles.from_file") as from_file:
        profiles = mock.Mock()
        profiles.resolve_profile.return_value = mock.Mock(gpu_query="gpu_name == RTX 4090")
        profiles.profiles = []
        from_file.return_value = profiles

        with mock.patch("hostai.commands.monitor.search_instance_offers", return_value=[]) as search:
            runner = CliRunner()
            result = runner.invoke(cmd_monitor_once, ["--profile", "test"], obj=config)

    assert result.exit_code == 0
    assert "no matching offers" in result.output
    assert search.call_count == 1
    assert search.call_args.kwargs == {"limit": 10, "order": "dph_total"}
