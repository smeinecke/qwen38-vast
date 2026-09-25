"""Tests for hostai.commands.lookup with mocked Vast search."""

import json
from unittest import mock

import click
import pytest
from click.testing import CliRunner

from hostai.commands.lookup import _filter_offers, _resolve_query, cmd_lookup


def make_offer(overrides=None):
    base = {
        "id": 1,
        "ask_contract_id": 1,
        "gpu_name": "RTX 4090",
        "dph_total": 0.5,
        "dph_base": 0.5,
        "discounted_dph_total": 0.45,
        "reliability2": 0.99,
        "verification": "verified",
        "geolocation": "US",
        "inet_down_cost": 0.0001,
        "inet_up_cost": 0.0001,
        "num_gpus": 1,
    }
    if overrides:
        base.update(overrides)
    return base


def test_resolve_query_basic(config):
    profile = mock.Mock()
    profile.gpu_query = "gpu_name == RTX 4090"
    profile.ctx_size = 32768
    profiles = mock.Mock()
    profiles.market_policy.require_free_traffic = False

    query, max_dph, ctx_size = _resolve_query(config, profiles, profile, None, False)
    assert "RTX 4090" in query
    assert f"disk_space>={config.market.disk_gb}" in query
    assert max_dph == 1.0
    assert ctx_size == 32768


def test_resolve_query_negative_max_price(config):
    profile = mock.Mock()
    profile.gpu_query = "gpu_name == RTX 4090"
    profile.ctx_size = 32768
    profiles = mock.Mock()
    with pytest.raises(click.ClickException, match="non-negative"):
        _resolve_query(config, profiles, profile, -1.0, False)


def test_resolve_query_ctx_fallback(config):
    profile = mock.Mock()
    profile.gpu_query = "gpu_name == RTX 4090"
    profile.ctx_size = None
    profiles = mock.Mock()
    profiles.market_policy.require_free_traffic = False
    _, _, ctx_size = _resolve_query(config, profiles, profile, None, False)
    assert ctx_size == 0


def test_filter_offers_respects_max_dph():
    offers = [make_offer({"dph_total": 0.4}), make_offer({"dph_total": 1.5})]
    filtered = _filter_offers(offers, max_dph=1.0, require_free=False, max_down=0.001, max_up=0.001)
    assert len(filtered) == 1
    assert filtered[0]["dph_total"] == 0.4


def test_filter_offers_free_traffic():
    offers = [
        make_offer({"inet_down_cost": 0.0, "inet_up_cost": 0.0}),
        make_offer({"inet_down_cost": 0.01, "inet_up_cost": 0.0}),
    ]
    filtered = _filter_offers(offers, max_dph=1.0, require_free=True, max_down=0.001, max_up=0.001)
    assert len(filtered) == 1
    assert filtered[0]["inet_down_cost"] == 0.0


def _mock_provider(offers=None):
    m = mock.Mock()
    m.search_offers.return_value = offers if offers is not None else []
    return m


def test_cmd_lookup_no_offers(config, project_dir):
    with mock.patch("hostai.commands.lookup.Profiles.from_file") as from_file:
        profile = mock.Mock()
        profile.name = "test"
        profile.gpu_query = "gpu_name == RTX 4090"
        profile.ctx_size = 32768
        profile.image = "test-image"
        image = mock.Mock()
        image.cuda_arch = "89"
        profiles = mock.Mock()
        profiles.resolve_profile.return_value = profile
        profiles.image_by_name.return_value = image
        profiles.market_policy.require_free_traffic = False
        from_file.return_value = profiles

        with mock.patch("hostai.commands.lookup.get_provider", return_value=_mock_provider([])):
            runner = CliRunner()
            result = runner.invoke(cmd_lookup, [], obj=config)

    assert result.exit_code == 0
    assert "No matching offers" in result.output


def test_cmd_lookup_with_offers(config, project_dir):
    provider = _mock_provider([make_offer()])
    with (
        mock.patch("hostai.commands.lookup.Profiles.from_file") as from_file,
        mock.patch("hostai.commands.lookup.get_provider", return_value=provider),
    ):
        profile = mock.Mock()
        profile.name = "test"
        profile.gpu_query = "gpu_name == RTX 4090"
        profile.ctx_size = 32768
        profile.image = "test-image"
        image = mock.Mock()
        image.cuda_arch = "89"
        profiles = mock.Mock()
        profiles.resolve_profile.return_value = profile
        profiles.image_by_name.return_value = image
        profiles.market_policy.require_free_traffic = False
        from_file.return_value = profiles

        runner = CliRunner(env={"COLUMNS": "160"})
        result = runner.invoke(cmd_lookup, ["--max-results", "1"], obj=config)

    assert result.exit_code == 0
    assert "RTX 4090" in result.output
    assert "United States \U0001f1fa\U0001f1f8" in result.output
    assert provider.search_offers.call_count == 1
    call_args = provider.search_offers.call_args
    assert "RTX 4090" in call_args.args[0]
    assert call_args.kwargs == {"limit": 50, "order": "dph_total", "storage": config.market.disk_gb}


def test_cmd_lookup_country_fallback(config, project_dir):
    provider = _mock_provider([make_offer({"geolocation": "local"})])
    with (
        mock.patch("hostai.commands.lookup.Profiles.from_file") as from_file,
        mock.patch("hostai.commands.lookup.get_provider", return_value=provider),
    ):
        profile = mock.Mock()
        profile.name = "test"
        profile.gpu_query = "gpu_name == RTX 4090"
        profile.ctx_size = 32768
        profile.image = "test-image"
        image = mock.Mock()
        image.cuda_arch = "89"
        profiles = mock.Mock()
        profiles.resolve_profile.return_value = profile
        profiles.image_by_name.return_value = image
        profiles.market_policy.require_free_traffic = False
        from_file.return_value = profiles

        runner = CliRunner(env={"COLUMNS": "160"})
        result = runner.invoke(cmd_lookup, ["--max-results", "1"], obj=config)

    assert result.exit_code == 0
    assert "local" in result.output


def test_cmd_lookup_invalid_max_results(config):
    runner = CliRunner()
    result = runner.invoke(cmd_lookup, ["--max-results", "0"], obj=config)
    assert result.exit_code != 0
    assert "positive" in result.output


def _mock_profile(name, gpu_query="gpu_name == RTX 4090"):
    profile = mock.Mock()
    profile.name = name
    profile.gpu_query = gpu_query
    profile.ctx_size = 32768
    profile.image = "test-image"
    profile.disk_gb = None
    return profile


def test_cmd_lookup_star_searches_all_profiles(config, project_dir):
    p1, p2 = _mock_profile("p1"), _mock_profile("p2", "gpu_name == A40")
    # The same offer id matches both profiles; p2 additionally has its own.
    shared = make_offer({"id": 10, "dph_total": 0.4})
    extra = make_offer({"id": 11, "gpu_name": "A40", "dph_total": 0.3})

    provider = mock.Mock()
    provider.search_offers.side_effect = lambda q, **kw: (
        [dict(shared)] if "RTX 4090" in q else [dict(shared), dict(extra)]
    )

    with (
        mock.patch("hostai.commands.lookup.Profiles.from_file") as from_file,
        mock.patch("hostai.commands.lookup.get_provider", return_value=provider),
    ):
        image = mock.Mock()
        image.cuda_arch = "89"
        profiles = mock.Mock()
        profiles.profiles = [p1, p2]
        profiles.image_by_name.return_value = image
        profiles.market_policy.require_free_traffic = False
        from_file.return_value = profiles

        runner = CliRunner(env={"COLUMNS": "200"})
        result = runner.invoke(cmd_lookup, ["*", "--json"], obj=config)

    assert result.exit_code == 0, result.output
    assert provider.search_offers.call_count == 2
    lines = result.output.splitlines()
    offers = json.loads("\n".join(lines[lines.index("["):]))
    by_id = {o["id"]: o for o in offers}
    # Cheapest first; shared offer deduped with both profile names.
    assert [o["id"] for o in offers] == [11, 10]
    assert by_id[10]["_profiles"] == ["p1", "p2"]
    assert by_id[11]["_profiles"] == ["p2"]


def test_cmd_lookup_star_warns_and_continues(config, project_dir):
    p1, p2 = _mock_profile("bad"), _mock_profile("good")
    provider = mock.Mock()
    provider.search_offers.side_effect = [RuntimeError("boom"), [make_offer()]]

    with (
        mock.patch("hostai.commands.lookup.Profiles.from_file") as from_file,
        mock.patch("hostai.commands.lookup.get_provider", return_value=provider),
    ):
        image = mock.Mock()
        image.cuda_arch = "89"
        profiles = mock.Mock()
        profiles.profiles = [p1, p2]
        profiles.image_by_name.return_value = image
        profiles.market_policy.require_free_traffic = False
        from_file.return_value = profiles

        runner = CliRunner(env={"COLUMNS": "200"})
        result = runner.invoke(cmd_lookup, ["*"], obj=config)

    assert result.exit_code == 0, result.output
    assert "[warn] bad:" in result.output
    assert "RTX 4090" in result.output
