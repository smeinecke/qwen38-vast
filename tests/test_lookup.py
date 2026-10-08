"""Tests for hostai.commands.lookup with mocked Vast search."""

import json
from unittest import mock

import click
import pytest
from click.testing import CliRunner

from hostai import market
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


def test_filter_offers_exclusions():
    offers = [
        make_offer({"id": 1, "machine_id": 111, "geolocation": "DE"}),
        make_offer({"id": 2, "machine_id": 222, "geolocation": "DE"}),
        make_offer({"id": 3, "ask_contract_id": 303, "geolocation": "DE"}),
        make_offer({"id": 4, "machine_id": "111", "geolocation": "DE"}),
        make_offer({"id": 5, "geolocation": "FR"}),
    ]
    filtered = _filter_offers(
        offers,
        max_dph=1.0,
        require_free=False,
        max_down=0.001,
        max_up=0.001,
        exclusions=market.OfferExclusions(machines=[111], offers=[303], countries=["France"]),
    )
    # id 1+4 via machine 111 (int and str form), id 3 via ask_contract_id,
    # id 5 via country name.
    assert [o["id"] for o in filtered] == [2]


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
    profile.aliases = []
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


def test_cmd_lookup_multi_profiles(config, project_dir):
    p1, p2 = _mock_profile("p1"), _mock_profile("p2", "gpu_name == A40")
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
        profiles.resolve_profile.side_effect = lambda n: {"p1": p1, "p2": p2}.get(n)
        profiles.image_by_name.return_value = image
        profiles.market_policy.require_free_traffic = False
        from_file.return_value = profiles

        runner = CliRunner(env={"COLUMNS": "200"})
        result = runner.invoke(cmd_lookup, ["p1,p2", "--json"], obj=config)

    assert result.exit_code == 0, result.output
    assert provider.search_offers.call_count == 2
    lines = result.output.splitlines()
    offers = json.loads("\n".join(lines[lines.index("["):]))
    by_id = {o["id"]: o for o in offers}
    assert [o["id"] for o in offers] == [11, 10]
    assert by_id[10]["_profiles"] == ["p1", "p2"]
    assert by_id[11]["_profiles"] == ["p2"]


def test_cmd_lookup_multi_profiles_semicolon_and_dedup(config, project_dir):
    p1, p2 = _mock_profile("p1"), _mock_profile("p2", "gpu_name == A40")

    provider = _mock_provider([make_offer()])
    with (
        mock.patch("hostai.commands.lookup.Profiles.from_file") as from_file,
        mock.patch("hostai.commands.lookup.get_provider", return_value=provider),
    ):
        image = mock.Mock()
        image.cuda_arch = "89"
        profiles = mock.Mock()
        profiles.resolve_profile.side_effect = lambda n: {"p1": p1, "p2": p2}.get(n)
        profiles.image_by_name.return_value = image
        profiles.market_policy.require_free_traffic = False
        from_file.return_value = profiles

        runner = CliRunner(env={"COLUMNS": "200"})
        result = runner.invoke(cmd_lookup, ["p1; p2;p1"], obj=config)

    assert result.exit_code == 0, result.output
    # p1 resolves twice via the list but is only searched once.
    assert provider.search_offers.call_count == 2


def test_cmd_lookup_multi_profiles_unknown_name(config, project_dir):
    provider = _mock_provider([make_offer()])
    with (
        mock.patch("hostai.commands.lookup.Profiles.from_file") as from_file,
        mock.patch("hostai.commands.lookup.get_provider", return_value=provider),
    ):
        profiles = mock.Mock()
        profiles.resolve_profile.return_value = None
        from_file.return_value = profiles

        runner = CliRunner(env={"COLUMNS": "200"})
        result = runner.invoke(cmd_lookup, ["p1,nope"], obj=config)

    assert result.exit_code != 0
    assert "unknown profile 'p1'" in result.output
    provider.search_offers.assert_not_called()


def test_cmd_lookup_glob_suffix(config, project_dir):
    p128 = _mock_profile("a-128k")
    p256a = _mock_profile("b-256k", "gpu_name == A40")
    p256b = _mock_profile("c-256k", "gpu_name == A100")
    p256b.aliases = ["alias-256k"]

    provider = _mock_provider([make_offer()])
    with (
        mock.patch("hostai.commands.lookup.Profiles.from_file") as from_file,
        mock.patch("hostai.commands.lookup.get_provider", return_value=provider),
    ):
        image = mock.Mock()
        image.cuda_arch = "89"
        profiles = mock.Mock()
        profiles.profiles = [p128, p256a, p256b]
        profiles.image_by_name.return_value = image
        profiles.market_policy.require_free_traffic = False
        from_file.return_value = profiles

        runner = CliRunner(env={"COLUMNS": "200"})
        result = runner.invoke(cmd_lookup, ["*-256k"], obj=config)

    assert result.exit_code == 0, result.output
    # only the two *-256k profiles are searched
    assert provider.search_offers.call_count == 2
    assert "b-256k" in result.output
    assert "c-256k" in result.output
    assert "a-128k" not in result.output


def test_cmd_lookup_glob_no_match(config, project_dir):
    provider = _mock_provider([make_offer()])
    with (
        mock.patch("hostai.commands.lookup.Profiles.from_file") as from_file,
        mock.patch("hostai.commands.lookup.get_provider", return_value=provider),
    ):
        profiles = mock.Mock()
        profiles.profiles = [_mock_profile("a-128k")]
        from_file.return_value = profiles

        runner = CliRunner(env={"COLUMNS": "200"})
        result = runner.invoke(cmd_lookup, ["*-999k"], obj=config)

    assert result.exit_code != 0
    assert "matched no profiles" in result.output
    provider.search_offers.assert_not_called()


def _mock_profiles(from_file, profile=None):
    profile = profile or _mock_profile("test")
    image = mock.Mock()
    image.cuda_arch = "89"
    profiles = mock.Mock()
    profiles.resolve_profile.return_value = profile
    profiles.image_by_name.return_value = image
    profiles.market_policy.require_free_traffic = False
    from_file.return_value = profiles
    return profiles


def test_cmd_lookup_skip_machine(config, project_dir):
    offers = [
        make_offer({"id": 1, "machine_id": 111, "dph_total": 0.3}),
        make_offer({"id": 2, "machine_id": 222, "dph_total": 0.4}),
    ]
    provider = _mock_provider(offers)
    with (
        mock.patch("hostai.commands.lookup.Profiles.from_file") as from_file,
        mock.patch("hostai.commands.lookup.get_provider", return_value=provider),
    ):
        _mock_profiles(from_file)
        runner = CliRunner(env={"COLUMNS": "200"})
        result = runner.invoke(
            cmd_lookup, ["--skip-machine", "111", "--max-results", "10"], obj=config
        )

    assert result.exit_code == 0, result.output
    assert "[exclude] machines [111]" in result.output
    # The table shows machine ids; only the offer on machine 222 survives.
    assert "Vast offers (1 of 1)" in result.output
    assert "222" in result.output


def test_cmd_lookup_config_blocklist(config, project_dir):
    """The [blocklist] config section applies even without --skip-* flags."""
    config.blocklist.machines = [111]
    offers = [
        make_offer({"id": 1, "machine_id": 111, "dph_total": 0.3}),
        make_offer({"id": 2, "machine_id": 222, "dph_total": 0.4}),
    ]
    provider = _mock_provider(offers)
    with (
        mock.patch("hostai.commands.lookup.Profiles.from_file") as from_file,
        mock.patch("hostai.commands.lookup.get_provider", return_value=provider),
    ):
        _mock_profiles(from_file)
        runner = CliRunner(env={"COLUMNS": "200"})
        result = runner.invoke(cmd_lookup, [], obj=config)

    assert result.exit_code == 0, result.output
    assert "[exclude] machines [111]" in result.output
    assert "Vast offers (1 of 1)" in result.output
    assert "222" in result.output


def test_cmd_lookup_skip_all_offers_reports_none(config, project_dir):
    provider = _mock_provider([make_offer({"id": 7, "machine_id": 111})])
    with (
        mock.patch("hostai.commands.lookup.Profiles.from_file") as from_file,
        mock.patch("hostai.commands.lookup.get_provider", return_value=provider),
    ):
        _mock_profiles(from_file)
        runner = CliRunner(env={"COLUMNS": "200"})
        result = runner.invoke(cmd_lookup, ["--skip-offer", "7"], obj=config)

    assert result.exit_code == 0, result.output
    assert "No matching offers" in result.output
