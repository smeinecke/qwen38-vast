"""Tests for hostai.commands.up helpers and CLI validation."""

import json
from unittest import mock

import click
import pytest
from click.testing import CliRunner

from hostai.commands.up import _do_fresh, _extra_args, cmd_up


def make_offer(**overrides):
    base = {
        "id": 42,
        "ask_contract_id": 42,
        "gpu_name": "RTX 4090",
        "dph_total": 0.5,
        "dph_base": 0.5,
        "inet_down": 1000,
        "inet_up": 1000,
        "inet_down_cost": 0.0,
        "inet_up_cost": 0.0,
    }
    base.update(overrides)
    return base


def _mock_provider():
    """Return a mock provider for tests that bypass real Vast."""
    m = mock.Mock()
    m.name = "vast"
    m.create_instance.return_value = {"new_contract": 123}
    return m


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


def test_cmd_up_rejects_non_positive_bid(config):
    runner = CliRunner()
    result = runner.invoke(cmd_up, ["--bid", "0"], obj=config)
    assert result.exit_code != 0
    assert "positive" in result.output


def test_cmd_up_rejects_bad_expected_session(config):
    runner = CliRunner()
    result = runner.invoke(cmd_up, ["--expected-session", "abc"], obj=config)
    assert result.exit_code != 0
    assert "invalid duration" in result.output


def test_cmd_up_rejects_bad_scoring_mode(config):
    runner = CliRunner()
    result = runner.invoke(cmd_up, ["--scoring-mode", "foo"], obj=config)
    assert result.exit_code != 0
    assert "dph, perf, or session" in result.output


def _write_test_profiles(project_dir, with_disk_gb=None):
    profile = {
        "name": "test",
        "image": "a",
        "ctx_size": 32768,
        "gpu_query": "gpu_name == RTX_4090",
        "monitor_group": "",
    }
    if with_disk_gb is not None:
        profile["disk_gb"] = with_disk_gb
    profiles = {
        "schema_version": 1,
        "default_profile": "test",
        "images": [{"name": "a", "cuda_arch": "86", "image_tag": "test", "description": ""}],
        "profiles": [profile],
        "monitor_hardware": {"policy": "same_or_better", "gpu_ranks": []},
        "market_policy": {"require_free_traffic": False},
    }
    (project_dir / "hostai.profiles.toml").write_text(json.dumps(profiles))
    return profiles


def _prep_config_for_cli(config, project_dir):
    _write_test_profiles(project_dir)
    config.hostai.profiles_file = "hostai.profiles.toml"
    config.image.base = "example/hostai"


def test_do_fresh_uses_resolved_profile_disk(config, project_dir):
    """When a profile defines disk_gb, it is used for search and instance creation."""
    _write_test_profiles(project_dir, with_disk_gb=150)
    config.hostai.profiles_file = "hostai.profiles.toml"
    config.image.base = "example/hostai"

    with (
        mock.patch("hostai.commands.up.market.select_offer", return_value=make_offer()) as select,
        mock.patch("hostai.commands.up._provider", return_value=_mock_provider()) as provider,
        mock.patch("hostai.commands.up._do_fresh_core"),
    ):
        _do_fresh(config, "test", None, None, None, False, False, False, False, None)

    assert select.call_args.kwargs["storage"] == 150
    assert provider.return_value.create_instance.call_args.kwargs["disk"] == 150
    # The search query must derive the host disk-space eligibility from the
    # same resolved disk allocation.
    query = select.call_args.args[2]
    assert "disk_space>=150" in query


def test_do_fresh_falls_back_to_market_disk_when_profile_has_none(config, project_dir):
    _write_test_profiles(project_dir)
    config.hostai.profiles_file = "hostai.profiles.toml"
    config.image.base = "example/hostai"

    with (
        mock.patch("hostai.commands.up.market.select_offer", return_value=make_offer()) as select,
        mock.patch("hostai.commands.up._provider", return_value=_mock_provider()),
        mock.patch("hostai.commands.up._do_fresh_core"),
    ):
        _do_fresh(config, "test", None, None, None, False, False, False, False, None)

    assert select.call_args.kwargs["storage"] == config.market.disk_gb
    query = select.call_args.args[2]
    assert f"disk_space>={config.market.disk_gb}" in query


def test_do_fresh_uses_bid_price_for_interruptible(config, project_dir):
    _write_test_profiles(project_dir)
    config.hostai.profiles_file = "hostai.profiles.toml"
    config.image.base = "example/hostai"

    with (
        mock.patch("hostai.commands.up.market.select_offer", return_value=make_offer()) as select,
        mock.patch("hostai.commands.up._provider", return_value=_mock_provider()) as provider,
        mock.patch("hostai.commands.up._do_fresh_core"),
    ):
        _do_fresh(config, "test", None, None, None, False, False, False, False, None, bid_price=0.35)

    assert select.call_args.kwargs["offer_type"] == "bid"
    assert provider.return_value.create_instance.call_args.kwargs["bid_price"] == 0.35


def test_do_fresh_dry_run_does_not_create(config, project_dir):
    _write_test_profiles(project_dir)
    config.hostai.profiles_file = "hostai.profiles.toml"
    config.image.base = "example/hostai"

    with (
        mock.patch("hostai.commands.up.market.select_offer", return_value=make_offer()),
        mock.patch("hostai.commands.up._provider") as provider,
    ):
        _do_fresh(
            config,
            "test",
            None,
            None,
            None,
            False,
            False,
            False,
            False,
            None,
            bid_price=0.35,
            dry_run=True,
        )

    provider.return_value.create_instance.assert_not_called()


def test_do_fresh_blocks_running_instance(config, project_dir):
    """A running instance in state.json must raise before renting another."""
    from hostai.state import State, state_dir

    sdir = state_dir(project_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    state = State(sdir / "state.json")
    state.instance_id = 123
    state.save()

    provider = _mock_provider()
    provider.get_instance.return_value = {"id": 123, "actual_status": "running"}
    with mock.patch("hostai.commands.up._provider", return_value=provider):
        with pytest.raises(click.ClickException, match="run hostai down first"):
            _do_fresh(config, "test", None, None, None, False, False, False, False, None)


def test_do_fresh_allows_recreate_when_instance_is_exited(config, project_dir):
    """An exited instance should not block a new hostai up."""
    from hostai.state import State, state_dir

    sdir = state_dir(project_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    state = State(sdir / "state.json")
    state.instance_id = 123
    state.save()

    provider = _mock_provider()
    provider.get_instance.return_value = {"id": 123, "actual_status": "exited"}
    with mock.patch("hostai.commands.up._provider", return_value=provider):
        with mock.patch("hostai.commands.up._resolve_profile", side_effect=RuntimeError("later step")):
            with pytest.raises(RuntimeError, match="later step"):
                _do_fresh(config, "test", None, None, None, False, False, False, False, None)


def test_cmd_up_config_interruptible_uses_bid_price(config, project_dir):
    """[vast] interruptible = true with a bid_price must select bid offers."""
    _prep_config_for_cli(config, project_dir)
    config.vast.interruptible = True
    config.vast.bid_price = 0.35

    with mock.patch("hostai.commands.up._do_fresh") as do_fresh:
        runner = CliRunner()
        result = runner.invoke(cmd_up, ["--dry-run"], obj=config)

    assert result.exit_code == 0, result.output
    assert do_fresh.call_args.kwargs["bid_price"] == 0.35


def test_cmd_up_config_interruptible_no_bid_falls_back_to_max_dph(config, project_dir):
    """[vast] interruptible = true without a bid must fall back to [market].max_dph."""
    _prep_config_for_cli(config, project_dir)
    config.vast.interruptible = True
    config.vast.bid_price = None
    config.market.max_dph = 0.8

    with mock.patch("hostai.commands.up._do_fresh") as do_fresh:
        runner = CliRunner()
        result = runner.invoke(cmd_up, ["--dry-run"], obj=config)

    assert result.exit_code == 0, result.output
    assert do_fresh.call_args.kwargs["bid_price"] == 0.8


def test_cmd_up_cli_bid_overrides_config_bid(config, project_dir):
    """--bid takes precedence over [vast].bid_price."""
    _prep_config_for_cli(config, project_dir)
    config.vast.bid_price = 0.25

    with mock.patch("hostai.commands.up._do_fresh") as do_fresh:
        runner = CliRunner()
        result = runner.invoke(cmd_up, ["--bid", "0.50", "--dry-run"], obj=config)

    assert result.exit_code == 0, result.output
    assert do_fresh.call_args.kwargs["bid_price"] == 0.50


def test_cmd_up_config_interruptible_false_with_bid_still_bid(config, project_dir):
    """A configured bid_price implies interruptible even when the flag is false."""
    _prep_config_for_cli(config, project_dir)
    config.vast.interruptible = False
    config.vast.bid_price = 0.30

    with mock.patch("hostai.commands.up._do_fresh") as do_fresh:
        runner = CliRunner()
        result = runner.invoke(cmd_up, ["--dry-run"], obj=config)

    assert result.exit_code == 0, result.output
    assert do_fresh.call_args.kwargs["bid_price"] == 0.30


def test_cmd_up_on_demand_default_no_bid(config, project_dir):
    """Without --interruptible, [vast].interruptible, or a bid, launch is on-demand."""
    _prep_config_for_cli(config, project_dir)
    config.vast.interruptible = False
    config.vast.bid_price = None

    with mock.patch("hostai.commands.up._do_fresh") as do_fresh:
        runner = CliRunner()
        result = runner.invoke(cmd_up, ["--dry-run"], obj=config)

    assert result.exit_code == 0, result.output
    assert do_fresh.call_args.kwargs["bid_price"] is None


def test_cmd_up_config_interruptible_no_price_fails(config, project_dir):
    """[vast] interruptible = true with no price source must fail cleanly."""
    _prep_config_for_cli(config, project_dir)
    config.vast.interruptible = True
    config.vast.bid_price = None
    config.market.max_dph = 0.0

    runner = CliRunner()
    result = runner.invoke(cmd_up, ["--dry-run"], obj=config)
    assert result.exit_code != 0
    assert "interruptible mode requires a positive bid price" in result.output


def test_do_fresh_search_query_caps_at_bid(config, project_dir):
    """Interruptible search must use dph_total <= bid_price."""
    _write_test_profiles(project_dir)
    config.hostai.profiles_file = "hostai.profiles.toml"
    config.image.base = "example/hostai"
    config.market.max_dph = 0.8

    with (
        mock.patch("hostai.commands.up.market.select_offer", return_value=make_offer()) as select,
        mock.patch("hostai.commands.up._provider", return_value=_mock_provider()),
        mock.patch("hostai.commands.up._do_fresh_core"),
    ):
        _do_fresh(config, "test", None, None, None, False, False, False, False, None, bid_price=0.35)

    query = select.call_args.args[2]
    assert "dph_total <= 0.35" in query
    assert select.call_args.kwargs["offer_type"] == "bid"


def test_extra_args_uses_cache_shm_minimum(config):
    config.cache.enabled = True
    config.cache.use_shm = True
    config.cache.shm_min_gb = 32
    config.vast.shm_size_gb = None

    assert _extra_args(config) == "--shm-size=32g"
    assert _extra_args(config, no_cache=True) == ""


def test_extra_args_prefers_explicit_shm_size(config):
    config.cache.use_shm = True
    config.cache.shm_min_gb = 32
    config.vast.shm_size_gb = 48

    assert _extra_args(config) == "--shm-size=48g"
