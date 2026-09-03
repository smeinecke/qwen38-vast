"""Tests for hostai.commands.monitor with mocked market search."""

import json
from unittest import mock

import click
import pytest
from click.testing import CliRunner

from hostai.commands.monitor import (
    _ranked_best_for_monitor,
    _resolve_monitor_targets,
    _search_profiles,
    cmd_monitor_once,
)
from hostai.profiles import HardwareRank, MonitorHardware, Profile, Profiles


def make_profile(name="test", group="", disk_gb=None, query="gpu_name == RTX_4090"):
    return Profile(
        name=name,
        image="a",
        ctx_size=32768,
        gpu_query=query,
        monitor_group=group or None,
        disk_gb=disk_gb,
    )


def test_resolve_monitor_targets_from_profile(config):
    profiles = mock.Mock()
    profiles.profiles = []
    p = make_profile("test")
    profiles.resolve_profile.return_value = p
    targets = _resolve_monitor_targets(config, profiles, "test", None)
    assert [t.name for t in targets] == ["test"]


def test_resolve_monitor_targets_from_group(config):
    profiles = mock.Mock()
    p = make_profile("test", group="cheap")
    profiles.profiles = [p]
    config.monitor.group = "cheap"
    targets = _resolve_monitor_targets(config, profiles, None, None)
    assert [t.name for t in targets] == ["test"]


def test_resolve_monitor_targets_unknown_group(config):
    profiles = mock.Mock()
    profiles.profiles = []
    config.monitor.group = "missing"
    with pytest.raises(click.ClickException, match="no profiles"):
        _resolve_monitor_targets(config, profiles, None, None)


def test_search_profiles_uses_profile_disk_and_storage(config):
    """Monitor searches must pass the resolved disk size to the market layer."""
    p = make_profile("test", disk_gb=200)
    profiles = mock.Mock()
    profiles.market_policy.require_free_traffic = False

    with mock.patch("hostai.commands.monitor.market.build_search_query", return_value=("query", 1.0)) as build:
        with mock.patch("hostai.commands.monitor.market.search_offers", return_value=[]) as search:
            _search_profiles(config, profiles, [p])

    assert build.call_args.kwargs == {"max_price": None, "unverified": config.market.allow_unverified, "offer": None}
    assert search.call_args.kwargs["storage"] == 200


def test_ranked_best_for_monitor_enforces_hardware_rank(config):
    """A worse-ranked GPU must not be returned as a cheaper upgrade."""

    profiles = Profiles(
        schema_version=1,
        images=[],
        profiles=[make_profile("a6000")],
        monitor_hardware=MonitorHardware(
            policy="same_or_better",
            gpu_ranks=[
                HardwareRank(gpu="A40", aliases=["A40"], rank=100),
                HardwareRank(gpu="RTX A6000", aliases=["RTX_A6000"], rank=110),
            ],
        ),
        market_policy=mock.Mock(require_free_traffic=False, max_inet_down_cost=0.0, max_inet_up_cost=0.0),
    )

    current = mock.Mock(gpu="RTX A6000", dph=0.5, ctx_size=32768, exists=True)
    candidates = [
        {"id": 1, "gpu_name": "A40", "dph_total": 0.3},
        {"id": 2, "gpu_name": "RTX A6000", "dph_total": 0.4},
    ]
    best = _ranked_best_for_monitor(config, profiles, current, candidates)
    assert best is not None
    assert best["id"] == 2


def test_ranked_best_for_monitor_rejects_worse_gpu(config):

    profiles = Profiles(
        schema_version=1,
        images=[],
        profiles=[make_profile("a6000")],
        monitor_hardware=MonitorHardware(
            policy="same_or_better",
            gpu_ranks=[
                HardwareRank(gpu="A40", aliases=["A40"], rank=100),
                HardwareRank(gpu="RTX A6000", aliases=["RTX_A6000"], rank=110),
            ],
        ),
        market_policy=mock.Mock(require_free_traffic=False, max_inet_down_cost=0.0, max_inet_up_cost=0.0),
    )

    current = mock.Mock(gpu="RTX A6000", dph=0.5, ctx_size=32768, exists=True)
    candidates = [
        {"id": 1, "gpu_name": "A40", "dph_total": 0.3},
    ]
    best = _ranked_best_for_monitor(config, profiles, current, candidates)
    assert best is None


def test_ranked_best_for_monitor_respects_zero_max_dph(config):
    """A max_dph of 0 should still be applied, not treated as unset."""
    config.market.max_dph = 0.0
    profiles = Profiles(
        schema_version=1,
        images=[],
        profiles=[make_profile("test")],
        monitor_hardware=MonitorHardware(policy="same_or_better", gpu_ranks=[]),
        market_policy=mock.Mock(require_free_traffic=False, max_inet_down_cost=0.0, max_inet_up_cost=0.0),
    )
    # If current.dph is 0 and state does not exist, fall back to config max_dph which is 0.
    current = mock.Mock(gpu="RTX 4090", dph=0.0, ctx_size=32768, exists=False)
    candidates = [
        {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.1},
    ]
    best = _ranked_best_for_monitor(config, profiles, current, candidates)
    assert best is None


def test_cmd_monitor_once_no_offers(config, project_dir):
    profiles_path = project_dir / "hostai.profiles.toml"
    profiles_path.write_text(
        '{"schema_version": 1, "default_profile": "test", "images": [{"name": "a", "cuda_arch": "86", "image_tag": "test", "description": ""}], '
        '"profiles": [{"name": "test", "image": "a", "ctx_size": 32768, "gpu_query": "gpu_name == RTX_4090", "monitor_group": ""}], '
        '"monitor_hardware": {"policy": "same_or_better", "gpu_ranks": []}, "market_policy": {"require_free_traffic": false}}'
    )
    config.hostai.profiles_file = "hostai.profiles.toml"

    with mock.patch("hostai.commands.monitor._search_profiles", return_value=[]):
        runner = CliRunner()
        result = runner.invoke(cmd_monitor_once, ["--profile", "test"], obj=config)

    assert result.exit_code == 0
    assert "no matching offers" in result.output
