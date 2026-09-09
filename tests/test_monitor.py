"""Tests for hostai.commands.monitor with mocked market search."""

import json
from unittest import mock

import click
import pytest
from click.testing import CliRunner

from hostai.commands.monitor import (
    _monitor_is_running,
    _monitor_log_file,
    _monitor_pid_file,
    _ranked_best_for_monitor,
    _resolve_monitor_targets,
    _search_profiles,
    cmd_monitor_logs,
    cmd_monitor_once,
    cmd_monitor_start,
    cmd_monitor_status,
    cmd_monitor_stop,
    cmd_monitor_watch,
    maybe_start_monitor,
    stop_monitor,
)
from hostai.profiles import HardwareRank, MonitorHardware, Profile, Profiles
from hostai.state import State


def make_profile(
    name="test", group="", disk_gb=None, query="gpu_name == RTX_4090", ctx_size=32768, monitor_search=True
):
    return Profile(
        name=name,
        image="a",
        ctx_size=ctx_size,
        gpu_query=query,
        monitor_group=group or None,
        disk_gb=disk_gb,
        monitor_search=monitor_search,
    )


def _make_profiles(*profiles):
    return Profiles(
        schema_version=1,
        images=[],
        profiles=list(profiles),
        monitor_hardware=MonitorHardware(policy="same_or_better", gpu_ranks=[]),
        market_policy=mock.Mock(require_free_traffic=False, max_inet_down_cost=0.0, max_inet_up_cost=0.0),
    )


def _state(project_dir, **data):
    state_file = project_dir / ".hostai-vast" / "state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(data))
    return State.load(state_file)


def test_resolve_monitor_targets_from_profile(config, project_dir):
    profiles = _make_profiles(make_profile("test"))
    current = _state(project_dir)
    targets = _resolve_monitor_targets(config, profiles, "test", None, current)
    assert [t.name for t in targets] == ["test"]


def test_resolve_monitor_targets_rejects_monitor_disabled(config, project_dir):
    profiles = _make_profiles(make_profile("test", monitor_search=False))
    current = _state(project_dir)
    with pytest.raises(click.ClickException, match="unknown or monitor-disabled"):
        _resolve_monitor_targets(config, profiles, "test", None, current)


def test_resolve_monitor_targets_from_group(config, project_dir):
    profiles = _make_profiles(make_profile("test", group="cheap"))
    config.monitor.group = "cheap"
    current = _state(project_dir)
    targets = _resolve_monitor_targets(config, profiles, None, None, current)
    assert [t.name for t in targets] == ["test"]


def test_resolve_monitor_targets_unknown_group(config, project_dir):
    profiles = _make_profiles()
    config.monitor.group = "missing"
    current = _state(project_dir)
    with pytest.raises(click.ClickException, match="no profiles"):
        _resolve_monitor_targets(config, profiles, None, None, current)


def test_resolve_monitor_targets_prefers_active_state(config, project_dir):
    """A running 128k instance must not fall back to the default 64k profile."""
    p64 = make_profile(name="default", ctx_size=65536)
    p128 = make_profile(name="big", ctx_size=131072)
    profiles = _make_profiles(p64, p128)
    config.hostai.default_profile = "default"

    current = _state(project_dir, profile="big", ctx_size=131072, instance_id=123, dph=0.5)

    targets = _resolve_monitor_targets(config, profiles, None, None, current)
    assert [t.name for t in targets] == ["big"]
    assert targets[0].ctx_size == 131072


def test_resolve_monitor_targets_active_uses_ctx_override(config, project_dir):
    """The active profile context is updated from state when it differs."""
    p = make_profile(name="big", ctx_size=131072)
    profiles = _make_profiles(p)

    current = _state(project_dir, profile="big", ctx_size=262144, instance_id=123, dph=0.5)

    targets = _resolve_monitor_targets(config, profiles, None, None, current)
    assert targets[0].ctx_size == 262144


def test_resolve_monitor_targets_uses_all_monitor_profiles_for_group(config, project_dir):
    """A running profile with a monitor_group searches all same-context, same-group profiles."""
    p64 = make_profile(name="a6000", group="q4-64k", ctx_size=65536, monitor_search=False)
    p_value = make_profile(name="ampere-value", group="q4-64k", ctx_size=65536, monitor_search=True)
    p128 = make_profile(name="a6000-128k", group="q4-128k", ctx_size=131072, monitor_search=False)
    profiles = _make_profiles(p64, p_value, p128)
    config.hostai.default_profile = "a6000"

    current = _state(project_dir, profile="a6000", ctx_size=65536, instance_id=123, dph=0.5)

    targets = _resolve_monitor_targets(config, profiles, None, None, current)
    names = [t.name for t in targets]
    assert "ampere-value" in names
    assert "a6000" in names
    assert "a6000-128k" not in names
    assert all(t.ctx_size == 65536 for t in targets)


def test_resolve_monitor_targets_ignores_wrong_context_alternatives(config, project_dir):
    """Only same-context monitor profiles are candidates, even if cheaper."""
    p64 = make_profile(name="a6000", ctx_size=65536, monitor_search=True)
    p128 = make_profile(name="a6000-128k", ctx_size=131072, monitor_search=True)
    profiles = _make_profiles(p64, p128)
    config.hostai.default_profile = "a6000"

    current = _state(project_dir, profile="a6000-128k", ctx_size=131072, instance_id=123, dph=0.5)

    targets = _resolve_monitor_targets(config, profiles, None, None, current)
    assert [t.name for t in targets] == ["a6000-128k"]
    assert targets[0].ctx_size == 131072


def test_search_profiles_uses_profile_disk_and_storage(config, project_dir):
    """Monitor searches must pass the resolved disk size to the market layer."""
    p = make_profile("test", disk_gb=200)
    profiles = _make_profiles(p)
    current = _state(project_dir)

    with mock.patch("hostai.commands.monitor.market.build_search_query", return_value=("query", 1.0)) as build:
        with mock.patch("hostai.commands.monitor.market.search_offers", return_value=[]) as search:
            _search_profiles(config, profiles, [p], current)

    assert build.call_args.kwargs["max_price"] is None
    assert build.call_args.kwargs["bid_price"] is None
    assert build.call_args.kwargs["unverified"] == config.market.allow_unverified
    assert build.call_args.kwargs["offer"] is None
    assert search.call_args.kwargs["storage"] == 200
    assert search.call_args.kwargs["offer_type"] == "on-demand"


def test_search_profiles_passes_bid_price_for_interruptible_instance(config, project_dir):
    """A running bid instance must search bid offers using its bid price."""
    p = make_profile("test")
    profiles = _make_profiles(p)

    current = _state(project_dir, instance_id=123, dph=0.3, bid_price=0.35)

    with mock.patch("hostai.commands.monitor.market.build_search_query", return_value=("query", 0.3)) as build:
        with mock.patch("hostai.commands.monitor.market.search_offers", return_value=[]) as search:
            _search_profiles(config, profiles, [p], current)

    assert build.call_args.kwargs["max_price"] == 0.3
    assert build.call_args.kwargs["bid_price"] == 0.35
    assert search.call_args.kwargs["offer_type"] == "bid"


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


def test_ranked_best_for_monitor_rejects_mismatching_vast_ctx_size(config):
    """A Vast offer with a non-matching ctx_size must not be considered an alternative."""
    profiles = _make_profiles(make_profile("test", ctx_size=32768))
    current = mock.Mock(gpu="RTX 4090", dph=0.5, ctx_size=32768, exists=True)
    candidates = [
        {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.3, "ctx_size": 65536},
    ]
    best = _ranked_best_for_monitor(config, profiles, current, candidates)
    assert best is None


def test_ranked_best_for_monitor_accepts_matching_vast_ctx_size(config):
    """A Vast offer with a matching ctx_size is allowed through."""
    profiles = _make_profiles(make_profile("test", ctx_size=32768))
    current = mock.Mock(gpu="RTX 4090", dph=0.5, ctx_size=32768, exists=True)
    candidates = [
        {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.3, "ctx_size": 32768},
    ]
    best = _ranked_best_for_monitor(config, profiles, current, candidates)
    assert best is not None
    assert best["id"] == 1


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


def test_monitor_is_running_with_missing_pid():
    assert _monitor_is_running(9999999) is False


def test_monitor_paths(config, project_dir):
    config.root_dir = project_dir
    assert _monitor_pid_file(config) == project_dir / ".hostai-cache" / "monitor.pid"
    assert _monitor_log_file(config) == project_dir / ".hostai-cache" / "monitor.log"


def test_maybe_start_monitor_skips_when_disabled(config, project_dir, running_state):
    config.root_dir = project_dir
    config.monitor.auto_start = False
    maybe_start_monitor(config, running_state)
    assert not _monitor_pid_file(config).exists()


def test_maybe_start_monitor_launches_daemon(config, project_dir, running_state):
    config.root_dir = project_dir
    config.monitor.auto_start = True
    running_state.instance_id = 12345

    def fake_callback(config, profile, group, interval, threshold):
        _monitor_pid_file(config).parent.mkdir(parents=True, exist_ok=True)
        _monitor_pid_file(config).write_text("12345")

    with mock.patch("hostai.commands.monitor._start_monitor", fake_callback):
        maybe_start_monitor(config, running_state)
    assert _monitor_pid_file(config).read_text() == "12345"


def test_resolve_monitor_targets_by_group(config, project_dir):
    profiles = Profiles(
        schema_version=1,
        images=[],
        profiles=[
            make_profile(name="test"),
            make_profile(name="other", group="g1"),
        ],
    )
    current = State(project_dir / ".hostai-vast" / "state.json")
    targets = _resolve_monitor_targets(config, profiles, None, "g1", current)
    assert len(targets) == 1
    assert targets[0].name == "other"


def test_resolve_monitor_targets_unknown_profile(config, project_dir):
    profiles = Profiles(schema_version=1, images=[], profiles=[make_profile()])
    current = State(project_dir / ".hostai-vast" / "state.json")
    with pytest.raises(click.ClickException, match="unknown"):
        _resolve_monitor_targets(config, profiles, "missing", None, current)


def test_search_profiles_collects_offers(config, project_dir):
    profiles = Profiles(schema_version=1, images=[], profiles=[make_profile()])
    target = profiles.profiles[0]
    current = State(project_dir / ".hostai-vast" / "state.json")
    with mock.patch("hostai.market.search_offers", return_value=[{"id": 1, "dph_total": 0.3}]):
        offers = _search_profiles(config, profiles, [target], current)
    assert len(offers) == 1


def test_ranked_best_for_monitor_upgrade(config, project_dir):
    profiles = Profiles(schema_version=1, images=[], profiles=[make_profile()])
    current = State(project_dir / ".hostai-vast" / "state.json")
    current.gpu = "RTX 4090"
    current.dph = 0.5
    candidates = [
        {"id": 1, "dph_total": 0.3, "gpu_name": "RTX 4090"},
        {"id": 2, "dph_total": 0.4, "gpu_name": "RTX 4090"},
    ]
    best = _ranked_best_for_monitor(config, profiles, current, candidates)
    assert best["id"] == 1


def test_cmd_monitor_watch_single_iteration(config, project_dir):
    config.root_dir = project_dir
    profiles = Profiles(schema_version=1, images=[], profiles=[make_profile()])
    with mock.patch("hostai.profiles.Profiles.from_file", return_value=profiles):
        with mock.patch("hostai.state.State.load", return_value=State(project_dir / ".hostai-vast" / "state.json")):
            with mock.patch("hostai.commands.monitor._search_profiles", return_value=[{"id": 1, "dph_total": 0.3, "gpu_name": "RTX 4090"}]):
                with mock.patch("time.sleep", side_effect=KeyboardInterrupt):
                    runner = CliRunner()
                    result = runner.invoke(cmd_monitor_watch, ["--interval", "1"], obj=config)
    assert result.exit_code == 0


def test_cmd_monitor_status_not_running(config, project_dir):
    config.root_dir = project_dir
    runner = CliRunner()
    result = runner.invoke(cmd_monitor_status, [], obj=config)
    assert result.exit_code == 0
    assert "not running" in result.output.lower()


def test_cmd_monitor_logs_empty(config, project_dir):
    config.root_dir = project_dir
    runner = CliRunner()
    result = runner.invoke(cmd_monitor_logs, [], obj=config)
    assert result.exit_code == 0


def test_cmd_monitor_once_no_offers_mocked(config, project_dir):
    config.root_dir = project_dir
    profiles = Profiles(schema_version=1, images=[], profiles=[make_profile()])
    with mock.patch("hostai.profiles.Profiles.from_file", return_value=profiles):
        with mock.patch("hostai.state.State.load", return_value=State(project_dir / ".hostai-vast" / "state.json")):
            with mock.patch("hostai.commands.monitor._search_profiles", return_value=[]):
                runner = CliRunner()
                result = runner.invoke(cmd_monitor_once, [], obj=config)
    assert result.exit_code == 0
    assert "no matching offers" in result.output


def test_maybe_start_monitor_inside_click_context(config, project_dir, running_state):
    """Regression test: maybe_start_monitor used to double-pass config to the command callback."""
    config.root_dir = project_dir
    config.monitor.auto_start = True
    running_state.instance_id = 12345

    @click.command()
    @click.pass_obj
    def inner(cfg):
        maybe_start_monitor(cfg, running_state)

    def fake_start(cfg, profile, group, interval, threshold):
        _monitor_pid_file(cfg).parent.mkdir(parents=True, exist_ok=True)
        _monitor_pid_file(cfg).write_text("12345")

    with mock.patch("hostai.commands.monitor._start_monitor", fake_start):
        runner = CliRunner()
        result = runner.invoke(inner, [], obj=config)
    assert result.exit_code == 0
    assert _monitor_pid_file(config).read_text() == "12345"


def test_stop_monitor_with_missing_pid(config, project_dir):
    config.root_dir = project_dir
    stop_monitor(config)
    # No-op should not raise.


def test_cmd_monitor_stop_kills_running(config, project_dir):
    config.root_dir = project_dir
    _monitor_pid_file(config).parent.mkdir(parents=True, exist_ok=True)
    _monitor_pid_file(config).write_text("12345")
    with mock.patch("os.kill") as kill:
        with mock.patch("hostai.commands.monitor._monitor_is_running", return_value=True):
            runner = CliRunner()
            result = runner.invoke(cmd_monitor_stop, [], obj=config)
    assert result.exit_code == 0
    assert kill.called


def test_cmd_monitor_start_and_stop(config, project_dir):
    config.root_dir = project_dir
    config.monitor.enabled = True
    config.monitor.interval_seconds = 60
    with mock.patch("subprocess.Popen") as popen:
        popen.return_value.pid = 12345
        runner = CliRunner()
        result = runner.invoke(cmd_monitor_start, [], obj=config)
        assert result.exit_code == 0

    runner = CliRunner()
    with mock.patch("os.kill"):
        with mock.patch("hostai.commands.monitor._monitor_is_running", return_value=True):
            result = runner.invoke(cmd_monitor_stop, [], obj=config)
    assert result.exit_code == 0


def test_cmd_monitor_status_no_daemon(config, project_dir):
    config.root_dir = project_dir
    runner = CliRunner()
    result = runner.invoke(cmd_monitor_status, [], obj=config)
    assert result.exit_code == 0
    assert "not running" in result.output.lower()


def test_cmd_monitor_logs(config, project_dir):
    config.root_dir = project_dir
    _monitor_log_file(config).parent.mkdir(parents=True, exist_ok=True)
    _monitor_log_file(config).write_text("log line 1\nlog line 2\n")
    runner = CliRunner()
    result = runner.invoke(cmd_monitor_logs, ["--lines", "1"], obj=config)
    assert result.exit_code == 0
    assert "log line 2" in result.output
