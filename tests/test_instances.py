"""Tests for multi-instance support: named deployments running in parallel."""

import contextlib
import json
import threading
from unittest import mock

import click
import pytest
from click.testing import CliRunner

from hostai import state as state_mod
from hostai.commands import _common
from hostai.commands import up as up_mod
from hostai.commands.down import cmd_down
from hostai.commands.monitor import _monitor_pid_file
from hostai.commands.status import cmd_status
from hostai.commands.up import _find_unclaimed_port, _start_proxy, _write_env_file, cmd_up
from hostai.commands.watchdog import _watchdog_pid_file
from hostai.state import State


def _save_state(config, instance, instance_id, **extra):
    """Persist a state.json for ``instance`` (None = default layout)."""
    sf = state_mod.instance_state_file(config.root_dir, instance)
    st = State(sf, {"instance_id": instance_id, "status": "running", **extra})
    st.save()
    return st


# ---------------------------------------------------------------------------
# Instance naming & state layout


def test_default_state_path_is_legacy(config):
    p = state_mod.instance_state_file(config.root_dir, None)
    assert p == config.root_dir / ".hostai-vast" / "state.json"
    assert state_mod.instance_state_file(config.root_dir, "default") == p


def test_named_state_path_is_scoped(config):
    p = state_mod.instance_state_file(config.root_dir, "foo")
    assert p == config.root_dir / ".hostai-vast" / "instances" / "foo" / "state.json"


def test_normalize_instance_name():
    assert state_mod.normalize_instance_name(None) == "default"
    assert state_mod.normalize_instance_name("") == "default"
    assert state_mod.normalize_instance_name("  ") == "default"
    assert state_mod.normalize_instance_name("default") == "default"
    assert state_mod.normalize_instance_name(" foo ") == "foo"


@pytest.mark.parametrize("bad", ["", "default", "a b", "a/b", "-x", ".x", "_x", "x" * 65])
def test_validate_instance_name_rejects(bad):
    with pytest.raises(ValueError):
        state_mod.validate_instance_name(bad)


@pytest.mark.parametrize("good", ["f", "foo", "foo-2", "a.b_c", "A9", "x" * 64])
def test_validate_instance_name_accepts(good):
    assert state_mod.validate_instance_name(good) == good


# ---------------------------------------------------------------------------
# Discovery and selector resolution


def test_find_instance_states_empty(config):
    assert state_mod.find_instance_states(config.root_dir) == {}


def test_find_instance_states_default_first_then_sorted(config):
    _save_state(config, "b", 222)
    _save_state(config, None, 111)
    _save_state(config, "a", 333)
    found = state_mod.find_instance_states(config.root_dir)
    assert list(found) == ["default", "a", "b"]
    assert found["a"] == config.root_dir / ".hostai-vast" / "instances" / "a" / "state.json"


def test_find_instance_states_ignores_junk(config):
    inst_root = config.root_dir / ".hostai-vast" / "instances"
    (inst_root / "empty-dir").mkdir(parents=True)
    (inst_root / "default").mkdir()  # must never shadow the real default
    (inst_root / "a-file").write_text("{}")
    _save_state(config, None, 1)
    found = state_mod.find_instance_states(config.root_dir)
    assert list(found) == ["default"]


def test_state_root_and_name_derivation(config):
    default_sf = state_mod.instance_state_file(config.root_dir, None)
    named_sf = state_mod.instance_state_file(config.root_dir, "foo")
    assert state_mod.state_root_for(default_sf) == config.root_dir / ".hostai-vast"
    assert state_mod.state_root_for(named_sf) == config.root_dir / ".hostai-vast"
    assert state_mod.instance_name_for_state_file(default_sf) == "default"
    assert state_mod.instance_name_for_state_file(named_sf) == "foo"


def test_sibling_state_files(config):
    _save_state(config, None, 1)
    named = _save_state(config, "foo", 2)
    siblings = dict(state_mod.sibling_state_files(named.state_file))
    assert set(siblings) == {"default", "foo"}


def test_resolve_selector_default(config):
    _save_state(config, None, 1)
    assert state_mod.resolve_instance_selector(config.root_dir, None) == "default"
    assert state_mod.resolve_instance_selector(config.root_dir, "default") == "default"


def test_resolve_selector_by_name(config):
    _save_state(config, "foo", 11)
    assert state_mod.resolve_instance_selector(config.root_dir, "foo") == "foo"


def test_resolve_selector_by_instance_id(config):
    _save_state(config, "foo", 4242)
    assert state_mod.resolve_instance_selector(config.root_dir, "4242") == "foo"


def test_resolve_selector_ambiguous_id(config):
    _save_state(config, "a", 4242)
    _save_state(config, "b", 4242)
    with pytest.raises(ValueError, match="multiple"):
        state_mod.resolve_instance_selector(config.root_dir, "4242")


def test_resolve_selector_unknown_id(config):
    _save_state(config, "a", 4242)
    with pytest.raises(ValueError, match="no instance with instance id"):
        state_mod.resolve_instance_selector(config.root_dir, "9999")


def test_resolve_selector_unknown_name(config):
    _save_state(config, "a", 1)
    with pytest.raises(ValueError, match="no instance named"):
        state_mod.resolve_instance_selector(config.root_dir, "zzz")


# ---------------------------------------------------------------------------
# _common.resolve_state


def test_resolve_state_none_tracked_required(config):
    with pytest.raises(click.ClickException, match="no hostai instances"):
        _common.resolve_state(config)


def test_resolve_state_none_tracked_optional(config):
    name, st = _common.resolve_state(config, required=False)
    assert name == "default"
    assert not st.exists
    assert st.state_file == config.root_dir / ".hostai-vast" / "state.json"


def test_resolve_state_auto_selects_single(config):
    _save_state(config, "foo", 7)
    name, st = _common.resolve_state(config)
    assert name == "foo"
    assert st.instance_id == 7


def test_resolve_state_multiple_requires_selector(config):
    _save_state(config, "a", 1)
    _save_state(config, "b", 2)
    with pytest.raises(click.ClickException, match="use --name"):
        _common.resolve_state(config)


def test_resolve_state_by_numeric_id(config):
    _save_state(config, "a", 55)
    name, st = _common.resolve_state(config, "55")
    assert name == "a"
    assert st.instance_id == 55


# ---------------------------------------------------------------------------
# Claimed ports


def test_claimed_local_ports_skips_dead_and_self(config):
    _save_state(config, "a", 1, local_port=18080)
    _save_state(config, "b", 2, local_port=19000)
    _save_state(config, "dead", None, local_port=17000)  # no instance_id -> free
    assert _common.claimed_local_ports(config) == {18080, 19000}
    assert _common.claimed_local_ports(config, exclude_instance="a") == {19000}
    assert _common.claimed_local_ports(config, exclude_instance=None) == {18080, 19000}


def test_find_unclaimed_port_skips_claimed():
    with mock.patch("hostai.utils.find_free_port") as ffp:
        ffp.side_effect = lambda start, **kw: start  # every port 'free'
        # 18080/18081 claimed -> must land on 18082
        assert _find_unclaimed_port(18080, {18080, 18081}) == 18082


def test_resolve_client_port_avoids_sibling_claim(config):
    """A sibling's recorded port must be skipped even though it is not bound."""
    _save_state(config, "sib", 9, local_port=18080)
    config.ssh.local_port = 18080
    config.ssh.local_port_auto = True
    chosen = up_mod._resolve_client_port(config, instance="other")
    assert chosen != 18080
    assert chosen > 18080


def test_resolve_client_port_claimed_user_port_errors(config):
    _save_state(config, "sib", 9, local_port=18080)
    with pytest.raises(click.ClickException, match="claimed by another"):
        up_mod._resolve_client_port(config, user_port=18080, instance="other")


# ---------------------------------------------------------------------------
# Locks


def test_lifecycle_lock_same_instance_blocks(config):
    with _common.lifecycle_lock(config, "up", instance="foo"):
        with pytest.raises(click.ClickException, match="in progress"):
            with _common.lifecycle_lock(config, "up", instance="foo"):
                pass


def test_lifecycle_lock_different_instances_do_not_block(config):
    with _common.lifecycle_lock(config, "up", instance="foo"):
        with _common.lifecycle_lock(config, "up", instance="bar"):
            pass
    # default and named are also independent
    with _common.lifecycle_lock(config, "up"):
        with _common.lifecycle_lock(config, "up", instance="foo"):
            pass


def test_allocation_lock_serializes(config):
    acquired = threading.Event()

    def contender():
        with _common.allocation_lock(config):
            acquired.set()

    t = threading.Thread(target=contender)
    with _common.allocation_lock(config):
        t.start()
        assert not acquired.wait(0.3)
    t.join(5)
    assert acquired.is_set()


# ---------------------------------------------------------------------------
# Scoped daemon files


def test_daemon_files_default_unsuffixed(config):
    assert _common.daemon_pid_file(config, "monitor") == config.root_dir / ".hostai-cache" / "monitor.pid"
    assert _common.daemon_log_file(config, "watchdog") == config.root_dir / ".hostai-cache" / "watchdog.log"


def test_daemon_files_named_suffixed(config):
    assert _common.daemon_pid_file(config, "monitor", "foo") == config.root_dir / ".hostai-cache" / "monitor-foo.pid"
    assert _common.daemon_log_file(config, "watchdog", "foo") == config.root_dir / ".hostai-cache" / "watchdog-foo.log"
    assert _watchdog_pid_file(config, "foo").name == "watchdog-foo.pid"
    assert _monitor_pid_file(config, "foo").name == "monitor-foo.pid"


def test_daemon_needles_instance_tagged():
    default_needles = _common._daemon_needles("watchdog", "default")
    named_needles = _common._daemon_needles("watchdog", "foo")
    assert b"\x00default\x00" in default_needles
    assert b"\x00foo\x00" in named_needles
    # 'foo' needle must not match a 'foo2' daemon cmdline (NUL-delimited token)
    fake_cmdline = b"hostai\x00watchdog\x00run\x00--name\x00foo2\x00"
    assert all(n in fake_cmdline for n in named_needles) is False
    real_cmdline = b"hostai\x00watchdog\x00run\x00--name\x00foo\x00"
    assert all(n in real_cmdline for n in named_needles)


# ---------------------------------------------------------------------------
# up --name (command level)


def _up_profile_mock():
    profile = mock.Mock()
    profile.name = "test"
    profile.ctx_size = 32768
    profile.monitor_group = ""
    profile.image = "test-img"
    profile.min_gpu_vram_mb = None
    return profile


def _up_image_mock():
    image = mock.Mock()
    image.cuda_arch = "sm_80"
    image.image_tag = "test"
    return image


def _fresh_up_mocks(config):
    """Patch everything up to _do_fresh_core, mirroring test_up.test_cmd_up_fresh."""
    provider = mock.Mock(name="provider")
    provider.create_instance.return_value = {"new_contract": 777}
    stack = [
        mock.patch("hostai.commands.up._resolve_client_port", return_value=18080),
        mock.patch(
            "hostai.commands.up._resolve_profile",
            return_value=(mock.Mock(), _up_profile_mock(), _up_image_mock()),
        ),
        mock.patch("hostai.commands.up.image_for_profile", return_value="ghcr.io/test"),
        mock.patch("hostai.commands.up.market.resolved_disk_gb", return_value=35),
        mock.patch("hostai.commands.up.market.build_search_query", return_value=("query", 1.0)),
        mock.patch(
            "hostai.commands.up.market.select_offer",
            return_value={"id": 1, "dph_total": 0.5, "gpu_name": "A100"},
        ),
        mock.patch("hostai.commands.up.market.offer_summary", return_value="summary"),
        mock.patch("hostai.commands.up._provider", return_value=provider),
        mock.patch("hostai.commands.up.cache._default_local_dir", return_value="/tmp/cache"),
        mock.patch("hostai.commands.up.utils.make_run_id", return_value="run-x"),
        mock.patch("hostai.commands.up.utils.make_api_key", return_value="api-key"),
        mock.patch("hostai.commands.up._do_fresh_core"),
    ]
    ctx = contextlib.ExitStack()
    for p in stack:
        ctx.enter_context(p)
    return ctx, provider


def test_cmd_up_named_writes_named_state(config, project_dir):
    runner = CliRunner()
    ctx, provider = _fresh_up_mocks(config)
    with ctx:
        result = runner.invoke(cmd_up, ["--name", "foo"], obj=config)
    assert result.exit_code == 0, result.output
    sf = project_dir / ".hostai-vast" / "instances" / "foo" / "state.json"
    assert sf.is_file()
    data = json.loads(sf.read_text())
    assert data["instance_name"] == "foo"
    assert data["instance_id"] == 777
    # Named deployments get a name-tagged label for easy identification.
    label = provider.create_instance.call_args.kwargs.get("label") or ""
    assert "foo" in label
    # The legacy default path stays untouched.
    assert not (project_dir / ".hostai-vast" / "state.json").exists()


def test_cmd_up_default_uses_legacy_path(config, project_dir):
    runner = CliRunner()
    ctx, _provider = _fresh_up_mocks(config)
    with ctx:
        result = runner.invoke(cmd_up, [], obj=config)
    assert result.exit_code == 0, result.output
    assert (project_dir / ".hostai-vast" / "state.json").is_file()
    assert not (project_dir / ".hostai-vast" / "instances").exists()


def test_cmd_up_name_env_var(config, project_dir):
    runner = CliRunner()
    ctx, _provider = _fresh_up_mocks(config)
    with ctx:
        result = runner.invoke(cmd_up, [], obj=config, env={"HOSTAI_INSTANCE": "envfoo"})
    assert result.exit_code == 0, result.output
    assert (project_dir / ".hostai-vast" / "instances" / "envfoo" / "state.json").is_file()


@pytest.mark.parametrize("bad", ["default", "bad name", "../escape", "x" * 65])
def test_cmd_up_invalid_name_rejected(config, bad):
    runner = CliRunner()
    result = runner.invoke(cmd_up, ["--name", bad], obj=config)
    assert result.exit_code != 0
    assert "invalid instance name" in result.output


def test_cmd_up_second_name_does_not_clobber(config, project_dir):
    """Two sequential named ups must leave both state files in place."""
    runner = CliRunner()
    for name in ("alpha", "beta"):
        ctx, _provider = _fresh_up_mocks(config)
        with ctx:
            result = runner.invoke(cmd_up, ["--name", name], obj=config)
        assert result.exit_code == 0, (name, result.output)
    found = state_mod.find_instance_states(config.root_dir)
    assert set(found) == {"alpha", "beta"}
    assert _common.claimed_local_ports(config) == {18080}


# ---------------------------------------------------------------------------
# down --name / --all


def _patch_down_pipeline():
    provider = mock.Mock()
    provider.get_instance.return_value = None  # remote gone -> skip remote steps
    stack = [
        mock.patch("hostai.commands.down._provider", return_value=provider),
        mock.patch("hostai.commands.down.down_instance", return_value="destroyed"),
        mock.patch("hostai.commands.watchdog.stop_watchdog"),
        mock.patch("hostai.commands.down.stop_monitor"),
    ]
    ctx = contextlib.ExitStack()
    mocks = [ctx.enter_context(p) for p in stack]
    return ctx, mocks


def test_cmd_down_all_stops_every_instance(config):
    _save_state(config, None, 100)
    _save_state(config, "foo", 200)
    _save_state(config, "bar", 300)
    runner = CliRunner()
    ctx, (_prov, down_mock, wd_mock, _mon_mock) = _patch_down_pipeline()
    with ctx:
        result = runner.invoke(cmd_down, ["--all", "--yes"], obj=config)
    assert result.exit_code == 0, result.output
    assert down_mock.call_count == 3
    assert {c.args[1].instance_id for c in down_mock.call_args_list} == {100, 200, 300}
    # Per-instance daemon shutdown for each deployment.
    assert {c.kwargs.get("instance") for c in wd_mock.call_args_list} == {"default", "foo", "bar"}


def test_cmd_down_name_selects_one(config):
    _save_state(config, "foo", 200)
    _save_state(config, "bar", 300)
    runner = CliRunner()
    ctx, (_prov, down_mock, wd_mock, _mon_mock) = _patch_down_pipeline()
    with ctx:
        result = runner.invoke(cmd_down, ["--name", "bar", "--yes"], obj=config)
    assert result.exit_code == 0, result.output
    down_mock.assert_called_once()
    assert down_mock.call_args.args[1].instance_id == 300
    assert wd_mock.call_args.kwargs.get("instance") == "bar"


def test_cmd_down_all_with_name_rejected(config):
    _save_state(config, "foo", 1)
    runner = CliRunner()
    result = runner.invoke(cmd_down, ["--all", "--name", "foo"], obj=config)
    assert result.exit_code != 0
    assert "cannot combine" in result.output


def test_cmd_down_multiple_no_name_asks(config):
    _save_state(config, "foo", 1)
    _save_state(config, "bar", 2)
    runner = CliRunner()
    result = runner.invoke(cmd_down, ["--yes"], obj=config)
    assert result.exit_code != 0
    assert "use --name" in result.output


def test_cmd_down_nothing_tracked(config):
    runner = CliRunner()
    result = runner.invoke(cmd_down, ["--all"], obj=config)
    assert result.exit_code == 0
    assert "No local hostai Vast state found" in result.output


# ---------------------------------------------------------------------------
# status overview


def test_cmd_status_overview_lists_instances(config):
    _save_state(config, None, 100, profile="test", gpu="A100", dph=0.5, local_port=18080)
    _save_state(config, "foo", 200, profile="test", gpu="RTX 4090", dph=0.3, local_port=19000)
    provider = mock.Mock()
    provider.get_instance.return_value = {"actual_status": "running"}
    with mock.patch("hostai.commands.status._provider", return_value=provider):
        runner = CliRunner()
        result = runner.invoke(cmd_status, [], obj=config, env={"COLUMNS": "160"})
    assert result.exit_code == 0, result.output
    assert "default" in result.output
    assert "foo" in result.output
    assert "--name" in result.output


def test_cmd_status_single_auto_selects(config):
    _save_state(config, "only", 200, profile="test")
    provider = mock.Mock()
    provider.get_instance.return_value = {"actual_status": "running", "gpu_name": "A100"}
    with mock.patch("hostai.commands.status._provider", return_value=provider):
        with mock.patch("hostai.commands.status.ssh.is_tunnel_healthy", return_value=False):
            with mock.patch("hostai.commands.status._refresh_ssh_state", return_value=False):
                runner = CliRunner()
                result = runner.invoke(cmd_status, [], obj=config)
    assert result.exit_code == 0, result.output


def test_cmd_status_name_selects(config):
    _save_state(config, "a", 1)
    _save_state(config, "b", 2)
    provider = mock.Mock()
    provider.get_instance.return_value = {"actual_status": "running", "gpu_name": "A100"}
    with mock.patch("hostai.commands.status._provider", return_value=provider):
        with mock.patch("hostai.commands.status.ssh.is_tunnel_healthy", return_value=False):
            with mock.patch("hostai.commands.status._refresh_ssh_state", return_value=False):
                runner = CliRunner()
                result = runner.invoke(cmd_status, ["--name", "b"], obj=config)
    assert result.exit_code == 0, result.output
    provider.get_instance.assert_called_once_with(2)


def test_cmd_status_no_state(config):
    runner = CliRunner()
    result = runner.invoke(cmd_status, [], obj=config)
    assert result.exit_code == 0
    assert "No local hostai Vast state found" in result.output


# ---------------------------------------------------------------------------
# Per-instance artifacts


def test_write_env_file_named_instance(config, project_dir):
    sf = state_mod.instance_state_file(config.root_dir, "foo")
    st = State(sf, {"instance_name": "foo", "unsecure": True})
    _write_env_file(config, st, "http://127.0.0.1:8080/v1", "http://127.0.0.1:8080")
    env_path = project_dir / ".hostai-vast" / "instances" / "foo" / "env"
    assert env_path.is_file()
    text = env_path.read_text()
    assert "HOSTAI_INSTANCE='foo'" in text
    # The default env file must not be touched.
    assert not (project_dir / ".hostai-vast" / "env").exists()


def test_start_proxy_named_scopes_argv_and_state(config, project_dir):
    sf = state_mod.instance_state_file(config.root_dir, "foo")
    st = State(sf, {"instance_name": "foo", "local_port": 18081, "unsecure": True})
    config.proxy.tokenized_only = True
    config.proxy.port = 0
    config.ssh.local_port = None
    proc = mock.Mock()
    proc.pid = 4242
    with mock.patch("hostai.commands.up._hostai_binary", return_value=project_dir / "hostai"):
        with mock.patch("hostai.commands.up.utils.port_is_free", return_value=False):
            with mock.patch("hostai.commands.up.utils.find_free_port", return_value=18083):
                with mock.patch("hostai.commands.up.subprocess.Popen", return_value=proc) as popen:
                    with mock.patch("hostai.commands.up.time.sleep"):
                        with mock.patch("hostai.commands.up.time.monotonic", side_effect=[0, 100]):
                            port = _start_proxy(config, st, "http")
    assert port == 18083
    argv = popen.call_args.args[0]
    assert "--name" in argv and "foo" in argv
    # Proxy pid is recorded in the *named* state; logs/env land in its dir.
    assert st.data["proxy_pid"] == 4242
    assert st.local_port == 18083
    assert (project_dir / ".hostai-vast" / "instances" / "foo" / "proxy.log").is_file()
    assert not (project_dir / ".hostai-vast" / "proxy.log").exists()


def test_start_proxy_default_argv(config, project_dir):
    st = State(project_dir / ".hostai-vast" / "state.json", {"local_port": 18081, "unsecure": True})
    config.proxy.tokenized_only = True
    config.proxy.port = 0
    config.ssh.local_port = None
    proc = mock.Mock()
    proc.pid = 4242
    with mock.patch("hostai.commands.up._hostai_binary", return_value=project_dir / "hostai"):
        with mock.patch("hostai.commands.up.utils.port_is_free", return_value=False):
            with mock.patch("hostai.commands.up.utils.find_free_port", return_value=18083):
                with mock.patch("hostai.commands.up.subprocess.Popen", return_value=proc) as popen:
                    with mock.patch("hostai.commands.up.time.sleep"):
                        with mock.patch("hostai.commands.up.time.monotonic", side_effect=[0, 100]):
                            port = _start_proxy(config, st, "http")
    assert port == 18083
    argv = popen.call_args.args[0]
    assert "--name" in argv and "default" in argv
    assert (project_dir / ".hostai-vast" / "proxy.log").is_file()


def test_env_file_contains_hostai_instance_for_default(config, project_dir):
    st = State(project_dir / ".hostai-vast" / "state.json", {"unsecure": True})
    _write_env_file(config, st, "http://127.0.0.1:8080/v1", "http://127.0.0.1:8080")
    text = (project_dir / ".hostai-vast" / "env").read_text()
    assert "HOSTAI_INSTANCE='default'" in text
