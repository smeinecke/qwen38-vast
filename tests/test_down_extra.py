"""Additional tests for hostai.commands.down."""

import json
from unittest import mock

import click
import pytest
from click.testing import CliRunner

from hostai.cache import format_upload_log, parse_rsync_transferred_bytes, save_slot
from hostai.commands import _common, down
from hostai.state import State


def fake_completed(returncode=0, stdout="", stderr=""):
    return mock.Mock(returncode=returncode, stdout=stdout, stderr=stderr)


def _write_state(project_dir, **kwargs):
    state_file = project_dir / ".hostai-vast" / "state.json"
    state_file.parent.mkdir(parents=True)
    State(state_file, kwargs).save()


def test_cmd_down_with_defaults(config, project_dir):
    _write_state(
        project_dir,
        instance_id=12345,
        profile="test",
        local_port=18080,
        dph=0.5,
        ctx_size=32768,
    )

    with (
        mock.patch("hostai.commands.down.down_instance", return_value="destroyed") as di,
        mock.patch("hostai.commands.watchdog.stop_watchdog"),
        mock.patch("hostai.commands.monitor.stop_monitor"),
    ):
        runner = CliRunner()
        result = runner.invoke(down.cmd_down, [], obj=config)

    assert result.exit_code == 0, result.output
    di.assert_called_once()


def test_cmd_down_no_state(config, project_dir):
    runner = CliRunner()
    result = runner.invoke(down.cmd_down, [], obj=config)
    assert result.exit_code == 0
    assert "No local hostai Vast state found" in result.output


def test_cmd_down_no_instance_id(config, project_dir):
    _write_state(project_dir)
    runner = CliRunner()
    result = runner.invoke(down.cmd_down, [], obj=config)
    assert result.exit_code == 0
    assert "No Vast instance id" in result.output


def test_client_log_writes_to_file(tmp_path):
    down._client_log(tmp_path, "hello")
    text = (tmp_path / "client-down.log").read_text()
    assert "hello" in text


def test_format_upload_log_handles_bytes():
    res = mock.Mock(stdout="uploaded 12345 bytes", stderr="")
    assert format_upload_log(res) == "=== STDOUT ===\nuploaded 12345 bytes"


def _make_slot_response(payload, status=200):
    resp = mock.Mock(status_code=status, text=json.dumps(payload))
    resp.json.return_value = payload
    return resp


def test_slot_save_success(config, running_state):
    running_state.instance_id = 12345
    running_state.local_port = 18080
    payload = {"n_saved": 100, "n_written": 1000, "timings": {"save_ms": 50}}
    with mock.patch("requests.post", return_value=_make_slot_response(payload)):
        details = save_slot(config, running_state)
    assert details is not None
    assert details["n_saved"] == 100


def test_slot_save_bad_status(config, running_state):
    running_state.instance_id = 12345
    running_state.local_port = 18080
    response = mock.Mock(status_code=500, text="")
    response.json.return_value = {}
    with mock.patch("requests.post", return_value=response):
        details = save_slot(config, running_state)
    assert details is None


def test_parse_rsync_transferred_bytes_kilobytes():
    stdout = "Total bytes sent: 1.5K\n"
    assert parse_rsync_transferred_bytes(stdout) == 1536


def test_parse_rsync_transferred_bytes_sent_line():
    stdout = "sent 2.25M bytes  received 79 bytes\n"
    assert parse_rsync_transferred_bytes(stdout) == 2359296


def test_parse_rsync_transferred_bytes_no_match():
    assert parse_rsync_transferred_bytes("") is None


def test_cmd_down_pause_and_no_cache(config, project_dir):
    _write_state(
        project_dir,
        instance_id=12345,
        profile="test",
        local_port=18080,
        dph=0.5,
        ctx_size=32768,
        ssh_url="ssh://root@10.0.0.1:22",
    )

    with (
        mock.patch("hostai.commands.down.down_instance", return_value="paused") as di,
        mock.patch("hostai.commands.watchdog.stop_watchdog"),
    ):
        runner = CliRunner()
        result = runner.invoke(down.cmd_down, ["--yes", "--pause", "--no-cache"], obj=config)

    assert result.exit_code == 0, result.output
    di.assert_called_once()
    _, kwargs = di.call_args
    assert kwargs["pause"] is True
    assert kwargs["no_cache"] is True


def test_down_instance_cancel_on_confirm(config, project_dir):
    state = _write_state(
        project_dir,
        instance_id=12345,
        profile="test",
        local_port=18080,
        dph=0.5,
        ctx_size=32768,
    )
    state = State(project_dir / ".hostai-vast" / "state.json")
    state.instance_id = 12345
    with mock.patch("click.confirm", return_value=False):
        outcome = down.down_instance(config, state)
    assert outcome == "cancelled"


def test_down_instance_pause_with_skip_confirm(config, project_dir):
    _write_state(
        project_dir,
        instance_id=12345,
        profile="test",
        local_port=18080,
        dph=0.5,
        ctx_size=32768,
        ssh_url="ssh://root@10.0.0.1:22",
    )
    state = State(project_dir / ".hostai-vast" / "state.json")
    state.instance_id = 12345
    run_dir = project_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    with (
        mock.patch("hostai.commands.down.init_run_dir", return_value=run_dir),
        mock.patch("hostai.commands.down._stop_proxy"),
        mock.patch("hostai.commands.down._refresh_ssh_state"),
        mock.patch("hostai.commands.down.ssh.ensure_tunnel"),
        mock.patch("hostai.cache.save_and_upload_slot_cache", return_value=None),
        mock.patch("hostai.commands.down._archive_session"),
        mock.patch("hostai.commands.down._stop_remote_model"),
        mock.patch("hostai.commands.down.ssh.stop_tunnel"),
        mock.patch("hostai.commands.down._pause_or_destroy", return_value="paused"),
    ):
        outcome = down.down_instance(config, state, pause=True, skip_confirm=True)
    assert outcome == "paused"


def _run_down(config, project_dir, **kwargs):
    _write_state(
        project_dir,
        instance_id=12345,
        profile="test",
        local_port=18080,
        dph=0.5,
        ctx_size=32768,
        ssh_url="ssh://root@10.0.0.1:22",
    )
    state = State(project_dir / ".hostai-vast" / "state.json")
    state.instance_id = 12345
    state.ssh_url = "ssh://root@10.0.0.1:22"
    run_dir = project_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    with (
        mock.patch("hostai.commands.down.init_run_dir", return_value=run_dir),
        mock.patch("hostai.commands.down._stop_proxy"),
        mock.patch("hostai.commands.down._refresh_ssh_state"),
        mock.patch("hostai.commands.down.ssh.ensure_tunnel"),
        mock.patch("hostai.cache.save_and_upload_slot_cache", return_value=None),
        mock.patch("hostai.commands.down._archive_session"),
        mock.patch("hostai.commands.down._stop_remote_model") as stop_model,
        mock.patch("hostai.commands.down.ssh.stop_tunnel"),
        mock.patch("hostai.commands.down._pause_or_destroy", return_value="destroyed"),
    ):
        outcome = down.down_instance(config, state, skip_confirm=True, **kwargs)
    return outcome, stop_model


def test_down_instance_stops_llama_by_default(config, project_dir):
    outcome, stop_model = _run_down(config, project_dir)
    assert outcome == "destroyed"
    stop_model.assert_called_once()


def test_down_instance_skip_llama(config, project_dir):
    outcome, stop_model = _run_down(config, project_dir, skip_llama=True)
    assert outcome == "destroyed"
    stop_model.assert_not_called()


def test_down_instance_unix_upstream_uses_proxy_endpoint(config, project_dir):
    """Tokenized-only sessions must not open a raw TCP tunnel: the proxy is
    the local API endpoint and owns the unix-socket SSH forward."""
    sock_path = project_dir / "upstream.sock"
    _write_state(
        project_dir,
        instance_id=12345,
        profile="test",
        local_port=18081,
        dph=0.5,
        ctx_size=32768,
        ssh_url="ssh://root@10.0.0.1:22",
        upstream_socket=str(sock_path),
    )
    state = State(project_dir / ".hostai-vast" / "state.json")
    state.instance_id = 12345
    state.ssh_url = "ssh://root@10.0.0.1:22"
    state.data["upstream_socket"] = str(sock_path)
    run_dir = project_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    with (
        mock.patch("hostai.commands.down.init_run_dir", return_value=run_dir),
        mock.patch("hostai.commands.down._stop_proxy") as stop_proxy,
        mock.patch("hostai.commands.down._refresh_ssh_state"),
        mock.patch("hostai.commands.down.ssh.ensure_tunnel") as ensure,
        mock.patch("hostai.cache.save_and_upload_slot_cache", return_value=None),
        mock.patch("hostai.commands.down._archive_session"),
        mock.patch("hostai.commands.down._stop_remote_model"),
        mock.patch("hostai.commands.down.ssh.stop_tunnel"),
        mock.patch("hostai.commands.down._pause_or_destroy", return_value="destroyed"),
    ):
        outcome = down.down_instance(config, state, skip_confirm=True)

    assert outcome == "destroyed"
    ensure.assert_not_called()
    stop_proxy.assert_called_once()


def test_cmd_down_skip_llama_flag(config, project_dir):
    _write_state(
        project_dir,
        instance_id=12345,
        profile="test",
        local_port=18080,
        dph=0.5,
        ctx_size=32768,
    )

    with (
        mock.patch("hostai.commands.down.down_instance", return_value="destroyed") as di,
        mock.patch("hostai.commands.watchdog.stop_watchdog"),
        mock.patch("hostai.commands.monitor.stop_monitor"),
    ):
        runner = CliRunner()
        result = runner.invoke(down.cmd_down, ["--yes", "--skip-llama"], obj=config)

    assert result.exit_code == 0, result.output
    di.assert_called_once()
    assert di.call_args.kwargs["skip_llama"] is True


def test_archive_session_creates_json(config, state, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state.run_dir = str(run_dir)
    state.instance_id = 12345
    state.profile = "test"
    state.started_epoch = 0
    state.dph = 0.5
    state.data["slot_cache_save"] = "not-yet"
    state.ssh_url = "ssh://root@10.0.0.1:22"
    with (
        mock.patch(
            "hostai.commands.down._provider", return_value=mock.Mock(get_instance=mock.Mock(return_value={}))
        ) as pi,
        mock.patch("hostai.commands.down.ssh.run_remote", return_value=fake_completed()),
        mock.patch(
            "hostai.commands.down.api.LlamaClient", return_value=mock.Mock(health=mock.Mock(return_value=False))
        ),
    ):
        down._archive_session(config, state, run_dir, no_archive=False)
    assert (run_dir / "metadata.json").exists()
    pi.assert_called_once()


def test_instance_remote_status_gone(config, project_dir):
    state = State(project_dir / ".hostai-vast" / "state.json")
    state.instance_id = 12345
    provider = mock.Mock()
    provider.get_instance.return_value = None
    with mock.patch("hostai.commands.down._provider", return_value=provider):
        remote_ok, inst, reason = down._instance_remote_status(config, state)
    assert remote_ok is False
    assert inst is None
    assert "no longer exists" in reason


def test_instance_remote_status_terminal(config, project_dir):
    state = State(project_dir / ".hostai-vast" / "state.json")
    state.instance_id = 12345
    provider = mock.Mock()
    provider.get_instance.return_value = {"actual_status": "stopped"}
    with mock.patch("hostai.commands.down._provider", return_value=provider):
        remote_ok, inst, reason = down._instance_remote_status(config, state)
    assert remote_ok is False
    assert inst is not None
    assert "stopped" in reason


def test_instance_remote_status_provider_error_fails_open(config, project_dir):
    state = State(project_dir / ".hostai-vast" / "state.json")
    state.instance_id = 12345
    provider = mock.Mock()
    provider.get_instance.side_effect = RuntimeError("api down")
    with mock.patch("hostai.commands.down._provider", return_value=provider):
        remote_ok, inst, _ = down._instance_remote_status(config, state)
    assert remote_ok is True
    assert inst is None


def test_down_instance_gone_skips_remote_steps(config, project_dir):
    _write_state(
        project_dir,
        instance_id=12345,
        profile="test",
        local_port=18080,
        dph=0.5,
        ctx_size=32768,
        ssh_url="ssh://root@10.0.0.1:22",
    )
    state = State(project_dir / ".hostai-vast" / "state.json")
    state.instance_id = 12345
    state.ssh_url = "ssh://root@10.0.0.1:22"
    run_dir = project_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    provider = mock.Mock()
    provider.get_instance.return_value = None

    with (
        mock.patch("hostai.commands.down._provider", return_value=provider),
        mock.patch("hostai.commands.down.init_run_dir", return_value=run_dir),
        mock.patch("hostai.commands.down._stop_proxy"),
        mock.patch("hostai.commands.down._refresh_ssh_state") as refresh,
        mock.patch("hostai.commands.down.ssh.ensure_tunnel") as ensure,
        mock.patch("hostai.cache.save_and_upload_slot_cache", return_value=None) as save_cache,
        mock.patch("hostai.commands.down._archive_session") as archive,
        mock.patch("hostai.commands.down._stop_remote_model") as stop_model,
        mock.patch("hostai.commands.down.ssh.stop_tunnel"),
        mock.patch("hostai.commands.down._pause_or_destroy", return_value="destroyed"),
    ):
        outcome = down.down_instance(config, state, skip_confirm=True)

    assert outcome == "destroyed"
    refresh.assert_not_called()
    ensure.assert_not_called()
    save_cache.assert_not_called()
    stop_model.assert_not_called()
    archive.assert_called_once()
    assert archive.call_args.kwargs["remote_ok"] is False


def test_down_instance_alive_runs_remote_steps(config, project_dir):
    _write_state(
        project_dir,
        instance_id=12345,
        profile="test",
        local_port=18080,
        dph=0.5,
        ctx_size=32768,
        ssh_url="ssh://root@10.0.0.1:22",
    )
    state = State(project_dir / ".hostai-vast" / "state.json")
    state.instance_id = 12345
    state.ssh_url = "ssh://root@10.0.0.1:22"
    run_dir = project_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    provider = mock.Mock()
    provider.get_instance.return_value = {"actual_status": "running"}

    with (
        mock.patch("hostai.commands.down._provider", return_value=provider),
        mock.patch("hostai.commands.down.init_run_dir", return_value=run_dir),
        mock.patch("hostai.commands.down._stop_proxy"),
        mock.patch("hostai.commands.down._refresh_ssh_state"),
        mock.patch("hostai.commands.down.ssh.ensure_tunnel") as ensure,
        mock.patch("hostai.cache.save_and_upload_slot_cache", return_value=None) as save_cache,
        mock.patch("hostai.commands.down._archive_session") as archive,
        mock.patch("hostai.commands.down._stop_remote_model") as stop_model,
        mock.patch("hostai.commands.down.ssh.stop_tunnel"),
        mock.patch("hostai.commands.down._pause_or_destroy", return_value="destroyed"),
    ):
        outcome = down.down_instance(config, state, skip_confirm=True)

    assert outcome == "destroyed"
    ensure.assert_called_once()
    save_cache.assert_called_once()
    stop_model.assert_called_once()
    archive.assert_called_once()
    assert archive.call_args.kwargs["remote_ok"] is True


def test_archive_session_skips_remote_when_gone(config, state, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state.run_dir = str(run_dir)
    state.instance_id = 12345
    state.ssh_url = "ssh://root@10.0.0.1:22"
    provider = mock.Mock()
    with (
        mock.patch("hostai.commands.down._provider", return_value=provider) as pi,
        mock.patch("hostai.commands.down.ssh.run_remote") as run_remote,
    ):
        down._archive_session(config, state, run_dir, no_archive=False, remote_ok=False)
    assert (run_dir / "metadata.json").exists()
    pi.assert_not_called()
    run_remote.assert_not_called()


def test_lifecycle_lock_blocks_second_holder(config, project_dir):
    config.root_dir = project_dir
    with _common.lifecycle_lock(config, "up"):
        with pytest.raises(click.ClickException, match="in progress"):
            with _common.lifecycle_lock(config, "up"):
                pass


def test_lifecycle_lock_released_after_exit(config, project_dir):
    config.root_dir = project_dir
    with _common.lifecycle_lock(config, "up"):
        pass
    with _common.lifecycle_lock(config, "up --restart"):  # must not raise
        pass


def test_pid_is_hostai_proxy_rejects_foreign_pid():
    assert _pid_check(b"/usr/bin/python3\x00-m\x00hostai\x00proxy\x00")
    assert _pid_check(b"/home/u/.venv/bin/hostai\x00proxy\x00")
    assert not _pid_check(b"/usr/bin/vim\x00file.txt\x00")


def _pid_check(cmdline: bytes) -> bool:
    """Exercise _common.pid_cmdline_contains without depending on a live pid."""
    real_pid = 12345
    with mock.patch("hostai.commands._common.Path.read_bytes", return_value=cmdline):
        return _common.pid_cmdline_contains(real_pid, b"hostai", b"proxy")


def test_daemon_pid_running(project_dir):
    pid_file = project_dir / "d.pid"
    assert not _common.daemon_pid_running(pid_file, b"hostai")  # missing file
    pid_file.write_text("garbage")
    assert not _common.daemon_pid_running(pid_file, b"hostai")  # unparsable
    pid_file.write_text("999999999")
    assert not _common.daemon_pid_running(pid_file, b"hostai")  # dead pid

    pid_file.write_text("12345")
    with mock.patch("os.kill"):  # process exists
        with mock.patch("hostai.commands._common.pid_cmdline_contains", return_value=True):
            assert _common.daemon_pid_running(pid_file, b"hostai", b"watchdog")
        with mock.patch("hostai.commands._common.pid_cmdline_contains", return_value=False):
            # Alive but wrong identity (PID reuse) — must NOT count as running.
            assert not _common.daemon_pid_running(pid_file, b"hostai", b"watchdog")


def test_running_proxy_pid_falls_back_to_pid_file(project_dir):
    """When state.proxy_pid was never persisted, proxy.pid still finds the daemon."""
    state_file = project_dir / ".hostai-vast" / "state.json"
    state_file.parent.mkdir(parents=True)
    state = State(state_file, {"instance_id": 1})
    (state_file.parent / "proxy.pid").write_text("7777")

    with mock.patch("os.kill"):
        with mock.patch("hostai.commands._common.pid_cmdline_contains", return_value=True):
            assert _common.running_proxy_pid(state) == 7777
        with mock.patch("hostai.commands._common.pid_cmdline_contains", return_value=False):
            assert _common.running_proxy_pid(state) is None


def test_stop_proxy_uses_pid_file(config, project_dir):
    """down must stop a proxy that is only recorded in proxy.pid."""
    import signal

    _write_state(project_dir, instance_id=1)  # no proxy_pid in state
    state = State.load(project_dir / ".hostai-vast" / "state.json")
    pid_file = state.state_file.parent / "proxy.pid"
    pid_file.write_text("7777")

    kills = []

    def fake_kill(pid, sig=0):
        kills.append((pid, sig))
        # The process is "alive" for liveness checks until SIGTERM lands.
        if sig == 0 and any(s == signal.SIGTERM for _, s in kills):
            raise ProcessLookupError

    with mock.patch("hostai.commands._common.os.kill", side_effect=fake_kill):
        with mock.patch("hostai.commands._common.pid_cmdline_contains", return_value=True):
            with mock.patch("hostai.commands.down.os.kill", side_effect=fake_kill):
                down._stop_proxy(state)

    assert (7777, signal.SIGTERM) in kills
    assert not pid_file.exists()
