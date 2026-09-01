"""Tests for hostai monitor daemon commands (start/stop/status/logs)."""

import os
import signal
from unittest import mock

import pytest
from click.testing import CliRunner

from hostai.commands.monitor import (
    cmd_monitor_logs,
    cmd_monitor_start,
    cmd_monitor_status,
    cmd_monitor_stop,
    _monitor_is_running,
    _monitor_log_file,
    _monitor_pid_file,
)


def test_monitor_pid_and_log_files(config):
    assert _monitor_pid_file(config) == config.root_dir / ".hostai-cache" / "monitor.pid"
    assert _monitor_log_file(config) == config.root_dir / ".hostai-cache" / "monitor.log"


def test_monitor_is_running():
    assert _monitor_is_running(os.getpid()) is True
    assert _monitor_is_running(99999999) is False


def test_monitor_start(config):
    pid_file = _monitor_pid_file(config)
    log_file = _monitor_log_file(config)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fake_proc = mock.Mock(pid=12345)
    with mock.patch("hostai.commands.monitor.subprocess.Popen", return_value=fake_proc) as popen:
        with mock.patch("hostai.commands.monitor._hostai_executable", return_value="hostai"):
            runner = CliRunner()
            result = runner.invoke(cmd_monitor_start, ["--profile", "test"], obj=config)

    assert result.exit_code == 0, result.output
    assert "started daemon" in result.output
    assert pid_file.read_text() == "12345"
    assert "test" in log_file.read_text()
    popen.assert_called_once()
    cmd = popen.call_args.args[0]
    assert cmd[:4] == ["hostai", "monitor", "watch", "--interval"]


def test_monitor_start_already_running(config):
    pid_file = _monitor_pid_file(config)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text("12345")

    with mock.patch("hostai.commands.monitor._monitor_is_running", return_value=True):
        runner = CliRunner()
        result = runner.invoke(cmd_monitor_start, [], obj=config)

    assert result.exit_code == 0
    assert "already running" in result.output


def test_monitor_stop(config):
    pid_file = _monitor_pid_file(config)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text("12345")

    with mock.patch("hostai.commands.monitor._monitor_is_running", side_effect=[True, False, False]):
        with mock.patch("os.kill") as kill:
            runner = CliRunner()
            result = runner.invoke(cmd_monitor_stop, [], obj=config)

    assert result.exit_code == 0
    assert "stopped" in result.output
    assert not pid_file.exists()
    kill.assert_called_with(12345, signal.SIGTERM)


def test_monitor_stop_not_running(config):
    pid_file = _monitor_pid_file(config)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text("12345")

    with mock.patch("hostai.commands.monitor._monitor_is_running", return_value=False):
        runner = CliRunner()
        result = runner.invoke(cmd_monitor_stop, [], obj=config)

    assert result.exit_code == 0
    assert "not running" in result.output
    assert not pid_file.exists()


def test_monitor_status_running(config):
    pid_file = _monitor_pid_file(config)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text("12345")

    with mock.patch("hostai.commands.monitor._monitor_is_running", return_value=True):
        runner = CliRunner()
        result = runner.invoke(cmd_monitor_status, [], obj=config)

    assert result.exit_code == 0
    assert "running (pid 12345)" in result.output


def test_monitor_status_stale(config):
    pid_file = _monitor_pid_file(config)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text("99999999")

    with mock.patch("hostai.commands.monitor._monitor_is_running", return_value=False):
        runner = CliRunner()
        result = runner.invoke(cmd_monitor_status, [], obj=config)

    assert result.exit_code == 0
    assert "stale" in result.output
    assert not pid_file.exists()


def test_monitor_logs(config):
    log_file = _monitor_log_file(config)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("line1\nline2\n")

    with mock.patch("hostai.commands.monitor.subprocess.run", return_value=mock.Mock(stdout="line2\n")) as run:
        runner = CliRunner()
        result = runner.invoke(cmd_monitor_logs, ["--lines", "1"], obj=config)

    assert result.exit_code == 0
    assert "line2" in result.output
    run.assert_called_once_with(["tail", "-n", "1", str(log_file)], capture_output=True, text=True, check=False)
