"""Tests for the idle/max-runtime watchdog."""

from types import SimpleNamespace
from unittest import mock

import pytest
from click.testing import CliRunner

from hostai.commands.watchdog import (
    _is_request_active,
    _run_once,
    cmd_watchdog_run,
    cmd_watchdog_start,
)


def make_client(metrics=None, slots=None, healthy=True):
    client = mock.Mock()
    client.get_metrics.return_value = metrics or {}
    client.slots.return_value = slots or []
    client.health.return_value = healthy
    return client


def test_is_request_active_first_call_is_active():
    client = make_client({"llamacpp:prompt_tokens_total": 10}, [])
    active, snapshot = _is_request_active(client, {})
    assert active is True
    assert snapshot["llamacpp:prompt_tokens_total"] == 10


def test_is_request_active_counters_changed():
    client = make_client({"llamacpp:prompt_tokens_total": 20})
    active, _ = _is_request_active(client, {"llamacpp:prompt_tokens_total": 10})
    assert active is True


def test_is_request_active_slots_processing():
    client = make_client({"llamacpp:prompt_tokens_total": 10}, [{"id": 0, "is_processing": True}])
    active, _ = _is_request_active(client, {"llamacpp:prompt_tokens_total": 10})
    assert active is True


def test_is_request_active_idle():
    client = make_client({"llamacpp:prompt_tokens_total": 10}, [])
    active, _ = _is_request_active(client, {"llamacpp:prompt_tokens_total": 10})
    assert active is False


def test_run_once_idle_timeout_triggers_down(config):
    config.vast.idle_timeout_seconds = 60
    config.vast.max_runtime_seconds = None

    state = mock.Mock()
    state.instance_id = 42
    state.started_epoch = 0
    state.exists = True

    with mock.patch("hostai.commands.watchdog.api.LlamaClient") as Client:
        Client.return_value = make_client({"llamacpp:prompt_tokens_total": 10}, [])
        with mock.patch("hostai.commands.watchdog.down_instance") as down:
            last_activity = 0.0
            with mock.patch("hostai.commands.watchdog.time.time", return_value=100):
                _run_once(config, state, {"llamacpp:prompt_tokens_total": 10}, last_activity, None)

    down.assert_called_once()
    assert down.call_args.kwargs["reason"] == "idle-timeout"


def test_run_once_max_runtime_waits_while_active(config):
    config.vast.idle_timeout_seconds = None
    config.vast.max_runtime_seconds = 60

    state = mock.Mock()
    state.instance_id = 42
    state.started_epoch = 0
    state.exists = True

    with mock.patch("hostai.commands.watchdog.api.LlamaClient") as Client:
        Client.return_value = make_client({"llamacpp:prompt_tokens_total": 10}, [{"is_processing": True}])
        with mock.patch("hostai.commands.watchdog.down_instance") as down:
            with mock.patch("hostai.commands.watchdog.time.time", side_effect=[100, 100]):
                _run_once(config, state, {}, 0, 60)

    down.assert_not_called()


def test_run_once_max_runtime_triggers_down_when_idle(config):
    config.vast.idle_timeout_seconds = None
    config.vast.max_runtime_seconds = 60

    state = mock.Mock()
    state.instance_id = 42
    state.started_epoch = 0
    state.exists = True

    with mock.patch("hostai.commands.watchdog.api.LlamaClient") as Client:
        Client.return_value = make_client({"llamacpp:prompt_tokens_total": 10}, [])
        with mock.patch("hostai.commands.watchdog.down_instance") as down:
            with mock.patch("hostai.commands.watchdog.time.time", return_value=100):
                _run_once(config, state, {"llamacpp:prompt_tokens_total": 10}, 0, 60)

    down.assert_called_once()
    assert down.call_args.kwargs["reason"] == "max-runtime"


def test_cmd_watchdog_run_exits_without_state(config, project_dir):
    runner = CliRunner()
    result = runner.invoke(cmd_watchdog_run, [], obj=config)
    assert result.exit_code == 0


def test_cmd_watchdog_start_creates_pid_and_log(config, project_dir):
    with mock.patch("hostai.commands.watchdog.subprocess.Popen") as Popen:
        Popen.return_value = SimpleNamespace(pid=12345)
        runner = CliRunner()
        result = runner.invoke(cmd_watchdog_start, [], obj=config)
    assert result.exit_code == 0
    assert "pid 12345" in result.output
    assert (project_dir / ".hostai-cache" / "watchdog.pid").exists()
