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
    maybe_start_watchdog,
)


def make_client(metrics=None, slots=None, healthy=True, metrics_error=False, slots_error=False):
    client = mock.Mock()
    client.health.return_value = healthy

    def _get_metrics(raise_on_error=False):
        if metrics_error:
            if raise_on_error:
                raise RuntimeError("metrics unreachable")
            return {}
        return metrics or {}

    def _slots(raise_on_error=False):
        if slots_error:
            if raise_on_error:
                raise RuntimeError("slots unreachable")
            return []
        if slots is not None and not isinstance(slots, list):
            if raise_on_error:
                raise RuntimeError("slots returned a non-list payload")
            return []
        return slots or []

    client.get_metrics.side_effect = _get_metrics
    client.slots.side_effect = _slots
    return client


def test_is_request_active_first_call_is_active():
    client = make_client({"llamacpp:prompt_tokens_total": 10}, [])
    state, snapshot = _is_request_active(client, {})
    assert state == "active"
    assert snapshot["llamacpp:prompt_tokens_total"] == 10


def test_is_request_active_counters_changed():
    client = make_client({"llamacpp:prompt_tokens_total": 20})
    state, _ = _is_request_active(client, {"llamacpp:prompt_tokens_total": 10})
    assert state == "active"


def test_is_request_active_slots_processing():
    client = make_client({"llamacpp:prompt_tokens_total": 10}, [{"id": 0, "state": 1}])
    state, _ = _is_request_active(client, {"llamacpp:prompt_tokens_total": 10})
    assert state == "active"


# Backwards-compatible alias for is_processing used by older slots payloads.
def test_is_request_active_slots_processing_legacy():
    client = make_client({"llamacpp:prompt_tokens_total": 10}, [{"id": 0, "is_processing": True}])
    state, _ = _is_request_active(client, {"llamacpp:prompt_tokens_total": 10})
    assert state == "active"


def test_is_request_active_idle():
    client = make_client({"llamacpp:prompt_tokens_total": 10}, [])
    state, _ = _is_request_active(client, {"llamacpp:prompt_tokens_total": 10})
    assert state == "inactive"


def test_is_request_active_unknown_on_metrics_error():
    client = make_client(metrics_error=True)
    state, _ = _is_request_active(client, {})
    assert state == "unknown"


def test_is_request_active_unknown_on_slots_error():
    client = make_client({"llamacpp:prompt_tokens_total": 10}, slots_error=True)
    state, _ = _is_request_active(client, {})
    assert state == "unknown"


def test_is_request_active_unknown_on_malformed_slots():
    client = make_client({"llamacpp:prompt_tokens_total": 10}, slots={"not_a_list": True})
    state, _ = _is_request_active(client, {})
    assert state == "unknown"


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
            with mock.patch("hostai.commands.watchdog.time.time", return_value=100):
                _run_once(config, state, {"llamacpp:prompt_tokens_total": 10}, 0, None)

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
        Client.return_value = make_client({"llamacpp:prompt_tokens_total": 10}, [{"state": 1}])
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


def test_run_once_unknown_activity_resets_idle_timer(config):
    """An unreachable API must reset the idle timer, not trigger destruction."""
    config.vast.idle_timeout_seconds = 60
    config.vast.max_runtime_seconds = None

    state = mock.Mock()
    state.instance_id = 42
    state.started_epoch = 0
    state.exists = True

    with mock.patch("hostai.commands.watchdog.api.LlamaClient") as Client:
        Client.return_value = make_client(metrics_error=True)
        with mock.patch("hostai.commands.watchdog.down_instance") as down:
            with mock.patch("hostai.commands.watchdog.time.time", return_value=100):
                current, last_activity, done, failures = _run_once(
                    config, state, {}, 0, None, consecutive_failures=0
                )

    down.assert_not_called()
    assert last_activity == 100
    assert failures == 1


def test_run_once_max_runtime_triggers_down_when_idle_unknown_activity(config):
    """Unknown activity is not idle, so max-runtime must not destroy."""
    config.vast.idle_timeout_seconds = None
    config.vast.max_runtime_seconds = 60

    state = mock.Mock()
    state.instance_id = 42
    state.started_epoch = 0
    state.exists = True

    with mock.patch("hostai.commands.watchdog.api.LlamaClient") as Client:
        Client.return_value = make_client(metrics_error=True)
        with mock.patch("hostai.commands.watchdog.down_instance") as down:
            with mock.patch("hostai.commands.watchdog.time.time", return_value=100):
                _run_once(config, state, {}, 0, 60, consecutive_failures=0)

    down.assert_not_called()


def test_run_once_consecutive_api_failures_are_logged(config):
    config.vast.idle_timeout_seconds = 60
    config.vast.max_runtime_seconds = None

    state = mock.Mock()
    state.instance_id = 42
    state.started_epoch = 0
    state.exists = True

    with mock.patch("hostai.commands.watchdog.api.LlamaClient") as Client:
        Client.return_value = make_client(metrics_error=True)
        with mock.patch("hostai.commands.watchdog._log") as log:
            with mock.patch("hostai.commands.watchdog.time.time", return_value=100):
                for _ in range(6):
                    _run_once(config, state, {}, 0, None, consecutive_failures=0)

    assert log.call_count >= 2


def test_run_once_temporary_api_failure_recovers(config):
    """A single unknown response followed by a recovery must not shut down."""
    config.vast.idle_timeout_seconds = 60
    config.vast.max_runtime_seconds = None

    state = mock.Mock()
    state.instance_id = 42
    state.started_epoch = 0
    state.exists = True

    client = mock.Mock()

    def _metrics(raise_on_error=False):
        if _metrics.call_count < 1:
            _metrics.call_count += 1
            if raise_on_error:
                raise RuntimeError("unreachable")
            return {}
        _metrics.call_count += 1
        return {"llamacpp:prompt_tokens_total": 10}

    _metrics.call_count = 0
    client.get_metrics.side_effect = _metrics
    client.slots.return_value = []

    with mock.patch("hostai.commands.watchdog.api.LlamaClient") as Client:
        Client.return_value = client
        with mock.patch("hostai.commands.watchdog.down_instance") as down:
            with mock.patch("hostai.commands.watchdog.time.time", return_value=100):
                _, _, done1, failures1 = _run_once(config, state, {}, 0, None)
                _, _, done2, failures2 = _run_once(config, state, {}, 0, None, consecutive_failures=failures1)

    assert done1 is False
    assert done2 is False
    down.assert_not_called()


def test_run_once_malformed_metrics_payload_fails_safe(config):
    """A non-dict metrics payload must be treated as unknown."""
    config.vast.idle_timeout_seconds = 60
    config.vast.max_runtime_seconds = None

    state = mock.Mock()
    state.instance_id = 42
    state.started_epoch = 0
    state.exists = True

    client = mock.Mock()
    client.get_metrics.return_value = "not a dict"
    client.slots.return_value = []

    with mock.patch("hostai.commands.watchdog.api.LlamaClient") as Client:
        Client.return_value = client
        with mock.patch("hostai.commands.watchdog.down_instance") as down:
            with mock.patch("hostai.commands.watchdog.time.time", return_value=100):
                state_str, _ = _is_request_active(client, {})

    assert state_str == "unknown"
    down.assert_not_called()


def test_run_once_malformed_slots_payload_fails_safe(config):
    """A non-list slots payload must be treated as unknown."""
    config.vast.idle_timeout_seconds = 60
    config.vast.max_runtime_seconds = None

    state = mock.Mock()
    state.instance_id = 42
    state.started_epoch = 0
    state.exists = True

    client = mock.Mock()
    client.get_metrics.return_value = {"llamacpp:prompt_tokens_total": 10}
    client.slots.return_value = {"not_a_list": True}

    with mock.patch("hostai.commands.watchdog.api.LlamaClient") as Client:
        Client.return_value = client
        with mock.patch("hostai.commands.watchdog.down_instance") as down:
            state_str, _ = _is_request_active(client, {})

    assert state_str == "unknown"
    down.assert_not_called()


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


def test_maybe_start_watchdog_respects_config(config):
    state = mock.Mock(instance_id=12345)
    config.vast.watchdog_auto_start = False
    with mock.patch("hostai.commands.watchdog.cmd_watchdog_start") as start:
        maybe_start_watchdog(config, state)
        start.callback.assert_not_called()

    config.vast.watchdog_auto_start = True
    config.vast.idle_timeout_seconds = 300
    with mock.patch("hostai.commands.watchdog.cmd_watchdog_start") as start:
        maybe_start_watchdog(config, state)
        start.callback.assert_called_once_with(config)
