"""Tests for hostai.commands.watchdog."""

from unittest import mock

from click.testing import CliRunner

from hostai.commands import watchdog


def test_watchdog_paths(config, project_dir):
    config.root_dir = project_dir
    assert watchdog._watchdog_pid_file(config) == project_dir / ".hostai-cache" / "watchdog.pid"
    assert watchdog._watchdog_log_file(config) == project_dir / ".hostai-cache" / "watchdog.log"


def test_is_running_with_missing_pid():
    assert watchdog._is_running(9999999) is False


def test_hostai_executable_exists():
    assert isinstance(watchdog._hostai_executable(), str)


def test_log_writes_to_file(config, project_dir):
    config.root_dir = project_dir
    watchdog._log(config, "test")
    assert (project_dir / ".hostai-cache" / "watchdog.log").exists()


def make_client(metrics, slots):
    client = mock.Mock()
    client.get_metrics.return_value = metrics
    client.slots.return_value = slots
    return client


def test_is_request_active_first_observation():
    client = make_client({"llamacpp:prompt_tokens_total": 0, "llamacpp:tokens_predicted_total": 0}, [])
    state, prev = watchdog._is_request_active(client, {})
    assert state == "active"


def test_is_request_active_inactive():
    client = make_client({"llamacpp:prompt_tokens_total": 1, "llamacpp:tokens_predicted_total": 2}, [])
    previous = {"llamacpp:prompt_tokens_total": 1, "llamacpp:tokens_predicted_total": 2, "slots": [], "n_processing_slots": 0}
    state, prev = watchdog._is_request_active(client, previous)
    assert state == "inactive"


def test_is_request_active_active():
    client = make_client({"llamacpp:prompt_tokens_total": 2, "llamacpp:tokens_predicted_total": 2}, [])
    previous = {"llamacpp:prompt_tokens_total": 1, "llamacpp:tokens_predicted_total": 2, "slots": [], "n_processing_slots": 0}
    state, _ = watchdog._is_request_active(client, previous)
    assert state == "active"


def test_cmd_watchdog_status_no_daemon(config, project_dir):
    config.root_dir = project_dir
    runner = CliRunner()
    result = runner.invoke(watchdog.cmd_watchdog_status, [], obj=config)
    assert result.exit_code == 0
    assert "not running" in result.output.lower()


def test_cmd_watchdog_stop_no_pid(config, project_dir):
    config.root_dir = project_dir
    runner = CliRunner()
    result = runner.invoke(watchdog.cmd_watchdog_stop, [], obj=config)
    assert result.exit_code == 0
    assert "not running" in result.output.lower()


def test_cmd_watchdog_start_and_stop(config, project_dir):
    config.root_dir = project_dir
    config.vast.watchdog_auto_start = True
    with mock.patch("subprocess.Popen") as popen:
        popen.return_value.pid = 12345
        runner = CliRunner()
        result = runner.invoke(watchdog.cmd_watchdog_start, [], obj=config)
        assert result.exit_code == 0

    runner = CliRunner()
    with mock.patch("os.kill"):
        with mock.patch("hostai.commands.watchdog._is_running", return_value=True):
            result = runner.invoke(watchdog.cmd_watchdog_stop, [], obj=config)
    assert result.exit_code == 0


def test_maybe_start_watchdog_skips_when_disabled(config, project_dir, running_state):
    config.root_dir = project_dir
    config.vast.watchdog_auto_start = False
    watchdog.maybe_start_watchdog(config, running_state)
    assert not watchdog._watchdog_pid_file(config).exists()


def test_maybe_start_watchdog_launches(config, project_dir, running_state):
    config.root_dir = project_dir
    config.vast.watchdog_auto_start = True
    config.vast.idle_timeout_seconds = 60
    running_state.instance_id = 12345

    def fake_callback(config):
        watchdog._watchdog_pid_file(config).parent.mkdir(parents=True, exist_ok=True)
        watchdog._watchdog_pid_file(config).write_text("12345")

    with mock.patch.object(watchdog, "_start_watchdog", fake_callback):
        watchdog.maybe_start_watchdog(config, running_state)
    assert watchdog._watchdog_pid_file(config).read_text() == "12345"


def test_stop_watchdog_with_missing_pid(config, project_dir):
    config.root_dir = project_dir
    watchdog.stop_watchdog(config)


def make_llama_client(metrics, slots):
    client = mock.Mock()
    client.get_metrics.return_value = metrics
    client.slots.return_value = slots
    return client


def test_run_once_active(config, running_state):
    running_state.instance_id = 12345
    client = make_llama_client({"llamacpp:prompt_tokens_total": 1, "llamacpp:tokens_predicted_total": 2}, [])
    with mock.patch("hostai.commands.watchdog.api.LlamaClient", return_value=client):
        current, last, shutdown, fails, inactive = watchdog._run_once(
            config, running_state, {}, 0, None, 0, 0
        )
    assert shutdown is False
    assert fails == 0


def test_run_once_unknown(config, running_state):
    running_state.instance_id = 12345
    client = mock.Mock()
    client.get_metrics.side_effect = Exception("down")
    with mock.patch("hostai.commands.watchdog.api.LlamaClient", return_value=client):
        current, last, shutdown, fails, inactive = watchdog._run_once(
            config, running_state, {}, 0, None, 0, 0
        )
    assert fails == 1
    assert shutdown is False


def test_run_once_idle_timeout_triggers_shutdown(config, running_state):
    running_state.instance_id = 12345
    config.vast.idle_timeout_seconds = 1
    previous = {"llamacpp:prompt_tokens_total": 1, "llamacpp:tokens_predicted_total": 2, "slots": [], "n_processing_slots": 0}
    client = make_llama_client({"llamacpp:prompt_tokens_total": 1, "llamacpp:tokens_predicted_total": 2}, [])
    with mock.patch("hostai.commands.watchdog.api.LlamaClient", return_value=client):
        with mock.patch("hostai.commands.watchdog.down_instance") as down:
            with mock.patch("time.time", return_value=10):
                current, last, shutdown, fails, inactive = watchdog._run_once(
                    config, running_state, previous, 0, None, 0, 2
                )
    assert shutdown is True
    down.assert_called_once()


def test_run_once_max_runtime_active_waits(config, running_state):
    running_state.instance_id = 12345
    config.vast.max_runtime_seconds = 60
    client = make_llama_client({"llamacpp:prompt_tokens_total": 2, "llamacpp:tokens_predicted_total": 2}, [])
    with mock.patch("hostai.commands.watchdog.api.LlamaClient", return_value=client):
        with mock.patch("time.time", return_value=10):
            current, last, shutdown, fails, inactive = watchdog._run_once(
                config, running_state, {}, 0, 10, 0, 0
            )
    assert shutdown is False


def test_run_watchdog_no_state_file(config, project_dir, capsys):
    config.root_dir = project_dir
    watchdog.run_watchdog(config)


def test_run_watchdog_no_instance(config, project_dir, running_state):
    config.root_dir = project_dir
    running_state.instance_id = None
    running_state.save()
    watchdog.run_watchdog(config)


def test_run_watchdog_loop_exits(config, project_dir, running_state):
    config.root_dir = project_dir
    config.vast.idle_timeout_seconds = 60
    running_state.instance_id = 12345
    running_state.started_epoch = 1
    running_state.save()
    with mock.patch("hostai.commands.watchdog._run_once", return_value=({}, 0, True, 0, 0)):
        with mock.patch("time.sleep"):
            with mock.patch("time.time", return_value=0):
                watchdog.run_watchdog(config)
