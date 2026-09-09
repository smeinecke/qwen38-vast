"""Additional tests for hostai.commands.down."""

import json
from unittest import mock

from click.testing import CliRunner

from hostai.commands import down
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
    assert down._format_upload_log(res) == "=== STDOUT ===\nuploaded 12345 bytes"


def _make_slot_response(payload, status=200):
    resp = mock.Mock(status_code=status, text=json.dumps(payload))
    resp.json.return_value = payload
    return resp


def test_slot_save_success(config, running_state):
    running_state.instance_id = 12345
    running_state.local_port = 18080
    payload = {"n_saved": 100, "n_written": 1000, "timings": {"save_ms": 50}}
    with mock.patch("requests.post", return_value=_make_slot_response(payload)):
        details = down._slot_save(config, running_state)
    assert details is not None
    assert details["n_saved"] == 100


def test_slot_save_bad_status(config, running_state):
    running_state.instance_id = 12345
    running_state.local_port = 18080
    response = mock.Mock(status_code=500, text="")
    response.json.return_value = {}
    with mock.patch("requests.post", return_value=response):
        details = down._slot_save(config, running_state)
    assert details is None


def test_parse_rsync_transferred_bytes_kilobytes():
    stdout = "Total bytes sent: 1.5K\n"
    assert down._parse_rsync_transferred_bytes(stdout) == 1536


def test_parse_rsync_transferred_bytes_sent_line():
    stdout = "sent 2.25M bytes  received 79 bytes\n"
    assert down._parse_rsync_transferred_bytes(stdout) == 2359296


def test_parse_rsync_transferred_bytes_no_match():
    assert down._parse_rsync_transferred_bytes("") is None


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
        mock.patch("hostai.commands.down._save_and_upload_slot_cache", return_value=None),
        mock.patch("hostai.commands.down._archive_session"),
        mock.patch("hostai.commands.down._stop_remote_model"),
        mock.patch("hostai.commands.down.ssh.stop_tunnel"),
        mock.patch("hostai.commands.down._pause_or_destroy", return_value="paused"),
    ):
        outcome = down.down_instance(config, state, pause=True, skip_confirm=True)
    assert outcome == "paused"


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
        mock.patch("hostai.commands.down._provider", return_value=mock.Mock(get_instance=mock.Mock(return_value={}))) as pi,
        mock.patch("hostai.commands.down.ssh.run_remote", return_value=fake_completed()),
        mock.patch("hostai.commands.down.api.LlamaClient", return_value=mock.Mock(health=mock.Mock(return_value=False))),
    ):
        down._archive_session(config, state, run_dir, no_archive=False)
    assert (run_dir / "metadata.json").exists()
    pi.assert_called_once()
