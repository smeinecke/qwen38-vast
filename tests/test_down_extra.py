"""Additional tests for hostai.commands.down."""

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
