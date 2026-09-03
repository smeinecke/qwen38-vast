"""Tests for hostai.commands.status with mocked instance/SSH."""

from unittest import mock

from click.testing import CliRunner

from hostai.commands.status import cmd_status


def test_status_no_state(config, project_dir):
    runner = CliRunner()
    result = runner.invoke(cmd_status, [], obj=config)
    assert result.exit_code == 0
    assert "No local hostai Vast state found" in result.output


def test_status_no_instance(config, project_dir):
    state_file = project_dir / ".hostai-vast" / "state.json"
    state_file.parent.mkdir(parents=True)
    state_file.write_text("{}")

    runner = CliRunner()
    result = runner.invoke(cmd_status, [], obj=config)
    assert result.exit_code != 0
    assert "no running instance" in result.output


def test_status_happy_path(config, project_dir):
    state_file = project_dir / ".hostai-vast" / "state.json"
    state_file.parent.mkdir(parents=True)
    state_file.write_text(
        '{"instance_id": 12345, "profile": "test", "local_port": 18080, "dph": 0.5, "ctx_size": 32768}'
    )

    class FakeClient:
        def health(self):
            return True

        def get_metrics(self):
            return {}

    with (
        mock.patch(
            "hostai.commands.status._provider",
            return_value=mock.Mock(get_instance=mock.Mock(return_value={
                "actual_status": "running",
                "gpu_name": "RTX 4090",
            })),
        ),
        mock.patch("hostai.commands.status.ssh.ensure_tunnel"),
        mock.patch("hostai.commands.status.ssh.is_tunnel_healthy", return_value=True),
        mock.patch("hostai.commands.status.api.LlamaClient", return_value=FakeClient()),
        mock.patch("hostai.commands.status.ssh.resolve_ssh_endpoint", return_value=None),
    ):
        runner = CliRunner()
        result = runner.invoke(cmd_status, [], obj=config)

    assert result.exit_code == 0, result.output
    assert "hostai status" in result.output


def test_status_logs(config, project_dir):
    state_file = project_dir / ".hostai-vast" / "state.json"
    state_file.parent.mkdir(parents=True)
    state_file.write_text(
        '{"instance_id": 12345, "profile": "test", "local_port": 18080, "dph": 0.5, "ctx_size": 32768, "ssh_url": "ssh://root@10.0.0.1:2222"}'
    )

    with mock.patch("hostai.commands.status.ssh.run_remote", return_value=mock.Mock(returncode=0, stdout="log line\n")):
        runner = CliRunner()
        result = runner.invoke(cmd_status, ["--logs"], obj=config)

    assert result.exit_code == 0, result.output
    assert "log line" in result.output
