"""Tests for hostai.commands.status with mocked instance/SSH."""

from unittest import mock

from click.testing import CliRunner

from hostai.commands.status import _perf_summary, cmd_status


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
            return_value=mock.Mock(
                get_instance=mock.Mock(
                    return_value={
                        "actual_status": "running",
                        "gpu_name": "RTX 4090",
                    }
                )
            ),
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


def test_status_shows_tok_per_s(config, project_dir):
    state_file = project_dir / ".hostai-vast" / "state.json"
    state_file.parent.mkdir(parents=True)
    state_file.write_text(
        '{"instance_id": 12345, "profile": "test", "local_port": 18080, "dph": 0.5, "ctx_size": 32768}'
    )

    class FakeClient:
        def health(self):
            return True

        def get_metrics(self):
            return {
                "llamacpp:tokens_predicted_total": 800.0,
                "llamacpp:tokens_predicted_seconds_total": 10.0,
                "llamacpp:prompt_tokens_total": 4000.0,
                "llamacpp:prompt_seconds_total": 5.0,
                "llamacpp:spec_decode_num_draft_tokens_total": 500.0,
                "llamacpp:spec_decode_num_accepted_tokens_total": 300.0,
            }

    with (
        mock.patch(
            "hostai.commands.status._provider",
            return_value=mock.Mock(get_instance=mock.Mock(return_value={"actual_status": "running"})),
        ),
        mock.patch("hostai.commands.status.ssh.is_tunnel_healthy", return_value=True),
        mock.patch("hostai.commands.status.api.LlamaClient", return_value=FakeClient()),
        mock.patch("hostai.commands.status.ssh.resolve_ssh_endpoint", return_value=None),
    ):
        runner = CliRunner()
        result = runner.invoke(cmd_status, [], obj=config)

    assert result.exit_code == 0, result.output
    assert "decode=80.0 tok/s" in result.output
    assert "prompt=800.0 tok/s" in result.output
    assert "draft-accept=60%" in result.output


def test_status_shows_machine_id_from_state(config, project_dir):
    state_file = project_dir / ".hostai-vast" / "state.json"
    state_file.parent.mkdir(parents=True)
    state_file.write_text(
        '{"instance_id": 12345, "machine_id": 4242, "profile": "test", "local_port": 18080, "dph": 0.5, "ctx_size": 32768}'
    )

    with (
        mock.patch(
            "hostai.commands.status._provider",
            return_value=mock.Mock(get_instance=mock.Mock(return_value={"actual_status": "running"})),
        ),
        mock.patch("hostai.commands.status.ssh.is_tunnel_healthy", return_value=False),
        mock.patch("hostai.commands.status._refresh_ssh_state", return_value=False),
    ):
        runner = CliRunner()
        result = runner.invoke(cmd_status, [], obj=config)

    assert result.exit_code == 0, result.output
    assert "Machine" in result.output
    assert "4242" in result.output


def test_status_machine_id_prefers_provider(config, project_dir):
    state_file = project_dir / ".hostai-vast" / "state.json"
    state_file.parent.mkdir(parents=True)
    state_file.write_text(
        '{"instance_id": 12345, "machine_id": 4242, "profile": "test", "local_port": 18080, "dph": 0.5, "ctx_size": 32768}'
    )

    with (
        mock.patch(
            "hostai.commands.status._provider",
            return_value=mock.Mock(
                get_instance=mock.Mock(return_value={"actual_status": "running", "machine_id": 9999})
            ),
        ),
        mock.patch("hostai.commands.status.ssh.is_tunnel_healthy", return_value=False),
        mock.patch("hostai.commands.status._refresh_ssh_state", return_value=False),
    ):
        runner = CliRunner()
        result = runner.invoke(cmd_status, [], obj=config)

    assert result.exit_code == 0, result.output
    assert "9999" in result.output
    assert "4242" not in result.output


def test_perf_summary_none_without_decode():
    assert _perf_summary({}) is None
    assert _perf_summary({"llamacpp:tokens_predicted_total": 10.0}) is None


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


def test_status_logs_save(config, project_dir):
    state_file = project_dir / ".hostai-vast" / "state.json"
    state_file.parent.mkdir(parents=True)
    state_file.write_text(
        '{"instance_id": 12345, "profile": "test", "local_port": 18080, "dph": 0.5, "ctx_size": 32768, "ssh_url": "ssh://root@10.0.0.1:2222", "run_dir": "' + str(project_dir) + '/run"}'
    )

    with mock.patch("hostai.commands.status.ssh.run_remote", return_value=mock.Mock(returncode=0, stdout="log line\n")):
        runner = CliRunner()
        result = runner.invoke(cmd_status, ["--logs"], obj=config)

    assert result.exit_code == 0, result.output
    assert (project_dir / "run" / "server-live.log").exists()


def test_status_follow_requires_ssh_binary(config, project_dir):
    state_file = project_dir / ".hostai-vast" / "state.json"
    state_file.parent.mkdir(parents=True)
    state_file.write_text(
        '{"instance_id": 12345, "profile": "test", "local_port": 18080, "dph": 0.5, "ctx_size": 32768, "ssh_url": "ssh://root@10.0.0.1:2222"}'
    )

    with mock.patch("hostai.commands.status.shutil.which", return_value=None):
        runner = CliRunner()
        result = runner.invoke(cmd_status, ["--logs", "--follow"], obj=config)

    assert result.exit_code != 0
    assert "ssh binary" in result.output.lower()


def test_status_gpu_snapshot_printed(config, project_dir):
    state_file = project_dir / ".hostai-vast" / "state.json"
    state_file.parent.mkdir(parents=True)
    state_file.write_text(
        '{"instance_id": 12345, "profile": "test", "local_port": 18080, "dph": 0.5, "ctx_size": 32768, "ssh_url": "ssh://root@10.0.0.1:2222"}'
    )

    with (
        mock.patch(
            "hostai.commands.status._provider",
            return_value=mock.Mock(get_instance=mock.Mock(return_value={"actual_status": "running"})),
        ),
        mock.patch("hostai.commands.status.ssh.is_tunnel_healthy", return_value=False),
        mock.patch("hostai.commands.status.ssh.run_remote", return_value=mock.Mock(returncode=0, stdout="GPU, 0 %, 0 MiB, 24 MiB")),
    ):
        runner = CliRunner()
        result = runner.invoke(cmd_status, [], obj=config)

    assert result.exit_code == 0, result.output
    assert "[gpu]" in result.output
