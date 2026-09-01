"""Tests for the hostai down CLI with mocked dependencies."""

from pathlib import Path
from unittest import mock

import click
from click.testing import CliRunner

from hostai.commands.down import cmd_down


def test_down_no_state(config):
    runner = CliRunner()
    result = runner.invoke(cmd_down, ["--yes"], obj=config)
    assert result.exit_code == 0
    assert "No local hostai Vast state found" in result.output


def test_down_no_instance(config, project_dir):
    state_file = project_dir / ".hostai-vast" / "state.json"
    state_file.parent.mkdir(parents=True)
    state_file.write_text("{}")

    runner = CliRunner()
    result = runner.invoke(cmd_down, ["--yes"], obj=config)
    assert result.exit_code == 0
    assert "No Vast instance id" in result.output


def test_down_happy_path(config, project_dir):
    from hostai.state import State

    state = State(
        project_dir / ".hostai-vast" / "state.json",
        {
            "instance_id": 12345,
            "profile": "test",
            "model": "model.gguf",
            "ctx_size": 32768,
            "dph": 0.5,
            "ssh_url": "ssh://root@10.0.0.1:2222",
            "run_dir": str(project_dir / ".hostai-runs" / "run-1"),
        },
    )
    state.save()
    Path(state.run_dir).mkdir(parents=True)

    with mock.patch("hostai.commands.down.State.load", return_value=state):
        with mock.patch("hostai.commands.down._refresh_ssh_state"):
            with mock.patch("hostai.commands.down.ssh.ensure_tunnel"):
                with mock.patch("hostai.commands.down._save_and_upload_slot_cache", return_value=True):
                    with mock.patch("hostai.commands.down._archive_session"):
                        with mock.patch("hostai.commands.down._stop_remote_model"):
                            with mock.patch("hostai.commands.down.ssh.stop_tunnel"):
                                with mock.patch("hostai.commands.down._pause_or_destroy", return_value="destroyed. Session duration: 10s"):
                                    runner = CliRunner()
                                    result = runner.invoke(cmd_down, ["--yes", "--no-archive", "--no-cache"], obj=config)

    assert result.exit_code == 0, result.output
    assert "destroyed" in result.output
