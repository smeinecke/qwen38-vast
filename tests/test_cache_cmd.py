"""Tests for hostai.commands.cache_cmd with mocked SSH/cache APIs."""

import json
from unittest import mock

from click.testing import CliRunner

from hostai.commands.cache_cmd import cmd_cache_copy, cmd_cache_setup


def test_cache_setup_rejects_invalid_target(config):
    runner = CliRunner()
    result = runner.invoke(cmd_cache_setup, ["not-a-valid-target"], obj=config)
    assert result.exit_code != 0
    assert "user@host" in result.output


def test_cache_setup_requires_host(config):
    config.cache.host = ""
    runner = CliRunner()
    result = runner.invoke(cmd_cache_setup, [], obj=config)
    assert result.exit_code != 0
    assert "cache.host is not configured" in result.output


def test_cache_setup_happy_path(config, project_dir):
    key_path = project_dir / "cache_key"
    key_path.write_text("private")
    (key_path.with_suffix(".pub")).write_text("ssh-ed25519 AAA test\n")

    with mock.patch("hostai.commands.cache_cmd.ensure_cache_key", return_value=key_path):
        with mock.patch("hostai.commands.cache_cmd.copy_cache_key", return_value=True):
            with mock.patch("hostai.commands.cache_cmd.preflight_remote", return_value=True):
                runner = CliRunner()
                result = runner.invoke(cmd_cache_setup, ["cache@cache.example.com"], obj=config)

    assert result.exit_code == 0, result.output
    assert "READY" in result.output
    assert "cache.example.com" in result.output


def test_cache_setup_key_install_fails(config, project_dir):
    key_path = project_dir / "cache_key"
    key_path.write_text("private")

    with mock.patch("hostai.commands.cache_cmd.ensure_cache_key", return_value=key_path):
        with mock.patch("hostai.commands.cache_cmd.copy_cache_key", return_value=False):
            runner = CliRunner()
            result = runner.invoke(cmd_cache_setup, [], obj=config)

    assert result.exit_code != 0
    assert "failed to install" in result.output


def test_cache_copy_no_instance(config, project_dir):
    from hostai.state import State

    state = State(project_dir / ".hostai-vast" / "state.json")
    with mock.patch("hostai.commands.cache_cmd.State.load", return_value=state):
        runner = CliRunner()
        result = runner.invoke(cmd_cache_copy, [], obj=config)

    assert result.exit_code != 0
    assert "no running instance" in result.output


def test_cache_copy_cache_disabled(config, running_state):
    running_state.slot_cache_enabled = False
    with mock.patch("hostai.commands.cache_cmd.State.load", return_value=running_state):
        runner = CliRunner()
        result = runner.invoke(cmd_cache_copy, [], obj=config)

    assert result.exit_code != 0
    assert "slot cache is disabled" in result.output


def test_cache_copy_happy_path(config, running_state, project_dir):
    run_dir = project_dir / ".hostai-runs" / "copy-1"
    run_dir.mkdir(parents=True)
    (run_dir / "cache-save.json").write_text(json.dumps({"n_written": 1024}))

    class FakeClient:
        def health(self):
            return True

    def fake_save(cfg, st, rd, **kwargs):
        # Simulate _save_and_upload_slot_cache writing the save file.
        (rd / "cache-save.json").write_text(json.dumps({"n_written": 1024}))
        return True

    key_path = project_dir / "cache_key"
    key_path.write_text("private")

    with mock.patch("hostai.commands.cache_cmd.State.load", return_value=running_state):
        with mock.patch("hostai.commands.cache_cmd.ssh.is_tunnel_healthy", return_value=True):
            with mock.patch("hostai.commands.cache_cmd.LlamaClient", return_value=FakeClient()):
                with mock.patch("hostai.commands.cache_cmd.down._save_and_upload_slot_cache", side_effect=fake_save):
                    with mock.patch("hostai.commands.cache_cmd.ensure_cache_key", return_value=key_path):
                        with mock.patch("hostai.commands.cache_cmd.utils.run", return_value=mock.Mock(returncode=0, stdout="1024")):
                            runner = CliRunner()
                            result = runner.invoke(cmd_cache_copy, ["--slot", "1"], obj=config)

    assert result.exit_code == 0, result.output
    assert "persisted" in result.output
    assert "1024 bytes" in result.output
