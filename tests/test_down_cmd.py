"""Tests for the hostai down CLI with mocked dependencies."""

import json
from pathlib import Path
from unittest import mock

import click
import pytest
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
                with mock.patch(
                    "hostai.cache.save_and_upload_slot_cache",
                    return_value={"save_ms": 50, "n_written": 1024, "uploaded": True, "upload_duration_s": 1.0},
                ):
                    with mock.patch("hostai.commands.down._archive_session"):
                        with mock.patch("hostai.commands.down._stop_remote_model"):
                            with mock.patch("hostai.commands.down.ssh.stop_tunnel"):
                                with mock.patch(
                                    "hostai.commands.down._pause_or_destroy",
                                    return_value="destroyed. Session duration: 10s",
                                ):
                                    runner = CliRunner()
                                    result = runner.invoke(
                                        cmd_down, ["--yes", "--no-archive", "--no-cache"], obj=config
                                    )

    assert result.exit_code == 0, result.output
    assert "destroyed" in result.output


def test_down_destroy_removes_state_and_writes_shutdown_tail(config, project_dir):
    from hostai.commands.down import down_instance
    from hostai.state import State

    state = State(
        project_dir / ".hostai-vast" / "state.json",
        {
            "instance_id": 12345,
            "profile": "test",
            "ctx_size": 32768,
            "dph": 0.5,
            "ssh_url": "ssh://root@10.0.0.1:2222",
            "run_dir": str(project_dir / ".hostai-runs" / "run-1"),
        },
    )
    state.save()
    run_dir = Path(state.run_dir)
    run_dir.mkdir(parents=True)

    def fake_pause_or_destroy(cfg, st, pause, rd):
        st.status = "paused" if pause else "destroyed"
        st.save_metadata(rd, status=st.status)
        if not pause:
            try:
                st.state_file.unlink()
            except FileNotFoundError:
                pass
        return "paused" if pause else "destroyed. Session duration: 10s"

    with mock.patch("hostai.commands.down._refresh_ssh_state"):
        with mock.patch("hostai.commands.down.ssh.ensure_tunnel"):
            with mock.patch(
                "hostai.cache.save_and_upload_slot_cache",
                return_value={"save_ms": 50, "n_written": 1024, "uploaded": True, "upload_duration_s": 1.0},
            ):
                with mock.patch("hostai.commands.down._archive_session"):
                    with mock.patch("hostai.commands.down._stop_remote_model"):
                        with mock.patch("hostai.commands.down.ssh.stop_tunnel"):
                            with mock.patch(
                                "hostai.commands.down._pause_or_destroy", side_effect=fake_pause_or_destroy
                            ):
                                down_instance(config, state, no_archive=True, no_cache=True, skip_confirm=True)

    assert not state.state_file.exists(), "destroyed state must stay deleted"
    assert (run_dir / "shutdown-tail.json").exists()
    assert (run_dir / "metadata.json").exists()
    assert json.loads((run_dir / "shutdown-tail.json").read_text())["reason"] == "manual"


def test_down_pause_keeps_state(config, project_dir):
    from hostai.commands.down import down_instance
    from hostai.state import State

    state = State(
        project_dir / ".hostai-vast" / "state.json",
        {
            "instance_id": 12345,
            "profile": "test",
            "ctx_size": 32768,
            "dph": 0.5,
            "ssh_url": "ssh://root@10.0.0.1:2222",
            "run_dir": str(project_dir / ".hostai-runs" / "run-1"),
        },
    )
    state.save()
    run_dir = Path(state.run_dir)
    run_dir.mkdir(parents=True)

    def fake_pause_or_destroy(cfg, st, pause, rd):
        st.status = "paused"
        st.save_metadata(rd, status="paused")
        st.save()
        return "paused. Session duration: 10s"

    with mock.patch("hostai.commands.down._refresh_ssh_state"):
        with mock.patch("hostai.commands.down.ssh.ensure_tunnel"):
            with mock.patch("hostai.cache.save_and_upload_slot_cache", return_value=None):
                with mock.patch("hostai.commands.down._archive_session"):
                    with mock.patch("hostai.commands.down._stop_remote_model"):
                        with mock.patch("hostai.commands.down.ssh.stop_tunnel"):
                            with mock.patch(
                                "hostai.commands.down._pause_or_destroy", side_effect=fake_pause_or_destroy
                            ):
                                down_instance(
                                    config, state, pause=True, no_archive=True, no_cache=True, skip_confirm=True
                                )

    assert state.state_file.exists(), "paused state must be preserved"
    assert (run_dir / "shutdown-tail.json").exists()


def _fake_provider_with_instance(config):
    from hostai.providers.fake import FakeProvider

    provider = FakeProvider(config)
    created = provider.create_instance(90001, image="img", disk=10, env={})
    iid = created["instance_id"]
    provider.start_instance(iid)
    return provider, iid


def test_down_by_id_untracked_destroys(config):
    """`down --id` destroys a provider instance that has no local state."""
    provider, iid = _fake_provider_with_instance(config)
    with mock.patch("hostai.commands.down._provider", return_value=provider):
        result = CliRunner().invoke(cmd_down, ["--id", str(iid), "--yes"], obj=config)
    assert result.exit_code == 0, result.output
    assert provider.get_instance(iid) is None
    assert "untracked" in result.output


def test_down_by_id_pause_stops_instance(config):
    provider, iid = _fake_provider_with_instance(config)
    with mock.patch("hostai.commands.down._provider", return_value=provider):
        result = CliRunner().invoke(cmd_down, ["--id", str(iid), "--yes", "--pause"], obj=config)
    assert result.exit_code == 0, result.output
    inst = provider.get_instance(iid)
    assert inst is not None
    assert inst["actual_status"] == "stopped"


def test_down_by_id_not_found(config):
    provider, _ = _fake_provider_with_instance(config)
    provider.destroy_instance(provider.list_instances()[0]["id"])
    with mock.patch("hostai.commands.down._provider", return_value=provider):
        result = CliRunner().invoke(cmd_down, ["--id", "424242", "--yes"], obj=config)
    assert result.exit_code == 0, result.output
    assert "does not exist" in result.output


def test_down_by_id_cancelled(config):
    provider, iid = _fake_provider_with_instance(config)
    with (
        mock.patch("hostai.commands.down._provider", return_value=provider),
        mock.patch("click.confirm", return_value=False),
    ):
        result = CliRunner().invoke(cmd_down, ["--id", str(iid)], obj=config)
    assert result.exit_code == 0, result.output
    assert "Cancelled." in result.output
    assert provider.get_instance(iid) is not None


def test_down_by_id_multiple_untracked(config):
    from hostai.providers.fake import FakeProvider

    provider = FakeProvider(config)
    ids = [provider.create_instance(90001, image="img", disk=10, env={})["instance_id"] for _ in range(2)]
    args = ["--yes"] + [arg for iid in ids for arg in ("--id", str(iid))]
    with mock.patch("hostai.commands.down._provider", return_value=provider):
        result = CliRunner().invoke(cmd_down, args, obj=config)
    assert result.exit_code == 0, result.output
    assert provider.list_instances() == []


def test_down_by_id_tracked_routes_to_full_path(config, project_dir):
    """An --id matching local state runs down_instance, not the bare destroy."""
    from hostai.state import State

    State(
        project_dir / ".hostai-vast" / "state.json",
        {"instance_id": 555, "profile": "test", "dph": 0.5, "ctx_size": 32768},
    ).save()

    with (
        mock.patch("hostai.commands.down.down_instance", return_value="destroyed") as di,
        mock.patch("hostai.commands.down.down_remote_instance") as dri,
        mock.patch("hostai.commands.watchdog.stop_watchdog"),
        mock.patch("hostai.commands.monitor.stop_monitor"),
    ):
        result = CliRunner().invoke(cmd_down, ["--id", "555", "--yes"], obj=config)

    assert result.exit_code == 0, result.output
    di.assert_called_once()
    dri.assert_not_called()


def test_down_by_id_mixed_tracked_and_untracked(config, project_dir):
    from hostai.state import State

    State(
        project_dir / ".hostai-vast" / "state.json",
        {"instance_id": 555, "profile": "test", "dph": 0.5, "ctx_size": 32768},
    ).save()
    provider, iid = _fake_provider_with_instance(config)

    with (
        mock.patch("hostai.commands.down.down_instance", return_value="destroyed") as di,
        mock.patch("hostai.commands.down._provider", return_value=provider),
        mock.patch("hostai.commands.watchdog.stop_watchdog"),
        mock.patch("hostai.commands.monitor.stop_monitor"),
    ):
        result = CliRunner().invoke(cmd_down, ["--id", "555", "--id", str(iid), "--yes"], obj=config)

    assert result.exit_code == 0, result.output
    di.assert_called_once()
    assert provider.get_instance(iid) is None


def test_down_by_id_conflicts_with_all(config):
    result = CliRunner().invoke(cmd_down, ["--id", "1", "--all"], obj=config)
    assert result.exit_code != 0
    assert "cannot combine --id" in result.output


def test_down_by_id_conflicts_with_name(config):
    result = CliRunner().invoke(cmd_down, ["--id", "1", "--name", "foo"], obj=config)
    assert result.exit_code != 0
    assert "cannot combine --id" in result.output


def test_down_failed_destroy_preserves_state(config, project_dir):
    from hostai.commands.down import down_instance
    from hostai.state import State

    state = State(
        project_dir / ".hostai-vast" / "state.json",
        {
            "instance_id": 12345,
            "profile": "test",
            "ctx_size": 32768,
            "dph": 0.5,
            "ssh_url": "ssh://root@10.0.0.1:2222",
            "run_dir": str(project_dir / ".hostai-runs" / "run-1"),
        },
    )
    state.save()
    run_dir = Path(state.run_dir)
    run_dir.mkdir(parents=True)

    with mock.patch("hostai.commands.down._refresh_ssh_state"):
        with mock.patch("hostai.commands.down.ssh.ensure_tunnel"):
            with mock.patch("hostai.cache.save_and_upload_slot_cache", return_value=None):
                with mock.patch("hostai.commands.down._archive_session"):
                    with mock.patch("hostai.commands.down._stop_remote_model"):
                        with mock.patch("hostai.commands.down.ssh.stop_tunnel"):
                            with mock.patch(
                                "hostai.commands.down._pause_or_destroy",
                                side_effect=click.ClickException("Vast destroy failed (timeout)"),
                            ):
                                with pytest.raises(click.ClickException):
                                    down_instance(config, state, no_archive=True, no_cache=True, skip_confirm=True)

    assert state.state_file.exists(), "failed destroy must keep state for recovery"
