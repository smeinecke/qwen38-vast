"""Tests for the hostai info command."""

from unittest import mock

from click.testing import CliRunner

from hostai.commands.info import cmd_info
from hostai.state import State


def _provider(account=None, rows=None):
    provider = mock.Mock()
    provider.name = "vast"
    provider.get_account_info.return_value = account
    provider.list_instances.return_value = rows or []
    return provider


def _write_state(project_dir, **kwargs):
    state_file = project_dir / ".hostai-vast" / "state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    State(state_file, kwargs).save()


def test_info_renders_account_and_instances(config, project_dir):
    _write_state(project_dir, instance_id=555, profile="test")
    provider = _provider(
        account={"username": "me", "email": "me@example.com", "balance": "12.34", "credit": 1.5},
        rows=[
            {
                "id": 555,
                "actual_status": "running",
                "num_gpus": 1,
                "gpu_name": "RTX 5090",
                "dph_total": 0.45,
                "image_uuid": "img:v1",
                "ssh_host": "1.2.3.4",
                "ssh_port": 22001,
                "duration": 3723.0,
                "label": "hostai",
            },
            {
                "id": 777,
                "actual_status": "stopped",
                "num_gpus": 2,
                "gpu_name": "RTX 4090",
                "dph_total": 0.4,
                "label": "other",
            },
        ],
    )
    with mock.patch("hostai.commands.info.get_provider", return_value=provider):
        result = CliRunner().invoke(cmd_info, [], obj=config)

    assert result.exit_code == 0, result.output
    assert "$12.34" in result.output
    assert "me@example.com" in result.output
    assert "RTX 5090" in result.output
    assert "default" in result.output  # tracked instance name
    assert "01:02:03" in result.output  # age formatting
    assert "running burn" in result.output
    # id 777 has no local state -> the untracked hint must appear
    assert "down --id" in result.output


def test_info_no_instances(config):
    provider = _provider(account={"balance": "5.00"}, rows=[])
    with mock.patch("hostai.commands.info.get_provider", return_value=provider):
        result = CliRunner().invoke(cmd_info, [], obj=config)
    assert result.exit_code == 0, result.output
    assert "No instances on this account." in result.output


def test_info_no_billing_for_provider(config):
    provider = _provider(account=None, rows=[])
    with mock.patch("hostai.commands.info.get_provider", return_value=provider):
        result = CliRunner().invoke(cmd_info, [], obj=config)
    assert result.exit_code == 0, result.output
    assert "no billing information" in result.output


def test_info_account_lookup_failure_still_lists_instances(config):
    provider = mock.Mock()
    provider.name = "vast"
    provider.get_account_info.side_effect = RuntimeError("billing api down")
    provider.list_instances.return_value = [
        {"id": 1, "actual_status": "running", "gpu_name": "RTX 4090", "dph_total": 0.3}
    ]
    with mock.patch("hostai.commands.info.get_provider", return_value=provider):
        result = CliRunner().invoke(cmd_info, [], obj=config)
    assert result.exit_code == 0, result.output
    assert "lookup failed" in result.output
    assert "RTX 4090" in result.output


def test_info_instance_lookup_failure_is_fatal(config):
    provider = mock.Mock()
    provider.name = "vast"
    provider.get_account_info.return_value = {"balance": "1.00"}
    provider.list_instances.side_effect = RuntimeError("api down")
    with mock.patch("hostai.commands.info.get_provider", return_value=provider):
        result = CliRunner().invoke(cmd_info, [], obj=config)
    assert result.exit_code != 0
    assert "instance lookup failed" in result.output


def test_info_marks_only_tracked_names(config, project_dir):
    _write_state(project_dir, instance_id=111, profile="test")
    provider = _provider(
        account=None,
        rows=[
            {"id": 111, "actual_status": "running", "gpu_name": "A", "dph_total": 0.1},
            {"id": 222, "actual_status": "running", "gpu_name": "B", "dph_total": 0.2},
        ],
    )
    with mock.patch("hostai.commands.info.get_provider", return_value=provider):
        result = CliRunner().invoke(cmd_info, [], obj=config)
    assert result.exit_code == 0, result.output
    assert "1 instance(s) not tracked" in result.output


def test_info_all_tracked_no_hint(config, project_dir):
    _write_state(project_dir, instance_id=111, profile="test")
    provider = _provider(
        account=None,
        rows=[{"id": 111, "actual_status": "running", "gpu_name": "A", "dph_total": 0.1}],
    )
    with mock.patch("hostai.commands.info.get_provider", return_value=provider):
        result = CliRunner().invoke(cmd_info, [], obj=config)
    assert result.exit_code == 0, result.output
    assert "not tracked" not in result.output
    assert "down --id" not in result.output


def test_info_shows_machine_id(config, project_dir):
    _write_state(project_dir, instance_id=555, profile="test", machine_id=4242)
    provider = _provider(
        account=None,
        rows=[
            # Provider payload omits machine_id -> falls back to local state.
            {"id": 555, "actual_status": "running", "gpu_name": "A", "dph_total": 0.1},
            # Provider payload wins when it carries machine_id.
            {"id": 666, "actual_status": "running", "gpu_name": "B", "dph_total": 0.2, "machine_id": 7777},
        ],
    )
    with mock.patch("hostai.commands.info.get_provider", return_value=provider):
        result = CliRunner().invoke(cmd_info, [], obj=config, env={"COLUMNS": "160"})
    assert result.exit_code == 0, result.output
    assert "Machine" in result.output
    assert "4242" in result.output
    assert "7777" in result.output


def test_info_machine_id_missing_everywhere(config, project_dir):
    provider = _provider(
        account=None,
        rows=[{"id": 111, "actual_status": "running", "gpu_name": "A", "dph_total": 0.1}],
    )
    with mock.patch("hostai.commands.info.get_provider", return_value=provider):
        result = CliRunner().invoke(cmd_info, [], obj=config)
    assert result.exit_code == 0, result.output
    assert "Machine" in result.output
    assert "-" in result.output
