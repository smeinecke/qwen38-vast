"""Tests for hostai.commands.cost."""

import json
from unittest import mock
from click.testing import CliRunner

from hostai.commands.cost import _count_starts_per_month, cmd_volume_break_even


def test_count_starts_per_month(config, project_dir):
    runs_dir = project_dir / ".hostai-runs"
    run_dir = runs_dir / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({"started_epoch": 9999999999, "profile": "test"}))

    counts = _count_starts_per_month(config)
    assert sum(counts.values()) == 1


def test_volume_break_even_defaults(config):
    runner = CliRunner()
    result = runner.invoke(
        cmd_volume_break_even,
        ["--volume-gb", "100", "--volume-cost-month", "5.00"],
        obj=config,
    )
    assert result.exit_code == 0
    assert "break-even starts/month" in result.output


def test_volume_break_even_cheaper(config):
    runner = CliRunner()
    result = runner.invoke(
        cmd_volume_break_even,
        ["--volume-gb", "100", "--volume-cost-month", "1.00", "--starts-per-month", "50", "--dph", "1.0"],
        obj=config,
    )
    assert result.exit_code == 0
    assert "persistent volume is cheaper" in result.output


def test_volume_break_even_rejects_invalid(config):
    runner = CliRunner()
    result = runner.invoke(
        cmd_volume_break_even,
        ["--volume-gb", "-1", "--volume-cost-month", "5.00"],
        obj=config,
    )
    assert result.exit_code != 0
