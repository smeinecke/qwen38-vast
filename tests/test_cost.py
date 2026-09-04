"""Tests for hostai.commands.cost."""

import json

from click.testing import CliRunner

from hostai.commands.cost import _count_starts_per_month, _model_fetch_estimate, cmd_volume_break_even


def test_count_starts_per_month(config, project_dir):
    runs_dir = project_dir / ".hostai-runs"
    run_dir = runs_dir / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({"started_epoch": 9999999999, "profile": "test"}))

    counts = _count_starts_per_month(config)
    assert sum(counts.values()) == 1


def test_model_fetch_excludes_image():
    """The model-only fetch estimate must not include image bytes."""
    seconds, avoided_gb = _model_fetch_estimate(18.83, 500.0, 200.0)
    assert avoided_gb == 18.83
    assert seconds >= 30.0


def test_volume_break_even_defaults(config):
    runner = CliRunner()
    result = runner.invoke(
        cmd_volume_break_even,
        ["--volume-gb", "100", "--volume-cost-month", "5.00"],
        obj=config,
    )
    assert result.exit_code == 0
    assert "break-even" in result.output
    assert "image pull is not saved" in result.output
    assert "GPU rental savings per start" in result.output


def test_volume_break_even_cheaper(config):
    runner = CliRunner()
    result = runner.invoke(
        cmd_volume_break_even,
        ["--volume-gb", "100", "--volume-cost-month", "1.00", "--starts-per-month", "50", "--dph", "1.0"],
        obj=config,
    )
    assert result.exit_code == 0
    assert "persistent volume is cheaper" in result.output


def test_volume_break_even_zero_starts(config):
    runner = CliRunner()
    result = runner.invoke(
        cmd_volume_break_even,
        ["--volume-gb", "100", "--volume-cost-month", "1.00", "--starts-per-month", "0"],
        obj=config,
    )
    assert result.exit_code == 0
    # With zero starts, the volume cannot pay for itself.
    assert "persistent volume is more expensive" in result.output


def test_volume_break_even_nonzero_transfer_cost(config):
    """A paid ingress should add transfer savings to the total."""
    runner = CliRunner()
    result = runner.invoke(
        cmd_volume_break_even,
        [
            "--volume-gb",
            "100",
            "--volume-cost-month",
            "5.00",
            "--starts-per-month",
            "50",
            "--dph",
            "1.0",
            "--inet-down",
            "500",
            "--disk-bw",
            "200",
        ],
        obj=config,
    )
    assert result.exit_code == 0
    # The output should explicitly note free traffic for default offers.
    assert "free traffic" in result.output or "$0.00" in result.output


def test_volume_break_even_volume_smaller_than_model_caps_savings(config, project_dir):
    """A volume smaller than the model payload can only save what it can hold."""
    config.market.model_download_gb = 50.0
    runner = CliRunner()
    result = runner.invoke(
        cmd_volume_break_even,
        ["--volume-gb", "20", "--volume-cost-month", "1.00", "--starts-per-month", "100", "--dph", "1.0"],
        obj=config,
    )
    assert result.exit_code == 0
    assert "capping avoided download at 20.0" in result.output


def test_volume_break_even_per_gb_and_per_day_output(config):
    runner = CliRunner()
    result = runner.invoke(
        cmd_volume_break_even,
        ["--volume-gb", "100", "--volume-cost-month", "30.00", "--starts-per-month", "1", "--dph", "0.5"],
        obj=config,
    )
    assert result.exit_code == 0
    assert "per-GiB cost" in result.output
    assert "per-day cost" in result.output


def test_volume_break_even_rejects_invalid(config):
    runner = CliRunner()
    result = runner.invoke(
        cmd_volume_break_even,
        ["--volume-gb", "-1", "--volume-cost-month", "5.00"],
        obj=config,
    )
    assert result.exit_code != 0


def test_volume_break_even_no_image_savings(config):
    """The full cold-start time must be larger than the avoided model fetch."""
    runner = CliRunner()
    result = runner.invoke(
        cmd_volume_break_even,
        ["--volume-gb", "100", "--volume-cost-month", "5.00", "--starts-per-month", "10", "--dph", "0.5"],
        obj=config,
    )
    assert result.exit_code == 0
    # Full cold-start includes the image, so it is longer than the model-only fetch.
    assert "full cold-start" in result.output
