"""Tests for hostai.validate and the validate command."""

from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from hostai.commands.validate_repo import cmd_validate
from hostai.validate import validate_repo


def test_validate_repo_real_root():
    """The actual repository should pass validation."""
    root = Path(__file__).parent.parent
    errors = validate_repo(root)
    assert errors == [], errors


def test_validate_repo_missing_files(tmp_path):
    errors = validate_repo(tmp_path)
    assert any("missing required file" in e for e in errors)


def test_validate_repo_invalid_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "wrong"\n')
    errors = validate_repo(tmp_path)
    assert any("missing project name 'hostai'" in e for e in errors)


def test_validate_repo_bad_toml_example(tmp_path):
    (tmp_path / "hostai.toml.example").write_text("[not valid toml")
    errors = validate_repo(tmp_path)
    assert any("not valid TOML" in e for e in errors)


def test_validate_repo_detects_disk_contradiction(config, tmp_path):
    """A profile gpu_query with disk_space lower than resolved disk_gb is an error."""
    import json
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "hostai"\n')
    (tmp_path / "hostai.toml").write_text("")
    profiles = {
        "schema_version": 1,
        "default_profile": "test",
        "images": [{"name": "a", "cuda_arch": "89", "image_tag": "a"}],
        "profiles": [{
            "name": "test",
            "image": "a",
            "ctx_size": 32768,
            "gpu_query": "disk_space>=10 num_gpus=1",
            "disk_gb": 35,
        }],
        "monitor_hardware": {"policy": "same_or_better", "gpu_ranks": []},
        "market_policy": {"require_free_traffic": False},
    }
    (tmp_path / "profiles.json").write_text(json.dumps(profiles))
    (tmp_path / "hostai.toml.example").write_text("")
    (tmp_path / "Dockerfile").write_text("libgomp1 llama-server CCACHE_SEED\n")
    (tmp_path / "start.sh").write_text("")
    (tmp_path / "entrypoint.sh").write_text("")
    (tmp_path / "hostai-init-ssh.sh").write_text("")
    errors = validate_repo(tmp_path, config)
    assert any("disk_space>=10" in e and "lower" in e for e in errors)


def test_cmd_validate(config):
    with mock.patch("hostai.commands.validate_repo.validate_repo", return_value=[]) as validate:
        runner = CliRunner()
        result = runner.invoke(cmd_validate, [], obj=config)
    assert result.exit_code == 0
    assert "OK" in result.output
    validate.assert_called_once_with(config.root_dir, config)
