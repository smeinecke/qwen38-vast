"""Tests for hostai.validate and the validate command."""

from pathlib import Path
from unittest import mock

import pytest
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


def test_cmd_validate(config):
    with mock.patch("hostai.commands.validate_repo.validate_repo", return_value=[]) as validate:
        runner = CliRunner()
        result = runner.invoke(cmd_validate, [], obj=config)
    assert result.exit_code == 0
    assert "OK" in result.output
    validate.assert_called_once_with(config.root_dir)
