"""Regression tests for the validation/gating machinery."""

import json

import pytest
from click.testing import CliRunner

from hostai import utils
from hostai.commands.up import _check_production_validation
from hostai.commands.validate_repo import cmd_validate
from hostai.validate import (
    ValidationRecord,
    compare_validations,
    load_last_validation,
    record_validation,
    validation_digest,
)


def _make_record(**overrides):
    defaults = {
        "timestamp": "2026-09-03T00:00:00Z",
        "result": "ok",
        "duration_seconds": 1.0,
        "git_commit": "abc1234def5678",
        "dirty": False,
        "image": "hostai-test:latest",
        "image_id": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
        "image_digest": "",
        "profile_hash": "aabbccddeeff0011",
        "errors": [],
        "level": "production",
        "checks_run": ["repo", "docker", "image-exists", "integration-tests"],
    }
    defaults.update(overrides)
    return ValidationRecord(**defaults)


def test_failed_production_validation_does_not_satisfy_gate(tmp_path, config, monkeypatch):
    """A failed production validation must not become the "last success" record."""
    config.root_dir = tmp_path
    record_validation(
        tmp_path,
        result="failed",
        duration=1.0,
        errors=["tests failed"],
        image="hostai-test:latest",
        level="production",
        checks_run=["repo", "docker", "image-exists", "integration-tests"],
    )

    # There should be no last-success record, and success=True must not fall back
    # to the failed validation.json.
    assert load_last_validation(tmp_path, success=True) is None


def test_load_last_success_ignores_failed_record(tmp_path):
    """load_last_validation(success=True) rejects a corrupted success file."""
    last_success = tmp_path / ".hostai-vast" / "validation-last-success.json"
    last_success.parent.mkdir(parents=True, exist_ok=True)
    last_success.write_text(json.dumps(_make_record(result="failed").to_dict()))

    assert load_last_validation(tmp_path, success=True) is None


def test_validation_digest_includes_level_and_checks(tmp_path):
    """A repo-level validation cannot impersonate a production validation."""
    production = _make_record(level="production", checks_run=["integration-tests"])
    repo = _make_record(level="repo", checks_run=["repo"])

    assert validation_digest(production) != validation_digest(repo)


def test_compare_detects_image_id_drift():
    """Rebuilding the integration image changes image_id and compare must flag it."""
    previous = _make_record(image_id="sha256:old")
    current = _make_record(image_id="sha256:new")
    diffs = compare_validations(current, previous)
    assert any("image ID" in d for d in diffs)


def test_compare_detects_profile_hash_drift():
    previous = _make_record(profile_hash="oldhash")
    current = _make_record(profile_hash="newhash")
    diffs = compare_validations(current, previous)
    assert any("profiles.json" in d for d in diffs)


def test_compare_detects_git_commit_drift():
    previous = _make_record(git_commit="abc123")
    current = _make_record(git_commit="def456")
    diffs = compare_validations(current, previous)
    assert any("git commit" in d for d in diffs)


def test_check_production_validation_rejects_missing_record(config, monkeypatch):
    """With the gate enabled and no success record, up is blocked."""
    config.vast.require_production_validation = True
    monkeypatch.setattr("hostai.commands.up._provider", lambda c: None)

    with pytest.raises(Exception) as exc_info:
        _check_production_validation(config, allow_unvalidated=False)
    assert "no successful production validation" in str(exc_info.value).lower()


def test_check_production_validation_skips_with_allow_unvalidated(config, monkeypatch):
    """--allow-unvalidated must bypass the gate even with no records."""
    config.vast.require_production_validation = True

    # Should return without raising.
    previous = _check_production_validation(config, allow_unvalidated=True)
    assert previous is None


def test_check_production_validation_rejects_failed_record(config, tmp_path, monkeypatch):
    """A failed record stored as last-success must still block."""
    config.root_dir = tmp_path
    config.vast.require_production_validation = True

    record = _make_record(result="failed")
    record_path = tmp_path / ".hostai-vast" / "validation-last-success.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record.to_dict()))

    with pytest.raises(Exception) as exc_info:
        _check_production_validation(config, allow_unvalidated=False)
    assert "no successful" in str(exc_info.value).lower()


def test_cmd_validate_production_rejects_dirty_tree(config, tmp_path, monkeypatch):
    """Production validation must refuse a dirty working tree."""
    config.root_dir = tmp_path
    monkeypatch.setattr(utils, "is_dirty_tree", lambda root: True)
    monkeypatch.setattr(utils, "git_commit", lambda root: "abc1234")
    runner = CliRunner()
    result = runner.invoke(cmd_validate, ["--production"], obj=config)
    assert result.exit_code != 0
    assert "dirty" in result.output.lower()


def test_cmd_validate_production_does_not_write_success_on_failure(config, tmp_path, monkeypatch):
    """A failed production run must not update validation-last-success.json."""
    config.root_dir = tmp_path
    monkeypatch.setattr(utils, "is_dirty_tree", lambda root: True)
    monkeypatch.setattr(utils, "git_commit", lambda root: "abc1234")
    runner = CliRunner()
    runner.invoke(cmd_validate, ["--production"], obj=config)

    success_path = tmp_path / ".hostai-vast" / "validation-last-success.json"
    assert not success_path.exists()


def test_cmd_validate_repo_lightweight_is_not_production(config, tmp_path, monkeypatch):
    """A repo-only validation must be level=repo and not write a success record."""
    config.root_dir = tmp_path
    monkeypatch.setattr("hostai.commands.validate_repo.validate_repo", lambda root, cfg=None: [])
    monkeypatch.setattr(utils, "git_commit", lambda root: "abc1234")
    monkeypatch.setattr(utils, "is_dirty_tree", lambda root: False)
    monkeypatch.setattr("hostai.validate._image_info", lambda image: ("sha256:111111111111", ""))

    runner = CliRunner()
    result = runner.invoke(cmd_validate, [], obj=config)
    assert result.exit_code == 0
    assert "level=repo" in result.output
    assert not (tmp_path / ".hostai-vast" / "validation-last-success.json").exists()


def test_record_validation_overwrites_current_but_not_success_on_repo_only(tmp_path):
    """Repo validation writes validation.json but must not touch the success file."""
    success = tmp_path / ".hostai-vast" / "validation-last-success.json"
    success.parent.mkdir(parents=True, exist_ok=True)
    success.write_text(json.dumps(_make_record().to_dict()))

    record_validation(
        tmp_path,
        result="ok",
        duration=1.0,
        errors=[],
        image="hostai-test:latest",
        level="repo",
        checks_run=["repo"],
    )

    current = load_last_validation(tmp_path, success=False)
    assert current is not None
    assert current.level == "repo"
    last_success = load_last_validation(tmp_path, success=True)
    assert last_success is not None
    assert last_success.level == "production"
