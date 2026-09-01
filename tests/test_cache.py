"""Tests for hostai.cache helpers with mocked SSH/subprocess calls."""

import json
from pathlib import Path
from unittest import mock

import pytest

from hostai import cache
from hostai.cache import (
    ensure_cache_key,
    install_cache_key_on_vast,
    remote_cache_dir,
    rclone_prefetch_script,
    rclone_remote_name,
    rclone_upload_script,
    upload_cache,
    validate_cache,
    validate_cache_config,
)


def fake_completed(returncode=0, stdout="", stderr=""):
    return mock.Mock(returncode=returncode, stdout=stdout, stderr=stderr)


def test_cache_signature_deterministic(config):
    sig1 = cache.cache_signature("abc123", "model.gguf", "rev", 32768, 1, "q4_0", "q4_0")
    sig2 = cache.cache_signature("abc123", "model.gguf", "rev", 32768, 1, "q4_0", "q4_0")
    assert sig1 == sig2
    assert len(sig1) == 20


def test_remote_cache_dir(config):
    path = remote_cache_dir(config, "sig123")
    assert path == "qwen-slot-cache/default/sig123"


def test_ensure_cache_key_generates_when_missing(config, project_dir, monkeypatch):
    key_path = project_dir / "cache_key"
    monkeypatch.setattr(cache, "_cache_key_path", lambda root: key_path)

    def fake_run(cmd, **kwargs):
        # Simulate ssh-keygen creating the key pair.
        key_path.write_text("private-key")
        key_path.with_suffix(".pub").write_text("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI test\n")
        return fake_completed()

    with mock.patch("hostai.cache.utils.run", side_effect=fake_run) as run:
        returned = ensure_cache_key(config, project_dir)

    assert returned == key_path
    assert run.call_count >= 1
    # First call should be ssh-keygen.
    assert run.call_args_list[0].args[0][0] == "ssh-keygen"


def test_ensure_cache_key_reuses_existing_key(config, project_dir, monkeypatch):
    key_path = project_dir / "cache_key"
    pub_path = key_path.with_suffix(".pub")
    key_path.write_text("private")
    pub_path.write_text("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDIhz2GK/XCuP5gP1qr0ZzB/5rO test\n")
    monkeypatch.setattr(cache, "_cache_key_path", lambda root: key_path)

    with mock.patch("hostai.cache.utils.run") as run:
        key = ensure_cache_key(config, project_dir)

    assert key == key_path
    run.assert_not_called()


def test_upload_cache_creates_remote_dir_and_uploads(config, state, project_dir, tmp_path):
    local_dir = tmp_path / "slot"
    local_dir.mkdir()
    (local_dir / "current.bin").write_bytes(b"bin")
    (local_dir / "current.json").write_text(json.dumps({"schema_version": 1}))

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "ssh" and "mkdir" in " ".join(cmd):
            return fake_completed()
        if cmd[0] == "rsync":
            return fake_completed()
        if cmd[0] == "ssh" and "chmod 600" in " ".join(cmd):
            return fake_completed()
        return fake_completed()

    with mock.patch("hostai.cache.utils.run", side_effect=fake_run):
        with mock.patch("hostai.cache.ensure_cache_key", return_value=project_dir / "key"):
            ok = upload_cache(config, state, local_dir, llama_commit="abc123")

    assert ok is True
    assert any("mkdir -p" in " ".join(c) for c in calls)
    rsync_calls = [c for c in calls if c[0] == "rsync"]
    assert len(rsync_calls) == 2
    joined = " ".join(str(x) for call in rsync_calls for x in call)
    assert "current.bin" in joined
    assert "current.json" in joined
    assert any("chmod 600" in " ".join(c) for c in calls)


def test_upload_cache_retries_on_rsync_failure(config, state, project_dir, tmp_path):
    local_dir = tmp_path / "slot"
    local_dir.mkdir()
    (local_dir / "current.bin").write_bytes(b"bin")

    attempts = []

    def fake_run(cmd, **kwargs):
        if cmd[0] == "rsync":
            attempts.append(cmd)
            if len(attempts) < 2:
                return fake_completed(returncode=23)
            return fake_completed()
        return fake_completed()

    with mock.patch("hostai.cache.utils.run", side_effect=fake_run):
        with mock.patch("hostai.cache.ensure_cache_key", return_value=project_dir / "key"):
            ok = upload_cache(config, state, local_dir, llama_commit="abc123")

    assert ok is True
    assert len(attempts) == 2


def test_upload_cache_returns_false_when_no_files(config, state, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert upload_cache(config, state, empty_dir) is False


def test_validate_cache(tmp_path):
    local_dir = tmp_path / "cache"
    local_dir.mkdir()
    (local_dir / "current.bin").write_bytes(b"x")
    (local_dir / "current.json").write_text(json.dumps({"schema_version": 1}))
    assert validate_cache(local_dir) is True


def test_validate_cache_missing_json(tmp_path):
    # Metadata is optional; only current.bin is required.
    local_dir = tmp_path / "cache"
    local_dir.mkdir()
    (local_dir / "current.bin").write_bytes(b"x")
    assert validate_cache(local_dir) is True


def test_install_cache_key_on_vast(running_state, config, project_dir):
    key_path = project_dir / "key"
    key_path.write_text("private")
    with mock.patch("hostai.cache.ssh.run_remote", return_value=fake_completed()) as run:
        with mock.patch("hostai.cache.ssh.scp_to", return_value=fake_completed()) as scp:
            with mock.patch("hostai.cache.ensure_cache_key", return_value=key_path):
                ok = install_cache_key_on_vast(running_state, config)

    assert ok is True
    run.assert_called_once()
    scp.assert_called_once()


def test_validate_cache_config_ssh(config):
    assert validate_cache_config(config) is True


def test_validate_cache_config_rclone_missing_url(config):
    config.cache.rclone = True
    config.cache.rclone_url = ""
    config.cache.host = ""
    assert validate_cache_config(config) is False


def test_validate_cache_config_rclone_url(config):
    config.cache.rclone = True
    config.cache.rclone_url = "https://cache.example.com/"
    config.cache.rclone_password = "secret"
    assert validate_cache_config(config) is True


def test_rclone_remote_name_default(config):
    assert rclone_remote_name(config) == "hostai"


def test_rclone_remote_name_configured(config):
    config.cache.rclone_remote = "mywebdav"
    assert rclone_remote_name(config) == "mywebdav"


def test_rclone_prefetch_script_contains_rclone_copyto(config):
    config.cache.rclone = True
    config.cache.rclone_url = "https://cache.example.com/"
    config.cache.rclone_password = "secret"
    script = rclone_prefetch_script(config, "/var/lib/qwen38/slots", "qwen-slot-cache/default/sig")
    assert "rclone copyto" in script
    assert '"$remote_name:$remote_dir/current.bin"' in script
    assert "RCLONE_CONFIG_HOSTAI_URL" in script
    assert "pass_plain=" in script
    assert "rclone obscure" in script


def test_rclone_upload_script_contains_rclone_copyto(config):
    config.cache.rclone = True
    config.cache.rclone_url = "https://cache.example.com/"
    config.cache.rclone_password = "secret"
    script = rclone_upload_script(config, "/var/lib/qwen38/slots", "qwen-slot-cache/default/sig")
    assert "rclone copyto" in script
    assert '"$remote_name:$remote_dir/current.bin"' in script
    assert "RCLONE_CONFIG_HOSTAI_URL" in script
    assert "pass_plain=" in script
    assert "rclone obscure" in script
