"""Tests for hostai.cache helpers with mocked SSH/subprocess calls."""

import json
from unittest import mock

from hostai import cache
from hostai.cache import (
    ensure_cache_key,
    install_cache_key_on_vast,
    rclone_prefetch_script,
    rclone_remote_name,
    rclone_upload_script,
    remote_cache_dir,
    rsync_prefetch_script,
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


def _bash_n(script: str) -> None:
    """Assert that *script* is syntactically valid bash."""
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "script.sh"
        path.write_text(script)
        subprocess.run(["bash", "-n", str(path)], check=True, text=True)


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


def test_rclone_prefetch_script_is_valid_bash(config):
    config.cache.rclone = True
    config.cache.rclone_url = "https://cache.example.com/"
    script = rclone_prefetch_script(config, "/var/lib/qwen38/slots", "qwen-slot-cache/default/sig")
    _bash_n(script)


def test_rclone_prefetch_script_has_speed_monitoring(config):
    config.cache.rclone = True
    config.cache.rclone_url = "https://cache.example.com/"
    script = rclone_prefetch_script(config, "/var/lib/qwen38/slots", "qwen-slot-cache/default/sig")
    assert "209715200" in script
    assert "total_bytes" in script
    assert "rclone size" in script
    assert "rclone --help" in script
    assert "final_size" in script
    assert "too_slow" in script
    assert "trap cleanup EXIT" in script


def test_rsync_prefetch_script_contains_rsync(config):
    script = rsync_prefetch_script(config, "/var/lib/qwen38/slots", "qwen-slot-cache/default/sig")
    assert "rsync -av" in script
    assert ".current.bin.part" in script
    assert "ssh -n" in script


def test_rsync_prefetch_script_is_valid_bash(config):
    script = rsync_prefetch_script(config, "/var/lib/qwen38/slots", "qwen-slot-cache/default/sig")
    _bash_n(script)


def test_rsync_prefetch_script_has_speed_monitoring(config):
    script = rsync_prefetch_script(config, "/var/lib/qwen38/slots", "qwen-slot-cache/default/sig")
    assert "209715200" in script
    assert "total_bytes" in script
    assert "stat -c %s" in script
    assert "final_size" in script
    assert "too_slow" in script
    assert "trap cleanup EXIT" in script


def _run_rclone_prefetch(
    config,
    tmp_path,
    mode,
    total_bytes,
    remote_dir="qwen-slot-cache/default/sig",
    json=True,
):
    """Run the rclone prefetch script with a fake rclone and a no-op sleep.

    This makes the 20 second monitoring loop execute quickly so slow/ETA
    cases can be tested without waiting 20 real seconds.
    """
    import os
    import subprocess
    import sys
    import textwrap

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    slot_dir = tmp_path / "slot"
    slot_dir.mkdir()

    fake_rclone = tmp_path / "fake_rclone.py"
    fake_rclone.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import json
            import os
            import sys
            import time

            def main():
                args = sys.argv[1:]
                if not args:
                    return 1
                if args == ["--help"]:
                    print("--inplace")
                    print("--max-duration")
                    print("--json")
                    return 0
                if args[0] == "obscure":
                    _ = sys.stdin.read()
                    print("obscured")
                    return 0
                if args[0] == "size":
                    total = int(os.environ.get("TOTAL_BYTES", "0"))
                    if "--json" in args:
                        if os.environ.get("RCLONE_NO_JSON"):
                            print("unknown flag: --json", file=sys.stderr)
                            return 2
                        print(json.dumps({{"count": 1 if total > 0 else 0, "bytes": total}}))
                    else:
                        print(f"Total objects: {{1 if total > 0 else 0}}")
                        print(f"Total size: {{total}} Byte ({{total}} Bytes)")
                    return 0
                if args[0] == "copyto":
                    non_opts = [a for a in args if not a.startswith("--")]
                    if len(non_opts) < 3:
                        return 2
                    src, dst = non_opts[-2], non_opts[-1]
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if dst.endswith("current.json") or "/current.json" in src:
                        with open(dst, "w") as f:
                            f.write("{{}}")
                        return 0
                    run_mode = os.environ.get("RCLONE_MODE", "ok")
                    if run_mode == "ok":
                        total = int(os.environ.get("TOTAL_BYTES", "0"))
                        with open(dst, "wb") as f:
                            f.truncate(total)
                        return 0
                    if run_mode == "slow":
                        with open(dst, "wb") as f:
                            f.write(b"a" * 100)
                        while True:
                            time.sleep(0.1)
                    if run_mode == "eta":
                        with open(dst, "wb") as f:
                            f.truncate(300_000_000)
                        while True:
                            time.sleep(0.1)
                return 1

            if __name__ == "__main__":
                sys.exit(main())
            """
        )
    )
    fake_rclone.chmod(0o755)

    rclone_wrapper = bin_dir / "rclone"
    rclone_wrapper.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        f"os.execv(sys.executable, [sys.executable, str({str(fake_rclone)!r})] + sys.argv[1:])\n"
    )
    rclone_wrapper.chmod(0o755)

    sleep_wrapper = bin_dir / "sleep"
    sleep_wrapper.write_text(f"#!{sys.executable}\nimport sys\nsys.exit(0)\n")
    sleep_wrapper.chmod(0o755)

    config.cache.rclone = True
    config.cache.rclone_url = "https://cache.example.com/"
    config.cache.rclone_password = "secret"
    script = rclone_prefetch_script(config, str(slot_dir), remote_dir)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    env["TOTAL_BYTES"] = str(total_bytes)
    env["RCLONE_MODE"] = mode
    if not json:
        env["RCLONE_NO_JSON"] = "1"

    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    return result, slot_dir


def test_rclone_prefetch_fast_ok(config, tmp_path):
    result, slot_dir = _run_rclone_prefetch(config, tmp_path, "ok", 12345)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
    assert (slot_dir / "current.bin").exists()
    assert (slot_dir / "current.bin").stat().st_size == 12345


def test_rclone_prefetch_text_fallback_ok(config, tmp_path):
    """Older rclone without --json still parses the text Byte(s) output."""
    result, slot_dir = _run_rclone_prefetch(config, tmp_path, "ok", 12345, json=False)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
    assert (slot_dir / "current.bin").exists()
    assert (slot_dir / "current.bin").stat().st_size == 12345


def test_rclone_prefetch_slow_aborts_and_cleans(config, tmp_path):
    result, slot_dir = _run_rclone_prefetch(config, tmp_path, "slow", 12345)
    assert result.returncode == 1
    assert "too slow" in result.stderr
    assert not (slot_dir / "current.bin").exists()


def test_rclone_prefetch_eta_aborts_and_cleans(config, tmp_path):
    # 10 GiB total; 300 MiB in 20 s projects to an ETA > 5 minutes.
    result, slot_dir = _run_rclone_prefetch(config, tmp_path, "eta", 10_737_418_240)
    assert result.returncode == 1
    assert "ETA exceeds 5 minutes" in result.stderr
    assert not (slot_dir / "current.bin").exists()


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
