"""Tests for hostai.tls."""

from pathlib import Path
from unittest import mock

from hostai import tls


def test_ensure_local_tls_dir_creates_private_dir(project_dir):
    d = tls.ensure_local_tls_dir(project_dir)
    assert d.is_dir()
    assert oct(d.stat().st_mode)[-3:] == "700"


def test_generate_cert_writes_key_and_cert(project_dir):
    def fake_run(*args, **kwargs):
        # Simulate openssl creating the files it was asked to create.
        cmd = args[0]
        key_idx = cmd.index("-keyout") + 1
        cert_idx = cmd.index("-out") + 1
        Path(cmd[key_idx]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[key_idx]).write_text("KEY")
        Path(cmd[cert_idx]).write_text("CERT")
        return mock.Mock(stdout="", stderr="", returncode=0)

    with mock.patch("hostai.tls.utils.run", side_effect=fake_run):
        cert, key = tls.generate_cert(project_dir / "tls")
    assert cert.name == "server.crt"
    assert key.name == "server.key"


def test_load_cert_pair_returns_text(project_dir):
    tls_dir = project_dir / "tls"
    tls_dir.mkdir(parents=True)
    (tls_dir / "server.crt").write_text("CERT")
    (tls_dir / "server.key").write_text("KEY")
    cert_text, key_text = tls.load_cert_pair(tls_dir)
    assert cert_text == "CERT"
    assert key_text == "KEY"


def test_deliver_cert_returns_false_when_missing(project_dir, running_state):
    assert tls.deliver_cert("ssh://root@10.0.0.1:22", project_dir / "nope", state=running_state) is False


def test_deliver_cert_sends_b64_payload(project_dir, running_state):
    tls_dir = project_dir / ".hostai-cache" / "tls"
    tls_dir.mkdir(parents=True)
    (tls_dir / "server.crt").write_text("CERT")
    (tls_dir / "server.key").write_text("KEY")
    with mock.patch("hostai.tls.ssh.run_remote", return_value=mock.Mock(returncode=0)) as run:
        assert tls.deliver_cert("ssh://root@10.0.0.1:22", tls_dir, state=running_state) is True
    assert "base64" in run.call_args.kwargs["input_data"]


def test_deliver_cert_propagates_failure(project_dir, running_state):
    tls_dir = project_dir / ".hostai-cache" / "tls"
    tls_dir.mkdir(parents=True)
    (tls_dir / "server.crt").write_text("CERT")
    (tls_dir / "server.key").write_text("KEY")
    with mock.patch("hostai.tls.ssh.run_remote", return_value=mock.Mock(returncode=1, stderr="nope")):
        assert tls.deliver_cert("ssh://root@10.0.0.1:22", tls_dir, state=running_state) is False
