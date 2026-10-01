"""Tests for hostai.ssh helpers."""

import socket
from pathlib import Path
from unittest import mock
from unittest.mock import AsyncMock

import pytest

from hostai import ssh
from hostai.state import State


def test_resolve_ssh_endpoint_legacy_fields():
    inst = {
        "ssh_host": "10.0.0.1",
        "ssh_port": 2222,
        "ssh_user": "admin",
        "direct_port_count": 1,
    }
    ep = ssh.resolve_ssh_endpoint(inst)
    assert ep["user"] == "admin"
    assert ep["host"] == "10.0.0.1"
    assert ep["port"] == 2222
    assert ep["direct"] is True


def test_resolve_ssh_endpoint_public_ip_ports():
    inst = {
        "public_ipaddr": "203.0.113.1",
        "ports": {"22/tcp": [{"HostPort": "2223"}]},
    }
    ep = ssh.resolve_ssh_endpoint(inst)
    assert ep["host"] == "203.0.113.1"
    assert ep["port"] == 2223
    assert ep["ssh_url"] == "ssh://root@203.0.113.1:2223"


def test_resolve_ssh_endpoint_returns_none_when_missing():
    assert ssh.resolve_ssh_endpoint({}) is None


def test_ssh_options_returns_empty():
    assert ssh.ssh_options("h", 22, Path("/tmp/known_hosts")) == []


def test_clear_known_hosts(tmp_path):
    known = tmp_path / "known_hosts"
    known.write_text("data")
    with mock.patch("hostai.ssh.utils.run") as run:
        ssh.clear_known_hosts("h", 22, known)
    assert run.call_count == 2


def test_clear_known_hosts_missing_file(tmp_path):
    known = tmp_path / "nope"
    with mock.patch("hostai.ssh.utils.run") as run:
        ssh.clear_known_hosts("h", 22, known)
    run.assert_not_called()


def test_connect_kwargs_with_identity():
    kwargs = ssh._connect_kwargs(identity=Path("/tmp/key"))
    assert kwargs["client_keys"] == ["/tmp/key"]
    assert kwargs["agent_path"] is None


def test_connect_kwargs_without_identity():
    kwargs = ssh._connect_kwargs()
    assert "client_keys" not in kwargs
    assert "agent_path" not in kwargs


def test_default_identity_from_state(project_dir):
    state = State(project_dir / ".hostai-vast" / "state.json")
    state.ssh_identity = Path("/tmp/state_key")
    assert ssh._default_identity(None, state) == Path("/tmp/state_key")


def test_default_identity_from_config(config):
    config.secrets["SSH_PRIVATE_KEY"] = "/tmp/config_key"
    assert ssh._default_identity(config, None) == Path("/tmp/config_key")


def test_default_identity_no_source(config):
    assert ssh._default_identity(config, None) is None


def test_parse_url():
    assert ssh._parse_url("ssh://root@host:22") == ("root", "host", 22)
    assert ssh._parse_url("ssh://user@host:2222") == ("user", "host", 2222)
    assert ssh._parse_url("ssh://host:2222") == ("root", "host", 2222)


def test_decode_output_bytes():
    assert ssh._decode_output(b"hello") == "hello"


def test_decode_output_string():
    assert ssh._decode_output("hello") == "hello"


def test_completed_process_defaults():
    cp = ssh.CompletedProcess(args=["ls"], returncode=0)
    assert cp.stdout is None
    assert cp.stderr is None


def test_connect_timeout_for_stage():
    assert ssh._connect_timeout_for_stage(10) == 10.0
    assert ssh._connect_timeout_for_stage(1000) == 120.0
    assert ssh._connect_timeout_for_stage(5) == 10.0


def test_is_remote_socket_present_detects(tmp_path):
    with mock.patch("hostai.ssh.run_remote", return_value=mock.Mock(returncode=0, stdout="present")) as run:
        assert ssh.is_remote_socket_present("ssh://root@h:22", "/s.sock", known_hosts=tmp_path / "kh") is True
    assert run.called


def test_is_remote_socket_present_not_there():
    with mock.patch("hostai.ssh.run_remote", return_value=mock.Mock(returncode=0, stdout="")):
        assert ssh.is_remote_socket_present("ssh://root@h:22", "/s.sock", known_hosts=Path("/tmp/kh")) is False


def test_wait_for_ssh_returns_false_without_url(tmp_path):
    assert ssh.wait_for_ssh(None, known_hosts=tmp_path / "kh", timeout=1) is False


def test_local_port_is_open_not_open():
    assert ssh._local_port_is_open(0, timeout=1) is False


def test_is_tunnel_healthy_unix_socket(config, tmp_path):
    sock_path = tmp_path / "upstream.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(sock_path))
    listener.listen(1)
    try:
        state = State(tmp_path / "state.json", {"upstream_socket": str(sock_path), "local_port": 0})
        assert ssh.is_tunnel_healthy(config, state, timeout=2) is True
    finally:
        listener.close()


def test_is_tunnel_healthy_unix_socket_missing_falls_back_to_port(config, tmp_path):
    # Recorded unix upstream but no socket file: fall back to the TCP check,
    # which fails when local_port is 0.
    state = State(tmp_path / "state.json", {"upstream_socket": str(tmp_path / "gone.sock"), "local_port": 0})
    assert ssh.is_tunnel_healthy(config, state, timeout=1) is False


def test_is_tunnel_healthy_no_endpoint(config, tmp_path):
    state = State(tmp_path / "state.json", {"local_port": 0})
    assert ssh.is_tunnel_healthy(config, state, timeout=1) is False


class _FakeConn:
    def __init__(self, result=None, fail=False):
        self.result = result
        self.fail = fail
    async def __aenter__(self):
        if self.fail:
            raise Exception("conn fail")
        return self
    async def __aexit__(self, *args):
        return None
    async def run(self, *args, **kwargs):
        return self.result


def _make_connect(result=None, fail=False):
    return mock.Mock(return_value=_FakeConn(result=result, fail=fail))


def test_run_remote_success(tmp_path):
    result = mock.Mock(returncode=0, stdout=b"ok", stderr=b"")
    with mock.patch("asyncssh.connect", _make_connect(result)):
        cp = ssh.run_remote("ssh://root@host:22", "echo ok", known_hosts=tmp_path / "kh")
    assert cp.returncode == 0
    assert cp.stdout == "ok"


def test_run_remote_no_url():
    cp = ssh.run_remote(None, "ls", known_hosts=Path("/tmp/kh"))
    assert cp.returncode == 1


def test_run_remote_exception():
    with mock.patch("asyncssh.connect", _make_connect(fail=True)):
        cp = ssh.run_remote("ssh://root@host:22", "ls", known_hosts=Path("/tmp/kh"))
    assert cp.returncode == 1


def test_run_remote_exception_names_class_when_message_empty(tmp_path):
    """Timeouts produce an empty str(exc); stderr must still name the cause."""
    with mock.patch("hostai.ssh._run_coro", side_effect=TimeoutError()):
        cp = ssh.run_remote("ssh://root@host:22", "ls", known_hosts=tmp_path / "kh")
    assert cp.returncode == 1
    assert cp.stderr == "TimeoutError"


def test_is_ssh_reachable_true():
    result = mock.Mock(returncode=0, stdout="ok", stderr="")
    with mock.patch("asyncssh.connect", _make_connect(result)):
        assert ssh.is_ssh_reachable("ssh://root@host:22", known_hosts=Path("/tmp/kh")) is True


def test_wait_for_ssh_returns_true(tmp_path):
    result = mock.Mock(returncode=0, stdout="ok", stderr="")
    with mock.patch("asyncssh.connect", _make_connect(result)):
        assert ssh.wait_for_ssh("ssh://root@host:22", known_hosts=tmp_path / "kh", timeout=3) is True


def test_wait_for_ssh_alive_check_aborts(tmp_path):
    def _gone():
        raise RuntimeError("instance gone")

    with mock.patch("asyncssh.connect", _make_connect(fail=True)):
        with mock.patch("hostai.ssh.time.sleep"):
            with pytest.raises(RuntimeError, match="instance gone"):
                ssh.wait_for_ssh(
                    "ssh://root@host:22",
                    known_hosts=tmp_path / "kh",
                    timeout=30,
                    quiet=True,
                    alive_check=_gone,
                )


def test_wait_for_ssh_alive_check_not_called_when_reachable(tmp_path):
    calls = []
    result = mock.Mock(returncode=0, stdout="ok", stderr="")
    with mock.patch("asyncssh.connect", _make_connect(result)):
        ok = ssh.wait_for_ssh(
            "ssh://root@host:22",
            known_hosts=tmp_path / "kh",
            timeout=3,
            quiet=True,
            alive_check=lambda: calls.append(1),
        )
    assert ok is True
    assert calls == []


def test_scp_to_success(tmp_path):
    local = tmp_path / "local.txt"
    local.write_text("x")
    with mock.patch("asyncssh.scp", new_callable=AsyncMock):
        cp = ssh.scp_to("ssh://root@host:22", local, "/remote", known_hosts=tmp_path / "kh")
    assert cp.returncode == 0


def test_scp_from_success(tmp_path):
    local = tmp_path / "down.txt"
    with mock.patch("asyncssh.scp", new_callable=AsyncMock):
        cp = ssh.scp_from("ssh://root@host:22", "/remote", local, known_hosts=tmp_path / "kh")
    assert cp.returncode == 0
