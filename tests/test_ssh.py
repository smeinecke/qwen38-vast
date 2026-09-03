"""Tests for hostai.ssh helpers with mocked asyncssh/socket."""

import socket
from pathlib import Path
from unittest import mock

from hostai import ssh
from hostai.ssh import CompletedProcess, resolve_ssh_endpoint, run_remote


def test_resolve_ssh_endpoint_legacy():
    inst = {
        "ssh_host": "10.0.0.1",
        "ssh_port": 2222,
        "ssh_user": "root",
        "direct_port_count": 1,
    }
    ep = resolve_ssh_endpoint(inst)
    assert ep == {
        "user": "root",
        "host": "10.0.0.1",
        "port": 2222,
        "ssh_url": "ssh://root@10.0.0.1:2222",
        "direct": True,
    }


def test_resolve_ssh_endpoint_public_ip():
    inst = {
        "public_ipaddr": "203.0.113.4",
        "ports": {"22/tcp": [{"HostPort": "2222"}]},
    }
    ep = resolve_ssh_endpoint(inst)
    assert ep == {
        "user": "root",
        "host": "203.0.113.4",
        "port": 2222,
        "ssh_url": "ssh://root@203.0.113.4:2222",
        "direct": False,
    }


def test_resolve_ssh_endpoint_missing():
    assert resolve_ssh_endpoint({}) is None


def test_resolve_ssh_endpoint_ignores_machine_management_port():
    inst = {
        "public_ipaddr": "203.0.113.4",
        "machine_dir_ssh_port": 2222,
    }
    assert resolve_ssh_endpoint(inst) is None


def test_is_tunnel_healthy_open(config, running_state):
    running_state.local_port = 18080
    with mock.patch("socket.create_connection", return_value=mock.MagicMock()) as conn:
        assert ssh.is_tunnel_healthy(config, running_state) is True
    conn.assert_called_once_with(("127.0.0.1", 18080), timeout=3)


def test_is_tunnel_healthy_refused(config, running_state):
    running_state.local_port = 18080
    with mock.patch("socket.create_connection", side_effect=socket.error):
        assert ssh.is_tunnel_healthy(config, running_state) is False


def test_run_remote_success():
    class FakeResult:
        returncode = 0
        stdout = "hello\n"
        stderr = ""

    class FakeConn:
        async def run(self, command, **kwargs):
            return FakeResult()

    class FakeSSH:
        async def __aenter__(self):
            return FakeConn()
        async def __aexit__(self, *args):
            return False

    with mock.patch("asyncssh.connect", return_value=FakeSSH()):
        res = run_remote(
            "ssh://root@10.0.0.1:2222",
            "echo hello",
            known_hosts=Path("/dev/null"),
        )

    assert res.returncode == 0
    assert res.stdout == "hello\n"


def test_run_remote_failure():
    with mock.patch("asyncssh.connect", side_effect=Exception("boom")):
        res = run_remote(
            "ssh://root@10.0.0.1:2222",
            "echo hello",
            known_hosts=Path("/dev/null"),
        )
    assert res.returncode == 1
    assert "boom" in (res.stderr or "")


def test_completed_process_defaults():
    cp = CompletedProcess(args=["x"], returncode=0)
    assert cp.stdout is None
    assert cp.stderr is None
