"""Tests for hostai utility functions."""

import re
import time
from unittest import mock

from hostai import utils


def test_now_rfc3339_format():
    s = utils.now_rfc3339()
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", s)


def test_now_epoch_returns_int():
    assert isinstance(utils.now_epoch(), int)
    assert utils.now_epoch() <= int(time.time()) + 1


def test_format_cost():
    assert utils.format_cost(3600, 1.0) == 1.0
    assert utils.format_cost(1800, 2.0) == 1.0


def test_format_dph():
    assert utils.format_dph(0.5) == "0.5000"


def test_safe_label():
    assert utils.safe_label("hello world") == "hello-world"
    assert utils.safe_label("a/b/c") == "a-b-c"
    assert re.match(r"^[A-Za-z0-9_.-]+$", utils.make_api_key())


def test_make_run_id():
    rid = utils.make_run_id("test")
    assert rid.startswith("20")
    assert "-test-" in rid


def test_port_is_free_when_free():
    with mock.patch("socket.socket") as sock:
        sock.return_value.__enter__.return_value.bind.return_value = None
        assert utils.port_is_free(18080) is True


def test_port_is_free_when_in_use():
    with mock.patch("socket.socket") as sock:
        sock.return_value.__enter__.return_value.bind.side_effect = OSError("Address already in use")
        assert utils.port_is_free(18080) is False


def test_find_free_port():
    with mock.patch("hostai.utils.port_is_free", side_effect=[False, False, True]):
        port = utils.find_free_port(start=18080)
    assert port == 18082


def test_sanitize_for_shell():
    assert utils.sanitize_for_shell("a b") == "'a b'"


def test_parse_ssh_url():
    user, host, port = utils.parse_ssh_url("ssh://root@10.0.0.1:2222")
    assert user == "root"
    assert host == "10.0.0.1"
    assert port == 2222


def test_parse_ssh_url_no_user():
    user, host, port = utils.parse_ssh_url("ssh://10.0.0.1:22")
    assert user == "root"  # default when no username is supplied
    assert host == "10.0.0.1"
    assert port == 22


def test_mkdir_private(tmp_path):
    p = tmp_path / "private"
    result = utils.mkdir_private(p)
    assert result == p
    assert result.exists()
    assert (result.stat().st_mode & 0o777) == 0o700
