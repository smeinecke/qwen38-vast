"""Tests for hostai.notify."""

from unittest import mock

from hostai.notify import notify, safe_apple


def test_safe_apple_escapes_quotes():
    assert safe_apple('say "hello"') == '"say \\"hello\\""'


def test_notify_linux_uses_notify_send():
    with (
        mock.patch("platform.system", return_value="Linux"),
        mock.patch("shutil.which", side_effect=lambda cmd: cmd == "notify-send"),
        mock.patch("subprocess.run") as run,
    ):
        assert notify("title", "msg") is True
        run.assert_called_once_with(["notify-send", "title", "msg"], check=False)


def test_notify_darwin_uses_osascript():
    with (
        mock.patch("platform.system", return_value="Darwin"),
        mock.patch("shutil.which", side_effect=lambda cmd: cmd == "osascript"),
        mock.patch("subprocess.run") as run,
    ):
        assert notify("title", "msg") is True
        assert run.call_args[0][0][0] == "osascript"


def test_notify_fallback_prints_and_returns_false(capsys):
    with (
        mock.patch("platform.system", return_value="OtherOS"),
        mock.patch("shutil.which", return_value=None),
    ):
        assert notify("title", "msg") is False
    captured = capsys.readouterr()
    assert "[notify] title: msg" in captured.err


def test_notify_catches_exceptions():
    with mock.patch("shutil.which", side_effect=RuntimeError("boom")):
        assert notify("title", "msg") is False
