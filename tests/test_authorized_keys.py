"""Tests for hostai.authorized_keys."""

from unittest import mock

import pytest

from hostai import authorized_keys as ak


def test_load_dotenv_parses_simple(tmp_path):
    env = tmp_path / ".env"
    env.write_text("FOO=bar\n# comment\nexport BAZ=qux\nBAD-KEY=val\n")
    result = ak._load_dotenv(env)
    # python-dotenv may keep keys with dashes, so only assert the known-good entries.
    assert result.get("FOO") == "bar"
    assert result.get("BAZ") == "qux"


def test_load_dotenv_fallback_parses_strict(tmp_path):
    env = tmp_path / ".env"
    env.write_text("FOO=bar\n# comment\nexport BAZ=qux\nBAD-KEY=val\n")
    with mock.patch("hostai.authorized_keys.dotenv_values", None):
        assert ak._load_dotenv(env) == {"FOO": "bar", "BAZ": "qux"}


def test_load_dotenv_strips_quotes(tmp_path):
    env = tmp_path / ".env"
    env.write_text('FOO="bar"\nBAZ=\'qux\'\n')
    assert ak._load_dotenv(env) == {"FOO": "bar", "BAZ": "qux"}


def test_load_dotenv_missing_returns_empty(tmp_path):
    assert ak._load_dotenv(tmp_path / "nope") == {}


def test_public_key_lines_filters_and_dedupes():
    text = "\n".join([
        "ssh-ed25519 AAAA comment",
        "not a key",
        "ssh-rsa BBBB another",
        "  ssh-ed25519 AAAA comment  ",
        "",
    ])
    assert ak._public_key_lines(text) == ["ssh-ed25519 AAAA comment", "ssh-rsa BBBB another"]


def test_github_user_from_https_remote():
    assert ak._github_user_from_remote_url("https://github.com/smeinecke/qwen38-vast.git") == "smeinecke"
    assert ak._github_user_from_remote_url("https://github.com/smeinecke/qwen38-vast/") == "smeinecke"


def test_github_user_from_ssh_remote():
    assert ak._github_user_from_remote_url("git@github.com:smeinecke/qwen38-vast.git") == "smeinecke"
    assert ak._github_user_from_remote_url("ssh://git@github.com/smeinecke/qwen38-vast.git") == "smeinecke"
    assert ak._github_user_from_remote_url("git://github.com/smeinecke/qwen38-vast.git") == "smeinecke"


def test_github_user_from_bad_remote():
    assert ak._github_user_from_remote_url("https://gitlab.com/foo/bar.git") is None


def test_github_user_from_git_config_with_git_command(tmp_path):
    (tmp_path / ".git" / "config").parent.mkdir(parents=True)
    (tmp_path / ".git" / "config").write_text("[remote \"origin\"]\nurl = https://github.com/smeinecke/qwen38-vast.git\n")
    with mock.patch("shutil.which", return_value="/usr/bin/git"):
        with mock.patch("hostai.authorized_keys.subprocess.run", return_value=mock.Mock(returncode=0, stdout="https://github.com/smeinecke/qwen38-vast.git")):
            assert ak._github_user_from_git_config(tmp_path) == "smeinecke"


def test_github_user_from_git_config_fallback(tmp_path):
    (tmp_path / ".git" / "config").parent.mkdir(parents=True)
    (tmp_path / ".git" / "config").write_text("[remote \"origin\"]\nurl = git@github.com:smeinecke/qwen38-vast.git\n")
    with mock.patch("shutil.which", return_value=None):
        assert ak._github_user_from_git_config(tmp_path) == "smeinecke"


def test_http_get_uses_requests(monkeypatch):
    with mock.patch("hostai.authorized_keys.requests") as requests_mod:
        requests_mod.get.return_value = mock.Mock(status_code=200, text="keys", raise_for_status=mock.Mock())
        assert ak._http_get("https://example.com/keys") == "keys"


def test_http_get_falls_back_to_urllib():
    resp = mock.MagicMock()
    resp.read.return_value = b"keys"
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    with mock.patch("hostai.authorized_keys.requests", None):
        with mock.patch("urllib.request.urlopen", return_value=resp) as urlopen:
            assert ak._http_get("https://example.com/keys") == "keys"
            urlopen.assert_called_once()


def test_http_get_returns_none_after_retries():
    with mock.patch("hostai.authorized_keys.requests") as requests_mod:
        requests_mod.get.side_effect = Exception("network")
        assert ak._http_get("https://example.com/keys") is None


def test_github_keys_fetches(monkeypatch):
    with mock.patch("hostai.authorized_keys._http_get", return_value="ssh-ed25519 AAAA"):
        assert ak._github_keys("smeinecke") == "ssh-ed25519 AAAA"


def test_github_keys_raises_when_fetch_fails():
    with mock.patch("hostai.authorized_keys._http_get", return_value=None):
        with pytest.raises(RuntimeError):
            ak._github_keys("smeinecke")


def test_prepare_authorized_keys_from_public_key(tmp_path, monkeypatch):
    monkeypatch.setenv("SSH_PUBLIC_KEY", "ssh-ed25519 AAAA test")
    try:
        out = ak.prepare_authorized_keys(root_dir=tmp_path)
        assert out.exists()
        assert out.read_text() == "ssh-ed25519 AAAA test\n"
    finally:
        monkeypatch.delenv("SSH_PUBLIC_KEY", raising=False)


def test_prepare_authorized_keys_from_github_user(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SSH_KEY_USER", "smeinecke")
    try:
        with mock.patch("hostai.authorized_keys._http_get", return_value="ssh-ed25519 AAAA test"):
            out = ak.prepare_authorized_keys(root_dir=tmp_path)
        assert out.exists()
        assert "ssh-ed25519 AAAA test" in out.read_text()
    finally:
        monkeypatch.delenv("GITHUB_SSH_KEY_USER", raising=False)


def test_prepare_authorized_keys_from_committed_file(tmp_path):
    ssh_dir = tmp_path / "ssh"
    ssh_dir.mkdir(parents=True)
    (ssh_dir / "authorized_keys").write_text("ssh-ed25519 BBBB committed")
    out = ak.prepare_authorized_keys(root_dir=tmp_path, from_github=False)
    assert out.exists()
    assert "ssh-ed25519 BBBB committed" in out.read_text()


def test_prepare_authorized_keys_strict_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("SSH_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("GITHUB_SSH_KEY_USER", raising=False)
    with pytest.raises(RuntimeError):
        ak.prepare_authorized_keys(root_dir=tmp_path, from_github=False, strict=True)


def test_prepare_authorized_keys_warnings_when_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("SSH_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("GITHUB_SSH_KEY_USER", raising=False)
    ak.prepare_authorized_keys(root_dir=tmp_path, from_github=False)
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
