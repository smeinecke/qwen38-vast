"""Tests for hostai.commands.down with mocked Vast/SSH APIs."""

import json
from unittest import mock

import requests

from hostai.commands.down import (
    _parse_rsync_transferred_bytes,
    _pause_or_destroy,
    _save_and_upload_slot_cache,
    _slot_save,
)


def fake_completed(returncode=0, stdout="", stderr=""):
    return mock.Mock(returncode=returncode, stdout=stdout, stderr=stderr)


def test_slot_save_parses_payload(config, running_state):
    payload = {"n_saved": 100, "n_written": 1024, "timings": {"save_ms": 50}}
    with mock.patch("requests.post", return_value=mock.Mock(status_code=200, text=json.dumps(payload), json=lambda: payload)) as post:
        result = _slot_save(config, running_state)
    assert result["n_saved"] == 100
    assert result["n_written"] == 1024
    assert post.call_args.args[0] == "https://127.0.0.1:18080/slots/0?action=save"
    assert post.call_args.kwargs["json"] == {"filename": "current.bin"}


def test_slot_save_empty_slot(config, running_state):
    payload = {"n_saved": 0}
    with mock.patch("requests.post", return_value=mock.Mock(status_code=200, text=json.dumps(payload), json=lambda: payload)):
        result = _slot_save(config, running_state)
    assert result is None


def test_slot_save_api_error(config, running_state):
    with mock.patch("requests.post", side_effect=requests.RequestException("boom")):
        result = _slot_save(config, running_state)
    assert result is None


def test_save_and_upload_slot_cache_happy_path(config, running_state, tmp_path):
    payload = {"n_saved": 100, "n_written": 1024, "timings": {"save_ms": 50}}
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with mock.patch("requests.post", return_value=mock.Mock(status_code=200, text=json.dumps(payload), json=lambda: payload)):
        with mock.patch("hostai.commands.down._install_cache_key_on_vast", return_value=True):
            with mock.patch("hostai.commands.down._fetch_llama_commit", return_value="abc123"):
                with mock.patch("hostai.commands.down.ssh.run_remote", return_value=fake_completed()):
                    with mock.patch("hostai.commands.down.ssh.scp_to", return_value=fake_completed()):
                        with mock.patch("hostai.commands.down._upload_slot_cache_from_vast", return_value=(True, 12345)):
                            result = _save_and_upload_slot_cache(
                                config,
                                running_state,
                                run_dir,
                                no_cache=False,
                                known_hosts=tmp_path / "known_hosts",
                            )

    assert result is not None
    assert result["uploaded"] is True
    assert running_state.data["slot_cache_save"] == "uploaded"
    assert running_state.data["slot_cache_n_saved"] == 100
    assert (run_dir / "cache-save.json").exists()


def test_save_and_upload_slot_cache_disabled(config, running_state, tmp_path):
    running_state.slot_cache_enabled = False
    result = _save_and_upload_slot_cache(
        config,
        running_state,
        tmp_path / "run",
        no_cache=False,
        known_hosts=tmp_path / "known_hosts",
    )
    assert result is None


def test_save_and_upload_slot_cache_no_ssh(config, running_state, tmp_path):
    running_state.ssh_url = None
    config.cache.require_save = False
    result = _save_and_upload_slot_cache(
        config,
        running_state,
        tmp_path / "run",
        no_cache=False,
        known_hosts=tmp_path / "known_hosts",
    )
    assert result is None


def test_parse_rsync_transferred_bytes_with_units():
    stdout = "...\nTotal bytes sent: 838.46K\n...\n"
    assert _parse_rsync_transferred_bytes(stdout) == int(838.46 * 1024)

    stdout = "...\nsent 12.5M bytes  received 100 bytes\n"
    assert _parse_rsync_transferred_bytes(stdout) == int(12.5 * 1024 * 1024)

    stdout = "...\nTotal bytes sent: 1234\n"
    assert _parse_rsync_transferred_bytes(stdout) == 1234


def test_parse_rsync_transferred_bytes_no_match():
    assert _parse_rsync_transferred_bytes("") is None
    assert _parse_rsync_transferred_bytes("no stats here") is None


def test_pause_or_destroy_destroy(config, running_state, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    running_state.started_epoch = 0
    running_state.dph = 1.0
    running_state.status = "running"

    with mock.patch("hostai.commands.down.vast.destroy", return_value=None):
        msg = _pause_or_destroy(config, running_state, pause=False, run_dir=run_dir)

    assert "destroyed" in msg or "Session duration" in msg
    assert running_state.status == "destroyed"


def test_pause_or_destroy_404_becomes_already_absent(config, running_state, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    running_state.started_epoch = 0
    running_state.dph = 1.0

    err = requests.exceptions.HTTPError(response=mock.Mock(status_code=404))
    with mock.patch("hostai.commands.down.vast.destroy", side_effect=err):
        msg = _pause_or_destroy(config, running_state, pause=False, run_dir=run_dir)

    assert "already_absent" in msg


def test_pause_or_destroy_removes_active_state(config, running_state, tmp_path):
    """A successful or already-absent destroy must leave no active state.json."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    running_state.started_epoch = 0
    running_state.dph = 1.0
    running_state.status = "running"
    running_state.save()  # create the active state file
    assert running_state.state_file.exists()

    with mock.patch("hostai.commands.down.vast.destroy", return_value=None):
        _ = _pause_or_destroy(config, running_state, pause=False, run_dir=run_dir)

    assert not running_state.state_file.exists()
    assert running_state.status == "destroyed"
    assert (run_dir / "metadata.json").exists()


def test_pause_or_destroy_pause_retains_state(config, running_state, tmp_path):
    """A pause must retain the active state file so it can be resumed."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    running_state.started_epoch = 0
    running_state.dph = 1.0
    running_state.status = "running"
    running_state.save()

    with mock.patch("hostai.commands.down.vast.pause", return_value=None):
        _ = _pause_or_destroy(config, running_state, pause=True, run_dir=run_dir)

    assert running_state.state_file.exists()
    assert running_state.status == "paused"
