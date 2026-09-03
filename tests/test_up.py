"""Tests for hostai.commands.up helpers and CLI validation."""

from types import SimpleNamespace
from unittest import mock

import click
import pytest
from click.testing import CliRunner

from hostai.commands.up import (
    _build_query,
    _do_fresh,
    _emit_instance_logs,
    _prefetch_slot_cache_to_vast,
    _select_offer,
    cmd_up,
)


def make_offer(**overrides):
    base = {
        "id": 42,
        "ask_contract_id": 42,
        "gpu_name": "RTX 4090",
        "dph_total": 0.5,
        "dph_base": 0.5,
        "inet_down": 1000,
        "inet_up": 1000,
    }
    base.update(overrides)
    return base


def test_build_query_basic(config):
    profile = mock.Mock()
    profiles = mock.Mock()
    profiles.market_policy.require_free_traffic = False
    query, max_dph = _build_query(config, profiles, profile, "gpu_name == RTX 4090", None, False, None)
    assert "RTX 4090" in query
    assert "dph_total" in query
    assert max_dph == config.market.max_dph


def test_build_query_with_offer_id_skips_dph(config):
    profile = mock.Mock()
    profiles = mock.Mock()
    profiles.market_policy.require_free_traffic = False
    query, max_dph = _build_query(config, profiles, profile, "gpu_name == RTX 4090", None, False, 123)
    assert "dph_total" not in query


def test_build_query_negative_max_price(config):
    profile = mock.Mock()
    profiles = mock.Mock()
    with pytest.raises(click.ClickException, match="non-negative"):
        _build_query(config, profiles, profile, "gpu_name == RTX 4090", -1.0, False, None)


def test_select_offer_chooses_cheapest(config):
    offers = [make_offer(dph_total=0.6), make_offer(dph_total=0.4)]
    with mock.patch("hostai.commands.up.search_instance_offers", return_value=offers) as search:
        selected = _select_offer(config, "gpu_name == RTX 4090", 1.0, False, None)
    assert selected["dph_total"] == 0.4
    assert search.call_args.args[1] == "gpu_name == RTX 4090"
    assert search.call_args.kwargs == {"limit": 25, "order": "dph_total", "storage": 100}


def test_select_offer_specific_id(config):
    offers = [make_offer(id=1), make_offer(id=2)]
    with mock.patch("hostai.commands.up.search_instance_offers", return_value=offers):
        selected = _select_offer(config, "gpu_name == RTX 4090", 1.0, False, 2)
    assert selected["id"] == 2


def test_select_offer_no_matches(config):
    with mock.patch("hostai.commands.up.search_instance_offers", return_value=[]):
        with pytest.raises(click.ClickException, match="no matching offer"):
            _select_offer(config, "gpu_name == RTX 4090", 1.0, False, None)


def test_fresh_instance_uses_custom_entrypoint_and_publishes_ssh(config):
    profiles = mock.Mock()
    profiles.market_policy.require_free_traffic = False
    profile = SimpleNamespace(
        name="test",
        ctx_size=32768,
        gpu_query="gpu_name == RTX 4090",
        cache_ram=None,
        ctx_checkpoints=None,
        monitor_group="",
    )
    image = SimpleNamespace(image_tag="test", cuda_arch="86")
    offer = make_offer(id=42)

    with (
        mock.patch("hostai.commands.up._resolve_profile", return_value=(profiles, profile, image)),
        mock.patch("hostai.commands.up.image_for_profile", return_value="example/image:test"),
        mock.patch("hostai.commands.up._select_offer", return_value=offer),
        mock.patch("hostai.commands.up.create_instance_from_offer", return_value={"new_contract": 123}) as create,
        mock.patch("hostai.commands.up._do_fresh_core"),
    ):
        _do_fresh(config, "test", None, None, None, False, False, False, False, None)

    launch = create.call_args.kwargs
    assert launch["env"]["-p 22:22"] == "1"
    assert launch["runtype"] == "args"
    assert launch["args"] is None


def test_cmd_up_rejects_invalid_port(config):
    runner = CliRunner()
    result = runner.invoke(cmd_up, ["--local-port", "70000"], obj=config)
    assert result.exit_code != 0
    assert "65535" in result.output


def test_cmd_up_rejects_negative_max_price(config):
    runner = CliRunner()
    result = runner.invoke(cmd_up, ["--max-price", "-1"], obj=config)
    assert result.exit_code != 0
    assert "non-negative" in result.output


def test_prefetch_slot_cache_uses_timeout_and_script(config, tmp_path):
    """_prefetch_slot_cache_to_vast runs the generated script with the 330s timeout."""
    calls = []

    def fake_run_remote(ssh_url, command, *, input_data, known_hosts, timeout=None):
        calls.append((ssh_url, command, timeout, input_data))
        return mock.Mock(returncode=0, stdout="ok", stderr="")

    with mock.patch("hostai.commands.up.ssh.run_remote", side_effect=fake_run_remote):
        result = _prefetch_slot_cache_to_vast(
            "ssh://root@1.2.3.4:22",
            config,
            "/var/lib/qwen38/slots",
            "qwen-slot-cache/default/sig",
            tmp_path / "known_hosts",
        )

    assert result is True
    assert len(calls) == 1
    ssh_url, command, timeout, input_data = calls[0]
    assert ssh_url == "ssh://root@1.2.3.4:22"
    assert command == "bash -s"
    assert timeout == 330
    assert "current.bin" in input_data
    assert "ok" in input_data


def test_do_fresh_blocks_running_instance(config, project_dir):
    """A running instance in state.json must raise before renting another."""
    from hostai.state import State, state_dir

    sdir = state_dir(project_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    state = State(sdir / "state.json")
    state.instance_id = 123
    state.save()

    with mock.patch("hostai.commands.up.get_instance", return_value={"id": 123, "actual_status": "running"}):
        with pytest.raises(click.ClickException, match="run hostai down first"):
            _do_fresh(config, "test", None, None, None, False, False, False, False, None)


def test_do_fresh_allows_recreate_when_instance_is_exited(config, project_dir):
    """An exited instance should not block a new hostai up."""
    from hostai.state import State, state_dir

    sdir = state_dir(project_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    state = State(sdir / "state.json")
    state.instance_id = 123
    state.save()

    with mock.patch("hostai.commands.up.get_instance", return_value={"id": 123, "actual_status": "exited"}):
        with mock.patch("hostai.commands.up._resolve_profile", side_effect=RuntimeError("later step")):
            with pytest.raises(RuntimeError, match="later step"):
                _do_fresh(config, "test", None, None, None, False, False, False, False, None)


def test_emit_instance_logs_skips_non_text_logs(config):
    """_emit_instance_logs must not crash when get_instance_logs returns a dict."""
    seen = {"container": set(), "daemon": set()}

    def fake_logs(config, instance_id, *, tail, daemon_logs, timeout):
        return {"not_ready": True} if daemon_logs else "line one\nline two"

    with mock.patch("hostai.commands.up.get_instance_logs", side_effect=fake_logs):
        _emit_instance_logs(config, 123, seen)

    assert "line one" in seen["container"]
    assert seen["daemon"] == set()
