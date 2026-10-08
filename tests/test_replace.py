"""Tests for hostai.commands.replace."""

import asyncio
from pathlib import Path
from unittest import mock

import click
import pytest
from click.testing import CliRunner

from hostai import proxy as proxy_mod
from hostai.commands import replace
from hostai.market import OfferExclusions
from hostai.state import State


def _live_state(project_dir, **overrides):
    """Write a state.json that looks like a running deployment."""
    data = {
        "instance_id": 12345,
        "machine_id": 9001,
        "profile": "test",
        "local_port": 18080,
        "api_key": "old-key",
        "ssh_url": "ssh://root@10.0.0.1:2222",
        "unsecure": True,
        "upstream_socket": str(project_dir / ".hostai-vast" / "upstream.sock"),
    }
    data.update(overrides)
    state = State(project_dir / ".hostai-vast" / "state.json", data)
    state.save()
    return state


def _do_replace_kwargs(**overrides):
    kw = dict(
        instance="default",
        profile_name="test",
        cache_session=None,
        max_price=None,
        unverified=False,
        unsecure=True,
        no_cache=True,
        abort_if_shm_too_small=False,
        offer=None,
        machine=None,
        exclusions=OfferExclusions(),
        bid_price=None,
        session_seconds=None,
        dry_run=False,
        allow_unvalidated=False,
        allow_same_machine=False,
        no_archive=True,
    )
    kw.update(overrides)
    return kw


def test_cmd_replace_validation_errors(config):
    config.hostai.default_profile = "test"
    runner = CliRunner()
    for args, expected in [
        (["--max-price", "-1"], "non-negative"),
        (["--bid", "0"], "positive"),
        (["--scoring-mode", "invalid"], "dph, perf, or session"),
    ]:
        result = runner.invoke(replace.cmd_replace, args, obj=config)
        assert result.exit_code != 0
        assert expected in result.output


def test_cmd_replace_no_profile(config):
    config.hostai.default_profile = ""
    runner = CliRunner()
    result = runner.invoke(replace.cmd_replace, [], obj=config)
    assert result.exit_code != 0
    assert "no profile" in result.output


def test_cmd_replace_nothing_to_replace(config, project_dir):
    config.hostai.default_profile = "test"
    runner = CliRunner()
    result = runner.invoke(replace.cmd_replace, [], obj=config)
    assert result.exit_code != 0
    assert "nothing to replace" in result.output


def test_cmd_replace_conflicts(config):
    config.hostai.default_profile = "test"
    runner = CliRunner()
    result = runner.invoke(replace.cmd_replace, ["--offer", "7", "--skip-offer", "7"], obj=config)
    assert result.exit_code != 0
    assert "conflicts with a skipped offer id" in result.output
    result = runner.invoke(replace.cmd_replace, ["--machine", "7", "--skip-machine", "7"], obj=config)
    assert result.exit_code != 0
    assert "conflicts with a skipped machine id" in result.output


def _dry_run_plan():
    return mock.Mock(offer_id=77, machine_id=9002)


def test_cmd_replace_dry_run_auto_excludes_current_machine(config, project_dir):
    """By default replace avoids re-renting on the current machine."""
    _live_state(project_dir)
    runner = CliRunner()
    with mock.patch(
        "hostai.commands.replace._resolve_fresh_offer", return_value=_dry_run_plan()
    ) as resolve:
        result = runner.invoke(replace.cmd_replace, ["--dry-run"], obj=config)
    assert result.exit_code == 0
    assert "dry-run" in result.output
    # positional args: (config, profile, local_port, max_price, unverified, offer, exclusions, ...)
    assert resolve.call_args.args[2] is None  # old port held by the live proxy
    exclusions = resolve.call_args.args[6]
    assert 9001 in exclusions.machines


def test_cmd_replace_allow_same_machine(config, project_dir):
    _live_state(project_dir)
    runner = CliRunner()
    with mock.patch(
        "hostai.commands.replace._resolve_fresh_offer", return_value=_dry_run_plan()
    ) as resolve:
        result = runner.invoke(replace.cmd_replace, ["--dry-run", "--allow-same-machine"], obj=config)
    assert result.exit_code == 0
    assert 9001 not in resolve.call_args.args[6].machines


def test_cmd_replace_machine_pin_skips_auto_exclusion(config, project_dir):
    """An explicit --machine pin overrides the auto-exclusion."""
    _live_state(project_dir)
    runner = CliRunner()
    with mock.patch(
        "hostai.commands.replace._resolve_fresh_offer", return_value=_dry_run_plan()
    ) as resolve:
        result = runner.invoke(replace.cmd_replace, ["--dry-run", "--machine", "9001"], obj=config)
    assert result.exit_code == 0
    assert resolve.call_args.kwargs["machine"] == 9001
    assert 9001 not in resolve.call_args.args[6].machines


def test_remote_health_probe_unix_socket(config):
    state = State(Path("/tmp/s.json"), {"api_key": "k", "unsecure": False})
    cmd = replace._remote_health_probe(state)
    assert "--unix-socket /dev/shm/qwen38/llama.sock" in cmd
    assert "Bearer k" in cmd
    assert "https://localhost/health" in cmd


def test_remote_health_probe_tcp(config):
    state = State(Path("/tmp/s.json"), {"api_key": "k", "unsecure": True})
    cmd = replace._remote_health_probe(state)
    assert "http://127.0.0.1:8080/health" in cmd
    assert "Bearer k" in cmd


def test_wait_for_remote_health_success(config, project_dir):
    state = State(
        project_dir / ".hostai-vast" / "state.replace.json",
        {"ssh_url": "ssh://root@h:22", "api_key": "k", "unsecure": True},
    )
    res = mock.Mock(returncode=0)
    with mock.patch("hostai.commands.replace.ssh.run_remote", return_value=res):
        with mock.patch("hostai.commands.replace._log"):
            replace._wait_for_remote_health(config, state, project_dir / "kh")


def test_wait_for_remote_health_timeout(config, project_dir):
    config.ssh.start_timeout = 1
    state = State(
        project_dir / ".hostai-vast" / "state.replace.json",
        {"ssh_url": "ssh://root@h:22", "api_key": "k", "unsecure": True},
    )
    res = mock.Mock(returncode=7, stdout="")
    with mock.patch("hostai.commands.replace.ssh.run_remote", return_value=res):
        with mock.patch("hostai.commands.replace.time.sleep"):
            with mock.patch("hostai.commands.replace._log"):
                with pytest.raises(click.ClickException, match="remote-health"):
                    replace._wait_for_remote_health(config, state, project_dir / "kh")


def test_destroy_staged_only_tears_down_remote(config, project_dir):
    """Staged cleanup must never signal the live proxy or tunnels."""
    staged_file = project_dir / ".hostai-vast" / "state.replace.json"
    staged_file.parent.mkdir(parents=True)
    staged_file.write_text("{}")
    staged = State(staged_file, {"instance_id": 99999})
    provider = mock.Mock()
    with mock.patch("hostai.commands.replace._provider", return_value=provider):
        replace._destroy_staged(config, staged, staged_file)
    provider.destroy_instance.assert_called_once_with(99999)
    assert not staged_file.exists()


def test_destroy_staged_keep_on_failure(config, project_dir):
    config.vast.keep_on_failure = True
    staged_file = project_dir / ".hostai-vast" / "state.replace.json"
    staged_file.parent.mkdir(parents=True)
    staged_file.write_text("{}")
    staged = State(staged_file, {"instance_id": 99999})
    provider = mock.Mock()
    with mock.patch("hostai.commands.replace._provider", return_value=provider):
        replace._destroy_staged(config, staged, staged_file)
    provider.destroy_instance.assert_not_called()
    assert not staged_file.exists()


def test_wait_for_proxy_cutover_success(config, project_dir):
    state_file = project_dir / ".hostai-vast" / "state.json"
    status = mock.AsyncMock(side_effect=[(12345, 200), (99999, 503), (99999, 200)])
    with mock.patch("hostai.commands.replace._proxy_status_once", new=status):
        with mock.patch("hostai.commands.replace.time.sleep"):
            assert replace._wait_for_proxy_cutover(config, state_file, 99999, timeout=5) is True


def test_wait_for_proxy_cutover_timeout(config, project_dir):
    state_file = project_dir / ".hostai-vast" / "state.json"
    status = mock.AsyncMock(return_value=(12345, 200))
    with mock.patch("hostai.commands.replace._proxy_status_once", new=status):
        with mock.patch("hostai.commands.replace.time.sleep"):
            assert replace._wait_for_proxy_cutover(config, state_file, 99999, timeout=0.1) is False


def test_wait_for_proxy_cutover_legacy_proxy(config, project_dir):
    """Proxies without /_hostai/backend pass once health dips and recovers."""
    state_file = project_dir / ".hostai-vast" / "state.json"
    status = mock.AsyncMock(side_effect=[(None, 200), (None, 503), (None, 200)])
    with mock.patch("hostai.commands.replace._proxy_status_once", new=status):
        with mock.patch("hostai.commands.replace.time.sleep"):
            assert replace._wait_for_proxy_cutover(config, state_file, 99999, timeout=5) is True


def test_proxy_status_once(config, project_dir):
    """Round-trip against a real aiohttp server exposing both routes."""
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def backend(r):
        return web.json_response({"instance_id": 99999})

    async def health(r):
        return web.Response(status=200)

    app = web.Application()
    app.router.add_get("/_hostai/backend", backend)
    app.router.add_get("/health", health)
    state_file = project_dir / ".hostai-vast" / "state.json"

    async def go():
        async with TestServer(app, host="127.0.0.1") as server:
            config.proxy.port = server.port
            return await replace._proxy_status_once(config, state_file)

    backend_id, health = asyncio.run(go())
    assert backend_id == 99999
    assert health == 200


def test_do_replace_happy_path(config, project_dir):
    """Successful replace: state flips, port kept, old instance destroyed last."""
    config.cache.enabled = False
    _live_state(project_dir, run_dir=None)
    state_file = project_dir / ".hostai-vast" / "state.json"
    run_old = project_dir / "run-old"
    run_old.mkdir()
    run_new = project_dir / "run-new"
    run_new.mkdir()

    staged_file = project_dir / ".hostai-vast" / "state.replace.json"
    staged = State(
        staged_file,
        {
            "instance_id": 99999,
            "machine_id": 9002,
            "local_port": 18080,
            "api_key": "old-key",
            "profile": "test",
            "run_dir": str(run_new),
        },
    )
    plan = mock.Mock(offer_id=77, machine_id=9002, image=mock.Mock(cuda_arch="sm_90"))
    provider = mock.Mock()

    with mock.patch("hostai.commands.replace._resolve_fresh_offer", return_value=plan):
        with mock.patch("hostai.commands.replace._create_fresh_instance", return_value=staged):
            with mock.patch("hostai.commands.replace._boot_staged"):
                with mock.patch(
                    "hostai.commands.replace._common.running_proxy_pid", return_value=4321
                ):
                    with mock.patch(
                        "hostai.commands.replace._wait_for_proxy_cutover", return_value=True
                    ):
                        with mock.patch("hostai.commands.replace._archive_session"):
                            with mock.patch(
                                "hostai.commands.replace._resolve_run_dir", return_value=run_old
                            ):
                                with mock.patch(
                                    "hostai.commands.replace._provider", return_value=provider
                                ):
                                    replace._do_replace(config, **_do_replace_kwargs())

    provider.destroy_instance.assert_called_once_with(12345)
    final = State.load(state_file)
    assert final.instance_id == 99999
    assert final.local_port == 18080
    assert final.api_key == "old-key"
    assert final.data["replaced_from_instance_id"] == 12345
    assert final.data["replaced_from_machine_id"] == 9001
    assert not staged_file.exists()


def test_do_replace_aborts_when_state_changed(config, project_dir):
    """If `down` ran mid-replace, the flip must not resurrect the deployment."""
    _live_state(project_dir)
    state_file = project_dir / ".hostai-vast" / "state.json"
    staged_file = project_dir / ".hostai-vast" / "state.replace.json"
    staged = State(staged_file, {"instance_id": 99999})
    plan = mock.Mock(offer_id=77, machine_id=9002, image=mock.Mock(cuda_arch="sm_90"))
    provider = mock.Mock()

    def _wipe_state(*a, **kw):
        State(state_file, {}).save()  # `hostai down` deleted the deployment

    with mock.patch("hostai.commands.replace._resolve_fresh_offer", return_value=plan):
        with mock.patch("hostai.commands.replace._create_fresh_instance", return_value=staged):
            with mock.patch("hostai.commands.replace._boot_staged", side_effect=_wipe_state):
                with mock.patch("hostai.commands.replace._common.running_proxy_pid", return_value=None):
                    with mock.patch("hostai.commands.replace._provider", return_value=provider):
                        with pytest.raises(click.ClickException, match="state.json changed"):
                            replace._do_replace(config, **_do_replace_kwargs())

    provider.destroy_instance.assert_called_once_with(99999)
    assert not staged_file.exists()
    assert State.load(state_file).instance_id is None


def test_do_replace_boot_failure_leaves_old_untouched(config, project_dir):
    """A staged boot failure destroys the new instance but not the old one."""
    _live_state(project_dir)
    state_file = project_dir / ".hostai-vast" / "state.json"
    staged_file = project_dir / ".hostai-vast" / "state.replace.json"
    staged = State(staged_file, {"instance_id": 99999})
    plan = mock.Mock(offer_id=77, machine_id=9002, image=mock.Mock(cuda_arch="sm_90"))
    provider = mock.Mock()

    with mock.patch("hostai.commands.replace._resolve_fresh_offer", return_value=plan):
        with mock.patch("hostai.commands.replace._create_fresh_instance", return_value=staged):
            with mock.patch(
                "hostai.commands.replace._boot_staged", side_effect=RuntimeError("boom")
            ):
                with mock.patch("hostai.commands.replace._common.running_proxy_pid", return_value=None):
                    with mock.patch("hostai.commands.replace._provider", return_value=provider):
                        with pytest.raises(RuntimeError):
                            replace._do_replace(config, **_do_replace_kwargs())

    provider.destroy_instance.assert_called_once_with(99999)
    final = State.load(state_file)
    assert final.instance_id == 12345  # old deployment intact
    assert not staged_file.exists()


def test_do_replace_cutover_failure_keeps_both(config, project_dir):
    """When the proxy never retargets, nothing is destroyed: state.json stays
    on the new instance and the old one is left running for manual recovery."""
    _live_state(project_dir)
    run_old = project_dir / "run-old"
    run_old.mkdir()
    state_file = project_dir / ".hostai-vast" / "state.json"
    staged_file = project_dir / ".hostai-vast" / "state.replace.json"
    staged = State(staged_file, {"instance_id": 99999, "local_port": 18080})
    plan = mock.Mock(offer_id=77, machine_id=9002, image=mock.Mock(cuda_arch="sm_90"))
    provider = mock.Mock()

    with mock.patch("hostai.commands.replace._resolve_fresh_offer", return_value=plan):
        with mock.patch("hostai.commands.replace._create_fresh_instance", return_value=staged):
            with mock.patch("hostai.commands.replace._boot_staged"):
                with mock.patch("hostai.commands.replace._common.running_proxy_pid", return_value=4321):
                    with mock.patch(
                        "hostai.commands.replace._wait_for_proxy_cutover", return_value=False
                    ):
                        with mock.patch("hostai.commands.replace._archive_session"):
                            with mock.patch(
                                "hostai.commands.replace._resolve_run_dir", return_value=run_old
                            ):
                                with mock.patch(
                                    "hostai.commands.replace._provider", return_value=provider
                                ):
                                    with pytest.raises(click.ClickException, match="did not retarget"):
                                        replace._do_replace(config, **_do_replace_kwargs())

    provider.destroy_instance.assert_not_called()
    assert State.load(state_file).instance_id == 99999
    assert not staged_file.exists()


def test_create_fresh_instance_sidecar_and_api_key(config, project_dir):
    """_create_fresh_instance must write the sidecar file and reuse the key."""
    from hostai.commands import up

    plan = mock.Mock()
    plan.profile = mock.Mock(
        image="img", cache_ram=None, ctx_checkpoints=None,
        disk_gb=None, min_gpu_vram_mb=None, monitor_group="", model=None, model_sha256=None,
    )
    plan.profile.name = "test"
    plan.image = mock.Mock(cuda_arch="sm_90", image_tag="img")
    plan.ctx_size = 32768
    plan.model = "model.gguf"
    plan.selected_image = "img:tag"
    plan.disk_gb = 35
    plan.interruptible = False
    plan.query = "q"
    plan.max_dph = 1.0
    plan.offer_type = "on-demand"
    plan.offer_data = {}
    plan.offer_id = 77
    plan.machine_id = 9002
    plan.gpu_name = "A100"
    plan.dph = 0.5
    plan.bid_price = None
    plan.session_seconds = None
    plan.exclusions = OfferExclusions()
    plan.local_port = 18080

    provider = mock.Mock()
    provider.create_instance.return_value = {"new_contract": 99999}

    sidecar = project_dir / ".hostai-vast" / "state.replace.json"
    with mock.patch("hostai.commands.up._provider", return_value=provider):
        staged = up._create_fresh_instance(
            config,
            "default",
            plan,
            None,
            True,
            True,
            state_file=sidecar,
            api_key="old-key",
        )
    assert staged.instance_id == 99999
    assert staged.api_key == "old-key"
    assert staged.state_file == sidecar
    assert sidecar.exists()
    # The live state.json was never touched.
    assert not (project_dir / ".hostai-vast" / "state.json").exists()


def test_proxy_retarget_swaps_backend(config, project_dir):
    state_file = project_dir / ".hostai-vast" / "state.json"
    old = State(state_file, {"instance_id": 1, "local_port": 18080, "api_key": "k1", "unsecure": True})
    fresh = State(state_file, {"instance_id": 2, "local_port": 18081, "api_key": "k2", "unsecure": True})
    proxy = proxy_mod.TokenizedProxy(config, old, mock.Mock(), project_dir / "p.sock", port=0)
    proxy.ready = True

    async def go():
        await proxy.retarget(fresh)
        await proxy.session.close()

    asyncio.run(go())
    assert proxy.state is fresh
    assert proxy.upstream == "http://127.0.0.1:18081"
    assert proxy.api_key == "k2"
    assert proxy.ready is False
    assert proxy.retarget_pending is True


def test_maybe_retarget_on_instance_change(config, project_dir):
    state_file = project_dir / ".hostai-vast" / "state.json"
    state_file.parent.mkdir(parents=True)
    State(state_file, {"instance_id": 2, "local_port": 18080, "unsecure": True, "api_key": "k"}).save()

    proxy = mock.Mock()
    proxy.state = State(state_file, {"instance_id": 1})
    proxy.upstream_socket = None
    proxy.retarget = mock.AsyncMock()

    with mock.patch("hostai.proxy._retarget_tunnel"):
        assert asyncio.run(proxy_mod._maybe_retarget(proxy)) is True
    proxy.retarget.assert_awaited_once()


def test_maybe_retarget_no_change(config, project_dir):
    state_file = project_dir / ".hostai-vast" / "state.json"
    state_file.parent.mkdir(parents=True)
    State(state_file, {"instance_id": 1}).save()

    proxy = mock.Mock()
    proxy.state = State(state_file, {"instance_id": 1})
    proxy.retarget = mock.AsyncMock()

    assert asyncio.run(proxy_mod._maybe_retarget(proxy)) is False
    proxy.retarget.assert_not_awaited()
