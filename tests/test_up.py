"""Tests for hostai.commands.up helpers."""

import base64
import contextlib
from pathlib import Path
from unittest import mock

import click
import pytest
from click.testing import CliRunner

from hostai.commands import up
from hostai.state import State
from hostai.validate import ValidationRecord


def test_now_helpers():
    assert up._now_epoch() > 0
    assert "T" in up._now_rfc()


def test_default_proxy_socket(config, project_dir):
    config.root_dir = project_dir
    assert up._default_proxy_socket(config) == project_dir / ".hostai-vast" / "proxy.sock"


def test_hostai_binary_points_to_executable(config, project_dir):
    bin_path = up._hostai_binary()
    assert bin_path is not None


def test_shm_preflight_disabled(config):
    config.cache.use_shm = False
    assert up._shm_preflight("ssh://root@h:22", config, Path("/tmp/kh"), 30) == 0


def test_shm_preflight_missing_ssh_url(config):
    assert up._shm_preflight(None, config, Path("/tmp/kh"), 30) == 2


def test_shm_preflight_sufficient_space(config, project_dir):
    config.cache.use_shm = True
    with mock.patch(
        "hostai.commands.up.ssh.run_remote", return_value=mock.Mock(returncode=0, stdout=str(64 * 1024 * 1024 * 1024))
    ):
        assert up._shm_preflight("ssh://root@h:22", config, Path("/tmp/kh"), 30) == 0


def test_shm_preflight_insufficient_space(config, project_dir):
    config.cache.use_shm = True
    with mock.patch(
        "hostai.commands.up.ssh.run_remote", return_value=mock.Mock(returncode=0, stdout=str(10 * 1024 * 1024 * 1024))
    ):
        assert up._shm_preflight("ssh://root@h:22", config, Path("/tmp/kh"), 30) == 1


def test_extra_args_adds_shm_size(config):
    config.vast.shm_size_gb = 64
    config.cache.enabled = True
    config.cache.use_shm = True
    assert "--shm-size=64g" in up._extra_args(config)


def test_extra_args_no_cache(config):
    config.vast.shm_size_gb = None
    config.cache.enabled = False
    assert up._extra_args(config, no_cache=True) == ""


def test_env_dict_includes_ssh_public_key(config):
    key = "ssh-ed25519 AAAA test"
    config.secrets["SSH_PUBLIC_KEY"] = key
    profile = mock.Mock(name="p", image="img", cache_ram=None, ctx_checkpoints=None)
    image = mock.Mock(name="img")
    env = up._env_dict(config, profile, image, "model", 32768, "apikey", False, False, "session")
    assert env["HOSTAI_SSH_PUBLIC_KEY_B64"] == base64.b64encode(key.encode()).decode()
    assert env["-p 22:22"] == "1"


def test_env_dict_slot_cache_enabled(config):
    config.cache.enabled = True
    config.cache.host = "cache.example.com"
    config.cache.rclone = False
    profile = mock.Mock(name="p", image="img", cache_ram=None, ctx_checkpoints=None)
    image = mock.Mock(name="img")
    env = up._env_dict(config, profile, image, "model", 32768, "apikey", False, False, "session")
    assert env["HOSTAI_SLOT_CACHE_ENABLED"] == "1"
    assert env["HOSTAI_SLOT_CACHE_MIN_GB"] == "30"


def test_env_dict_default_reasoning_effort(config):
    config.model.reasoning_effort = "medium"
    profile = mock.Mock(name="p", image="img", cache_ram=None, ctx_checkpoints=None)
    image = mock.Mock(name="img")
    env = up._env_dict(config, profile, image, "model", 32768, "apikey", False, False, "session")
    assert env["REASONING_EFFORT"] == "medium"


def test_resolve_client_port_free(config, project_dir):
    config.root_dir = project_dir
    config.ssh.local_port = None
    config.proxy.port = None
    with mock.patch("hostai.utils.port_is_free", return_value=True):
        assert up._resolve_client_port(config) == 18081


def test_resolve_client_port_user_specified_in_use(config):
    with mock.patch("hostai.utils.port_is_free", return_value=False):
        with pytest.raises(Exception):
            up._resolve_client_port(config, user_port=18081)


def test_resolve_client_port_finds_free(config):
    with mock.patch("hostai.utils.port_is_free", side_effect=[False, True]):
        with mock.patch("hostai.utils.find_free_port", return_value=18082):
            assert up._resolve_client_port(config) == 18082


def test_write_env_file(config, project_dir):
    state = State(project_dir / ".hostai-vast" / "state.json")
    state.env_file = project_dir / ".hostai-vast" / "env"
    up._write_env_file(config, state, "http://127.0.0.1:8080/v1", "http://127.0.0.1:8080")
    text = state.env_file.read_text()
    assert "OPENAI_API_KEY" in text
    assert "OPENAI_BASE_URL" in text


def test_cleanup_instance_calls_down(config, project_dir):
    state = State(project_dir / ".hostai-vast" / "state.json")
    state.instance_id = 12345
    with mock.patch("hostai.commands.up._provider") as prov:
        with mock.patch("hostai.commands.up._log"):
            up._cleanup_instance(config, state, "test reason")
    prov.assert_called_once_with(config)


def test_wait_for_ssh_endpoint(config, project_dir, running_state):
    state = State(project_dir / ".hostai-vast" / "state.json")
    state.instance_id = 12345
    inst = {
        "public_ipaddr": "203.0.113.1",
        "ports": {"22/tcp": [{"HostPort": "2222"}]},
        "actual_status": "running",
    }
    provider = mock.Mock()
    provider.get_instance.return_value = inst
    with mock.patch("hostai.commands.up._provider", return_value=provider):
        with mock.patch("hostai.commands.up.time.sleep"):
            with mock.patch("hostai.commands.up.time.monotonic", side_effect=[0, 0.1]):
                up._wait_for_ssh_endpoint(config, state, 10)
    assert state.ssh_url == "ssh://root@203.0.113.1:2222"


def test_wait_for_ssh_endpoint_offline(config, project_dir):
    state = State(project_dir / ".hostai-vast" / "state.json")
    state.instance_id = 12345
    provider = mock.Mock()
    provider.get_instance.return_value = {"actual_status": "exited"}
    with mock.patch("hostai.commands.up._provider", return_value=provider):
        with mock.patch("hostai.commands.up.time.sleep"):
            with mock.patch("hostai.commands.up.time.monotonic", side_effect=[0, 0.1]):
                with pytest.raises(click.ClickException):
                    up._wait_for_ssh_endpoint(config, state, 10)


def test_wait_for_api_ready(config, project_dir):
    state = State(project_dir / ".hostai-vast" / "state.json")
    client = mock.Mock()
    client.health.return_value = True
    with mock.patch("hostai.commands.up.time.sleep"):
        with mock.patch("hostai.commands.up._log"):
            up._wait_for_api(config, state, 10, client)


def test_capture_disk_telemetry(config, project_dir):
    state = State(project_dir / ".hostai-vast" / "state.json")
    state.instance_id = 12345
    state.ssh_url = "ssh://root@h:22"
    state.disk_gb = 35
    run_dir = project_dir / "run-123"
    run_dir.mkdir()
    state.run_dir = run_dir
    res1 = mock.Mock(returncode=0, stdout='{"stage":"a"}\n')
    res2 = mock.Mock(returncode=0, stdout='{"stage":"up-final"}\n')
    with mock.patch("hostai.commands.up.ssh.run_remote", side_effect=[res1, res2]):
        telemetry = up._capture_disk_telemetry(config, state, project_dir / "kh")
    assert telemetry is not None
    assert telemetry["disk_gb"] == 35


def test_start_proxy(config, project_dir):
    state = State(project_dir / ".hostai-vast" / "state.json")
    config.proxy.tokenized_only = True
    config.proxy.port = 0
    config.ssh.local_port = None
    proc = mock.Mock()
    proc.pid = 12345
    with mock.patch("hostai.commands.up._hostai_binary", return_value=Path("/tmp/hostai")):
        with mock.patch("hostai.commands.up.utils.port_is_free", return_value=False):
            with mock.patch("hostai.commands.up.utils.find_free_port", return_value=18083):
                with mock.patch("hostai.commands.up.subprocess.Popen", return_value=proc) as popen:
                    with mock.patch("hostai.commands.up.time.sleep"):
                        with mock.patch("hostai.commands.up.time.monotonic", side_effect=[0, 100]):
                            port = up._start_proxy(config, state, "http")
    assert port == 18083
    assert popen.called


def _make_profile_mock():
    profile = mock.Mock()
    profile.name = "test"
    profile.ctx_size = 32768
    profile.monitor_group = ""
    profile.image = "test-img"
    profile.min_gpu_vram_mb = None
    return profile


def _make_image_mock():
    image = mock.Mock()
    image.cuda_arch = "sm_80"
    image.image_tag = "test"
    return image


def test_cmd_up_validation_errors(config, project_dir):
    config.hostai.default_profile = "test"
    runner = CliRunner()
    for args, expected in [
        (["--local-port", "0"], "must be between"),
        (["--max-price", "-1"], "non-negative"),
        (["--bid", "0"], "positive"),
        (["--scoring-mode", "invalid"], "dph, perf, or session"),
    ]:
        result = runner.invoke(up.cmd_up, args, obj=config)
        assert result.exit_code != 0
        assert expected in result.output


def test_cmd_up_no_profile(config, project_dir):
    config.hostai.default_profile = ""
    runner = CliRunner()
    result = runner.invoke(up.cmd_up, [], obj=config)
    assert result.exit_code != 0
    assert "no profile" in result.output


def test_cmd_up_scoring_and_keep_flags(config, project_dir):
    config.hostai.default_profile = "test"
    runner = CliRunner()
    with mock.patch("hostai.commands.up._resolve_client_port", return_value=18080):
        with mock.patch(
            "hostai.commands.up._resolve_profile", return_value=(mock.Mock(), _make_profile_mock(), _make_image_mock())
        ):
            with mock.patch("hostai.commands.up.image_for_profile", return_value="ghcr.io/test"):
                with mock.patch("hostai.commands.up.market.resolved_disk_gb", return_value=35):
                    with mock.patch("hostai.commands.up.market.build_search_query", return_value=("query", 1.0)):
                        with mock.patch(
                            "hostai.commands.up.market.select_offer",
                            return_value={"id": 1, "dph_total": 0.5, "gpu_name": "A100"},
                        ):
                            with mock.patch("hostai.commands.up.market.offer_summary", return_value="summary"):
                                provider = mock.Mock()
                                provider.name = "vast"
                                with mock.patch("hostai.commands.up._provider", return_value=provider):
                                    with mock.patch("hostai.commands.up._do_fresh_core"):
                                        result = runner.invoke(
                                            up.cmd_up,
                                            ["--scoring-mode", "perf", "--keep-on-failure", "--dry-run"],
                                            obj=config,
                                        )
    assert result.exit_code == 0
    assert config.market.scoring_mode == "perf"
    assert config.vast.keep_on_failure is True


def test_cmd_up_restart(config, project_dir):
    config.hostai.default_profile = "test"
    runner = CliRunner()
    with mock.patch("hostai.commands.up._do_restart") as restart:
        result = runner.invoke(up.cmd_up, ["--restart"], obj=config)
    assert result.exit_code == 0
    restart.assert_called_once()


def test_cmd_up_dry_run(config, project_dir):
    config.hostai.default_profile = "test"
    runner = CliRunner()
    with mock.patch("hostai.commands.up._resolve_client_port", return_value=18080):
        with mock.patch(
            "hostai.commands.up._resolve_profile", return_value=(mock.Mock(), _make_profile_mock(), _make_image_mock())
        ):
            with mock.patch("hostai.commands.up.image_for_profile", return_value="ghcr.io/test"):
                with mock.patch("hostai.commands.up.market.resolved_disk_gb", return_value=35):
                    with mock.patch("hostai.commands.up.market.build_search_query", return_value=("query", 1.0)):
                        with mock.patch(
                            "hostai.commands.up.market.select_offer",
                            return_value={"id": 1, "dph_total": 0.5, "gpu_name": "A100"},
                        ):
                            with mock.patch("hostai.commands.up.market.offer_summary", return_value="summary"):
                                provider = mock.Mock()
                                provider.name = "vast"
                                with mock.patch("hostai.commands.up._provider", return_value=provider):
                                    with mock.patch("hostai.commands.up._do_fresh_core") as core:
                                        result = runner.invoke(up.cmd_up, ["--dry-run"], obj=config)
    assert result.exit_code == 0
    provider.create_instance.assert_not_called()
    core.assert_not_called()


def test_do_fresh_core_tokenized(config, project_dir):
    state = _make_state(project_dir)
    config.proxy.tokenized_only = True
    config.cache.enabled = False
    client = mock.Mock()
    with mock.patch("hostai.commands.up._wait_for_ssh_endpoint"):
        with mock.patch("hostai.commands.up.ssh.wait_for_ssh", return_value=True):
            with mock.patch("hostai.commands.up.ssh.run_remote", return_value=mock.Mock(returncode=0)):
                with mock.patch("hostai.commands.up._start_proxy", return_value=18083):
                    with mock.patch("hostai.commands.up.State.load", return_value=state):
                        with mock.patch("hostai.commands.up._write_env_file") as write_env:
                            with mock.patch("hostai.commands.up.LlamaClient", return_value=client):
                                with mock.patch("hostai.commands.up._wait_for_api"):
                                    with mock.patch("hostai.commands.up._capture_disk_telemetry", return_value=None):
                                        with mock.patch("hostai.commands.up.maybe_start_watchdog"):
                                            with mock.patch("hostai.commands.up.maybe_start_monitor"):
                                                up._do_fresh_core(
                                                    config,
                                                    state,
                                                    _make_image_mock(),
                                                    no_cache=True,
                                                    abort_if_shm_too_small=False,
                                                )
    assert state.status == "running"
    write_env.assert_called_once()


def test_do_fresh_core_tokenized_proxy_fails(config, project_dir):
    state = _make_state(project_dir)
    config.proxy.tokenized_only = True
    config.cache.enabled = False
    with mock.patch("hostai.commands.up._wait_for_ssh_endpoint"):
        with mock.patch("hostai.commands.up.ssh.wait_for_ssh", return_value=True):
            with mock.patch("hostai.commands.up.ssh.run_remote", return_value=mock.Mock(returncode=0)):
                with mock.patch("hostai.commands.up._start_proxy", return_value=0):
                    with pytest.raises(click.ClickException, match="proxy.*failed"):
                        up._do_fresh_core(
                            config, state, _make_image_mock(), no_cache=True, abort_if_shm_too_small=False
                        )


def test_cmd_up_fresh(config, project_dir):
    config.hostai.default_profile = "test"
    runner = CliRunner()
    with mock.patch("hostai.commands.up._resolve_client_port", return_value=18080):
        with mock.patch(
            "hostai.commands.up._resolve_profile", return_value=(mock.Mock(), _make_profile_mock(), _make_image_mock())
        ):
            with mock.patch("hostai.commands.up.image_for_profile", return_value="ghcr.io/test"):
                with mock.patch("hostai.commands.up.market.resolved_disk_gb", return_value=35):
                    with mock.patch("hostai.commands.up.market.build_search_query", return_value=("query", 1.0)):
                        with mock.patch(
                            "hostai.commands.up.market.select_offer",
                            return_value={"id": 1, "dph_total": 0.5, "gpu_name": "A100"},
                        ):
                            with mock.patch("hostai.commands.up.market.offer_summary", return_value="summary"):
                                provider = mock.Mock()
                                provider.name = "vast"
                                provider.create_instance.return_value = {"new_contract": 123}
                                with mock.patch("hostai.commands.up._provider", return_value=provider):
                                    with mock.patch(
                                        "hostai.commands.up.cache._default_local_dir", return_value="/tmp/cache"
                                    ):
                                        with mock.patch("hostai.commands.up.utils.make_run_id", return_value="run-123"):
                                            with mock.patch(
                                                "hostai.commands.up.utils.make_api_key", return_value="api-key"
                                            ):
                                                with mock.patch("hostai.commands.up._do_fresh_core") as core:
                                                    result = runner.invoke(up.cmd_up, [], obj=config)
    assert result.exit_code == 0
    assert provider.create_instance.called
    assert core.called


def _make_state(project_dir, **overrides):
    state = State(project_dir / ".hostai-vast" / "state.json")
    state.instance_id = 12345
    state.ssh_url = "ssh://root@203.0.113.1:2222"
    state.unsecure = True
    state.slot_cache_enabled = False
    state.proxy_tokenized_only = False
    state.started_epoch = up._now_epoch()
    state.dph = 0.5
    state.profile = "test"
    state.image = "ghcr.io/test"
    state.gpu = "A100"
    state.ctx_size = 32768
    state.local_port = 18080
    state.run_dir = project_dir / "run-123"
    state.run_dir.mkdir()
    for k, v in overrides.items():
        setattr(state, k, v)
    state.save()
    return state


def _fresh_core_mocks(config, state):
    client = mock.Mock()
    patches = [
        mock.patch("hostai.commands.up._wait_for_ssh_endpoint"),
        mock.patch("hostai.commands.up.ssh.wait_for_ssh", return_value=True),
        mock.patch("hostai.commands.up.ssh.run_remote", return_value=mock.Mock(returncode=0)),
        mock.patch("hostai.commands.up.ssh.ensure_tunnel", return_value=18080),
        mock.patch("hostai.commands.up.State.load", return_value=state),
        mock.patch("hostai.commands.up.LlamaClient", return_value=client),
        mock.patch("hostai.commands.up._wait_for_api"),
        mock.patch("hostai.commands.up._capture_disk_telemetry", return_value=None),
        mock.patch("hostai.commands.up.maybe_start_watchdog"),
        mock.patch("hostai.commands.up.maybe_start_monitor"),
    ]
    stack = contextlib.ExitStack()
    for p in patches:
        stack.enter_context(p)
    write_env = stack.enter_context(mock.patch("hostai.commands.up._write_env_file"))
    return stack, write_env


def test_do_fresh_core_no_cache_unsecure(config, project_dir):
    state = _make_state(project_dir)
    config.proxy.tokenized_only = False
    config.cache.enabled = False
    stack, _ = _fresh_core_mocks(config, state)
    with stack:
        up._do_fresh_core(config, state, _make_image_mock(), no_cache=True, abort_if_shm_too_small=False)
    assert state.status == "running"


def test_do_fresh_core_with_cache_and_tls(config, project_dir):
    state = _make_state(project_dir, slot_cache_enabled=True, unsecure=False)
    config.proxy.tokenized_only = False
    config.cache.enabled = True
    config.cache.use_shm = False
    config.cache.rclone = False
    config.cache.local_dir = "/var/lib/qwen38/slots"
    stack, _ = _fresh_core_mocks(config, state)
    with stack:
        with mock.patch("hostai.commands.up.tls.ensure_local_tls_dir", return_value=Path("/tmp/tls")):
            with mock.patch("hostai.commands.up.tls.generate_cert"):
                with mock.patch("hostai.commands.up.tls.deliver_cert", return_value=True):
                    with mock.patch("hostai.commands.up.cache.validate_cache_config", return_value=True):
                        with mock.patch("hostai.commands.up.cache.install_cache_key_on_vast", return_value=True):
                            with mock.patch("hostai.cache.fetch_llama_commit", return_value="commit"):
                                with mock.patch("hostai.commands.up._shm_preflight", return_value=0):
                                    with mock.patch(
                                        "hostai.commands.up.cache._signature_for_state", return_value="sig"
                                    ):
                                        with mock.patch(
                                            "hostai.commands.up.cache.remote_cache_dir", return_value="remote"
                                        ):
                                            with mock.patch(
                                                "hostai.commands.up._prefetch_slot_cache_to_vast", return_value=False
                                            ):
                                                with mock.patch(
                                                    "hostai.commands.up.cache._default_local_dir",
                                                    return_value="/var/lib/qwen38/slots",
                                                ):
                                                    up._do_fresh_core(
                                                        config,
                                                        state,
                                                        _make_image_mock(),
                                                        no_cache=False,
                                                        abort_if_shm_too_small=False,
                                                    )
    assert state.status == "running"


def test_check_production_validation_non_vast(config):
    config.provider.backend = "local"
    up._check_production_validation(config, allow_unvalidated=False)


def test_check_production_validation_disabled(config):
    config.provider.backend = "vast"
    config.vast.require_production_validation = False
    up._check_production_validation(config, allow_unvalidated=False)


def test_check_production_validation_passes(config, project_dir):
    config.provider.backend = "vast"
    config.vast.require_production_validation = True
    previous = ValidationRecord(
        timestamp="",
        result="ok",
        duration_seconds=0.0,
        git_commit="abc",
        dirty=False,
        image="img",
        image_id="id",
        image_digest="dig",
        profile_hash="h",
        errors=[],
        level="production",
        checks_run=["integration-tests"],
    )
    with mock.patch("hostai.commands.up.load_last_validation", return_value=previous):
        with mock.patch("hostai.commands.up.utils.git_commit", return_value="abc"):
            with mock.patch("hostai.commands.up.utils.is_dirty_tree", return_value=False):
                with mock.patch("hostai.commands.up.utils.file_hash", return_value="h"):
                    with mock.patch("hostai.commands.up._image_info", return_value=("id", "dig")):
                        with mock.patch("hostai.commands.up.compare_validations", return_value=[]):
                            record = up._check_production_validation(config, allow_unvalidated=False)
    assert record is not None


def test_check_production_validation_drift_raises(config):
    config.provider.backend = "vast"
    config.vast.require_production_validation = True
    previous = ValidationRecord(
        timestamp="",
        result="ok",
        duration_seconds=0.0,
        git_commit="abc",
        dirty=False,
        image="img",
        image_id="id",
        image_digest="dig",
        profile_hash="h",
        errors=[],
        level="production",
        checks_run=["integration-tests"],
    )
    with mock.patch("hostai.commands.up.load_last_validation", return_value=previous):
        with mock.patch("hostai.commands.up.utils.git_commit", return_value="def"):
            with mock.patch("hostai.commands.up.utils.is_dirty_tree", return_value=False):
                with mock.patch("hostai.commands.up.utils.file_hash", return_value="h"):
                    with mock.patch("hostai.commands.up._image_info", return_value=("id", "dig")):
                        with mock.patch("hostai.commands.up.compare_validations", return_value=["git commit drift"]):
                            with pytest.raises(click.ClickException):
                                up._check_production_validation(config, allow_unvalidated=False)


def test_do_restart_no_cache_unsecure(config, project_dir):
    state = _make_state(project_dir)
    config.proxy.tokenized_only = False
    config.cache.enabled = False
    client = mock.Mock()
    provider = mock.Mock()
    provider.get_instance.return_value = {"actual_status": "stopped"}
    with mock.patch("hostai.commands.up._check_production_validation"):
        with mock.patch("hostai.commands.up._resolve_client_port", return_value=18080):
            with mock.patch("hostai.commands.up._provider", return_value=provider):
                with mock.patch("hostai.commands.up._wait_for_ssh_endpoint"):
                    with mock.patch("hostai.commands.up.ssh.wait_for_ssh", return_value=True):
                        with mock.patch("hostai.commands.up.ssh.ensure_tunnel", return_value=18080):
                            with mock.patch("hostai.commands.up.State.load", return_value=state):
                                with mock.patch("hostai.commands.up._write_env_file"):
                                    with mock.patch("hostai.commands.up.LlamaClient", return_value=client):
                                        with mock.patch("hostai.commands.up._wait_for_api"):
                                            with mock.patch("hostai.commands.up.maybe_start_watchdog"):
                                                with mock.patch("hostai.commands.up.maybe_start_monitor"):
                                                    up._do_restart(config, "test", None, True, no_cache=True)
    assert state.status == "running"
    provider.start_instance.assert_called_once()


def test_do_restart_with_cache_and_tls(config, project_dir):
    state = _make_state(project_dir, slot_cache_enabled=True, unsecure=False)
    state.data["llama_cpp_commit"] = "commit123"
    config.proxy.tokenized_only = False
    config.cache.enabled = True
    config.cache.use_shm = False
    config.cache.rclone = False
    client = mock.Mock()
    client.slot_restore.return_value = True
    provider = mock.Mock()
    provider.get_instance.return_value = {"actual_status": "running"}
    with mock.patch("hostai.commands.up._check_production_validation"):
        with mock.patch("hostai.commands.up._resolve_client_port", return_value=18080):
            with mock.patch("hostai.commands.up._provider", return_value=provider):
                with mock.patch("hostai.commands.up._wait_for_ssh_endpoint"):
                    with mock.patch("hostai.commands.up.ssh.wait_for_ssh", return_value=True):
                        with mock.patch("hostai.commands.up.ssh.ensure_tunnel", return_value=18080):
                            with mock.patch("hostai.commands.up.State.load", return_value=state):
                                with mock.patch(
                                    "hostai.commands.up.tls.ensure_local_tls_dir", return_value=Path("/tmp/tls")
                                ):
                                    with mock.patch("hostai.commands.up.Path.exists", return_value=True):
                                        with mock.patch("hostai.commands.up.tls.deliver_cert", return_value=True):
                                            with mock.patch(
                                                "hostai.commands.up.cache.validate_cache_config", return_value=True
                                            ):
                                                with mock.patch(
                                                    "hostai.commands.up.cache.install_cache_key_on_vast",
                                                    return_value=True,
                                                ):
                                                    with mock.patch(
                                                        "hostai.commands.up.cache._signature_for_state",
                                                        return_value="sig",
                                                    ):
                                                        with mock.patch(
                                                            "hostai.commands.up.cache.remote_cache_dir",
                                                            return_value="remote",
                                                        ):
                                                            with mock.patch(
                                                                "hostai.commands.up._prefetch_slot_cache_to_vast",
                                                                return_value=False,
                                                            ):
                                                                with mock.patch("hostai.commands.up._write_env_file"):
                                                                    with mock.patch(
                                                                        "hostai.commands.up.LlamaClient",
                                                                        return_value=client,
                                                                    ):
                                                                        with mock.patch(
                                                                            "hostai.commands.up._wait_for_api"
                                                                        ):
                                                                            with mock.patch(
                                                                                "hostai.commands.up.maybe_start_watchdog"
                                                                            ):
                                                                                with mock.patch(
                                                                                    "hostai.commands.up.maybe_start_monitor"
                                                                                ):
                                                                                    up._do_restart(
                                                                                        config,
                                                                                        "test",
                                                                                        None,
                                                                                        False,
                                                                                        no_cache=False,
                                                                                    )
    assert state.status == "running"


def test_parse_nvidia_smi_vram_csv():
    out = "NVIDIA CMP 170HX, 65536\nNVIDIA A100-SXM4-40GB, 40960 MiB"
    gpus = up._parse_nvidia_smi_vram(out)
    assert gpus == [("NVIDIA CMP 170HX", 65536), ("NVIDIA A100-SXM4-40GB", 40960)]


def test_parse_nvidia_smi_vram_table():
    table = """+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.14              Driver Version: 550.54.14      CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA CMP 170HX                 |   00000000:00:00.0 Off |                    0 |
| N/A   45C    P0             35W /  300W |       0MiB /  65536MiB |      0%      Default |
|   1  NVIDIA A100-SXM4-40GB            |   00000000:00:00.0 Off |                    0 |
| N/A   45C    P0             35W /  300W |       0MiB /  40960MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+"""
    gpus = up._parse_nvidia_smi_vram(table)
    assert ("NVIDIA CMP 170HX", 65536) in gpus
    assert ("NVIDIA A100-SXM4-40GB", 40960) in gpus


def test_gpu_vram_preflight_accepts_cmp(config, project_dir):
    with mock.patch(
        "hostai.commands.up.ssh.run_remote", return_value=mock.Mock(returncode=0, stdout="NVIDIA CMP 170HX, 65536\n")
    ):
        rc = up._gpu_vram_preflight("ssh://root@h:22", project_dir / "kh", config, "cmp170hx-256k", 60000)
    assert rc == 0


def test_gpu_vram_preflight_rejects_locked_cmp(config, project_dir):
    with mock.patch(
        "hostai.commands.up.ssh.run_remote", return_value=mock.Mock(returncode=0, stdout="NVIDIA CMP 170HX, 8192\n")
    ):
        rc = up._gpu_vram_preflight("ssh://root@h:22", project_dir / "kh", config, "cmp170hx-256k", 60000)
    assert rc == 1


def test_gpu_vram_preflight_accepts_gb10(config, project_dir):
    with mock.patch(
        "hostai.commands.up.ssh.run_remote", return_value=mock.Mock(returncode=0, stdout="NVIDIA GB10, 121856\n")
    ):
        rc = up._gpu_vram_preflight("ssh://root@h:22", project_dir / "kh", config, "gb10-256k", 115000)
    assert rc == 0


def test_gpu_vram_preflight_rejects_low_memory_gb10(config, project_dir):
    with mock.patch(
        "hostai.commands.up.ssh.run_remote", return_value=mock.Mock(returncode=0, stdout="NVIDIA GB10, 96000\n")
    ):
        rc = up._gpu_vram_preflight("ssh://root@h:22", project_dir / "kh", config, "gb10-256k", 115000)
    assert rc == 1


def test_gpu_vram_preflight_rejects_cmp_for_gb10_requirement(config, project_dir):
    with mock.patch(
        "hostai.commands.up.ssh.run_remote", return_value=mock.Mock(returncode=0, stdout="NVIDIA CMP 170HX, 65536\n")
    ):
        rc = up._gpu_vram_preflight("ssh://root@h:22", project_dir / "kh", config, "gb10-256k", 115000)
    assert rc == 1


def test_gpu_vram_preflight_skips_without_requirement(config, project_dir):
    with mock.patch("hostai.commands.up.ssh.run_remote") as run:
        rc = up._gpu_vram_preflight("ssh://root@h:22", project_dir / "kh", config, "a6000", None)
    assert rc == 0
    run.assert_not_called()


def test_cpu_arch_preflight_captures_arch(config, project_dir):
    state = State(project_dir / "state.json")
    with mock.patch("hostai.commands.up.ssh.run_remote", return_value=mock.Mock(returncode=0, stdout="aarch64\n")):
        up._cpu_arch_preflight("ssh://root@h:22", project_dir / "kh", config, state=state)
    assert state.data.get("cpu_arch") == "aarch64"


def test_cpu_arch_preflight_is_non_fatal_on_failure(config, project_dir):
    state = State(project_dir / "state.json")
    with mock.patch("hostai.commands.up.ssh.run_remote", return_value=mock.Mock(returncode=1, stdout="")):
        up._cpu_arch_preflight("ssh://root@h:22", project_dir / "kh", config, state=state)
    assert state.data.get("cpu_arch") is None


def test_new_profiles_resolve():
    repo_root = Path(__file__).parent.parent
    profiles = up.Profiles.from_file(repo_root / "profiles.json")

    a100 = profiles.resolve_profile("a100-128k")
    assert a100 is not None
    assert a100.image == "ga100"
    assert a100.ctx_size == 131072
    assert a100.min_gpu_vram_mb == 39000

    cmp = profiles.resolve_profile("cmp170hx-256k")
    assert cmp is not None
    assert cmp.image == "ga100"
    assert cmp.ctx_size == 262144
    assert cmp.min_gpu_vram_mb == 60000

    gb10 = profiles.resolve_profile("gb10-256k")
    assert gb10 is not None
    assert gb10.image == "gb10"
    assert gb10.ctx_size == 262144
    assert gb10.min_gpu_vram_mb == 115000

    gb10_128 = profiles.resolve_profile("gb10-128k")
    assert gb10_128 is not None
    assert gb10_128.image == "gb10"
    assert gb10_128.ctx_size == 131072
    assert gb10_128.min_gpu_vram_mb == 115000

    ga100 = profiles.image_by_name("ga100")
    assert ga100 is not None
    assert ga100.cuda_arch == "80"
    assert ga100.image_tag == "ga100"

    gb10_img = profiles.image_by_name("gb10")
    assert gb10_img is not None
    assert gb10_img.cuda_arch == "121"
    assert gb10_img.image_tag == "gb10"
    assert gb10_img.platform == "linux/arm64"
    assert "13.3.1" in gb10_img.builder_base
    assert "13.3.1" in gb10_img.runtime_base
