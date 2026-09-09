"""Tests for hostai.commands.up helpers."""

import base64
from pathlib import Path
from unittest import mock

import click
import pytest
from click.testing import CliRunner

from hostai.commands import up
from hostai.state import State


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
    with mock.patch("hostai.commands.up.ssh.run_remote", return_value=mock.Mock(returncode=0, stdout=str(64 * 1024 * 1024 * 1024))):
        assert up._shm_preflight("ssh://root@h:22", config, Path("/tmp/kh"), 30) == 0


def test_shm_preflight_insufficient_space(config, project_dir):
    config.cache.use_shm = True
    with mock.patch("hostai.commands.up.ssh.run_remote", return_value=mock.Mock(returncode=0, stdout=str(10 * 1024 * 1024 * 1024))):
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
    return profile


def _make_image_mock():
    image = mock.Mock()
    image.cuda_arch = "sm_80"
    image.image_tag = "test"
    return image


def test_cmd_up_dry_run(config, project_dir):
    config.hostai.default_profile = "test"
    runner = CliRunner()
    with mock.patch("hostai.commands.up._resolve_client_port", return_value=18080):
        with mock.patch("hostai.commands.up._resolve_profile", return_value=(mock.Mock(), _make_profile_mock(), _make_image_mock())):
            with mock.patch("hostai.commands.up.image_for_profile", return_value="ghcr.io/test"):
                with mock.patch("hostai.commands.up.market.resolved_disk_gb", return_value=35):
                    with mock.patch("hostai.commands.up.market.build_search_query", return_value=("query", 1.0)):
                        with mock.patch("hostai.commands.up.market.select_offer", return_value={"id": 1, "dph_total": 0.5, "gpu_name": "A100"}):
                            with mock.patch("hostai.commands.up.market.offer_summary", return_value="summary"):
                                provider = mock.Mock()
                                provider.name = "vast"
                                with mock.patch("hostai.commands.up._provider", return_value=provider):
                                    with mock.patch("hostai.commands.up._do_fresh_core") as core:
                                        result = runner.invoke(up.cmd_up, ["--dry-run"], obj=config)
    assert result.exit_code == 0
    provider.create_instance.assert_not_called()
    core.assert_not_called()


def test_cmd_up_fresh(config, project_dir):
    config.hostai.default_profile = "test"
    runner = CliRunner()
    with mock.patch("hostai.commands.up._resolve_client_port", return_value=18080):
        with mock.patch("hostai.commands.up._resolve_profile", return_value=(mock.Mock(), _make_profile_mock(), _make_image_mock())):
            with mock.patch("hostai.commands.up.image_for_profile", return_value="ghcr.io/test"):
                with mock.patch("hostai.commands.up.market.resolved_disk_gb", return_value=35):
                    with mock.patch("hostai.commands.up.market.build_search_query", return_value=("query", 1.0)):
                        with mock.patch("hostai.commands.up.market.select_offer", return_value={"id": 1, "dph_total": 0.5, "gpu_name": "A100"}):
                            with mock.patch("hostai.commands.up.market.offer_summary", return_value="summary"):
                                provider = mock.Mock()
                                provider.name = "vast"
                                provider.create_instance.return_value = {"new_contract": 123}
                                with mock.patch("hostai.commands.up._provider", return_value=provider):
                                    with mock.patch("hostai.commands.up.cache._default_local_dir", return_value="/tmp/cache"):
                                        with mock.patch("hostai.commands.up.utils.make_run_id", return_value="run-123"):
                                            with mock.patch("hostai.commands.up.utils.make_api_key", return_value="api-key"):
                                                with mock.patch("hostai.commands.up._do_fresh_core") as core:
                                                    result = runner.invoke(up.cmd_up, [], obj=config)
    assert result.exit_code == 0
    assert provider.create_instance.called
    assert core.called
