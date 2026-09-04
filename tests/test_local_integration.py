"""End-to-end local provider lifecycle test.

This test uses a real Docker container and the LocalProvider.  It is gated on:
  * Docker being installed and running
  * The hostai-test integration image being available

The image can be built with:
    docker build -f tests/integration/Dockerfile.test -t hostai-test:latest .
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from hostai.cli import cli


def _has_docker() -> bool:
    return shutil.which("docker") is not None


def _image_exists(name: str) -> bool:
    if not _has_docker():
        return False
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", name],
            capture_output=True,
            check=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _integration_image() -> str:
    """Return the integration image selected by the environment."""
    return os.environ.get("HOSTAI_LOCAL_IMAGE", "hostai-test:latest")


def _container_count() -> int:
    result = subprocess.run(
        ["docker", "ps", "-a", "-q", "--filter", "label=hostai.provider=local"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return len([line for line in result.stdout.splitlines() if line.strip()])


@pytest.fixture
def project_dir(tmp_path):
    """A temporary project root with a copy of the repo profiles and empty env."""
    repo_root = Path(__file__).parent.parent
    (tmp_path / "hostai.toml").write_text('[hostai]\ndefault_profile = "v100-128k"\n')
    (tmp_path / ".env").write_text("GHCR_IMAGE_BASE=ghcr.io/smeinecke/qwen38-vast\n")
    # Use the repository's built-in profiles for real profile resolution.
    (tmp_path / "profiles.json").write_text((repo_root / "profiles.json").read_text())
    return tmp_path


@pytest.fixture
def local_env():
    env = dict(os.environ)
    env["HOSTAI_PROVIDER"] = "local"
    env["HOSTAI_LOCAL_IMAGE"] = os.environ.get("HOSTAI_LOCAL_IMAGE", "hostai-test:latest")
    env["HOSTAI_PROFILES_FILE"] = "profiles.json"
    env["GHCR_IMAGE_BASE"] = "ghcr.io/smeinecke/qwen38-vast"
    return env


@pytest.mark.skipif(not _has_docker(), reason="docker not available")
@pytest.mark.skipif(
    not _image_exists(_integration_image()), reason=f"integration image {_integration_image()} not built"
)
def test_local_up_down_lifecycle(project_dir, local_env, monkeypatch):
    """Run `hostai up` and `hostai down` against the local Docker backend."""
    runner = CliRunner(env=local_env)
    monkeypatch.chdir(project_dir)

    # Ensure a clean slate inside the isolated project directory.
    state_dir = project_dir / ".hostai-vast"
    if state_dir.exists():
        shutil.rmtree(state_dir)
    runs_dir = project_dir / ".hostai-runs"
    if runs_dir.exists():
        shutil.rmtree(runs_dir)

    before = _container_count()
    result = runner.invoke(
        cli,
        ["up", "--unsecure", "--no-cache", "v100-128k"],
        catch_exceptions=False,
        obj=None,
    )
    assert result.exit_code == 0, result.output
    assert "[provider] local" in result.output
    assert "READY" in result.output
    assert "http://127.0.0.1:" in result.output

    # The container should exist.
    assert _container_count() == before + 1

    # API should be healthy through the tunnel.
    api_base = [line for line in result.output.splitlines() if "API:" in line][0].split()[-1]
    import urllib.request

    health_url = api_base.rsplit("/v1", 1)[0] + "/health"
    req = urllib.request.Request(health_url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200

    down_result = runner.invoke(
        cli,
        ["down", "--yes"],
        catch_exceptions=False,
        obj=None,
    )
    assert down_result.exit_code == 0, down_result.output
    assert "destroyed" in down_result.output

    # Container should be gone.
    assert _container_count() == before


@pytest.mark.skipif(not _has_docker(), reason="docker not available")
@pytest.mark.skipif(
    not _image_exists(_integration_image()), reason=f"integration image {_integration_image()} not built"
)
def test_local_up_shm_size_configurable(project_dir):
    """HOSTAI_LOCAL_SHM_SIZE_GB is reflected in the container's --shm-size."""
    from hostai.config import Config
    from hostai.providers import LocalProvider

    cfg = Config(
        root_dir=project_dir,
        config_file=project_dir / "hostai.toml",
        env_file=project_dir / ".env",
    )
    cfg.provider.backend = "local"
    cfg.provider.local_image = _integration_image()
    cfg.provider.local_shm_size_gb = 2

    provider = LocalProvider(cfg)
    extra = provider._build_extra_args(None, 35)
    assert "--shm-size" in extra
    assert "2g" in extra


@pytest.mark.skipif(not _has_docker(), reason="docker not available")
@pytest.mark.skipif(
    not _image_exists(_integration_image()), reason=f"integration image {_integration_image()} not built"
)
def test_local_up_with_small_shm(project_dir, local_env, monkeypatch):
    """A /dev/shm smaller than the cache minimum falls back to disk slot cache."""
    runner = CliRunner(env=local_env)
    monkeypatch.chdir(project_dir)

    (project_dir / ".hostai-vast").mkdir(parents=True, exist_ok=True)
    (project_dir / ".hostai-runs").mkdir(parents=True, exist_ok=True)

    # Pass a small shm size and enable cache use_shm so the preflight trips.
    # Point the cache at the container itself so the cache code path runs far
    # enough to evaluate /dev/shm before the (empty) cache fetch fails.
    env = dict(local_env)
    env["HOSTAI_LOCAL_SHM_SIZE_GB"] = "1"
    env["HOSTAI_SLOT_CACHE_USE_SHM"] = "1"
    env["HOSTAI_SHM_MIN_GB"] = "2"
    env["HOSTAI_SLOT_CACHE_HOST"] = "127.0.0.1"
    env["HOSTAI_SLOT_CACHE_USER"] = "root"
    runner = CliRunner(env=env)

    before = _container_count()
    result = runner.invoke(
        cli,
        ["up", "--unsecure", "v100-128k"],
        catch_exceptions=False,
        obj=None,
    )
    assert result.exit_code == 0, result.output
    assert "[cache] /dev/shm is too small; falling back to disk slot cache" in result.output

    down_result = runner.invoke(cli, ["down", "--yes"], catch_exceptions=False, obj=None)
    assert down_result.exit_code == 0, down_result.output
    assert _container_count() == before


@pytest.mark.skipif(not _has_docker(), reason="docker not available")
@pytest.mark.skipif(
    not _image_exists(_integration_image()), reason=f"integration image {_integration_image()} not built"
)
def test_local_up_socket_delay_regression(project_dir, local_env, monkeypatch):
    """A 15-second socket delay is covered by a 60-second global boot deadline."""
    env = dict(local_env)
    env["START_TIMEOUT"] = "60"
    env["HOSTAI_FAULT_SOCKET_DELAY_SECONDS"] = "15"
    runner = CliRunner(env=env)
    monkeypatch.chdir(project_dir)

    before = _container_count()
    result = runner.invoke(
        cli,
        ["up", "--unsecure", "--no-cache", "v100-128k"],
        catch_exceptions=False,
        obj=None,
    )
    assert result.exit_code == 0, result.output
    assert "[boot:end-to-end] /health OK" in result.output
    assert _container_count() == before + 1

    down_result = runner.invoke(cli, ["down", "--yes"], catch_exceptions=False, obj=None)
    assert down_result.exit_code == 0, down_result.output
    assert _container_count() == before


@pytest.mark.skipif(not _has_docker(), reason="docker not available")
@pytest.mark.skipif(
    not _image_exists(_integration_image()), reason=f"integration image {_integration_image()} not built"
)
def test_local_up_socket_timeout_regression(project_dir, local_env, monkeypatch):
    """A socket delay that exceeds the global deadline fails with a precise stage error and cleans up."""
    env = dict(local_env)
    env["START_TIMEOUT"] = "30"
    env["HOSTAI_FAULT_SOCKET_DELAY_SECONDS"] = "60"
    runner = CliRunner(env=env)
    monkeypatch.chdir(project_dir)

    before = _container_count()
    result = runner.invoke(
        cli,
        ["up", "--unsecure", "--no-cache", "v100-128k"],
        catch_exceptions=False,
        obj=None,
    )
    # The CLI catches provisioning errors and exits non-zero.
    assert result.exit_code != 0
    assert "[boot:end-to-end] timeout" in result.output or "timeout" in result.output.lower()

    # Cleanup should remove the container.
    assert _container_count() == before
