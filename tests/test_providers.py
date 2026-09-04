"""Tests for the provider abstraction and its built-in implementations."""

import shutil
from unittest import mock

import pytest

from hostai.config import Config
from hostai.providers import FakeProvider, LocalProvider, VastProvider, get_provider


def _config(project_dir, **overrides):
    from hostai.state import state_dir

    state_dir(project_dir).mkdir(parents=True, exist_ok=True)
    cfg = Config(
        root_dir=project_dir,
        config_file=project_dir / "hostai.toml",
        env_file=project_dir / ".env",
    )
    for k, v in overrides.items():
        setattr(cfg.provider, k, v)
    return cfg


def test_get_provider_selects_vast_by_default(config):
    config.secrets["VAST_API_KEY"] = "test-key"
    provider = get_provider(config)
    assert provider.name == "vast"
    assert isinstance(provider, VastProvider)


def test_get_provider_selects_local_via_env(config, monkeypatch):
    monkeypatch.setenv("HOSTAI_PROVIDER", "local")
    config.provider.backend = "local"
    if not shutil.which("docker"):
        pytest.skip("docker not available")
    provider = get_provider(config)
    assert provider.name == "local"
    assert isinstance(provider, LocalProvider)


def test_get_provider_unknown_raises(config):
    config.provider.backend = "does-not-exist"
    with pytest.raises(Exception, match="unknown provider backend"):
        get_provider(config)


def test_fake_provider_search_offers():
    from hostai.config import Config

    cfg = Config(root_dir=Config().root_dir)
    provider = FakeProvider(cfg)
    offers = provider.search_offers("gpu_name=RTX_4090")
    assert len(offers) == 3
    assert all(o["dph_total"] == 0.0 for o in offers)


def test_fake_provider_lifecycle():
    from hostai.config import Config

    cfg = Config(root_dir=Config().root_dir)
    provider = FakeProvider(cfg)
    offers = provider.search_offers("v100")
    v100 = next(o for o in offers if "V100" in o["gpu_name"])

    created = provider.create_instance(
        v100["id"],
        image="test",
        disk=35,
        env={},
    )
    instance_id = created["instance_id"]

    assert provider.get_instance(instance_id)["actual_status"] == "loading"

    provider.start_instance(instance_id)
    assert provider.get_instance(instance_id)["actual_status"] == "running"

    provider.stop_instance(instance_id)
    assert provider.get_instance(instance_id)["actual_status"] == "stopped"

    logs = provider.get_logs(instance_id)
    assert logs and "fake" in logs

    provider.destroy_instance(instance_id)
    assert provider.get_instance(instance_id) is None


def test_local_provider_offers_match_query(project_dir):
    if not shutil.which("docker"):
        pytest.skip("docker not available")
    config = _config(project_dir)
    provider = LocalProvider(config)
    v100_offers = provider.search_offers('gpu_name in ["Tesla V100"]')
    assert len(v100_offers) == 1
    assert v100_offers[0]["gpu_name"] == "Tesla V100"

    all_offers = provider.search_offers("gpu_name=RTX_4090")
    assert len(all_offers) == 1
    assert all_offers[0]["gpu_name"] == "RTX 4090"


def test_local_provider_filters_invalid_env_keys(project_dir):
    if not shutil.which("docker"):
        pytest.skip("docker not available")
    config = _config(project_dir)
    provider = LocalProvider(config)
    assert provider._is_valid_env_key("VALID_KEY")
    assert not provider._is_valid_env_key("-p 22:22")
    assert not provider._is_valid_env_key("1INVALID")


def test_local_provider_respects_configured_image(project_dir):
    if not shutil.which("docker"):
        pytest.skip("docker not available")
    config = _config(project_dir, local_image="hostai-test:latest")
    provider = LocalProvider(config)
    assert provider._resolve_image("ghcr.io/smeinecke/qwen38-vast:v100") == "hostai-test:latest"
    assert provider._resolve_image("ghcr.io/smeinecke/qwen38-vast") == "hostai-test:latest"


def test_local_provider_resolve_image_with_unconfigured(project_dir):
    if not shutil.which("docker"):
        pytest.skip("docker not available")
    config = _config(project_dir)
    # Falls back to the production image when nothing is configured and no env.
    provider = LocalProvider(config)
    with mock.patch.dict("os.environ", {}, clear=True):
        resolved = provider._resolve_image("ghcr.io/smeinecke/qwen38-vast:v100")
    assert resolved == "ghcr.io/smeinecke/qwen38-vast:v100"
