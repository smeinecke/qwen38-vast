"""Tests for hostai.config loading and helpers."""

from pathlib import Path

import pytest

from hostai.config import Config, image_for_profile, load_config


def test_load_config_from_toml(project_dir):
    (project_dir / "hostai.toml").write_text(
        '[hostai]\ndefault_profile = "test"\n[market]\nmax_dph = 0.75\n[model]\nmodel = "my-model.gguf"\n'
    )
    cfg = load_config(project_dir)
    assert cfg.hostai.default_profile == "test"
    assert cfg.market.max_dph == 0.75
    assert cfg.model.model == "my-model.gguf"


def test_load_config_from_env_override(project_dir, monkeypatch):
    (project_dir / "hostai.toml").write_text('[hostai]\ndefault_profile = "test"\n')
    (project_dir / ".env").write_text("VAST_API_KEY=secret\n")
    monkeypatch.setenv("MAX_DPH", "0.99")
    cfg = load_config(project_dir)
    assert cfg.secrets["VAST_API_KEY"] == "secret"
    assert cfg.market.max_dph == 0.99


def test_load_config_blocklist(project_dir):
    (project_dir / "hostai.toml").write_text(
        '[hostai]\ndefault_profile = "t"\n'
        "[blocklist]\nmachines = [11, 22]\noffers = [99]\ncountries = [\"cn\"]\n"
    )
    cfg = load_config(project_dir)
    assert cfg.blocklist.machines == [11, 22]
    assert cfg.blocklist.offers == [99]
    assert cfg.blocklist.countries == ["cn"]


def test_load_config_blocklist_env(project_dir, monkeypatch):
    (project_dir / "hostai.toml").write_text('[hostai]\ndefault_profile = "t"\n')
    monkeypatch.setenv("HOSTAI_BLOCKLIST_MACHINES", "5, 7,9")
    monkeypatch.setenv("HOSTAI_BLOCKLIST_COUNTRIES", "de, fr")
    cfg = load_config(project_dir)
    assert cfg.blocklist.machines == [5, 7, 9]
    assert cfg.blocklist.countries == ["de", "fr"]


def test_load_config_blocklist_empty_default(project_dir):
    (project_dir / "hostai.toml").write_text('[hostai]\ndefault_profile = "t"\n')
    cfg = load_config(project_dir)
    assert cfg.blocklist.machines == []
    assert cfg.blocklist.offers == []
    assert cfg.blocklist.countries == []


def test_load_config_missing_project():
    # load_config tolerates a missing project root and uses defaults.
    cfg = load_config(Path("/nonexistent"))
    assert isinstance(cfg, Config)
    assert cfg.hostai.default_profile == "a6000"  # default from HostaiSection


def test_image_for_profile(config):
    config.image.base = "ghcr.io/example/hostai"
    assert image_for_profile(config, "cuda-12-1") == "ghcr.io/example/hostai:cuda-12-1"


def test_image_for_profile_unconfigured(config):
    config.image.base = ""
    with pytest.raises(ValueError, match="image.base is not configured"):
        image_for_profile(config, "cuda")
