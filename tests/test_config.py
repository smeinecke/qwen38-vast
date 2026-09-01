"""Tests for hostai.config loading and helpers."""

from pathlib import Path
from unittest import mock

import pytest

from hostai.config import Config, image_for_profile, load_config


def test_load_config_from_toml(project_dir):
    (project_dir / "hostai.toml").write_text(
        "[hostai]\ndefault_profile = \"test\"\n"
        "[market]\nmax_dph = 0.75\n"
        "[model]\nmodel = \"my-model.gguf\"\n"
    )
    cfg = load_config(project_dir)
    assert cfg.hostai.default_profile == "test"
    assert cfg.market.max_dph == 0.75
    assert cfg.model.model == "my-model.gguf"


def test_load_config_from_env_override(project_dir, monkeypatch):
    (project_dir / "hostai.toml").write_text("[hostai]\ndefault_profile = \"test\"\n")
    (project_dir / ".env").write_text("VAST_API_KEY=secret\n")
    monkeypatch.setenv("MAX_DPH", "0.99")
    cfg = load_config(project_dir)
    assert cfg.secrets["VAST_API_KEY"] == "secret"
    assert cfg.market.max_dph == 0.99


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
