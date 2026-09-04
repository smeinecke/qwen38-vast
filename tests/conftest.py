"""Shared fixtures for hostai tests."""

import pytest

from hostai.config import (
    Config,
)
from hostai.state import State


@pytest.fixture
def project_dir(tmp_path):
    """A temporary project root with empty hostai.toml and .env files."""
    (tmp_path / "hostai.toml").write_text('[hostai]\ndefault_profile = "test"\n')
    (tmp_path / ".env").write_text("")
    return tmp_path


@pytest.fixture
def config(project_dir):
    """A fully populated Config object pointing at the temporary project root."""
    cfg = Config(
        root_dir=project_dir,
        config_file=project_dir / "hostai.toml",
        env_file=project_dir / ".env",
    )
    cfg.hostai.default_profile = "test"
    cfg.hostai.profiles_file = "hostai.profiles.toml"
    cfg.market.max_dph = 1.0
    cfg.market.disk_gb = 100
    cfg.model.model = "model.gguf"
    cfg.model.hf_repo = "repo"
    cfg.model.hf_revision = "rev"
    cfg.model.use_fastmtp = True
    cfg.cache.host = "cache.example.com"
    cfg.cache.user = "qwen-cache"
    cfg.cache.port = 22
    cfg.cache.root = "qwen-slot-cache"
    cfg.cache.session = "default"
    cfg.ssh.local_port = 18080
    return cfg


@pytest.fixture
def state(project_dir):
    """An empty State object bound to a temporary state.json."""
    state_file = project_dir / ".hostai-vast" / "state.json"
    return State(state_file)


@pytest.fixture
def running_state(project_dir):
    """A State object that looks like a running instance."""
    state_file = project_dir / ".hostai-vast" / "state.json"
    return State(
        state_file,
        {
            "instance_id": 12345,
            "profile": "test",
            "model": "model.gguf",
            "hf_revision": "rev",
            "ctx_size": 32768,
            "local_port": 18080,
            "api_key": "test-api-key",
            "dph": 0.5,
            "ssh_url": "ssh://root@10.0.0.1:2222",
            "use_fastmtp": True,
            "slot_cache_enabled": True,
            "slot_cache_session": "default",
        },
    )
