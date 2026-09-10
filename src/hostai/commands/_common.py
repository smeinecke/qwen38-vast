"""Helpers shared by multiple hostai CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from hostai import ssh
from hostai.config import Config
from hostai.providers import get_provider
from hostai.state import State


def resolve_cache_enabled(cache: bool, no_cache: bool, default: bool) -> bool:
    """Resolve mutually exclusive --cache/--no-cache flags against the config default."""
    if cache and no_cache:
        raise click.ClickException("cannot use both --cache and --no-cache")
    if cache:
        return True
    if no_cache:
        return False
    return default


def _provider(config: Config):
    return get_provider(config)


def refresh_ssh_state(config: Config, state: State) -> bool:
    """Refresh SSH endpoint fields from the provider."""
    if not state.instance_id:
        return False
    try:
        instance = _provider(config).get_instance(state.instance_id)
    except Exception:
        return False
    if not instance:
        return False
    endpoint = ssh.resolve_ssh_endpoint(instance)
    if endpoint and endpoint.get("host") and endpoint.get("port"):
        state.ssh_url = str(endpoint["ssh_url"])
        state.set("ssh_host", str(endpoint["host"]))
        state.set("ssh_port", int(endpoint["port"]))
        state.set("ssh_user", str(endpoint["user"]))
        state.save()
        return True
    return False


def stop_remote_model(ssh_url: Optional[str], known_hosts: Path) -> None:
    """Stop the remote llama.cpp server gracefully."""
    if not ssh_url:
        click.echo("[down] no ssh_url; cannot stop model")
        return
    click.echo("[down] stopping llama.cpp server...")
    ssh.run_remote(
        ssh_url,
        "pkill -TERM llama-server 2>/dev/null || true; sleep 2; pkill -KILL llama-server 2>/dev/null || true",
        known_hosts=known_hosts,
        timeout=30,
    )
