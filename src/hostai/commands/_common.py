"""Helpers shared by multiple hostai CLI commands."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import click

from hostai import ssh, vast
from hostai.config import Config
from hostai.state import State


def refresh_ssh_state(config: Config, state: State) -> bool:
    """Refresh SSH endpoint fields from the Vast API."""
    if not state.instance_id:
        return False
    try:
        instance = vast.get_instance(config, state.instance_id)
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


def fetch_llama_commit(ssh_url: Optional[str], known_hosts: Path) -> str:
    """Read the llama.cpp commit from /etc/qwen38-build.json on the remote host."""
    if not ssh_url:
        return "unknown"
    res = ssh.run_remote(
        ssh_url,
        "cat /etc/qwen38-build.json 2>/dev/null || true",
        known_hosts=known_hosts,
        timeout=30,
    )
    if res.returncode != 0:
        return "unknown"
    try:
        data = json.loads(res.stdout or "{}")
        commit = data.get("llama_cpp_commit", "unknown")
        if not re.match(r"^[a-f0-9]+$", str(commit)) and commit != "unknown":
            return "unknown"
        return str(commit)
    except Exception:
        return "unknown"


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
