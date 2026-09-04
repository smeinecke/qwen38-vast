"""Show the current Vast instance status and live logs."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import click
from rich.console import Console
from rich.table import Table

from hostai import api, ssh, utils
from hostai.commands import _common
from hostai.config import Config
from hostai.providers import get_provider
from hostai.state import State, state_dir


def _provider(config: Config):
    return get_provider(config)

_refresh_ssh_state = _common.refresh_ssh_state


def _tail_logs(state: State, save: bool, lines: int = 100) -> None:
    if not state.ssh_url:
        raise click.ClickException("SSH endpoint not available")
    known_hosts = state.state_file.parent / "known_hosts"
    res = ssh.run_remote(
        state.ssh_url,
        f"tail -n {lines} /var/log/qwen38/server.log 2>/dev/null || true",
        known_hosts=known_hosts,
        timeout=60,
    )
    if res.returncode != 0 and res.stderr:
        click.echo(f"[status] warning: {res.stderr}", err=True)
    output = res.stdout or ""
    click.echo(output)
    if save and state.run_dir:
        run_dir = Path(state.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "server-live.log").open("a", encoding="utf-8") as f:
            f.write(output)
            if not output.endswith("\n"):
                f.write("\n")


def _tail_logs_follow(state: State, save: bool, lines: int = 100) -> None:
    """Stream the remote server log using the local OpenSSH client."""
    if not state.ssh_url:
        raise click.ClickException("SSH endpoint not available")
    if not shutil.which("ssh"):
        raise click.ClickException("local ssh binary not found; cannot follow logs")

    user, host, port = utils.parse_ssh_url(state.ssh_url)
    known_hosts = state.state_file.parent / "known_hosts"
    remote_cmd = f"tail -n {lines} -F /var/log/qwen38/server.log 2>/dev/null"
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-p",
        str(port),
        f"{user}@{host}",
        remote_cmd,
    ]

    run_dir = Path(state.run_dir) if state.run_dir else None
    log_file = run_dir / "server-live.log" if run_dir else None
    if save and log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        for line in proc.stdout or []:
            click.echo(line, nl=False)
            if save and log_file:
                with log_file.open("a", encoding="utf-8") as f:
                    f.write(line)
    except KeyboardInterrupt:
        click.echo("\n[status] log streaming stopped")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _fetch_gpu_snapshot(ssh_url: str, known_hosts: Path) -> Optional[str]:
    res = ssh.run_remote(
        ssh_url,
        "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader 2>/dev/null",
        known_hosts=known_hosts,
        timeout=30,
    )
    if res.returncode == 0 and res.stdout:
        return res.stdout.strip()
    return None


def _print_status(
    config: Config,
    state: State,
    instance: Optional[Dict[str, Any]],
    metrics: Dict[str, float],
    *,
    tunnel_healthy: bool = False,
    api_healthy: bool = False,
) -> None:
    console = Console()
    table = Table(title="hostai status", show_header=True, header_style="bold")
    table.add_column("Key")
    table.add_column("Value")

    instance_status = "unknown"
    if instance:
        instance_status = instance.get("actual_status") or instance.get("status") or "unknown"

    scheme = "http" if state.unsecure else "https"
    api_url = f"{scheme}://127.0.0.1:{state.local_port}"

    table.add_row("Instance ID", str(state.instance_id))
    table.add_row("Profile", state.profile)
    table.add_row("Image", state.image)
    table.add_row("GPU", state.gpu)
    table.add_row("Status", instance_status)
    table.add_row("Cost ($/h)", f"{state.dph:.4f}")
    table.add_row("Context", str(state.ctx_size))
    table.add_row("SSH", state.ssh_url or "not published")
    table.add_row("Local port", str(state.local_port))
    table.add_row("API URL", api_url)
    table.add_row("Tunnel healthy", str(tunnel_healthy))
    table.add_row("API healthy", str(api_healthy))

    if state.slot_cache_enabled:
        cache_session = state.slot_cache_session
        cache_sig = state.data.get("slot_cache_signature", "pending")
        cache_restore = state.data.get("slot_cache_restore", "pending")
        cache_save = state.data.get("slot_cache_save", "not-yet")
        table.add_row(
            "Slot cache",
            f"session={cache_session} | sig={cache_sig} | restore={cache_restore} | save={cache_save}",
        )

    if state.started_epoch:
        elapsed = max(0, utils.now_epoch() - state.started_epoch)
        cost = utils.format_cost(elapsed, state.dph)
        table.add_row("Elapsed", f"{elapsed}s")
        table.add_row("Est. cost", f"${cost:.4f}")

    console.print(table)

    if state.ssh_url:
        known_hosts = state.state_file.parent / "known_hosts"
        snapshot = _fetch_gpu_snapshot(state.ssh_url, known_hosts)
        if snapshot:
            click.echo("[gpu]")
            for line in snapshot.splitlines():
                click.echo(f"  {line}")

    if metrics:
        click.echo("[metrics]")
        for name, value in sorted(metrics.items()):
            click.echo(f"  {name}: {value}")


@click.command("status", help="Show the current instance status.")
@click.option("--logs", is_flag=True, help="Tail the remote llama-server log.")
@click.option("--lines", type=int, default=100, help="Number of log lines to show.")
@click.option("--follow", is_flag=True, help="Follow the log stream (uses local ssh client).")
@click.option("--no-save", is_flag=True, help="Do not append --logs output to run_dir/server-live.log.")
@click.pass_obj
def cmd_status(config: Config, logs: bool, lines: int, follow: bool, no_save: bool) -> None:
    sd = state_dir(config.root_dir)
    state_file = sd / "state.json"

    if not state_file.exists():
        click.echo("No local hostai Vast state found.")
        return

    state = State.load(state_file)
    if not state.instance_id:
        raise click.ClickException("no running instance; run hostai up first")

    if logs:
        _common.refresh_ssh_state(config, state)
        if follow:
            _tail_logs_follow(state, save=not no_save, lines=lines)
        else:
            _tail_logs(state, save=not no_save, lines=lines)
        return

    try:
        instance = _provider(config).get_instance(state.instance_id)
    except Exception:
        instance = None
    if not instance:
        raise click.ClickException(f"instance {state.instance_id} could not be found. Local state may be stale.")

    if _refresh_ssh_state(config, state):
        state.save()

    # status is a read-only report: do not open a new SSH tunnel/port-forward.
    # If a tunnel from a previous up is still healthy, use it for API health/
    # metrics; otherwise leave those fields empty/false.
    metrics: Dict[str, float] = {}
    tunnel_healthy = ssh.is_tunnel_healthy(config, state, timeout=3)
    api_healthy = False
    if tunnel_healthy:
        client = api.LlamaClient(config, state)
        api_healthy = client.health()
        if api_healthy:
            metrics = client.get_metrics()

    _print_status(config, state, instance, metrics, tunnel_healthy=tunnel_healthy, api_healthy=api_healthy)
