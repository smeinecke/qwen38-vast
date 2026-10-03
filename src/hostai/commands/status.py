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
from hostai import state as state_mod
from hostai.commands import _common
from hostai.config import Config
from hostai.providers import get_provider
from hostai.state import State


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


def _perf_summary(metrics: Dict[str, float]) -> Optional[str]:
    """Compact throughput summary from llama.cpp /metrics counters.

    Rates are cumulative averages since server start (tokens/seconds totals).
    """
    pred_tok = metrics.get("llamacpp:tokens_predicted_total")
    pred_sec = metrics.get("llamacpp:tokens_predicted_seconds_total")
    if not pred_tok or not pred_sec:
        return None
    parts = [f"decode={pred_tok / pred_sec:.1f} tok/s"]
    prompt_tok = metrics.get("llamacpp:prompt_tokens_total")
    prompt_sec = metrics.get("llamacpp:prompt_seconds_total")
    if prompt_tok and prompt_sec:
        parts.append(f"prompt={prompt_tok / prompt_sec:.1f} tok/s")
    draft = metrics.get("llamacpp:spec_decode_num_draft_tokens_total")
    accepted = metrics.get("llamacpp:spec_decode_num_accepted_tokens_total")
    if draft:
        parts.append(f"draft-accept={100.0 * (accepted or 0) / draft:.0f}%")
    return " | ".join(parts)


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

    # The local hostai proxy in tokenized-only mode is plain HTTP.
    scheme = "http" if (state.unsecure or config.proxy.tokenized_only) else "https"
    if state.local_port:
        api_url = f"{scheme}://127.0.0.1:{state.local_port}"
    elif config.proxy.tokenized_only:
        proxy_sock = config.proxy.socket_path or str(state.state_file.parent / "proxy.sock")
        api_url = f"unix:{proxy_sock}"
    else:
        api_url = "not assigned"

    table.add_row("Instance ID", str(state.instance_id))
    instance_name = state.data.get("instance_name") or state_mod.instance_name_for_state_file(state.state_file)
    table.add_row("Name", instance_name)
    table.add_row("Profile", state.profile)
    table.add_row("Image", state.image)
    table.add_row("GPU", state.gpu)
    table.add_row("Status", instance_status)
    table.add_row("Cost ($/h)", f"{state.dph:.4f}")
    table.add_row("Context", str(state.ctx_size))
    table.add_row("SSH", state.ssh_url or "not published")
    table.add_row("Local port", str(state.local_port) if state.local_port else "-")
    table.add_row("API URL", api_url)
    table.add_row("Tunnel healthy", str(tunnel_healthy))
    table.add_row("API healthy", str(api_healthy))

    perf = _perf_summary(metrics)
    if perf:
        table.add_row("Perf (avg)", perf)

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
        table.add_row("Elapsed", utils.format_duration(elapsed))
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


def _status_overview(config: Config, states: Dict[str, Path]) -> None:
    """Render one row per tracked instance (provider status when reachable)."""
    console = Console()
    table = Table(title="hostai instances", show_header=True, header_style="bold")
    for col in ("Name", "ID", "Profile", "GPU", "Status", "$/h", "Port", "Elapsed", "Est. cost"):
        table.add_column(col)

    provider = None
    for name, sf in states.items():
        state = State.load(sf)
        remote_status = state.status
        if provider is None and state.instance_id:
            try:
                provider = _provider(config)
            except Exception:
                provider = False  # type: ignore[assignment]
        if provider and state.instance_id:
            try:
                inst = provider.get_instance(state.instance_id)  # type: ignore[union-attr]
                if inst:
                    remote_status = inst.get("actual_status") or inst.get("status") or "unknown"
                else:
                    remote_status = "gone"
            except Exception:
                pass
        elapsed_s = utils.format_duration(max(0, utils.now_epoch() - state.started_epoch)) if state.started_epoch else "-"
        cost = f"${utils.format_cost(max(0, utils.now_epoch() - state.started_epoch), state.dph):.4f}" if state.started_epoch else "-"
        table.add_row(
            name,
            str(state.instance_id or "-"),
            state.profile,
            state.gpu,
            str(remote_status),
            f"{state.dph:.4f}",
            str(state.local_port or "-"),
            elapsed_s,
            cost,
        )
    console.print(table)
    click.echo("\nUse 'hostai status --name <name>' for details, '--logs' to tail logs, 'hostai down --name <name>' to stop.")


@click.command("status", help="Show instance status (all instances when several are tracked).")
@_common.instance_option
@click.option("--logs", is_flag=True, help="Tail the remote llama-server log.")
@click.option("--lines", type=int, default=100, help="Number of log lines to show.")
@click.option("--follow", is_flag=True, help="Follow the log stream (uses local ssh client).")
@click.option("--no-save", is_flag=True, help="Do not append --logs output to run_dir/server-live.log.")
@click.pass_obj
def cmd_status(config: Config, instance_name: Optional[str], logs: bool, lines: int, follow: bool, no_save: bool) -> None:
    states = state_mod.find_instance_states(config.root_dir)

    if not states:
        click.echo("No local hostai Vast state found.")
        return

    if instance_name is None and len(states) > 1:
        if logs:
            raise click.ClickException(
                f"multiple hostai instances are tracked ({', '.join(states)}); use --name <name|id> with --logs"
            )
        _status_overview(config, states)
        return

    _name, state = _common.resolve_state(config, instance_name)
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
