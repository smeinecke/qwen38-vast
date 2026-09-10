"""Stop, save cache, and destroy/pause the current Vast instance."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, Optional

import click
import requests

from hostai import api, cache, ssh, utils
from hostai.commands import _common
from hostai.commands.monitor import stop_monitor
from hostai.config import Config
from hostai.providers import get_provider
from hostai.state import State, init_run_dir, runs_dir, state_dir

_refresh_ssh_state = _common.refresh_ssh_state
_stop_remote_model = _common.stop_remote_model


def _provider(config: Config):
    return get_provider(config)


def _archive_session(config: Config, state: State, run_dir: Path, no_archive: bool) -> None:
    """Collect final telemetry before destroy/pause."""
    if no_archive or not run_dir:
        return

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"[archive] saving telemetry to {run_dir}")

    # Sanitized state copy.
    state.save_metadata(run_dir, status="archiving")

    try:
        instance = _provider(config).get_instance(state.instance_id) if state.instance_id else None
        if instance:
            (run_dir / "vast-final.json").write_text(json.dumps(instance, indent=2, default=str))
    except Exception:
        pass

    try:
        client = api.LlamaClient(config, state)
        if client.health():
            response = requests.get(
                f"{client.base_url}/metrics",
                headers=client._headers,
                verify=client._verify,
                timeout=(2, 10),
            )
            if response.status_code == 200:
                (run_dir / "metrics-final.prom").write_text(response.text)
    except Exception:
        pass

    if state.ssh_url:
        known_hosts = state.state_file.parent / "known_hosts"
        res = ssh.run_remote(
            state.ssh_url,
            "cat /var/log/qwen38/server.log 2>/dev/null || true",
            known_hosts=known_hosts,
            timeout=60,
        )
        if res.stdout is not None:
            (run_dir / "server.log").write_text(res.stdout)

        res = ssh.run_remote(
            state.ssh_url,
            "nvidia-smi --query-gpu=timestamp,index,name,driver_version,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu --format=csv,noheader 2>/dev/null || nvidia-smi 2>/dev/null || true",
            known_hosts=known_hosts,
            timeout=60,
        )
        if res.stdout:
            (run_dir / "gpu-final.txt").write_text(res.stdout)


def _client_log(run_dir: Path, message: str) -> None:
    if not run_dir:
        return
    log = Path(run_dir) / "client-down.log"
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{utils.now_rfc3339()} {message}\n")


def _stop_proxy(state: State) -> None:
    """Stop a proxy that was auto-started by `hostai up`."""
    pid = state.data.get("proxy_pid")
    if not pid:
        return
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return
    click.echo(f"[down] stopping proxy (pid {pid})")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return
        time.sleep(0.2)
    click.echo(f"[down] proxy did not stop; killing (pid {pid})", err=True)
    try:
        os.kill(pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


def _pause_or_destroy(config: Config, state: State, pause: bool, run_dir: Path) -> str:
    """Pause or destroy the instance and update metadata."""
    if not state.instance_id:
        raise click.ClickException("state missing instance_id")

    now = utils.now_rfc3339()
    epoch = utils.now_epoch()
    started = state.started_epoch or epoch
    duration = max(0, epoch - started)
    cost = utils.format_cost(duration, state.dph)

    if pause:
        click.echo(f"Pausing instance {state.instance_id}...")
        try:
            _provider(config).stop_instance(state.instance_id)
            pause_outcome = "paused"
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                pause_outcome = "not_found"
            else:
                raise click.ClickException(f"pause failed: {exc}")
        except requests.exceptions.RequestException as exc:
            raise click.ClickException(f"pause failed: {exc}")
        state.status = "paused"
        state.set("pause_outcome", pause_outcome)
        state.set("ended_at", now)
        state.set("ended_epoch", epoch)
        state.set("duration_seconds", duration)
        state.set("estimated_compute_cost_usd", cost)
        state.set("pause_outcome", "ok")
        state.tunnel_pid = None
        state.save()
        state.save_metadata(run_dir, status="paused")
        _client_log(run_dir, f"paused instance {state.instance_id}")
        return f"Paused. Session duration: {duration}s | estimated compute: ${cost:.4f}"

    click.echo(f"Destroying instance {state.instance_id}...")
    destroy_outcome = "destroyed"
    try:
        _provider(config).destroy_instance(state.instance_id)
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            destroy_outcome = "already_absent"
        else:
            raise click.ClickException(f"destroy failed: {exc}")
    except requests.exceptions.RequestException as exc:
        raise click.ClickException(f"destroy failed (timeout): {exc}")

    state.status = "destroyed"
    state.set("ended_at", now)
    state.set("ended_epoch", epoch)
    state.set("duration_seconds", duration)
    state.set("estimated_compute_cost_usd", cost)
    state.set("destroy_outcome", destroy_outcome)
    state.tunnel_pid = None
    state.save_metadata(run_dir, status="destroyed")

    try:
        state.state_file.unlink()
    except FileNotFoundError:
        pass

    _client_log(run_dir, f"destroyed instance {state.instance_id} ({destroy_outcome})")
    return f"{destroy_outcome}. Session duration: {duration}s | estimated compute: ${cost:.4f}"


def _resolve_run_dir(config: Config, state: State) -> Path:
    """Ensure the run directory exists, creating one if the state is legacy."""
    run_dir = state.run_dir
    if run_dir is None:
        run_dir = init_run_dir(
            runs_dir(config.root_dir),
            state.profile or "unknown",
            f"{utils.now_epoch()}-recovered-{state.instance_id}",
        )
        state.run_dir = run_dir
        state.save()
    return Path(run_dir)


def _build_shutdown_tail(
    slot_details: Optional[Dict[str, Any]],
    down_start_epoch: int,
    archive_duration: float,
    pause_or_destroy_duration: float,
    reason: Optional[str],
    dph: float,
) -> Dict[str, Any]:
    """Build the shutdown-tail metrics dict written with the run archive."""
    shutdown_tail_seconds = max(0, utils.now_epoch() - down_start_epoch)
    shutdown_cost = (dph or 0.0) * shutdown_tail_seconds / 3600.0

    snapshot_bytes = int(slot_details.get("n_written", 0)) if slot_details else 0
    transferred = slot_details.get("cache_bytes_transferred") if slot_details else None
    cache_upload_failed = bool(slot_details and not slot_details.get("uploaded"))
    cache_delta_seed_used = False
    if transferred is not None and snapshot_bytes and snapshot_bytes > 0 and transferred < snapshot_bytes * 0.95:
        cache_delta_seed_used = True

    tail: Dict[str, Any] = {
        "down_started_epoch": down_start_epoch,
        "down_ended_epoch": utils.now_epoch(),
        "shutdown_tail_seconds": shutdown_tail_seconds,
        "estimated_shutdown_tail_cost_usd": round(shutdown_cost, 6),
        "cache_save_ms": slot_details.get("save_ms", 0) if slot_details else 0,
        "cache_snapshot_bytes": snapshot_bytes,
        "cache_transfer_bytes": transferred if transferred is not None else 0,
        "cache_transfer_duration_seconds": round(slot_details["upload_duration_s"], 3)
        if slot_details and "upload_duration_s" in slot_details
        else 0,
        "cache_delta_seed_used": cache_delta_seed_used,
        "cache_upload_failed": cache_upload_failed,
        "telemetry_archive_duration_seconds": round(archive_duration, 3),
        "pause_or_destroy_duration_seconds": round(pause_or_destroy_duration, 3),
        "reason": reason or "manual",
    }
    if transferred and snapshot_bytes and snapshot_bytes > 0:
        tail["cache_delta_ratio"] = round(transferred / snapshot_bytes, 6)
    return tail


def down_instance(
    config: Config,
    state: State,
    *,
    pause: bool = False,
    no_archive: bool = False,
    no_cache: bool = False,
    reason: Optional[str] = None,
    skip_confirm: bool = False,
) -> str:
    """Stop, persist cache, archive telemetry, and destroy/pause an instance.

    This is the shared lifecycle path used by ``hostai down`` and the watchdog
    daemon.  It records a shutdown reason and shutdown-tail metrics.
    """
    if not state.instance_id:
        raise click.ClickException("state missing instance_id")

    action = "Pause" if pause else "Destroy"
    gpu = state.gpu
    dph = state.dph
    if not skip_confirm:
        if not click.confirm(f"{action} instance {state.instance_id} ({gpu}, ${dph:.4f}/h)?"):
            click.echo("Cancelled.")
            return "cancelled"

    run_dir = _resolve_run_dir(config, state)
    down_start_epoch = utils.now_epoch()
    state.set("down_reason", reason or "manual")
    state.set("down_started_epoch", down_start_epoch)
    state.set("down_started_at", utils.now_rfc3339())
    state.save()
    _client_log(run_dir, f"{action} initiated for instance {state.instance_id}: reason={reason or 'manual'}")

    _stop_proxy(state)

    known_hosts = state.state_file.parent / "known_hosts"

    _refresh_ssh_state(config, state)
    if state.ssh_url:
        try:
            ssh.ensure_tunnel(config, state)
        except Exception as exc:
            click.echo(f"[down] WARNING: tunnel not available: {exc}", err=True)

    slot_details = cache.save_and_upload_slot_cache(config, state, run_dir, no_cache, known_hosts)

    archive_start = time.monotonic()
    _archive_session(config, state, run_dir, no_archive)
    archive_duration = time.monotonic() - archive_start

    if state.ssh_url:
        _stop_remote_model(state.ssh_url, known_hosts)

    ssh.stop_tunnel(state)

    pause_or_destroy_start = time.monotonic()
    outcome = _pause_or_destroy(config, state, pause, run_dir)
    pause_or_destroy_duration = time.monotonic() - pause_or_destroy_start

    tail = _build_shutdown_tail(
        slot_details,
        down_start_epoch,
        archive_duration,
        pause_or_destroy_duration,
        reason,
        state.dph,
    )

    (run_dir / "shutdown-tail.json").write_text(json.dumps(tail, indent=2, ensure_ascii=False) + "\n")
    state.set("shutdown_tail", tail)
    # Update the run metadata so the shutdown tail is recorded in the
    # archived run, but do not recreate live state.json after a destroy.
    state.save_metadata(run_dir, status="paused" if pause else "destroyed")
    if pause:
        state.save()

    _client_log(
        run_dir,
        f"{outcome} | tail={tail['shutdown_tail_seconds']}s cost=${tail['estimated_shutdown_tail_cost_usd']:.6f} reason={reason or 'manual'}",
    )

    click.echo(outcome)
    if run_dir and not no_archive:
        click.echo(f"Archived run: {run_dir}")
    return outcome


@click.command("down", help="Stop, save cache, and destroy/pause the current instance.")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
@click.option("--no-archive", is_flag=True, help="Skip telemetry archive.")
@click.option("--cache", is_flag=True, help="Save/upload the slot cache for this shutdown.")
@click.option("--no-cache", is_flag=True, help="Do not save/upload the slot cache for this shutdown.")
@click.option("--pause", is_flag=True, help="Pause the instance instead of destroying it.")
@click.option("--reason", help="Shutdown reason (used by watchdog).")
@click.pass_obj
def cmd_down(
    config: Config, yes: bool, no_archive: bool, cache: bool, no_cache: bool, pause: bool, reason: Optional[str]
) -> None:
    sd = state_dir(config.root_dir)
    state_file = sd / "state.json"

    if not state_file.exists():
        click.echo("No local hostai Vast state found.")
        return

    state = State.load(state_file)
    if not state.instance_id:
        click.echo("No Vast instance id in local state.")
        return

    cache_enabled = _common.resolve_cache_enabled(cache, no_cache, config.cache.enabled)
    if cache_enabled:
        state.set("slot_cache_enabled", True)

    down_instance(
        config,
        state,
        pause=pause,
        no_archive=no_archive,
        no_cache=not cache_enabled,
        reason=reason,
        skip_confirm=yes,
    )
    from hostai.commands.watchdog import stop_watchdog

    stop_watchdog(config)
    stop_monitor(config)
