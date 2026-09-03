"""Stop, save cache, and destroy/pause the current Vast instance."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import click
import requests

from hostai import api, cache, ssh, utils, vast
from hostai.commands import _common
from hostai.config import Config
from hostai.state import State, init_run_dir, runs_dir, state_dir

_install_cache_key_on_vast = cache.install_cache_key_on_vast
_fetch_llama_commit = _common.fetch_llama_commit
_refresh_ssh_state = _common.refresh_ssh_state
_stop_remote_model = _common.stop_remote_model


def _slot_save(config: Config, state: State, slot_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """POST /slots/<slot_id>?action=save and parse the response."""
    client = api.LlamaClient(config, state)
    slot_id = slot_id if slot_id is not None else config.cache.slot_id
    url = f"{client.base_url}/slots/{slot_id}?action=save"
    try:
        response = requests.post(
            url,
            headers=client._headers,
            json={"filename": "current.bin"},
            verify=client._verify,
            timeout=(5, 1800),
        )
    except Exception as exc:
        click.echo(f"[slot-cache] WARNING: slot save API request failed: {exc}", err=True)
        return None

    if response.status_code != 200:
        click.echo(f"[slot-cache] WARNING: slot save API returned {response.status_code}", err=True)
        return None

    payload = response.json() if response.text else {}
    n_saved = payload.get("n_saved", 0)
    n_written = payload.get("n_written", 0)
    save_ms = payload.get("timings", {}).get("save_ms", 0)
    if not n_saved:
        click.echo("[slot-cache] slot is empty; nothing to persist.")
        return None
    return {
        "n_saved": n_saved,
        "n_written": n_written,
        "save_ms": save_ms,
        "payload": payload,
    }


def _upload_slot_cache_from_vast(
    ssh_url: str,
    config: Config,
    slot_dir: str,
    remote_dir: str,
    known_hosts: Path,
    upload_log: Optional[Path] = None,
) -> bool:
    """Push current.bin/json to the cache server (rsync or rclone)."""
    cache_configured = config.cache.host or config.cache.rclone_url or config.cache.rclone_remote
    if not cache_configured:
        return False

    if config.cache.rclone:
        script = cache.rclone_upload_script(config, slot_dir, remote_dir)
        res = ssh.run_remote(ssh_url, "bash -s", input_data=script, known_hosts=known_hosts, timeout=1800)
        if upload_log is not None:
            upload_log.parent.mkdir(parents=True, exist_ok=True)
            upload_log.write_text(res.stdout or "")
        return res.returncode == 0 and "ok" in (res.stdout or "")

    script = """set -Eeuo pipefail
umask 077
cache_host="$1"; cache_port="$2"; cache_user="$3"; remote_dir="$4"; slot_dir="$5"; cache_root="$6"
key=/root/.ssh/qwen-slot-cache
known=/root/.ssh/qwen-slot-cache-known_hosts
mkdir -p /root/.ssh
touch "$known"
chmod 600 "$known"
ssh_base="ssh -n -i $key -p $cache_port -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$known"
$ssh_base "${cache_user}@${cache_host}" "mkdir -p '$remote_dir' && chmod 700 '$cache_root' '$cache_root/'* 2>/dev/null || true; mkdir -p '$remote_dir'"
# Delta-seed: if a previous current.bin exists, copy it to .current.bin.part so
# rsync only has to ship the changed blocks.  cp --reflink=auto is best-effort.
$ssh_base "${cache_user}@${cache_host}" "if [ -f '$remote_dir/current.bin' ]; then cp --reflink=auto '$remote_dir/current.bin' '$remote_dir/.current.bin.part' 2>/dev/null || cp '$remote_dir/current.bin' '$remote_dir/.current.bin.part' 2>/dev/null || true; fi"
for attempt in 1 2 3; do
  if rsync -a --inplace --partial --info=progress2,stats2 -e "ssh -i $key -p $cache_port -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$known" "$slot_dir/current.bin" "${cache_user}@${cache_host}:${remote_dir}/.current.bin.part" < /dev/null; then
    break
  fi
  (( attempt == 3 )) && { echo >&2 "[slot-cache] upload failed after 3 attempts"; exit 15; }
  echo >&2 "[slot-cache] upload attempt $attempt failed; retrying in 3s..."
  sleep 3
done
for attempt in 1 2 3; do
  if rsync -a --inplace --partial -e "ssh -i $key -p $cache_port -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$known" "$slot_dir/current.json" "${cache_user}@${cache_host}:${remote_dir}/.current.json.part" < /dev/null; then
    break
  fi
  (( attempt == 3 )) && { echo >&2 "[slot-cache] metadata upload failed after 3 attempts"; exit 16; }
  echo >&2 "[slot-cache] metadata upload attempt $attempt failed; retrying in 3s..."
  sleep 3
done
$ssh_base "${cache_user}@${cache_host}" "chmod 600 '$remote_dir/.current.bin.part' '$remote_dir/.current.json.part' && mv -f '$remote_dir/.current.bin.part' '$remote_dir/current.bin' && mv -f '$remote_dir/.current.json.part' '$remote_dir/current.json'"
echo "ok"
"""
    import shlex

    args = [config.cache.host, str(config.cache.port), config.cache.user, remote_dir, slot_dir, config.cache.root]
    arg_str = " ".join(shlex.quote(str(a)) for a in args)
    res = ssh.run_remote(ssh_url, f"bash -s {arg_str}", input_data=script, known_hosts=known_hosts, timeout=1800)
    if upload_log is not None:
        upload_log.parent.mkdir(parents=True, exist_ok=True)
        upload_log.write_text(res.stdout or "")
    return res.returncode == 0 and "ok" in (res.stdout or "")


def _save_and_upload_slot_cache(
    config: Config,
    state: State,
    run_dir: Path,
    no_cache: bool,
    known_hosts: Path,
    slot_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Save slot, create metadata and upload to the cache server.

    Returns the slot-save details (including ``save_ms``, ``n_written``,
    ``upload_duration_s``, and ``uploaded``) when a cache is saved.  Returns
    ``None`` when the slot is empty, cache is disabled, or the upload fails
    without ``require_save``.
    """
    cache_configured = config.cache.host or config.cache.rclone_url or config.cache.rclone_remote
    if no_cache or not state.slot_cache_enabled or not config.cache.enabled or not cache_configured:
        return None

    if not state.ssh_url:
        click.echo("[slot-cache] WARNING: no SSH endpoint; cannot save slot", err=True)
        if config.cache.require_save:
            raise click.ClickException("slot cache save failed and require_save is set")
        return None

    if not config.cache.rclone and not _install_cache_key_on_vast(state, config):
        click.echo("[slot-cache] WARNING: could not install cache key on Vast", err=True)
        if config.cache.require_save:
            raise click.ClickException("slot cache key install failed and require_save is set")
        return None

    llama_commit = state.data.get("llama_cpp_commit")
    if not llama_commit or not re.match(r"^[a-f0-9]+$", str(llama_commit)):
        llama_commit = _fetch_llama_commit(state.ssh_url, known_hosts)

    signature = cache._signature_for_state(config, state, llama_commit)

    state.set("llama_cpp_commit", llama_commit)
    state.set("slot_cache_signature", signature)
    state.save()

    details = _slot_save(config, state, slot_id=slot_id)
    if not details:
        state.set("slot_cache_save", "empty")
        state.save()
        return None

    (run_dir / "cache-save.json").write_text(
        json.dumps(details.get("payload", {}), indent=2, ensure_ascii=False) + "\n"
    )

    n_saved = int(details.get("n_saved", 0))
    n_written = int(details.get("n_written", 0))
    save_ms = float(details.get("save_ms", 0))
    click.echo(
        f"[slot-cache] llama.cpp wrote {n_saved} tokens / {n_written} bytes "
        f"({save_ms} ms); uploading to {config.cache.host}..."
    )

    slot_dir = cache._default_local_dir(config)
    use_fastmtp = 1 if state.data.get("use_fastmtp", config.model.use_fastmtp) else 0
    metadata = {
        "schema_version": 1,
        "saved_at": utils.now_rfc3339(),
        "signature": signature,
        "session": state.slot_cache_session,
        "model": state.data.get("model", config.model.model),
        "hf_revision": state.data.get("hf_revision", config.model.hf_revision),
        "ctx_size": state.ctx_size,
        "use_fastmtp": use_fastmtp,
        "llama_cpp_commit": llama_commit,
        "profile": state.profile,
        "source_instance_id": state.instance_id,
        "slot_id": slot_id if slot_id is not None else config.cache.slot_id,
        "n_saved": n_saved,
        "n_written": n_written,
        "save_ms": save_ms,
    }

    meta_path = run_dir / "current.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    meta_path.chmod(0o600)

    ssh.run_remote(state.ssh_url, f"install -d -m 700 {slot_dir}", known_hosts=known_hosts, timeout=30)
    res = ssh.scp_to(state.ssh_url, meta_path, f"{slot_dir}/current.json.tmp", known_hosts=known_hosts, timeout=60)
    if res.returncode == 0:
        ssh.run_remote(
            state.ssh_url,
            f"mv {slot_dir}/current.json.tmp {slot_dir}/current.json && chmod 600 {slot_dir}/current.json",
            known_hosts=known_hosts,
            timeout=30,
        )

    remote_dir = cache.remote_cache_dir(config, signature, state.slot_cache_session)

    upload_log = run_dir / "cache-upload.log"
    upload_start = time.monotonic()
    ok = _upload_slot_cache_from_vast(state.ssh_url, config, slot_dir, remote_dir, known_hosts, upload_log)
    details["uploaded"] = ok
    details["upload_duration_s"] = time.monotonic() - upload_start
    details["n_saved"] = n_saved
    details["n_written"] = n_written
    details["save_ms"] = save_ms
    if ok:
        click.echo("[slot-cache] uploaded to cache server")
        state.set("slot_cache_save", "uploaded")
        state.set("slot_cache_n_saved", n_saved)
        state.set("slot_cache_bytes_saved", n_written)
    else:
        click.echo("[slot-cache] WARNING: upload to cache server failed", err=True)
        state.set("slot_cache_save", "upload-failed")
        if config.cache.require_save:
            state.save()
            raise click.ClickException("slot cache upload failed and require_save is set; instance not destroyed")
    state.save()
    return details if ok else None


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
        instance = vast.get_instance(config, state.instance_id) if state.instance_id else None
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
        click.echo(f"Pausing Vast instance {state.instance_id}...")
        try:
            vast.pause(config, state.instance_id, timeout=config.vast.pause_timeout_seconds)
            pause_outcome = "paused"
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                pause_outcome = "not_found"
            else:
                raise click.ClickException(f"Vast pause failed: {exc}")
        except requests.exceptions.RequestException as exc:
            raise click.ClickException(f"Vast pause failed: {exc}")
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

    click.echo(f"Destroying Vast instance {state.instance_id}...")
    destroy_outcome = "destroyed"
    try:
        vast.destroy(config, state.instance_id, timeout=config.vast.destroy_timeout_seconds)
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            destroy_outcome = "already_absent"
        else:
            raise click.ClickException(f"Vast destroy failed: {exc}")
    except requests.exceptions.RequestException as exc:
        raise click.ClickException(f"Vast destroy failed (timeout): {exc}")

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
        if not click.confirm(f"{action} Vast instance {state.instance_id} ({gpu}, ${dph:.4f}/h)?"):
            click.echo("Cancelled.")
            return "cancelled"

    # Ensure a run directory exists (legacy states may be missing it).
    run_dir = state.run_dir
    if run_dir is None:
        run_dir = init_run_dir(
            runs_dir(config.root_dir),
            state.profile or "unknown",
            f"{utils.now_epoch()}-recovered-{state.instance_id}",
        )
        state.run_dir = run_dir
        state.save()

    run_dir = Path(run_dir)
    down_start_epoch = utils.now_epoch()
    state.set("down_reason", reason or "manual")
    state.set("down_started_epoch", down_start_epoch)
    state.set("down_started_at", utils.now_rfc3339())
    state.save()
    _client_log(run_dir, f"{action} initiated for instance {state.instance_id}: reason={reason or 'manual'}")

    known_hosts = state.state_file.parent / "known_hosts"

    _refresh_ssh_state(config, state)
    if state.ssh_url:
        try:
            ssh.ensure_tunnel(config, state)
        except Exception as exc:
            click.echo(f"[down] WARNING: tunnel not available: {exc}", err=True)

    slot_details = _save_and_upload_slot_cache(config, state, run_dir, no_cache, known_hosts)

    archive_start = time.monotonic()
    _archive_session(config, state, run_dir, no_archive)
    archive_duration = time.monotonic() - archive_start

    if state.ssh_url:
        _stop_remote_model(state.ssh_url, known_hosts)

    ssh.stop_tunnel(state)

    pause_or_destroy_start = time.monotonic()
    outcome = _pause_or_destroy(config, state, pause, run_dir)
    pause_or_destroy_duration = time.monotonic() - pause_or_destroy_start

    shutdown_tail_seconds = max(0, utils.now_epoch() - down_start_epoch)
    shutdown_cost = (state.dph or 0.0) * shutdown_tail_seconds / 3600.0

    tail = {
        "down_started_epoch": down_start_epoch,
        "down_ended_epoch": utils.now_epoch(),
        "shutdown_tail_seconds": shutdown_tail_seconds,
        "estimated_shutdown_tail_cost_usd": round(shutdown_cost, 6),
        "cache_save_ms": slot_details.get("save_ms", 0) if slot_details else 0,
        "cache_bytes_uploaded": slot_details.get("n_written", 0) if slot_details else 0,
        "telemetry_archive_duration_s": round(archive_duration, 3),
        "pause_or_destroy_duration_s": round(pause_or_destroy_duration, 3),
        "reason": reason or "manual",
    }
    if slot_details and "upload_duration_s" in slot_details:
        tail["cache_upload_duration_s"] = round(slot_details["upload_duration_s"], 3)

    (run_dir / "shutdown-tail.json").write_text(json.dumps(tail, indent=2, ensure_ascii=False) + "\n")
    state.set("shutdown_tail", tail)
    state.save()

    _client_log(run_dir, f"{outcome} | tail={shutdown_tail_seconds}s cost=${shutdown_cost:.6f} reason={reason or 'manual'}")

    click.echo(outcome)
    if run_dir and not no_archive:
        click.echo(f"Archived run: {run_dir}")
    return outcome


@click.command("down", help="Stop, save cache, and destroy/pause the current instance.")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
@click.option("--no-archive", is_flag=True, help="Skip telemetry archive.")
@click.option("--no-cache", is_flag=True, help="Do not save/upload the slot cache.")
@click.option("--pause", is_flag=True, help="Pause the instance instead of destroying it.")
@click.option("--reason", help="Shutdown reason (used by watchdog).")
@click.pass_obj
def cmd_down(config: Config, yes: bool, no_archive: bool, no_cache: bool, pause: bool, reason: Optional[str]) -> None:
    sd = state_dir(config.root_dir)
    state_file = sd / "state.json"

    if not state_file.exists():
        click.echo("No local hostai Vast state found.")
        return

    state = State.load(state_file)
    if not state.instance_id:
        click.echo("No Vast instance id in local state.")
        return

    down_instance(
        config,
        state,
        pause=pause,
        no_archive=no_archive,
        no_cache=no_cache,
        reason=reason,
        skip_confirm=yes,
    )
    from hostai.commands.watchdog import stop_watchdog
    stop_watchdog(config)
