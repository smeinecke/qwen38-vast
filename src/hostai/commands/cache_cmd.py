import json
from pathlib import Path
from typing import Optional

import click

from hostai import cache, ssh, utils
from hostai.api import LlamaClient
from hostai.cache import (
    cache_config,
    cache_ssh_url,
    copy_cache_key,
    ensure_cache_key,
    preflight_remote,
    rclone_enabled,
    rclone_remote_name,
    remote_cache_dir,
)
from hostai.commands import down
from hostai.config import Config
from hostai.state import State, runs_dir, state_dir


@click.command("setup", help="One-time setup for the slot cache server.")
@click.argument("target", required=False)
@click.pass_obj
def cmd_cache_setup(config: Config, target: Optional[str]):
    """Prepare a dedicated cache key and install it on the cache server."""
    if target:
        if "@" not in target:
            raise click.ClickException("expected user@host")
        config.cache.user, config.cache.host = target.split("@", 1)

    cfg = cache_config(config)
    if rclone_enabled(config):
        cache_configured = cfg.rclone_remote or cfg.rclone_url or cfg.host
        if not cache_configured:
            raise click.ClickException("rclone cache is not configured; set rclone_url or rclone_remote")
        if not cache.validate_cache_config(config):
            raise click.ClickException("rclone cache configuration is invalid")
        click.echo(f"[cache] using rclone backend {rclone_remote_name(config)} for cache storage")
        click.echo("READY")
        click.echo(f"  remote root:  {config.cache.root}")
        click.echo(f"  max cache:    {config.cache.max_gb} GiB")
        click.echo()
        click.echo("hostai up will now automatically prefetch a matching slot snapshot via rclone")
        click.echo("and hostai down will save/upload slot 0 before destroying the Vast host.")
        return

    if not cfg.host:
        raise click.ClickException("cache.host is not configured; set it in hostai.toml or pass user@host")

    click.echo(f"[cache] using cache server {cfg.user}@{cfg.host}:{cfg.port}")
    click.echo(f"[cache] local key: {cfg.key}")
    key = ensure_cache_key(config, config.root_dir)
    click.echo(f"[cache] public key ready: {key}.pub")

    url = cache_ssh_url(config)
    if copy_cache_key(config, url):
        click.echo("[cache] public key installed on cache server")
    else:
        raise click.ClickException("failed to install cache key on cache server")

    if not preflight_remote(config):
        raise click.ClickException("cache server preflight failed")

    click.echo("[cache] preflight passed")
    click.echo()
    click.echo("READY")
    click.echo(f"  cache server: {config.cache.user}@{config.cache.host}:{config.cache.port}")
    click.echo(f"  remote root:  {config.cache.root}")
    click.echo(f"  private key:  {cfg.key}")
    click.echo(f"  max cache:    {config.cache.max_gb} GiB")
    click.echo()
    click.echo("hostai up will now automatically prefetch a matching slot snapshot from this")
    click.echo("server, and hostai down will save/upload slot 0 before destroying the Vast host.")


@click.command("copy", help="Save the current slot and upload it to the cache server.")
@click.option("--slot", type=int, default=None, help="Slot ID to save (default from config).")
@click.pass_obj
def cmd_cache_copy(config: Config, slot: Optional[int]):
    """Trigger a slot save on the running instance and upload it."""
    state = State.load(state_dir(config.root_dir) / "state.json")
    if not state.instance_id:
        raise click.ClickException("no running instance; run hostai up first")

    if not state.slot_cache_enabled:
        raise click.ClickException("slot cache is disabled; enable it in hostai.toml")

    if not ssh.is_tunnel_healthy(config, state, timeout=3):
        ssh.ensure_tunnel(config, state)

    client = LlamaClient(config, state)
    if not client.health():
        raise click.ClickException("llama-server is not healthy")

    known_hosts = state.state_file.parent / "known_hosts"

    rdir = Path(runs_dir(config.root_dir))
    run_id = f"{utils.now_epoch()}-copy-cache-{state.instance_id}"
    run_dir = rdir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir.chmod(0o700)

    ok = down._save_and_upload_slot_cache(
        config,
        state,
        run_dir,
        no_cache=False,
        known_hosts=known_hosts,
        slot_id=slot,
    )
    if not ok:
        raise click.ClickException("slot cache copy/upload failed")

    # Load save details for verification.
    save_path = run_dir / "cache-save.json"
    if not save_path.exists():
        raise click.ClickException("cache-save.json is missing after upload")
    payload = json.loads(save_path.read_text())
    n_written = int(payload.get("n_written", 0))

    remote_dir = remote_cache_dir(config, state.slot_cache_signature, state.slot_cache_session)
    if rclone_enabled(config):
        remote_size = n_written
        click.echo("[cache] rclone upload complete; skipping remote size verification")
    else:
        key_path = ensure_cache_key(config, config.root_dir)
        cache_known_hosts = cache._known_hosts_path(config.root_dir)
        verify = utils.run(
            [
                "ssh",
                "-i",
                str(key_path),
                "-p",
                str(config.cache.port),
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                f"UserKnownHostsFile={cache_known_hosts}",
                f"{config.cache.user}@{config.cache.host}",
                f"stat -c %s '{remote_dir}/current.bin'",
            ],
            check=False,
            timeout=30,
        )
        if verify.returncode != 0 or not verify.stdout.strip().isdigit():
            raise click.ClickException("could not verify uploaded cache size on cache server")
        remote_size = int(verify.stdout.strip())
        if remote_size != n_written:
            raise click.ClickException(f"remote cache size {remote_size} does not match expected {n_written}")

    state.set("slot_cache_save", "uploaded")
    state.set("slot_cache_remote_dir", remote_dir)
    state.set("slot_cache_bytes_saved", n_written)
    state.set("last_cache_copy_run", str(run_dir))
    state.set("last_cache_copy_run_id", run_id)
    state.save()

    if rclone_enabled(config):
        click.echo(f"[cache] persisted: {rclone_remote_name(config)}:{remote_dir}/current.bin")
    else:
        click.echo(f"[cache] persisted: {config.cache.user}@{config.cache.host}:{remote_dir}/current.bin")
    click.echo(f"[cache] verified: remote current.bin is {remote_size} bytes")
    click.echo(f"[cache] run log: {run_dir}")
