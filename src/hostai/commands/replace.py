"""Replace the running instance with a freshly provisioned machine.

The old instance keeps serving while the new one provisions (model download
takes minutes).  The new run is staged under a sidecar ``state.replace.json``
so the live ``state.json`` — and the proxy watching it — only flips at
cutover.  The flip is the proxy's retarget trigger: its upstream supervisor
rebuilds the aiohttp session and re-creates the SSH tunnel against the new
host while the client-facing socket/port stay bound.  The old instance is
destroyed only after the proxy reports healthy on the new backend; a retarget
timeout leaves both instances running (state tracks the new one — the proxy
keeps watching and can still retarget) instead of guessing which way to flip.
"""

from __future__ import annotations

import asyncio
import copy
import shlex
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import aiohttp
import click

from hostai import cache as cache_mod
from hostai import market, ssh
from hostai import state as state_mod
from hostai.commands import _common
from hostai.commands.down import _archive_session, _resolve_run_dir
from hostai.commands.up import (
    _cpu_arch_preflight,
    _create_fresh_instance,
    _deliver_tls_cert,
    _ensure_instance_alive,
    _gpu_preflight_or_fail,
    _log,
    _now_epoch,
    _now_rfc,
    _prepare_slot_cache,
    _provider,
    _resolve_bid_price,
    _resolve_fresh_offer,
    _resolve_session_seconds,
    _start_proxy,
    _validate_up_options,
    _wait_for_ssh_endpoint,
    _write_env_file,
)
from hostai.config import Config
from hostai.state import State

_SIDE_CAR_NAME = "state.replace.json"


def _remote_health_probe(state: State) -> str:
    """Shell probe that exits 0 only when the remote /health answers 2xx.

    Runs *inside* the container over SSH so the new machine can be checked
    before any local tunnel/socket is re-pointed at it.  curl ships in the
    image.
    """
    key = shlex.quote(state.api_key or "")
    dest = ssh._default_remote_dest(state)
    if ":" in dest:
        return f"curl -fsS --max-time 8 -o /dev/null -H 'Authorization: Bearer {key}' http://{dest}/health"
    return (
        f"curl -fsSk --max-time 8 --unix-socket {shlex.quote(dest)} -o /dev/null "
        f"-H 'Authorization: Bearer {key}' https://localhost/health"
    )


def _wait_for_remote_health(config: Config, state: State, known_hosts: Path) -> None:
    """Poll the staged machine's remote /health over SSH until it serves."""
    timeout = config.ssh.start_timeout
    probe = _remote_health_probe(state)
    start = time.monotonic()
    last_log = 0.0
    seen: set = set()
    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout:
            raise click.ClickException(f"[boot:remote-health] timeout after {elapsed:.1f}s")
        try:
            res = ssh.run_remote(
                state.ssh_url,
                probe,
                known_hosts=known_hosts,
                config=config,
                state=state,
                timeout=20,
            )
            if res.returncode == 0:
                _log(f"[boot:remote-health] /health OK after {elapsed:.1f}s")
                return
        except Exception:
            pass
        if elapsed - last_log >= 15:
            last_log = elapsed
            _log(f"[replace] waiting for remote llama-server ({int(elapsed)}s / {timeout}s)")
            _ensure_instance_alive(config, state, "remote health wait")
            # Tail the remote server.log like `_wait_for_api` does.
            try:
                tail = ssh.run_remote(
                    state.ssh_url,
                    "tail -n 50 /var/log/qwen38/server.log 2>/dev/null || true",
                    known_hosts=known_hosts,
                    config=config,
                    state=state,
                    timeout=10,
                )
                for line in (tail.stdout or "").splitlines():
                    line = line.rstrip()
                    if line and line not in seen:
                        seen.add(line)
                        _log(f"[logs:server] {line}")
            except Exception:
                pass
        time.sleep(3)


def _boot_staged(
    config: Config,
    state: State,
    image: Any,
    no_cache: bool,
    abort_if_shm_too_small: bool,
) -> None:
    """Boot the staged instance: SSH, preflights, TLS, cache prefetch, health.

    Mirrors ``_do_fresh_core`` minus the client-endpoint steps — the running
    proxy owns those and must not be touched while it still serves the old
    machine.
    """
    _log(f"[replace] instance {state.instance_id} created; waiting for SSH...")
    _wait_for_ssh_endpoint(config, state, config.ssh.start_timeout)

    known_hosts = state.state_file.parent / "known_hosts"
    ssh_start = time.monotonic()
    alive_check = lambda: _ensure_instance_alive(config, state, "SSH wait")  # noqa: E731
    if not ssh.wait_for_ssh(
        state.ssh_url,
        known_hosts=known_hosts,
        config=config,
        state=state,
        timeout=300,
        alive_check=alive_check,
    ):
        _ensure_instance_alive(config, state, "SSH wait")
        raise click.ClickException("[boot:ssh-command] timeout")
    _log(f"[boot:ssh-command] ready after {time.monotonic() - ssh_start:.1f}s")

    _gpu_preflight_or_fail(config, state, known_hosts)
    _cpu_arch_preflight(state.ssh_url, known_hosts, config, state=state)

    result = ssh.run_remote(
        state.ssh_url,
        "/usr/local/bin/llama-server --version",
        known_hosts=known_hosts,
        config=config,
        state=state,
        timeout=30,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no output").strip()
        raise click.ClickException(f"remote llama-server preflight failed (rc={result.returncode}): {detail}")
    state.data["cuda_arch"] = image.cuda_arch

    _deliver_tls_cert(config, state, known_hosts)
    _prepare_slot_cache(config, state, no_cache, abort_if_shm_too_small, known_hosts)
    _wait_for_remote_health(config, state, known_hosts)

    state.status = "running"
    state.data["ready_at"] = _now_rfc()
    state.data["ready_epoch"] = _now_epoch()
    state.data["startup_seconds"] = state.data["ready_epoch"] - (state.started_epoch or 0)
    state.save()


def _destroy_staged(config: Config, staged: State, staged_file: Path) -> None:
    """Tear down a failed/rolled-back replacement.

    Only destroys the provider instance and removes the sidecar — it must
    never signal the proxy or tunnels, which still serve the old machine.
    """
    if not config.vast.keep_on_failure and staged.instance_id:
        try:
            _provider(config).destroy_instance(staged.instance_id)
            _log(f"[replace] cleaned up staged instance {staged.instance_id}")
        except Exception as exc:
            _log(
                f"[replace] WARNING: failed to destroy staged instance {staged.instance_id}: {exc}",
                err=True,
            )
    staged_file.unlink(missing_ok=True)


async def _proxy_status_once(config: Config, state_file: Path) -> Tuple[Optional[int], int]:
    """Return ``(backend_instance_id, health_status)``.

    ``backend_instance_id`` comes from ``/_hostai/backend`` (None for older
    proxies without the route); ``health_status`` is the /health code or 0
    when the proxy is unreachable.  The client-facing unix socket is probed
    first — it always exists while the proxy runs — with the TCP port as
    fallback.
    """
    state = State.load(state_file)
    sock = Path(config.proxy.socket_path) if config.proxy.socket_path else state_file.parent / "proxy.sock"
    port = state.data.get("proxy_port") or config.proxy.port or state.local_port
    timeout = aiohttp.ClientTimeout(total=5)
    targets = []
    if sock.exists():
        targets.append(({"connector": aiohttp.UnixConnector(path=str(sock))}, "http://localhost"))
    if port:
        targets.append(({}, f"http://127.0.0.1:{port}"))

    backend_id: Optional[int] = None
    health = 0
    for kwargs, base in targets:
        try:
            async with aiohttp.ClientSession(timeout=timeout, **kwargs) as session:
                try:
                    async with session.get(f"{base}/_hostai/backend") as resp:
                        if resp.status == 200:
                            backend_id = (await resp.json()).get("instance_id")
                except aiohttp.ClientError:
                    pass
                try:
                    async with session.get(f"{base}/health") as resp:
                        health = resp.status
                except aiohttp.ClientError:
                    pass
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            continue
        if backend_id is not None or health:
            break
    return backend_id, health


def _wait_for_proxy_cutover(
    config: Config,
    state_file: Path,
    new_id: int,
    timeout: float = 240.0,
) -> bool:
    """Wait until the running proxy serves the new backend.

    Success means ``/_hostai/backend`` reports *new_id* AND /health is 200.
    Proxies without the backend route (pre-retarget builds) get a degraded
    check: healthy after at least one not-ready transition.
    """
    deadline = time.monotonic() + timeout
    saw_backend_route = False
    saw_not_ready = False
    while time.monotonic() < deadline:
        backend_id, health = asyncio.run(_proxy_status_once(config, state_file))
        if backend_id is not None:
            saw_backend_route = True
        if backend_id == new_id and health == 200:
            return True
        if not saw_backend_route:
            if health != 200:
                saw_not_ready = True
            elif saw_not_ready:
                return True
        time.sleep(1.5)
    return False


def _slot_restore_remote(config: Config, state: State, known_hosts: Path) -> None:
    """POST /slots/<id>?action=restore on the new machine over SSH."""
    key = shlex.quote(state.api_key or "")
    dest = ssh._default_remote_dest(state)
    body = shlex.quote('{"filename":"current.bin"}')
    slot = config.cache.slot_id
    if ":" in dest:
        cmd = (
            f"curl -fsS --max-time 1800 -o /dev/null -X POST "
            f"-H 'Authorization: Bearer {key}' -H 'Content-Type: application/json' "
            f"-d {body} 'http://{dest}/slots/{slot}?action=restore'"
        )
    else:
        cmd = (
            f"curl -fsSk --max-time 1800 --unix-socket {shlex.quote(dest)} -o /dev/null -X POST "
            f"-H 'Authorization: Bearer {key}' -H 'Content-Type: application/json' "
            f"-d {body} 'https://localhost/slots/{slot}?action=restore'"
        )
    try:
        res = ssh.run_remote(
            state.ssh_url,
            cmd,
            known_hosts=known_hosts,
            config=config,
            state=state,
            timeout=1810,
        )
        if res.returncode == 0:
            _log("[cache] slot restored on new machine")
            state.data["slot_cache_restore"] = "restored"
        else:
            _log("[cache] WARNING: slot restore failed; continuing cold", err=True)
            state.data["slot_cache_restore"] = "failed"
    except Exception as exc:
        _log(f"[cache] WARNING: slot restore failed: {exc}; continuing cold", err=True)
        state.data["slot_cache_restore"] = "failed"


def _do_replace(
    config: Config,
    instance: str,
    profile_name: str,
    cache_session: Optional[str],
    max_price: Optional[float],
    unverified: bool,
    unsecure: bool,
    no_cache: bool,
    abort_if_shm_too_small: bool,
    offer: Optional[int],
    machine: Optional[int],
    exclusions: market.OfferExclusions,
    bid_price: Optional[float],
    session_seconds: Optional[int],
    dry_run: bool,
    allow_unvalidated: bool,
    allow_same_machine: bool,
    no_archive: bool,
) -> None:
    state_file = state_mod.instance_state_file(config.root_dir, instance)
    old = State.load(state_file)
    if not old.instance_id:
        flag = "" if instance == state_mod.DEFAULT_INSTANCE else f" --name {instance}"
        raise click.ClickException(f"nothing to replace for instance '{instance}'; run 'hostai up{flag}' first")

    old_id = old.instance_id
    old_machine = old.data.get("machine_id")
    # Kept for the post-cutover metadata write that marks the old run replaced.
    old_snapshot = copy.deepcopy(old._data)
    proxy_pid = _common.running_proxy_pid(old)
    _log(
        f"[replace] current instance {old_id} (machine {old_machine})"
        + (f"; proxy pid {proxy_pid} will hot-retarget" if proxy_pid else "")
    )

    # "A newer machine" means a different host by default; explicit
    # --machine/--offer/--allow-same-machine override the auto-exclusion.
    if offer is None and machine is None and not allow_same_machine and old_machine is not None:
        try:
            exclusions = exclusions.merged(market.OfferExclusions(machines=(int(old_machine),)))
            _log(f"[replace] excluding current machine {old_machine}")
        except (TypeError, ValueError):
            pass

    known_hosts = state_file.parent / "known_hosts"

    # Warm-swap: snapshot the old machine's KV cache while it still serves so
    # the new machine's prefetch finds it (same model/ctx => same signature).
    if not no_cache and not dry_run:
        try:
            run_dir = _resolve_run_dir(config, old)
            cache_mod.save_and_upload_slot_cache(config, old, run_dir, False, known_hosts)
        except Exception as exc:
            _log(f"[replace] slot-cache save on old instance failed: {exc}; continuing", err=True)

    staged_file = state_file.with_name(_SIDE_CAR_NAME)
    staged_file.unlink(missing_ok=True)

    with _common.allocation_lock(config):
        # local_port=None: the old port is held by the running proxy, and the
        # staged port is overridden below anyway.
        plan = _resolve_fresh_offer(
            config,
            profile_name,
            None,
            max_price,
            unverified,
            offer,
            exclusions,
            bid_price,
            session_seconds,
            allow_unvalidated,
            instance,
            machine=machine,
            # The running proxy already owns the client port; resolving a new
            # one here would fail on "port in use" and is overridden below.
            resolve_client_port=False,
        )
        if dry_run:
            click.echo(
                f"[dry-run] would create instance on offer {plan.offer_id} "
                f"(machine {plan.machine_id}) to replace {old_id}"
            )
            return
        staged = _create_fresh_instance(
            config,
            instance,
            plan,
            cache_session,
            no_cache,
            unsecure,
            state_file=staged_file,
            api_key=old.api_key,
        )
        # Preserve the client-facing identity: the running proxy owns the TCP
        # port, the upstream socket path, and its pid; the api key already
        # matches the old deployment so env and proxy auth are unchanged.
        # Older states can lack upstream_socket/proxy_port (a stale save in
        # _start_proxy could clobber them); derive the same defaults the
        # proxy would use instead of writing nulls.
        staged.local_port = old.local_port
        staged.data["upstream_socket"] = old.data.get("upstream_socket")
        if not staged.unsecure and not staged.data["upstream_socket"]:
            staged.data["upstream_socket"] = str(state_file.parent / "upstream.sock")
        staged.data["proxy_pid"] = proxy_pid or old.data.get("proxy_pid")
        staged.data["proxy_port"] = old.data.get("proxy_port") or config.proxy.port or old.local_port
        staged.data["instance_name"] = instance
        staged.save()

    new_id = staged.instance_id
    if new_id is None:
        _destroy_staged(config, staged, staged_file)
        raise click.ClickException("create did not record an instance id")

    try:
        _boot_staged(config, staged, plan.image, no_cache, abort_if_shm_too_small)
    except Exception:
        _destroy_staged(config, staged, staged_file)
        raise

    # `down` takes no lock by design; if it ran mid-replace the live state is
    # gone — refuse to flip rather than resurrect a deleted deployment.
    current = State.load(state_file)
    if current.instance_id != old_id:
        _destroy_staged(config, staged, staged_file)
        raise click.ClickException("state.json changed during replace; aborted before cutover")

    # Archive the old run's telemetry while its API still answers (the proxy
    # still targets the old machine until the flip below).
    old_run_dir = _resolve_run_dir(config, old)
    try:
        _archive_session(config, old, old_run_dir, no_archive, remote_ok=True)
    except Exception as exc:
        _log(f"[replace] telemetry archive failed: {exc}", err=True)

    # --- cutover -----------------------------------------------------------
    new_data = copy.deepcopy(staged._data)
    new_data["instance_name"] = instance
    new_data["replaced_from_instance_id"] = old_id
    if old_machine is not None:
        new_data["replaced_from_machine_id"] = old_machine
    State(state_file, new_data).save()

    client_scheme = "http" if (config.proxy.tokenized_only or staged.unsecure) else "https"
    # In unsecure mode local_port is the SSH tunnel port — the client-facing
    # port is proxy_port (persisted by run_proxy) or falls back to it.
    client_port = new_data.get("proxy_port") or staged.local_port
    client_api_url = f"{client_scheme}://127.0.0.1:{client_port}"
    client_base_url = f"{client_api_url}/v1"
    _write_env_file(config, State(state_file, new_data), client_api_url, client_base_url)

    if proxy_pid:
        _log(f"[replace] waiting for proxy to retarget to instance {new_id}...")
        if not _wait_for_proxy_cutover(config, state_file, new_id):
            staged_file.unlink(missing_ok=True)
            raise click.ClickException(
                f"proxy did not retarget to the new instance {new_id}; state.json already tracks it. "
                f"Old instance {old_id} is still running — fix the proxy ('hostai proxy' restarts it; "
                f"the daemon keeps watching state.json and retargets when it can), "
                f"then 'hostai down --id {old_id}' once the new instance serves."
            )
        _log("[replace] proxy is serving the new instance on the same port")
    elif config.proxy.tokenized_only:
        # No proxy daemon alive — spawn one against the flipped state.
        final = State.load(state_file)
        try:
            _start_proxy(config, final, client_api_scheme="http")
        except Exception as exc:
            _log(f"[replace] proxy spawn failed: {exc}; run 'hostai proxy' manually", err=True)

    final = State.load(state_file)
    if final.data.get("slot_cache_prefetch") == "ok" and not no_cache:
        _slot_restore_remote(config, final, known_hosts)

    # Old instance goes away only after the new backend is confirmed serving.
    try:
        _provider(config).destroy_instance(old_id)
        _log(f"[replace] destroyed old instance {old_id}")
    except Exception as exc:
        _log(f"[replace] WARNING: failed to destroy old instance {old_id}: {exc}", err=True)

    State(state_file, old_snapshot).save_metadata(old_run_dir, status="replaced")
    if final.run_dir:
        final.save_metadata(final.run_dir, status="ready")
    staged_file.unlink(missing_ok=True)

    _log("\nREADY (replaced)")
    _log(f"  Profile:   {final.profile}")
    if instance != state_mod.DEFAULT_INSTANCE:
        _log(f"  Name:      {instance}")
    _log(f"  API:       {client_base_url}")
    _log(f"  Instance:  {final.instance_id}  (was {old_id})")
    if final.data.get("machine_id") is not None:
        _log(f"  Machine:   {final.data['machine_id']}  (was {old_machine})")
    _log(f"  Run log:   {final.run_dir}")


@click.command("replace", help="Replace the running instance with a new machine (proxy keeps its port).")
@click.argument("profile", required=False)
@click.option("-p", "--profile", "profile_opt", help="Profile to run.")
@click.option("-s", "--session", "cache_session", help="Slot-cache session name.")
@click.option("--max-price", type=float, help="Maximum all-in $/h.")
@click.option("--unverified", is_flag=True, help="Also consider unverified/unknown hosts.")
@click.option("--unsecure", is_flag=True, help="Use legacy TCP/no-TLS mode.")
@click.option("--cache", is_flag=True, help="Enable slot cache for this run.")
@click.option("--no-cache", is_flag=True, help="Disable slot cache for this run.")
@click.option("--keep-on-failure", is_flag=True, help="Do not destroy the staged instance if provisioning fails.")
@click.option("--abort-if-shm-too-small", is_flag=True, help="Fail if /dev/shm is too small.")
@click.option("--offer", type=int, help="Use a specific offer ID.")
@click.option("--machine", type=int, help="Pin to offers on this Vast machine ID.")
@click.option(
    "--allow-same-machine",
    is_flag=True,
    help="Allow the replacement to land on the current machine (excluded by default).",
)
@click.option(
    "--skip-machine",
    "skip_machines",
    type=int,
    multiple=True,
    help="Exclude offers hosted on this Vast machine ID (repeatable).",
)
@click.option(
    "--skip-offer",
    "skip_offers",
    type=int,
    multiple=True,
    help="Exclude this Vast offer ID (repeatable).",
)
@click.option(
    "--skip-country",
    "skip_countries",
    multiple=True,
    help="Exclude offers in this country (alpha-2 code or name, repeatable).",
)
@click.option("--interruptible", is_flag=True, help="Use an interruptible/bid instance.")
@click.option("--bid", type=float, help="Bid price (max $/h) for interruptible instances.")
@click.option("--expected-session", help="Expected session duration, e.g. 30m, 2h.")
@click.option("--dry-run", is_flag=True, help="Search and print the chosen offer without renting.")
@click.option("--scoring-mode", help="Override market scoring mode (dph, perf, session).")
@click.option("--no-archive", is_flag=True, help="Skip telemetry archiving of the old run.")
@click.option(
    "-n",
    "--name",
    "instance_name",
    envvar="HOSTAI_INSTANCE",
    default=None,
    help="Name of the instance to replace.",
)
@click.option(
    "--allow-unvalidated",
    is_flag=True,
    help="Skip the production-validation gate when [vast].require_production_validation is true.",
)
@click.pass_obj
def cmd_replace(
    config: Config,
    profile: Optional[str],
    profile_opt: Optional[str],
    cache_session: Optional[str],
    max_price: Optional[float],
    unverified: bool,
    unsecure: bool,
    cache: bool,
    no_cache: bool,
    keep_on_failure: bool,
    abort_if_shm_too_small: bool,
    offer: Optional[int],
    machine: Optional[int],
    allow_same_machine: bool,
    skip_machines: Tuple[int, ...],
    skip_offers: Tuple[int, ...],
    skip_countries: Tuple[str, ...],
    interruptible: bool,
    bid: Optional[float],
    expected_session: Optional[str],
    dry_run: bool,
    scoring_mode: Optional[str],
    no_archive: bool,
    instance_name: Optional[str],
    allow_unvalidated: bool,
):
    chosen_profile = profile or profile_opt or config.hostai.default_profile
    if not chosen_profile:
        raise click.ClickException("no profile specified and no default profile configured")
    _validate_up_options(None, max_price, bid, scoring_mode)
    try:
        instance = state_mod.validate_instance_name(instance_name) if instance_name else state_mod.DEFAULT_INSTANCE
    except ValueError as exc:
        raise click.ClickException(str(exc))
    cli_exclusions = market.OfferExclusions(machines=skip_machines, offers=skip_offers, countries=skip_countries)
    exclusions = market.config_exclusions(config).merged(cli_exclusions)
    if offer is not None and offer in exclusions.offers:
        raise click.ClickException(f"--offer {offer} conflicts with a skipped offer id")
    if machine is not None and machine in exclusions.machines:
        raise click.ClickException(f"--machine {machine} conflicts with a skipped machine id")

    if scoring_mode:
        config.market.scoring_mode = scoring_mode
    if config.image.unsecure:
        unsecure = True

    no_cache = not _common.resolve_cache_enabled(cache, no_cache, config.cache.enabled)

    if keep_on_failure:
        config.vast.keep_on_failure = True

    bid_price = _resolve_bid_price(config, interruptible, bid, max_price)
    session_seconds = _resolve_session_seconds(config, expected_session)

    with _common.lifecycle_lock(config, "replace", instance=instance):
        _do_replace(
            config,
            instance,
            chosen_profile,
            cache_session,
            max_price,
            unverified,
            unsecure,
            no_cache,
            abort_if_shm_too_small,
            offer,
            machine,
            exclusions,
            bid_price,
            session_seconds,
            dry_run,
            allow_unvalidated,
            allow_same_machine,
            no_archive,
        )
