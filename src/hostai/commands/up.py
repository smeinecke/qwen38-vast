"""Start a Vast instance for a profile."""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import click

from hostai import cache, market, ssh, tls, utils
from hostai import state as state_mod
from hostai.api import LlamaClient
from hostai.commands import _common
from hostai.commands.monitor import maybe_start_monitor
from hostai.commands.watchdog import maybe_start_watchdog
from hostai.config import Config, image_for_profile
from hostai.profiles import Image, Profile, Profiles
from hostai.providers import get_provider
from hostai.state import State, init_run_dir, runs_dir, state_dir
from hostai.validate import ValidationRecord, _image_info, compare_validations, load_last_validation


def _provider(config: Config):
    return get_provider(config)


def _log(message: str, err: bool = False) -> None:
    """Emit a console line with a wall-clock timestamp."""
    ts = time.strftime("%H:%M:%S", time.localtime())
    click.echo(f"[{ts}] {message}", err=err)


def _validate_up_options(
    local_port: Optional[int],
    max_price: Optional[float],
    bid: Optional[float],
    scoring_mode: Optional[str],
) -> None:
    if local_port is not None and not (1 <= local_port <= 65535):
        raise click.ClickException("--local-port must be between 1 and 65535")
    if max_price is not None and max_price < 0:
        raise click.ClickException("--max-price must be non-negative")
    if bid is not None and bid <= 0:
        raise click.ClickException("--bid must be positive")
    if scoring_mode is not None and scoring_mode not in ("dph", "perf", "session"):
        raise click.ClickException("--scoring-mode must be dph, perf, or session")


def _resolve_bid_price(
    config: Config,
    interruptible: bool,
    bid: Optional[float],
    max_price: Optional[float],
) -> Optional[float]:
    """Resolve the effective bid price for interruptible mode, or ``None``."""
    use_interruptible = (
        interruptible or config.vast.interruptible or bid is not None or config.vast.bid_price is not None
    )
    if not use_interruptible:
        return None
    bid_price = bid if bid is not None else config.vast.bid_price
    if bid_price is None:
        bid_price = max_price if max_price is not None else config.market.max_dph
    if bid_price is None or bid_price <= 0:
        raise click.ClickException(
            "interruptible mode requires a positive bid price: set --bid, [vast].bid_price, or [market].max_dph"
        )
    return bid_price


def _resolve_session_seconds(config: Config, expected_session: Optional[str]) -> Optional[int]:
    if expected_session is not None:
        try:
            return utils.parse_duration_to_seconds(expected_session)
        except ValueError as exc:
            raise click.ClickException(str(exc))
    return config.vast.expected_session_seconds


@click.command("up", help="Start a Vast instance for a profile.")
@click.argument("profile", required=False)
@click.option("-p", "--profile", "profile_opt", help="Profile to run.")
@click.option("-s", "--session", "cache_session", help="Slot-cache session name.")
@click.option("-l", "--local-port", type=int, help="Local tunnel port.")
@click.option("--max-price", type=float, help="Maximum all-in $/h.")
@click.option("--unverified", is_flag=True, help="Also consider unverified/unknown hosts.")
@click.option("--unsecure", is_flag=True, help="Use legacy TCP/no-TLS mode.")
@click.option("--cache", is_flag=True, help="Enable slot cache for this run.")
@click.option("--no-cache", is_flag=True, help="Disable slot cache for this run.")
@click.option("--keep-on-failure", is_flag=True, help="Do not destroy the instance if provisioning fails.")
@click.option("--abort-if-shm-too-small", is_flag=True, help="Fail if /dev/shm is too small.")
@click.option("--offer", type=int, help="Use a specific offer ID.")
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
@click.option("--restart", is_flag=True, help="Restart an existing paused instance.")
@click.option("--interruptible", is_flag=True, help="Use an interruptible/bid instance.")
@click.option("--bid", type=float, help="Bid price (max $/h) for interruptible instances.")
@click.option("--expected-session", help="Expected session duration, e.g. 30m, 2h.")
@click.option("--dry-run", is_flag=True, help="Search and print the chosen offer without renting.")
@click.option("--scoring-mode", help="Override market scoring mode (dph, perf, session).")
@click.option(
    "-n",
    "--name",
    "instance_name",
    envvar="HOSTAI_INSTANCE",
    default=None,
    help="Name for this instance; lets multiple instances run in parallel.",
)
@click.option(
    "--allow-unvalidated",
    is_flag=True,
    help="Skip the production-validation gate when [vast].require_production_validation is true.",
)
@click.pass_obj
def cmd_up(
    config: Config,
    profile: Optional[str],
    profile_opt: Optional[str],
    cache_session: Optional[str],
    local_port: Optional[int],
    max_price: Optional[float],
    unverified: bool,
    unsecure: bool,
    cache: bool,
    no_cache: bool,
    keep_on_failure: bool,
    abort_if_shm_too_small: bool,
    offer: Optional[int],
    skip_machines: Tuple[int, ...],
    skip_offers: Tuple[int, ...],
    skip_countries: Tuple[str, ...],
    restart: bool,
    interruptible: bool,
    bid: Optional[float],
    expected_session: Optional[str],
    dry_run: bool,
    scoring_mode: Optional[str],
    instance_name: Optional[str],
    allow_unvalidated: bool,
):
    chosen_profile = profile or profile_opt or config.hostai.default_profile
    if not chosen_profile:
        raise click.ClickException("no profile specified and no default profile configured")
    _validate_up_options(local_port, max_price, bid, scoring_mode)
    try:
        instance = state_mod.validate_instance_name(instance_name) if instance_name else state_mod.DEFAULT_INSTANCE
    except ValueError as exc:
        raise click.ClickException(str(exc))
    exclusions = market.OfferExclusions(machines=skip_machines, offers=skip_offers, countries=skip_countries)
    if offer is not None and offer in exclusions.offers:
        raise click.ClickException(f"--offer {offer} conflicts with --skip-offer {offer}")

    if scoring_mode:
        config.market.scoring_mode = scoring_mode

    no_cache = not _common.resolve_cache_enabled(cache, no_cache, config.cache.enabled)

    if keep_on_failure:
        config.vast.keep_on_failure = True

    # Interruptible mode is enabled by CLI flag, config flag, or an explicit bid.
    bid_price = _resolve_bid_price(config, interruptible, bid, max_price)
    session_seconds = _resolve_session_seconds(config, expected_session)

    # A second concurrent `up`/`--restart` for the same instance would race
    # `_check_for_live_instance` and rent (or restart) a duplicate; serialize
    # per instance on a flock.  Different names lock different files so
    # parallel instances provision concurrently.
    with _common.lifecycle_lock(config, "up --restart" if restart else "up", instance=instance):
        if restart:
            for flag, values in (
                ("--skip-machine", exclusions.machines),
                ("--skip-offer", exclusions.offers),
                ("--skip-country", exclusions.countries),
            ):
                if values:
                    _log(f"[up] {flag} is ignored for --restart (existing instance)", err=True)
            _do_restart(
                config,
                instance,
                chosen_profile,
                local_port,
                unsecure,
                no_cache,
                allow_unvalidated=allow_unvalidated,
            )
        else:
            _do_fresh(
                config,
                instance,
                chosen_profile,
                cache_session,
                local_port,
                max_price,
                unverified,
                unsecure,
                no_cache,
                abort_if_shm_too_small,
                offer,
                exclusions=exclusions,
                bid_price=bid_price,
                session_seconds=session_seconds,
                dry_run=dry_run,
                allow_unvalidated=allow_unvalidated,
            )


def _check_production_validation(config: Config, allow_unvalidated: bool) -> Optional[ValidationRecord]:
    """Block a Vast launch unless a current production validation exists.

    When `[vast].require_production_validation` is true, this compares the
    working tree against the last successful `hostai validate --production` run.
    If the git commit, dirty state, integration image, or profiles have drifted,
    we refuse to call Vast unless the user passes --allow-unvalidated.
    """
    if config.provider.backend != "vast":
        return
    if not config.vast.require_production_validation:
        return
    if allow_unvalidated:
        _log("[validation] --allow-unvalidated set; skipping production validation gate", err=True)
        return

    previous = load_last_validation(config.root_dir, success=True)
    if previous is None or previous.result != "ok":
        raise click.ClickException(
            "[validation] no successful production validation on record. "
            "Run 'hostai validate --production' before a real Vast rental, "
            "or pass --allow-unvalidated."
        )
    if previous.level != "production" or "integration-tests" not in previous.checks_run:
        raise click.ClickException(
            "[validation] the last successful validation was not a production validation. "
            "Run 'hostai validate --production', or pass --allow-unvalidated."
        )

    current_image_id, current_image_digest = _image_info(previous.image)
    current_record = ValidationRecord(
        timestamp="",
        result="ok",
        duration_seconds=0.0,
        git_commit=utils.git_commit(config.root_dir) or "",
        dirty=utils.is_dirty_tree(config.root_dir),
        image=previous.image,
        image_id=current_image_id,
        image_digest=current_image_digest,
        profile_hash=utils.file_hash(config.root_dir / "profiles.json"),
        errors=[],
        level="production",
        checks_run=["integration-tests"],
    )
    diffs = compare_validations(current_record, previous)
    if diffs:
        for d in diffs:
            _log(f"[validation] DRIFT: {d}", err=True)
        raise click.ClickException(
            "[validation] current state does not match the last successful production validation. "
            "Run 'hostai validate --production' again, or pass --allow-unvalidated."
        )

    _log("[validation] production validation gate passed")
    return previous


def _now_rfc() -> str:
    return utils.now_rfc3339()


def _now_epoch() -> int:
    return int(time.time())


def _cleanup_instance(config: Config, state: State, reason: str) -> None:
    """Destroy the instance if something went wrong during provisioning."""
    if not state.instance_id:
        return
    machine_id = state.data.get("machine_id")
    hint = f"; re-run with --skip-machine {machine_id} to avoid this host" if machine_id is not None else ""
    if config.vast.keep_on_failure:
        _log(f"[cleanup] {reason}; keep_on_failure is set, not destroying {state.instance_id}{hint}", err=True)
        state.status = "failed"
        state.set("failure_reason", reason)
        state.save()
        return
    _log(f"[cleanup] {reason}; destroying instance {state.instance_id}...{hint}", err=True)
    try:
        _provider(config).destroy_instance(state.instance_id)
        _log(f"[cleanup] instance {state.instance_id} destroyed", err=True)
    except Exception as exc:
        _log(f"[cleanup] destroy failed: {exc}; please remove it manually", err=True)
    # Release local helpers so a failed provision cannot leave an orphaned
    # proxy daemon or SSH tunnel holding the port/socket behind.
    try:
        from hostai.commands.down import _stop_proxy

        _stop_proxy(state)
    except Exception:
        pass
    try:
        ssh.stop_tunnel(state)
    except Exception:
        pass
    state.status = "failed"
    state.set("failure_reason", reason)
    state.save()


def _shm_preflight(
    ssh_url: Optional[str],
    config: Config,
    known_hosts: Path,
    min_gb: int,
) -> int:
    """Check /dev/shm on the Vast host. Returns 0=ok, 1=too-small, 2=error."""
    if not ssh_url:
        return 2
    if not config.cache.use_shm:
        return 0
    min_bytes = min_gb * 1024 * 1024 * 1024
    if min_bytes == 0:
        return 0
    res = ssh.run_remote(
        ssh_url,
        "df -P -B1 /dev/shm | awk 'NR==2 {print $4}'",
        known_hosts=known_hosts,
        config=config,
        timeout=30,
    )
    if res.returncode != 0:
        return 2
    try:
        free = int((res.stdout or "").strip() or 0)
    except (TypeError, ValueError):
        return 2
    if free < min_bytes:
        _log(
            f"[cache] /dev/shm free={free / 1024 / 1024 / 1024:.2f}GB is below shm_min_gb={min_gb}GB",
            err=True,
        )
        return 1
    _log(f"[cache] /dev/shm free={free / 1024 / 1024 / 1024:.2f}GB OK")
    return 0


# GPUs with unified memory (e.g. NVIDIA GB10 on DGX Spark / Grace Blackwell
# systems) have no dedicated VRAM, so nvidia-smi reports memory.total as
# "[N/A]" / "Not Supported".  Their usable GPU pool is system RAM instead.
_UMA_GPU_RE = re.compile(r"\bGB10\b", re.IGNORECASE)


def _parse_nvidia_smi_names(output: str) -> List[str]:
    """Extract GPU names from nvidia-smi CSV (--query-gpu) or table output."""
    names: List[str] = []
    for line in output.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if "," in line:
            name = line.split(",", 1)[0].strip()
            if name:
                names.append(name)
            continue
        name_match = re.search(r"\|\s*\d+\s+([A-Za-z][A-Za-z0-9\s\-/_]*?)(?:\s+\||\s*$)", line)
        if name_match:
            # The name cell is followed by Persistence-M state ("On"/"Off").
            name = re.sub(r"\s+(On|Off)$", "", name_match.group(1).strip())
            if name:
                names.append(name)
    return names


def _uma_vram_preflight(
    ssh_url: str,
    known_hosts: Path,
    config: Config,
    gpu_names: List[str],
    profile_name: str,
    required_mb: int,
    state: Optional[State] = None,
) -> int:
    """Verify a unified-memory GPU by checking total system RAM.

    On UMA parts the GPU shares the system DRAM pool, so MemTotal is the
    closest equivalent of "VRAM" for the profile's minimum-memory gate.
    """
    res = ssh.run_remote(
        ssh_url,
        "awk '/^MemTotal:/ {print $2}' /proc/meminfo",
        known_hosts=known_hosts,
        config=config,
        state=state,
        timeout=30,
    )
    out = res.stdout if isinstance(res.stdout, str) else ""
    try:
        total_mb = int(out.strip()) // 1024
    except (TypeError, ValueError):
        total_mb = 0
    if res.returncode != 0 or total_mb <= 0:
        _log("[gpu] unified-memory GPU but system RAM is unreadable; cannot verify", err=True)
        return 2
    for name in gpu_names:
        _log(f"[gpu] {name} | unified memory; RAM={total_mb} MiB | required>={required_mb} MiB")
    if total_mb < required_mb:
        _log(
            f"ERROR: unified-memory pool is only {total_mb} MiB; profile {profile_name} requires at least {required_mb} MiB",
            err=True,
        )
        return 1
    return 0


def _parse_nvidia_smi_vram(output: str) -> List[Tuple[str, int]]:
    """Parse nvidia-smi output into (gpu_name, total_mib) pairs.

    Supports the machine-readable --query-gpu CSV form as well as the
    human-readable table form used by integration test fixtures.
    """
    gpus: List[Tuple[str, int]] = []
    lines = [line.rstrip() for line in output.splitlines()]

    # CSV: "NVIDIA CMP 170HX, 65536" or "NVIDIA A100-SXM4-40GB, 40960 MiB"
    for line in lines:
        if "," not in line:
            continue
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) != 2:
            continue
        name, mem_part = parts
        mem_str = re.sub(r"\s*(MiB|MB|GB|GiB|MIB|GIB)\s*$", "", mem_part, flags=re.IGNORECASE)
        try:
            total = int(float(mem_str))
            gpus.append((name, total))
        except (TypeError, ValueError):
            continue

    if gpus:
        return gpus

    # Table form: first line has the GPU name, next line has memory usage.
    current_name: Optional[str] = None
    for line in lines:
        if not line:
            continue
        mem_match = re.search(r"(\d+)\s*(?:MiB|MB|GB|GiB)\s*/\s*(\d+)\s*(?:MiB|MB|GB|GiB)", line, re.IGNORECASE)
        if mem_match:
            if current_name:
                try:
                    total = int(mem_match.group(2))
                    gpus.append((current_name.strip(), total))
                except (TypeError, ValueError):
                    pass
                current_name = None
            continue
        # Look for a GPU name row: "|   0  NVIDIA ... | ..."
        name_match = re.search(r"\|\s*\d+\s+([A-Za-z][A-Za-z0-9\s\-/_]*?)(?:\s+\||\s*$)", line)
        if name_match:
            current_name = re.sub(r"\s+(On|Off)$", "", name_match.group(1).strip())

    return gpus


def _gpu_vram_preflight(
    ssh_url: Optional[str],
    known_hosts: Path,
    config: Config,
    profile_name: str,
    required_mb: Optional[int],
    state: Optional[State] = None,
) -> int:
    """Check each visible GPU has at least *required_mb* of memory.

    Returns 0 when the check passes or is not configured, 1 when a GPU has too
    little memory, and 2 when nvidia-smi cannot be read.  The caller is
    responsible for cleanup.
    """
    if not ssh_url or not required_mb:
        return 0

    res = ssh.run_remote(
        ssh_url,
        "nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader,nounits",
        known_hosts=known_hosts,
        config=config,
        state=state,
        timeout=30,
    )
    output = res.stdout if isinstance(res.stdout, str) else ""
    if res.returncode != 0 or not output:
        # Fall back to the default human-readable table.
        res = ssh.run_remote(
            ssh_url,
            "nvidia-smi",
            known_hosts=known_hosts,
            config=config,
            state=state,
            timeout=30,
        )
        output = res.stdout if isinstance(res.stdout, str) else ""
        if res.returncode != 0 or not output:
            _log("[gpu] nvidia-smi is not available; cannot verify GPU VRAM", err=True)
            return 2

    gpus = _parse_nvidia_smi_vram(output)
    if not gpus:
        # Unified-memory GPUs (e.g. NVIDIA GB10) cannot report dedicated VRAM
        # via nvidia-smi; the shared system RAM pool is the effective limit.
        names = _parse_nvidia_smi_names(output)
        if names and all(_UMA_GPU_RE.search(name) for name in names):
            return _uma_vram_preflight(
                ssh_url,
                known_hosts,
                config,
                names,
                profile_name,
                required_mb,
                state=state,
            )
        snippet = " | ".join(line.strip() for line in output.splitlines() if line.strip())[:200]
        _log(f"[gpu] nvidia-smi output did not contain GPU memory; cannot verify (raw: {snippet})", err=True)
        return 2

    for name, total in gpus:
        _log(f"[gpu] {name} | VRAM={total} MiB | required>={required_mb} MiB")

    smallest = min(total for _, total in gpus)
    if smallest < required_mb:
        _log(
            f"ERROR: GPU exposes only {smallest} MiB VRAM; profile {profile_name} requires at least {required_mb} MiB",
            err=True,
        )
        return 1
    return 0


def _cpu_arch_preflight(
    ssh_url: Optional[str],
    known_hosts: Path,
    config: Config,
    state: Optional[State] = None,
) -> None:
    """Capture the remote CPU architecture (e.g. aarch64 on GB10) for metadata."""
    if not ssh_url or not state:
        return
    res = ssh.run_remote(
        ssh_url,
        "uname -m",
        known_hosts=known_hosts,
        config=config,
        state=state,
        timeout=5,
    )
    stdout = res.stdout if isinstance(res.stdout, str) else ""
    arch = stdout.strip() if res.returncode == 0 and stdout else ""
    if arch:
        state.data["cpu_arch"] = arch
        _log(f"[cpu] architecture={arch}")
    else:
        _log("[cpu] could not determine remote architecture; continuing", err=True)


def _resolve_profile(config: Config, name: str) -> Tuple[Profiles, Any, Any]:
    profiles = Profiles.from_file(config.root_dir / config.hostai.profiles_file)
    p = profiles.resolve_profile(name)
    if not p:
        raise click.ClickException(f"unknown profile '{name}'")
    image = profiles.image_by_name(p.image)
    if not image:
        raise click.ClickException(f"profile '{p.name}' references unknown image '{p.image}'")
    return profiles, p, image


def _resolve_client_port(
    config: Config,
    *,
    user_port: Optional[int] = None,
    state_port: Optional[int] = None,
    instance: Optional[str] = None,
) -> int:
    """Return a local client port that is free on 127.0.0.1.

    Honors the user-specified port (``--local-port`` or ``[proxy].port``) and
    fails early if that exact port is in use.  For the default port the helper
    searches for the next free port instead of starting a rental that will
    later fail to bind.  Ports recorded in other instances' state files count
    as taken even when not currently bound (e.g. a sibling mid-provisioning).
    """
    user_set = False
    if user_port is not None:
        desired = user_port
        user_set = True
    elif config.proxy.port:
        desired = config.proxy.port
        user_set = True
    elif state_port is not None:
        desired = state_port
    else:
        desired = config.ssh.local_port or 18081

    if not (1 <= desired <= 65535):
        raise click.ClickException("client port must be between 1 and 65535")

    claimed = _common.claimed_local_ports(config, exclude_instance=instance)

    if desired not in claimed and utils.port_is_free(desired, host="127.0.0.1"):
        config.ssh.local_port = desired
        if user_port is not None:
            config.proxy.port = desired
        return desired

    if user_set:
        if desired in claimed:
            raise click.ClickException(
                f"client port {desired} is already claimed by another hostai instance; choose another with --local-port"
            )
        raise click.ClickException(f"client port {desired} is already in use; choose another with --local-port")

    _log(f"[up] client port {desired} is in use; searching for a free port", err=True)
    free_port = _find_unclaimed_port(desired, claimed)
    if free_port is None:
        raise click.ClickException("no free localhost port found")

    _log(f"[up] using client port {free_port}", err=True)
    config.ssh.local_port = free_port
    return free_port


def _find_unclaimed_port(start: int, claimed: Set[int]) -> Optional[int]:
    """First free port >= ``start`` that is also not claimed by a sibling instance."""
    candidate = start
    for _ in range(100):
        if candidate in claimed:
            candidate += 1
            continue
        try:
            port = utils.find_free_port(start=candidate, host="127.0.0.1")
        except RuntimeError:
            return None
        if port not in claimed:
            return port
        candidate = port + 1
    return None


def _env_model_overrides(config: Config, profile: Any, env: Dict[str, str]) -> None:
    cache_ram = config.model.cache_ram if config.model.cache_ram is not None else profile.cache_ram
    ctx_checkpoints = (
        config.model.ctx_checkpoints if config.model.ctx_checkpoints is not None else profile.ctx_checkpoints
    )
    if cache_ram:
        env["CACHE_RAM"] = str(cache_ram)
    if ctx_checkpoints:
        env["CTX_CHECKPOINTS"] = str(ctx_checkpoints)
    if config.model.cache_type_k and config.model.cache_type_k != "default":
        env["CACHE_TYPE_K"] = config.model.cache_type_k
    if config.model.cache_type_v and config.model.cache_type_v != "default":
        env["CACHE_TYPE_V"] = config.model.cache_type_v


def _env_cache_vars(config: Config, session: str, slot_dir: str, env: Dict[str, str]) -> None:
    env["HOSTAI_SLOT_CACHE_ENABLED"] = "1"
    env["HOSTAI_SLOT_CACHE_HOST"] = config.cache.host
    env["HOSTAI_SLOT_CACHE_PORT"] = str(config.cache.port)
    env["HOSTAI_SLOT_CACHE_USER"] = config.cache.user
    env["HOSTAI_SLOT_CACHE_ROOT"] = config.cache.root
    env["HOSTAI_SLOT_CACHE_SESSION"] = session
    env["HOSTAI_SLOT_CACHE_MAX_GB"] = str(config.cache.max_gb)
    env["HOSTAI_SLOT_CACHE_USE_SHM"] = "1" if config.cache.use_shm else "0"
    # A value of 0 means "use the runtime default" (30 GB).
    env["HOSTAI_SLOT_CACHE_MIN_GB"] = str(config.cache.shm_min_gb or 30)
    env["HOSTAI_SLOT_CACHE_LOCAL_DIR"] = slot_dir
    if not config.cache.rclone:
        return
    env["HOSTAI_SLOT_CACHE_RCLONE"] = "1"
    for key, value in (
        ("HOSTAI_SLOT_CACHE_RCLONE_REMOTE", config.cache.rclone_remote),
        ("HOSTAI_SLOT_CACHE_RCLONE_TYPE", config.cache.rclone_type),
        ("HOSTAI_SLOT_CACHE_RCLONE_URL", config.cache.rclone_url),
        ("HOSTAI_SLOT_CACHE_RCLONE_USER", config.cache.rclone_user),
        ("HOSTAI_SLOT_CACHE_RCLONE_PASSWORD", config.cache.rclone_password),
    ):
        if value:
            env[key] = value


def _env_dict(
    config: Config,
    profile: Any,
    image: Any,
    model: str,
    ctx_size: int,
    api_key: str,
    unsecure: bool,
    no_cache: bool,
    session: str,
) -> Dict[str, str]:
    slot_dir = cache._default_local_dir(config)
    env: Dict[str, str] = {
        "HOSTAI_PROFILE": profile.name,
        "LLAMA_API_KEY": api_key,
        "MODEL": model,
        "CTX_SIZE": str(ctx_size),
        "USE_FASTMTP": str(int(config.model.use_fastmtp)),
        "REASONING_EFFORT": config.model.reasoning_effort,
        "HF_REVISION": config.model.hf_revision,
        "HOSTAI_UNSECURE": "1" if unsecure else "0",
        "HOSTAI_TOKENIZED_ONLY": "1" if config.proxy.tokenized_only else "0",
        "SLOT_SAVE_PATH": slot_dir,
    }
    hf_token = config.secrets.get("HF_TOKEN") or config.secrets.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        env["HF_TOKEN"] = hf_token

    _env_model_overrides(config, profile, env)

    ssh_public_key = config.secrets.get("SSH_PUBLIC_KEY")
    if ssh_public_key:
        env["HOSTAI_SSH_PUBLIC_KEY_B64"] = base64.b64encode(ssh_public_key.encode()).decode()

    # Forward deterministic fault-injection variables so integration tests can
    # exercise boot deadlines without changing production start.sh.
    for key, value in os.environ.items():
        if key.startswith("HOSTAI_FAULT_") or key.startswith("HOSTAI_TEST_"):
            env[key] = value

    # Vast maps container port 22 to a public host port in args/entrypoint mode.
    env["-p 22:22"] = "1"

    cache_configured = config.cache.host or config.cache.rclone_url or config.cache.rclone_remote
    if not no_cache and cache_configured:
        _env_cache_vars(config, session, slot_dir, env)
    return env


def _extra_args(config: Config, no_cache: bool = False) -> str:
    parts = []
    shm_size_gb = config.vast.shm_size_gb
    if shm_size_gb is None and config.cache.use_shm and not no_cache:
        shm_size_gb = config.cache.shm_min_gb
    if shm_size_gb:
        parts.append(f"--shm-size={shm_size_gb}g")
    return " ".join(parts)


def _emit_instance_logs(config: Config, instance_id: int, seen: Dict[str, Set[str]]) -> None:
    """Fetch and emit any new container/daemon log lines from the provider."""
    for kind, daemon in (("container", False), ("daemon", True)):
        try:
            text = _provider(config).get_logs(
                instance_id,
                tail=50,
                daemon_logs=daemon,
                timeout=15.0,
            )
        except Exception:
            # Logs may not be ready while the container is still starting.
            continue
        if not text or not isinstance(text, str):
            continue
        for line in text.splitlines():
            line = line.rstrip()
            if not line or line in seen[kind]:
                continue
            seen[kind].add(line)
            _log(f"[logs:{kind}] {line}")


# Instance statuses that mean provisioning can never succeed.  "stopped"
# covers interruptible instances preempted by a higher bid.
_INSTANCE_DEAD_STATUSES = frozenset({"exited", "offline", "unknown", "stopped"})
# Minimum seconds between provider liveness polls inside retry loops.
_INSTANCE_CHECK_INTERVAL = 10.0


def _instance_alive(config: Config, state: State, *, stopped_is_dead: bool = True) -> Tuple[bool, str]:
    """Poll the provider once and return ``(alive, reason_if_dead)``.

    A live instance also refreshes ``state.ssh_url`` when Vast has remapped
    the SSH port since it was first resolved.  Provider errors are treated
    as "alive" so a transient API failure never aborts provisioning.

    ``stopped_is_dead=False`` skips the "stopped" status, for the restart
    path where Vast can still report "stopped" briefly after start_instance.
    """
    if not state.instance_id:
        return True, ""
    try:
        inst = _provider(config).get_instance(state.instance_id)
    except Exception:
        return True, ""
    if not inst:
        return False, "instance no longer exists (preempted or removed)"
    status = str(inst.get("actual_status") or inst.get("status") or "")
    dead = _INSTANCE_DEAD_STATUSES if stopped_is_dead else _INSTANCE_DEAD_STATUSES - {"stopped"}
    if status in dead:
        return False, f"instance status is '{status}'"
    endpoint = ssh.resolve_ssh_endpoint(inst)
    if endpoint and endpoint["ssh_url"] != state.ssh_url:
        _log(f"[instance] ssh endpoint moved {state.ssh_url} -> {endpoint['ssh_url']}")
        state.ssh_url = endpoint["ssh_url"]
        state.save()
    return True, ""


def _ensure_instance_alive(config: Config, state: State, context: str, *, stopped_is_dead: bool = True) -> None:
    """Raise ``ClickException`` when the instance is gone.

    Death is confirmed by a second poll so a single flaky API response
    cannot destroy a healthy provisioning run.
    """
    alive, reason = _instance_alive(config, state, stopped_is_dead=stopped_is_dead)
    if alive:
        return
    time.sleep(3)
    alive, reason2 = _instance_alive(config, state, stopped_is_dead=stopped_is_dead)
    if alive:
        return
    raise click.ClickException(f"instance {state.instance_id} lost during {context}: {reason2 or reason}")


def _wait_for_ssh_endpoint(config: Config, state: State, timeout: int) -> None:
    if not state.instance_id:
        raise click.ClickException("no instance id in state")
    start = time.monotonic()
    last_status = 0
    seen_instance = False
    seen_log_lines: Dict[str, Set[str]] = {"container": set(), "daemon": set()}
    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout:
            raise click.ClickException(f"[boot:ssh-endpoint] timeout after {elapsed:.1f}s")
        try:
            inst = _provider(config).get_instance(state.instance_id)
        except Exception as exc:
            # A flaky provider response must not abort a booting instance.
            _log(f"[boot] instance lookup failed: {exc}; retrying", err=True)
            time.sleep(5)
            continue
        if not inst:
            if seen_instance:
                # Was visible before and vanished — likely preempted or
                # destroyed.  _ensure_instance_alive confirms via a second
                # poll before raising.
                _ensure_instance_alive(config, state, "SSH endpoint wait")
            else:
                _log("[boot] waiting for instance to appear...")
            time.sleep(5)
            continue
        seen_instance = True
        status = inst.get("actual_status") or inst.get("status") or "loading"
        if status in ("exited", "offline", "unknown"):
            raise click.ClickException(f"instance entered status '{status}'")
        endpoint = ssh.resolve_ssh_endpoint(inst)
        if endpoint:
            state.ssh_url = endpoint["ssh_url"]
            state.status = "ssh-ready"
            state.save()
            _log(f"[boot:ssh-endpoint] {state.ssh_url} ready after {elapsed:.1f}s")
            return
        if elapsed - last_status >= 15:
            last_status = elapsed
            host = inst.get("public_ipaddr") or inst.get("public_ip") or "?"
            ports = inst.get("ports") or {}
            tcp = ports.get("22/tcp") or []
            port = tcp[0].get("HostPort") if tcp and isinstance(tcp[0], dict) else "?"
            _log(f"[boot] status={status} | ssh={host}:{port} | waiting...")
            _emit_instance_logs(config, state.instance_id, seen_log_lines)
        time.sleep(5)


def _wait_for_api(
    config: Config,
    state: State,
    timeout: int,
    client: LlamaClient,
    *,
    stage_label: str = "end-to-end",
) -> None:
    start = _now_epoch()
    interval = 1.0
    last_log = start
    known_hosts = state.state_file.parent / "known_hosts"
    seen_logs: Dict[str, Set[str]] = {"container": set(), "daemon": set(), "server": set()}
    while True:
        now = _now_epoch()
        if now - start > timeout:
            raise click.ClickException(f"timeout waiting for llama-server /health ({stage_label})")
        if client.health():
            _log(f"[boot:{stage_label}] /health OK after {now - start:.1f}s")
            return
        # In the direct-tunnel path a dead SSH connection kills the tunnel
        # worker (registry entry removed, local port closed); re-establish
        # it instead of waiting for a port that will never open.
        if not config.proxy.tokenized_only and state.local_port and not ssh._tunnel_is_running(state):
            try:
                ssh.ensure_tunnel(config, state)
                _log("[tunnel] re-established dead tunnel")
            except Exception as exc:
                _log(f"[tunnel] reconnect failed: {exc}", err=True)
        if now - last_log >= 15:
            last_log = now
            detail = f" | last: {client.last_health_error}" if client.last_health_error else ""
            _log(f"[api] waiting for llama-server ({now - start}s / {timeout}s){detail}")
            if state.instance_id:
                _ensure_instance_alive(config, state, "API health wait")
                _emit_instance_logs(config, state.instance_id, seen_logs)
            # best-effort server log tail from inside the container
            try:
                result = ssh.run_remote(
                    state.ssh_url,
                    "tail -n 50 /var/log/qwen38/server.log 2>/dev/null || true",
                    known_hosts=known_hosts,
                    config=config,
                    state=state,
                    timeout=10,
                )
                if result.stdout:
                    for line in result.stdout.splitlines():
                        line = line.rstrip()
                        if not line or line in seen_logs["server"]:
                            continue
                        seen_logs["server"].add(line)
                        _log(f"[logs:server] {line}")
            except Exception:
                pass
        time.sleep(interval)
        interval = min(interval * 2, 5.0)


def _capture_disk_telemetry(config: Config, state: State, known_hosts: Path) -> Optional[Dict[str, Any]]:
    """Capture container disk usage after a successful cold start.

    Reads the disk-usage log written by ``start.sh`` and appends a final
    snapshot so operators can verify that the allocated disk is realistically
    sized.  Returns the combined telemetry or ``None`` if it could not be
    gathered.
    """
    if not state.ssh_url:
        return None

    run_dir = state.run_dir
    if not run_dir:
        return None

    log_path = "/var/log/qwen38/disk-usage.log"
    res = ssh.run_remote(
        state.ssh_url,
        f"cat {log_path} 2>/dev/null || true",
        known_hosts=known_hosts,
        config=config,
        state=state,
        timeout=30,
    )
    records: List[Dict[str, Any]] = []
    if res.stdout:
        for line in res.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    # Always capture a final snapshot in case the start.sh log is missing
    # (older images) or the container's log path differs.
    final_script = r"""python3 - <<'PY'
import json, os, subprocess

def du(path):
    if not os.path.exists(path):
        return 0
    try:
        return int(subprocess.check_output(["du", "-sb", path], text=True).split()[0])
    except Exception:
        return 0

df = subprocess.check_output(["df", "-B1", "/"], text=True).strip().splitlines()[1].split()
record = {
    "stage": "up-final",
    "total_bytes": int(df[1]),
    "used_bytes": int(df[2]),
    "free_bytes": int(df[3]),
    "models_bytes": du("/models"),
    "slots_disk_bytes": du("/var/lib/qwen38/slots"),
    "slots_shm_bytes": du("/dev/shm/qwen38/slots"),
    "log_bytes": du("/var/log/qwen38"),
    "run_bytes": du("/run/qwen38"),
    "tmp_bytes": du("/dev/shm/qwen38/tmp"),
}
print(json.dumps(record))
PY"""
    res2 = ssh.run_remote(state.ssh_url, final_script, known_hosts=known_hosts, config=config, state=state, timeout=60)
    if res2.returncode == 0 and res2.stdout:
        try:
            record = json.loads(res2.stdout.strip().splitlines()[-1])
            records.append(record)
        except (json.JSONDecodeError, IndexError):
            pass

    if not records:
        return None

    telemetry = {
        "captured_at": utils.now_rfc3339(),
        "disk_gb": state.disk_gb,
        "records": records,
    }
    out = run_dir / "disk-telemetry.json"
    out.write_text(json.dumps(telemetry, indent=2, default=str) + "\n")
    out.chmod(0o600)
    return telemetry


def _write_env_file(config: Config, state: State, api_url: str, base_url: str) -> None:
    env_path = state.state_file.parent / "env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        env_path.parent.chmod(0o700)
    except OSError:
        pass

    client_base: Optional[str]
    if config.proxy.tokenized_only and config.proxy.port:
        # The proxy is the OpenAI-compatible endpoint for clients.
        client_base = f"http://127.0.0.1:{config.proxy.port}/v1"
    else:
        client_base = base_url

    instance_name = state.data.get("instance_name") or state_mod.instance_name_for_state_file(state.state_file)
    lines = [
        f"export OPENAI_API_KEY='{state.api_key}'",
        f"export HOSTAI_MODEL='{state.data.get('model')}'",
        f"export HOSTAI_PROFILE='{state.data.get('profile')}'",
        f"export HOSTAI_INSTANCE='{instance_name}'",
        f"export HOSTAI_VAST_INSTANCE_ID='{state.instance_id}'",
        f"export HOSTAI_BASE_URL='{base_url}'",
        f"export HOSTAI_API_URL='{api_url}'",
    ]
    if client_base:
        lines += [
            f"export OPENAI_BASE_URL='{client_base}'",
            f"export OPENAI_API_BASE='{client_base}'",
        ]

    if config.proxy.tokenized_only:
        socket_path = (
            Path(config.proxy.socket_path) if config.proxy.socket_path else state.state_file.parent / "proxy.sock"
        )
        lines += [
            f"export HOSTAI_PROXY_SOCKET='{socket_path}'",
            "export HOSTAI_TOKENIZED_ONLY=1",
        ]
        if not config.proxy.port:
            lines += [
                "# tokenized-only is enabled; run 'hostai proxy' and connect your OpenAI client to HOSTAI_PROXY_SOCKET",
                "# or set HOSTAI_PROXY_PORT to expose the proxy on a local TCP port as well.",
            ]
        lines += [
            "# In tokenized-only mode the raw HOSTAI_API_URL is for diagnostics only.",
            "# All chat traffic must go through the hostai proxy so prompts are tokenized locally.",
        ]

    if not state.unsecure and state.tls_ca:
        ca = str(state.tls_ca)
        lines += [
            f"export HOSTAI_CA_CERT='{ca}'",
            f"export SSL_CERT_FILE='{ca}'",
            f"export CURL_CA_BUNDLE='{ca}'",
            f"export REQUESTS_CA_BUNDLE='{ca}'",
        ]
    env_path.write_text("\n".join(lines) + "\n")
    env_path.chmod(0o600)


def _default_proxy_socket(config: Config) -> Path:
    return state_dir(config.root_dir) / "proxy.sock"


def _hostai_binary() -> Path:
    """Return the path to the hostai executable for spawning child processes."""
    candidate = Path(sys.executable).parent / "hostai"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    path = shutil.which("hostai")
    if path:
        return Path(path)
    raise RuntimeError("could not find hostai executable")


def _start_proxy(config: Config, state: State, client_api_scheme: str = "http") -> Optional[int]:
    """Auto-start the local OpenAI proxy.

    In tokenized-only mode it tokenizes /v1/chat/completions. In normal mode it
    passes traffic through. The proxy owns the SSH connection to the remote
    (Unix-socket upstream if secure) and stays up after `up` exits.

    Returns the TCP port the proxy listens on for clients, or None when the
    proxy is not needed (tokenized-only is false and there is no pass-through
    configured).
    """
    if not config.proxy.tokenized_only:
        return None

    instance_name = state.data.get("instance_name") or state_mod.instance_name_for_state_file(state.state_file)
    claimed = _common.claimed_local_ports(config, exclude_instance=instance_name)
    port = config.proxy.port or config.ssh.local_port or 0
    if port and port in claimed:
        _log(f"[proxy] WARNING: configured port {port} is claimed by another instance; skipping", err=True)
        return None
    if port and not utils.port_is_free(port, host="127.0.0.1"):
        # The port may be held by a stale hostai proxy from a previous run —
        # those can be killed safely (a foreign process must not be touched).
        stale_pid = _common.running_proxy_pid(state)
        if stale_pid:
            _log(f"[proxy] stopping stale proxy (pid {stale_pid}) holding port {port}")
            try:
                os.kill(stale_pid, signal.SIGTERM)
            except OSError:
                pass
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not utils.port_is_free(port, host="127.0.0.1"):
                time.sleep(0.2)
        if not utils.port_is_free(port, host="127.0.0.1"):
            _log(f"[proxy] WARNING: configured port {port} is in use; skipping", err=True)
            return None
    if not port:
        port = _find_unclaimed_port(18083, claimed) or 0
        if not port:
            _log("[proxy] WARNING: no free TCP port; skipping", err=True)
            return None
    config.proxy.port = port

    env = os.environ.copy()
    env["HOSTAI_PROXY_PORT"] = str(port)
    env["PYTHONUNBUFFERED"] = "1"
    log_path = state.state_file.parent / "proxy.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        proxy_bin = _hostai_binary()
    except RuntimeError as exc:
        _log(f"[proxy] WARNING: {exc}; skipping", err=True)
        return None

    _log(f"[proxy] auto-starting on {client_api_scheme}://127.0.0.1:{port}")
    try:
        with open(log_path, "a", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                [str(proxy_bin), "proxy", "--name", instance_name],
                env=env,
                start_new_session=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
    except OSError as exc:
        _log(f"[proxy] WARNING: failed to start: {exc}; skipping", err=True)
        return None

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if not utils.port_is_free(port, host="127.0.0.1"):
            break
        time.sleep(0.2)
    else:
        _log(f"[proxy] WARNING: did not see proxy on port {port}; check {log_path}", err=True)
        try:
            tail = log_path.read_text(encoding="utf-8").splitlines()[-50:]
            for line in tail:
                _log(f"[proxy] log: {line}", err=True)
        except Exception:
            pass
        return None

    # Record the proxy pid/port in the same state object the caller will keep
    # writing.  Reloading would create a fresh copy that the next state.save()
    # in the caller would overwrite.
    state.data["proxy_pid"] = proc.pid
    state.local_port = port
    state.save()

    api_url = f"{client_api_scheme}://127.0.0.1:{port}"
    base_url = f"{api_url}/v1"
    _write_env_file(config, state, api_url, base_url)
    _log(f"[proxy] ready (pid {proc.pid}); OPENAI_BASE_URL={base_url}")
    return port


def _prefetch_slot_cache_to_vast(
    ssh_url: Optional[str],
    config: Config,
    slot_dir: str,
    remote_dir: str,
    known_hosts: Path,
) -> bool:
    """Pull current.bin/json from the cache server (rsync or rclone)."""
    if not ssh_url:
        return False
    cache_configured = config.cache.host or config.cache.rclone_url or config.cache.rclone_remote
    if not cache_configured:
        return False

    if config.cache.rclone:
        script = cache.rclone_prefetch_script(config, slot_dir, remote_dir)
    else:
        script = cache.rsync_prefetch_script(config, slot_dir, remote_dir)
    _log(f"[cache] checking cache server for a warm slot ({'rclone' if config.cache.rclone else 'rsync'})...")
    res = ssh.run_remote(ssh_url, "bash -s", input_data=script, known_hosts=known_hosts, config=config, timeout=330)
    ok = res.returncode == 0 and "ok" in (res.stdout or "")
    if not ok:
        tail = [line.strip() for line in (res.stderr or "").splitlines() if line.strip()]
        if tail:
            _log(f"[cache] prefetch: {tail[-1]}", err=True)
    return ok


class _FreshOffer(NamedTuple):
    """Resolved provisioning inputs before an instance is created."""

    local_port: int
    profile: Profile
    image: Image
    ctx_size: int
    model: str
    selected_image: str
    disk_gb: int
    interruptible: bool
    query: str
    max_dph: float
    offer_type: str
    offer_data: Dict[str, Any]
    offer_id: int
    machine_id: Optional[int]
    gpu_name: str
    dph: float
    bid_price: Optional[float]
    session_seconds: Optional[int]
    exclusions: market.OfferExclusions


def _check_for_live_instance(config: Config, instance: str = state_mod.DEFAULT_INSTANCE) -> None:
    """Refuse to proceed if this instance's state still references a live remote."""
    existing = state_mod.instance_state_file(config.root_dir, instance)
    if not existing.exists():
        return
    old = State.load(existing)
    if not old.instance_id:
        return
    try:
        inst = _provider(config).get_instance(old.instance_id)
    except Exception:
        # Could not reach provider to verify; proceed rather than hard-block.
        inst = None
    if inst and (inst.get("actual_status") or inst.get("status")) not in ("exited", "offline"):
        if instance == state_mod.DEFAULT_INSTANCE:
            raise click.ClickException(
                f"state file already references instance {old.instance_id}; run hostai down first"
            )
        raise click.ClickException(
            f"instance '{instance}' already references live instance {old.instance_id}; "
            f"run 'hostai down --name {instance}' first or pick another --name"
        )


def _resolve_fresh_offer(
    config: Config,
    profile_name: str,
    local_port: Optional[int],
    max_price: Optional[float],
    unverified: bool,
    offer: Optional[int],
    exclusions: market.OfferExclusions,
    bid_price: Optional[float],
    session_seconds: Optional[int],
    allow_unvalidated: bool,
    instance: str = state_mod.DEFAULT_INSTANCE,
) -> _FreshOffer:
    """Resolve profile, search query and select a market offer."""
    local_port = _resolve_client_port(config, user_port=local_port, instance=instance)
    profiles, profile, image = _resolve_profile(config, profile_name)
    ctx_size = config.hostai.ctx_size_override if config.hostai.ctx_size_override else profile.ctx_size
    model = config.model.model
    selected_image = image_for_profile(config, image.image_tag)
    disk_gb = market.resolved_disk_gb(profile, config)
    interruptible = bid_price is not None

    query, max_dph = market.build_search_query(
        config, profiles, profile, max_price=max_price, unverified=unverified, offer=offer, bid_price=bid_price
    )
    _log(
        f"[profile] {profile.name} | sm_{image.cuda_arch} | ctx={ctx_size} | image={selected_image} | disk={disk_gb}GB"
    )
    _log(f"[search]  {query}")

    offer_type = "bid" if interruptible else "on-demand"
    configured_max_dph = max_price if max_price is not None else config.market.max_dph
    if interruptible and bid_price is not None:
        effective_cap = min(configured_max_dph, bid_price)
        _log(
            f"[search]  mode={offer_type} configured_max_dph=${configured_max_dph:.4f}/h "
            f"bid=${bid_price:.4f}/h effective_cap=${effective_cap:.4f}/h"
        )
    else:
        _log(f"[search]  mode={offer_type} max_dph=${configured_max_dph:.4f}/h")
    if exclusions:
        _log(f"[search]  excluding {exclusions.describe()}")
    provider = _provider(config)
    _log(f"[provider] {provider.name}")
    previous = _check_production_validation(config, allow_unvalidated)
    if config.vast.require_production_validation and previous is not None and not allow_unvalidated:
        # Use the immutable SHA-tagged image that the CI published for the
        # validated Git commit instead of the mutable profile tag.
        commit = previous.git_commit[:7]
        selected_image = image_for_profile(config, f"{image.image_tag}-sha-{commit}")
        _log(f"[image] gated to validated image: {selected_image}")

    offer_data = market.select_offer(
        config,
        profiles,
        query,
        max_dph=max_dph,
        unverified=unverified,
        offer=offer,
        storage=disk_gb,
        offer_type=offer_type,
        session_seconds=session_seconds,
        exclusions=exclusions,
        verbose=True,
    )
    offer_id_raw = offer_data.get("id") or offer_data.get("ask_contract_id")
    if offer_id_raw is None:
        raise click.ClickException("selected offer has no id")
    offer_id = int(offer_id_raw)
    machine_id_raw = offer_data.get("machine_id")
    try:
        machine_id = int(machine_id_raw) if machine_id_raw is not None else None
    except (TypeError, ValueError):
        machine_id = None
    gpu_name = offer_data.get("gpu_name", "unknown")
    dph = offer_data.get("dph_total", 0.0)
    _log(f"[rent] {market.offer_summary(offer_data)}")
    return _FreshOffer(
        local_port=local_port,
        profile=profile,
        image=image,
        ctx_size=ctx_size,
        model=model,
        selected_image=selected_image,
        disk_gb=disk_gb,
        interruptible=interruptible,
        query=query,
        max_dph=max_dph,
        offer_type=offer_type,
        offer_data=offer_data,
        offer_id=offer_id,
        machine_id=machine_id,
        gpu_name=gpu_name,
        dph=dph,
        bid_price=bid_price,
        session_seconds=session_seconds,
        exclusions=exclusions,
    )


def _fresh_metadata(
    config: Config,
    offer: _FreshOffer,
    session: str,
    no_cache: bool,
    unsecure: bool,
    run_id: str,
    run_started: str,
    instance: str = state_mod.DEFAULT_INSTANCE,
) -> Dict[str, Any]:
    profile = offer.profile
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "provisioning",
        "started_at": run_started,
        "instance_name": instance,
        "profile": profile.name,
        "monitor_group": profile.monitor_group or "",
        "gpu_query": offer.query,
        "machine_id": offer.machine_id,
        "disk_gb": offer.disk_gb,
        "interruptible": offer.interruptible,
        "bid_price": offer.bid_price if offer.interruptible else None,
        "expected_session_seconds": offer.session_seconds,
        "scoring_mode": config.market.scoring_mode,
        "cuda_arch": offer.image.cuda_arch,
        "image": offer.selected_image,
        "ctx_size": offer.ctx_size,
        "model": offer.model,
        "hf_revision": config.model.hf_revision,
        "use_fastmtp": config.model.use_fastmtp,
        "cache_type_k": config.model.cache_type_k or "default",
        "cache_type_v": config.model.cache_type_v or "default",
        "slot_cache_enabled": not no_cache,
        "slot_cache_host": config.cache.host,
        "slot_cache_port": config.cache.port,
        "slot_cache_user": config.cache.user,
        "slot_cache_root": config.cache.root,
        "slot_cache_session": session,
        "slot_cache_max_gb": config.cache.max_gb,
        "slot_cache_local_dir": cache._default_local_dir(config),
        "slot_cache_use_shm": config.cache.use_shm,
        "unsecure": unsecure,
    }


def _fill_fresh_state(
    state: State,
    config: Config,
    offer: _FreshOffer,
    instance_id: int,
    api_key: str,
    label: str,
    session: str,
    run_id: str,
    run_dir: Path,
    run_started: str,
    run_epoch: int,
    no_cache: bool,
    unsecure: bool,
    instance: str = state_mod.DEFAULT_INSTANCE,
) -> None:
    profile = offer.profile
    state.instance_id = int(instance_id)
    state.status = "provisioning"
    state.data["instance_name"] = instance
    state.offer_id = offer.offer_id
    state.data["machine_id"] = offer.machine_id
    # Exclusions stick to the deployment: the monitor merges them from state
    # so it never recommends an offer/host the user already ruled out.
    state.data.update(offer.exclusions.to_state())
    state.gpu = offer.gpu_name
    state.dph = float(offer.dph)
    state.local_port = offer.local_port if offer.local_port is not None else config.ssh.local_port
    state.location = offer.offer_data.get("geolocation", "") or offer.offer_data.get("location", "")
    state.inet_down = offer.offer_data.get("inet_down", 0.0)
    state.inet_down_cost = offer.offer_data.get("inet_down_cost", 0.0)
    state.inet_up = offer.offer_data.get("inet_up", 0.0)
    state.inet_up_cost = offer.offer_data.get("inet_up_cost", 0.0)
    state.disk_bw = offer.offer_data.get("disk_bw", 0.0)
    state.reliability = offer.offer_data.get("reliability", 0.0)
    state.model = offer.model
    state.hf_revision = config.model.hf_revision
    state.ctx_size = offer.ctx_size
    state.api_key = api_key
    state.image = offer.selected_image
    state.label = label
    state.profile = profile.name
    state.data["min_gpu_vram_mb"] = profile.min_gpu_vram_mb
    state.data["profile_name"] = profile.name
    state.monitor_group = profile.monitor_group or ""
    state.disk_gb = offer.disk_gb
    state.interruptible = offer.interruptible
    state.bid_price = offer.bid_price if offer.interruptible else None
    state.expected_session_seconds = offer.session_seconds
    state.run_id = run_id
    state.run_dir = run_dir
    state.started_at = _now_rfc()
    state.started_epoch = _now_epoch()
    state.run_started_at = run_started
    state.run_started_epoch = run_epoch
    state.unsecure = unsecure
    state.slot_cache_enabled = not no_cache
    state.slot_cache_host = config.cache.host
    state.slot_cache_port = config.cache.port
    state.slot_cache_user = config.cache.user
    state.slot_cache_root = config.cache.root
    state.slot_cache_session = session
    state.slot_cache_max_gb = config.cache.max_gb
    state.slot_cache_local_dir = cache._default_local_dir(config)
    state.slot_cache_use_shm = config.cache.use_shm


def _create_fresh_instance(
    config: Config,
    instance: str,
    offer: _FreshOffer,
    cache_session: Optional[str],
    no_cache: bool,
    unsecure: bool,
) -> State:
    """Create the provider instance and initialize a fresh ``State``."""
    sdir = state_mod.instance_state_dir(config.root_dir, instance)
    profile = offer.profile
    image = offer.image
    run_id = utils.make_run_id(profile.name)
    run_dir = init_run_dir(runs_dir(config.root_dir), profile.name, run_id)
    run_started = _now_rfc()
    run_epoch = _now_epoch()
    api_key = config.secrets.get("MODEL_API_KEY") or utils.make_api_key()
    session = cache_session or config.cache.session

    metadata = _fresh_metadata(config, offer, session, no_cache, unsecure, run_id, run_started, instance)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (run_dir / "metadata.json").chmod(0o600)

    env = _env_dict(config, profile, image, offer.model, offer.ctx_size, api_key, unsecure, no_cache, session)
    extra = _extra_args(config, no_cache=no_cache)
    if instance == state_mod.DEFAULT_INSTANCE:
        label = f"hostai-{profile.name}-{_now_epoch()}"
    else:
        label = f"hostai-{instance}-{profile.name}-{_now_epoch()}"

    volume_info: Optional[Dict[str, Any]] = None
    if config.vast.volume_id and config.vast.volume_mount_path:
        try:
            volume_info = {"volume_id": int(config.vast.volume_id), "mount_path": config.vast.volume_mount_path}
        except ValueError:
            volume_info = None

    create_kwargs: Dict[str, Any] = {
        "image": offer.selected_image,
        "disk": offer.disk_gb,
        "env": env,
        "label": label,
        "extra": extra,
        "runtype": "args",
        "args": None,
        "volume_info": volume_info,
    }
    if offer.interruptible:
        create_kwargs["bid_price"] = offer.bid_price
    try:
        create_raw = _provider(config).create_instance(offer.offer_id, **create_kwargs)
    except Exception as exc:
        raise click.ClickException(f"create instance failed: {exc}")

    instance_id = create_raw.get("new_contract") or create_raw.get("instance_id") or create_raw.get("id")
    if not instance_id:
        raise click.ClickException(f"create response did not contain an instance ID: {create_raw}")

    state = State.load(sdir / "state.json")
    _fill_fresh_state(
        state,
        config,
        offer,
        instance_id,
        api_key,
        label,
        session,
        run_id,
        run_dir,
        run_started,
        run_epoch,
        no_cache,
        unsecure,
        instance=instance,
    )

    provider = _provider(config)
    private_key: Optional[Path] = getattr(provider, "ssh_private_key", None)
    if private_key:
        state.ssh_identity = private_key

    state.save()
    state.save_metadata(run_dir, status="provisioning")
    return state


def _do_fresh(
    config: Config,
    instance: str,
    profile_name: str,
    cache_session: Optional[str],
    local_port: Optional[int],
    max_price: Optional[float],
    unverified: bool,
    unsecure: bool,
    no_cache: bool,
    abort_if_shm_too_small: bool,
    offer: Optional[int],
    *,
    exclusions: Optional[market.OfferExclusions] = None,
    bid_price: Optional[float] = None,
    session_seconds: Optional[int] = None,
    dry_run: bool = False,
    allow_unvalidated: bool = False,
) -> None:
    # The allocation lock serializes the claim phase (live-instance check,
    # client-port choice, provider create, state write) so two parallel `up`
    # runs cannot pick the same name or local port.  The slow boot afterwards
    # runs outside the lock.
    with _common.allocation_lock(config):
        _check_for_live_instance(config, instance)
        plan = _resolve_fresh_offer(
            config,
            profile_name,
            local_port,
            max_price,
            unverified,
            offer,
            exclusions or market.OfferExclusions(),
            bid_price,
            session_seconds,
            allow_unvalidated,
            instance,
        )

        if dry_run:
            _log("\nDRY RUN: not creating an instance")
            return

        state = _create_fresh_instance(config, instance, plan, cache_session, no_cache, unsecure)

    try:
        _do_fresh_core(config, state, plan.image, no_cache, abort_if_shm_too_small)
    except click.ClickException:
        _cleanup_instance(config, state, "provisioning failed")
        raise
    except Exception as exc:
        _cleanup_instance(config, state, f"provisioning error: {exc}")
        raise click.ClickException(str(exc)) from exc


def _gpu_preflight_or_fail(
    config: Config,
    state: State,
    known_hosts: Path,
    *,
    cleanup_on_fail: bool = False,
) -> None:
    """Verify GPU VRAM meets the profile minimum; raise ``ClickException`` on failure."""
    profile_label = state.data.get("profile_name") or state.profile or "unknown"
    vram_rc = _gpu_vram_preflight(
        state.ssh_url,
        known_hosts,
        config,
        profile_label,
        state.data.get("min_gpu_vram_mb"),
        state=state,
    )
    if vram_rc == 1:
        # The preflight already logged the exact GPU and requirement.
        if cleanup_on_fail:
            _cleanup_instance(config, state, "GPU memory below profile minimum")
        raise click.ClickException(f"GPU does not meet the {profile_label} memory requirement")
    if vram_rc == 2:
        if cleanup_on_fail:
            _cleanup_instance(config, state, "nvidia-smi failed")
        raise click.ClickException("[gpu] nvidia-smi failed; cannot verify GPU VRAM")


def _do_fresh_core(
    config: Config,
    state: State,
    image: Any,
    no_cache: bool,
    abort_if_shm_too_small: bool,
) -> None:
    """Provision a freshly created instance (SSH, cache, tunnel, TLS, API)."""
    _log(f"[boot] instance {state.instance_id} created; waiting for SSH...")
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

    # GPU memory preflight: fail before model download if the visible VRAM is
    # below the profile's minimum (e.g. a locked 8/10 GB CMP 170HX).
    _gpu_preflight_or_fail(config, state, known_hosts)

    _cpu_arch_preflight(state.ssh_url, known_hosts, config, state=state)

    # runtime preflight
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
    client_base_url, cache_remote, cache_enabled = _start_instance_runtime(
        config,
        state,
        no_cache=no_cache,
        abort_if_shm_too_small=abort_if_shm_too_small,
    )

    state = State.load(state.state_file)
    state.data["startup_seconds"] = state.data["ready_epoch"] - (state.started_epoch or 0)
    state.save()
    run_dir = state.run_dir
    if run_dir:
        state.save_metadata(run_dir, status="ready")

    if run_dir and state.ssh_url:
        telemetry = _capture_disk_telemetry(config, state, known_hosts)
        if telemetry:
            _log(
                f"[disk] telemetry: {len(telemetry['records'])} stages, free={telemetry['records'][-1]['free_bytes'] / 1e9:.2f}GB"
            )

    instance_name = state.data.get("instance_name") or state_mod.instance_name_for_state_file(state.state_file)
    env_path = state.state_file.parent / "env"
    try:
        env_display = env_path.relative_to(config.root_dir)
    except ValueError:
        env_display = env_path
    name_flag = "" if instance_name == state_mod.DEFAULT_INSTANCE else f" --name {instance_name}"

    _log("\nREADY")
    _log(f"  Profile:   {state.profile} (sm_{image.cuda_arch})")
    if instance_name != state_mod.DEFAULT_INSTANCE:
        _log(f"  Name:      {instance_name}")
    _log(f"  Image:     {state.image}")
    _log(f"  GPU:       {state.gpu}")
    _log(f"  Cost:      ${float(state.dph):.4f}/h")
    _log(f"  Context:   {state.ctx_size}")
    _log(f"  API:       {client_base_url}")
    _log(f"  Instance:  {state.instance_id}")
    if state.data.get("machine_id") is not None:
        _log(f"  Machine:   {state.data['machine_id']}")
    _log(f"  Run log:   {run_dir}")
    if cache_enabled:
        _log(f"  Slot cache: session={state.slot_cache_session} remote={cache_remote}")
    _log(f"\nRun: source {env_display}")
    _log(f"Stop: hostai down{name_flag}")

    maybe_start_watchdog(config, state)
    maybe_start_monitor(config, state)


def _deliver_tls_cert(config: Config, state: State, known_hosts: Path) -> str:
    """Deliver TLS certs to the container; return ``https`` or ``http``."""
    if state.unsecure:
        return "http"
    tls_dir = tls.ensure_local_tls_dir(config.root_dir)
    # Regeneration mutates a shared on-disk cert pair — serialize it so two
    # parallel `up` runs cannot interleave writes and ship a mismatched pair.
    with _common.allocation_lock(config):
        if tls.cert_needs_regeneration(tls_dir):
            if (tls_dir / "server.crt").exists():
                _log("[tls] certificate expired or near expiry; regenerating")
            tls.generate_cert(tls_dir)
    _log("[tls] delivering certificates to container")
    tls_deadline = time.monotonic() + min(120.0, float(config.ssh.start_timeout or 1200))
    last_alive_check = 0.0
    while True:
        if tls.deliver_cert(
            state.ssh_url,
            tls_dir,
            known_hosts=known_hosts,
            config=config,
            state=state,
            timeout=60,
        ):
            break
        now = time.monotonic()
        if now - last_alive_check >= _INSTANCE_CHECK_INTERVAL:
            last_alive_check = now
            _ensure_instance_alive(config, state, "TLS certificate delivery")
        if now >= tls_deadline:
            raise click.ClickException("TLS certificate delivery failed")
        _log("[tls] delivery attempt failed; retrying in 5s", err=True)
        time.sleep(5)
    state.tls_ca = tls_dir / "ca.crt"
    state.save()
    _log("[tls] certificates delivered")
    return "https"


def _prepare_slot_cache(
    config: Config,
    state: State,
    no_cache: bool,
    abort_if_shm_too_small: bool,
    known_hosts: Path,
) -> Tuple[bool, str]:
    """Validate/cache key/shm/prefetch; return ``(cache_enabled, cache_remote)``."""
    cache_enabled = state.slot_cache_enabled and not no_cache
    if cache_enabled and not cache.validate_cache_config(config):
        _log("[cache] WARNING: cache config invalid; continuing cold", err=True)
        cache_enabled = False
        state.slot_cache_enabled = False

    if cache_enabled and not config.cache.rclone and not cache.install_cache_key_on_vast(state, config):
        _log("[cache] WARNING: could not install cache key; continuing cold", err=True)
        cache_enabled = False
        state.slot_cache_enabled = False

    if not cache_enabled:
        return False, ""

    ssh_url = state.ssh_url or ""
    if not ssh_url:
        _log("[cache] no SSH endpoint; continuing without cache", err=True)
        return False, ""

    llama_commit = cache.resolve_llama_commit(state, ssh_url, known_hosts)
    state.data["llama_cpp_commit"] = llama_commit

    if config.cache.use_shm:
        shm_rc = _shm_preflight(
            ssh_url,
            config,
            known_hosts,
            config.cache.shm_min_gb or 30,
        )
        if shm_rc == 1:
            if abort_if_shm_too_small or config.cache.shm_require:
                raise click.ClickException(
                    "[cache] /dev/shm is too small and abort-if-shm-too-small/shm_require is set"
                )
            _log("[cache] /dev/shm is too small; falling back to disk slot cache", err=True)
            state.slot_cache_use_shm = False
            state.slot_cache_local_dir = "/var/lib/qwen38/slots"
        elif shm_rc == 2:
            raise click.ClickException("[cache] slot cache /dev/shm preflight failed")

    signature = cache._signature_for_state(config, state, llama_commit)
    cache_remote = cache.remote_cache_dir(config, signature, state.slot_cache_session)
    state.data["slot_cache_signature"] = signature
    state.data["slot_cache_remote_dir"] = cache_remote
    state.data["slot_cache_restore"] = "pending"
    state.save()

    if _prefetch_slot_cache_to_vast(ssh_url, config, state.slot_cache_local_dir, cache_remote, known_hosts):
        _log("[cache] prefetched slot from cache server")
        state.data["slot_cache_prefetch"] = "ok"
    else:
        _log("[cache] no slot cache on server; will start cold", err=True)
        state.data["slot_cache_prefetch"] = "empty"
    state.save()
    return cache_enabled, cache_remote


def _start_client_endpoint(config: Config, state: State) -> Tuple[int, State]:
    """Start the proxy or SSH tunnel and return ``(proxy_port, updated_state)``."""
    if config.proxy.tokenized_only:
        proxy_port = _start_proxy(config, state, client_api_scheme="http")
        if not proxy_port:
            raise click.ClickException("[proxy] failed to start")
        _log(f"[tunnel] proxy on localhost:{proxy_port}")
    else:
        proxy_port = ssh.ensure_tunnel(config, state)
        _log(f"[tunnel] localhost:{proxy_port}")

    # Reload state after the proxy/tunnel process has written its metadata
    # (proxy_pid, upstream_socket, etc.) so the rest of provisioning and the
    # final ready state are consistent with the running proxy.
    state = State.load(state.state_file)
    return proxy_port, state


def _wait_for_client_api(config: Config, state: State, proxy_port: int, api_scheme: str) -> Tuple[LlamaClient, str]:
    """Build the client URL, write the env file, and wait for API health."""
    client_scheme = "http" if config.proxy.tokenized_only else api_scheme
    client_api_url = f"{client_scheme}://127.0.0.1:{proxy_port}"
    client_base_url = f"{client_api_url}/v1"
    _write_env_file(config, state, client_api_url, client_base_url)

    # In tokenized mode the proxy is the endpoint and only speaks HTTP, so use a
    # temporary client state with unsecure=True to avoid requiring TLS for the local hop.
    if config.proxy.tokenized_only:
        client_state = State.load(state.state_file)
        client_state.local_port = proxy_port
        client_state.unsecure = True
        client = LlamaClient(config, client_state)
        _wait_for_api(config, client_state, config.ssh.start_timeout, client, stage_label="end-to-end via proxy")
        return client, client_base_url

    client = LlamaClient(config, state)
    _wait_for_api(config, state, config.ssh.start_timeout, client, stage_label="end-to-end")
    return client, client_base_url


def _restore_or_clear_slot_cache(config: Config, client: LlamaClient, state: State, cache_enabled: bool) -> None:
    if not cache_enabled:
        return
    if client.slot_restore(config.cache.slot_id):
        _log("[cache] slot restored")
        state.data["slot_cache_restore"] = "restored"
    else:
        _log("[cache] WARNING: slot restore failed; continuing cold", err=True)
        state.data["slot_cache_restore"] = "failed"
    state.save()


def _start_instance_runtime(
    config: Config,
    state: State,
    *,
    no_cache: bool,
    abort_if_shm_too_small: bool,
) -> Tuple[str, str, bool]:
    """Finish booting an SSH-reachable instance.

    Shared by ``_do_fresh_core`` and ``_do_restart``: TLS delivery, slot-cache
    prefetch, tunnel/proxy startup, API health wait, and slot restore.  Marks
    the state ``running`` and returns ``(client_base_url, cache_remote,
    cache_enabled)``.
    """
    known_hosts = state.state_file.parent / "known_hosts"

    # TLS: deliver certificates as soon as SSH is ready, before the slower cache
    # and slot-cache steps, so the remote start.sh does not time out waiting.
    api_scheme = _deliver_tls_cert(config, state, known_hosts)

    cache_enabled, cache_remote = _prepare_slot_cache(config, state, no_cache, abort_if_shm_too_small, known_hosts)

    # Start the local proxy if tokenized-only is enabled. The proxy owns the
    # SSH tunnel to the remote Unix socket and provides the client-facing
    # OpenAI endpoint on LOCAL_PORT (or [proxy].port).
    proxy_port, state = _start_client_endpoint(config, state)

    # Build client API URL and wait for the API to be reachable.
    client, client_base_url = _wait_for_client_api(config, state, proxy_port, api_scheme)

    # restore slot cache
    _restore_or_clear_slot_cache(config, client, state, cache_enabled)

    state.status = "running"
    state.data["ready_at"] = _now_rfc()
    state.data["ready_epoch"] = _now_epoch()
    state.save()

    return client_base_url, cache_remote, cache_enabled


def _do_restart(
    config: Config,
    instance: str,
    profile_name: str,
    local_port: Optional[int],
    unsecure: bool,
    no_cache: bool,
    *,
    allow_unvalidated: bool = False,
) -> None:
    _check_production_validation(config, allow_unvalidated)
    state = State.load(state_mod.instance_state_file(config.root_dir, instance))
    if not state.instance_id:
        flag = "" if instance == state_mod.DEFAULT_INSTANCE else f" --name {instance}"
        raise click.ClickException(f"no state to restart for instance '{instance}'; run 'hostai up{flag}' first")

    state.data["instance_name"] = instance
    state.unsecure = unsecure
    resolved = _resolve_client_port(config, user_port=local_port, state_port=state.local_port, instance=instance)
    if resolved != state.local_port:
        state.local_port = resolved
        state.save()

    inst = _provider(config).get_instance(state.instance_id)
    status = inst.get("actual_status") or inst.get("status") if inst else None
    if status != "running":
        try:
            _provider(config).start_instance(state.instance_id)
            _log(f"[restart] started instance {state.instance_id}")
        except Exception as exc:
            raise click.ClickException(f"failed to start instance: {exc}")

    _wait_for_ssh_endpoint(config, state, config.ssh.start_timeout)
    known_hosts = state.state_file.parent / "known_hosts"
    # stopped_is_dead=False: Vast can still report "stopped" briefly while
    # a restarted instance transitions to loading.
    alive_check = lambda: _ensure_instance_alive(config, state, "SSH wait", stopped_is_dead=False)  # noqa: E731
    if not ssh.wait_for_ssh(
        state.ssh_url,
        known_hosts=known_hosts,
        config=config,
        state=state,
        timeout=300,
        alive_check=alive_check,
    ):
        _ensure_instance_alive(config, state, "SSH wait", stopped_is_dead=False)
        raise click.ClickException("SSH daemon did not become reachable")

    # GPU memory preflight on restart: if the profile has a minimum, verify it.
    _gpu_preflight_or_fail(config, state, known_hosts, cleanup_on_fail=True)

    _cpu_arch_preflight(state.ssh_url, known_hosts, config, state=state)

    client_base_url, _cache_remote, _cache_enabled = _start_instance_runtime(
        config,
        state,
        no_cache=no_cache,
        abort_if_shm_too_small=False,
    )

    state = State.load(state.state_file)
    if state.run_dir:
        state.save_metadata(state.run_dir, status="restarted")

    _log("\nREADY")
    _log(f"  Profile:   {state.profile}")
    if instance != state_mod.DEFAULT_INSTANCE:
        _log(f"  Name:      {instance}")
    _log(f"  API:       {client_base_url}")
    _log(f"  Instance:  {state.instance_id}")
    if state.data.get("machine_id") is not None:
        _log(f"  Machine:   {state.data['machine_id']}")

    maybe_start_watchdog(config, state)
    maybe_start_monitor(config, state)
