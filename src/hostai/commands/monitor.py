"""Monitor Vast prices for cheaper, equivalent offers.

The monitor reuses the same query building, storage allocation, eligibility
filtering, and hardware-rank logic as ``hostai up``.  It only alerts for an
offer that uses the active/equivalent local profile, an equal-or-better GPU,
and has a lower hourly price.

The comparison context comes from HostAI profile configuration, not from Vast
offer fields.  When an instance is running, the active profile and context size
from ``state.json`` take precedence over the configured default so the monitor
compares apples-to-apples.
"""

import dataclasses
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from hostai import market
from hostai.config import Config
from hostai.profiles import Profile, Profiles
from hostai.state import State, state_dir


def _monitor_pid_file(config: Config) -> Path:
    return config.root_dir / ".hostai-cache" / "monitor.pid"


def _monitor_log_file(config: Config) -> Path:
    return config.root_dir / ".hostai-cache" / "monitor.log"


def _monitor_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _hostai_executable() -> str:
    exe = shutil.which("hostai")
    if exe:
        return exe
    return sys.argv[0]


@click.group("monitor", help="Monitor Vast prices for cheaper offers.")
@click.pass_obj
def cmd_monitor(config: Config):
    pass


def _resolve_monitor_targets(
    config: Config,
    profiles: Profiles,
    profile: Optional[str],
    group: Optional[str],
    current: State,
) -> List[Profile]:
    # A running instance is the authoritative source of context/profile.
    # Use all monitor-searchable profiles that are compatible with the running
    # workload (same context and monitor group) so the monitor does not fall
    # back to the default profile silently.
    if current.exists and current.instance_id and current.profile:
        active = profiles.resolve_profile(current.profile)
        if active:
            ctx = current.ctx_size or active.ctx_size
            active = dataclasses.replace(active, ctx_size=ctx)
            if active.monitor_group:
                targets = [p for p in profiles.all_monitor_profiles(ctx) if p.monitor_group == active.monitor_group]
                # The exact active profile is always a candidate even when it
                # has monitor_search=false, because we need to compare against
                # the same hardware class.
                if active not in targets:
                    targets.insert(0, active)
            else:
                targets = [active]
            if targets:
                return targets

    if profile:
        p = profiles.resolve_profile(profile)
        if p and p.monitor_search:
            return [p]
        raise click.ClickException(f"unknown or monitor-disabled profile '{profile}'")

    if config.monitor.profile:
        p = profiles.resolve_profile(config.monitor.profile)
        if p and p.monitor_search:
            return [p]

    group = group or config.monitor.group
    if group:
        targets = [p for p in profiles.profiles if p.monitor_group == group and p.monitor_search]
        if not targets:
            raise click.ClickException(f"no profiles in monitor group '{group}'")
        return targets

    p = profiles.resolve_profile(config.hostai.default_profile)
    if p and p.monitor_search:
        return [p]

    raise click.ClickException("no monitorable profile configured")


def _search_profiles(
    config: Config,
    profiles: Profiles,
    targets: List[Profile],
    current: State,
) -> List[Dict[str, Any]]:
    """Search a list of profiles and merge the results.

    If a running instance is active, its bid price and current dph constrain
    the search so the monitor only compares against offers that could actually
    be rented at the same economics.
    """
    bid_price = current.bid_price if (current.exists and current.bid_price is not None) else None
    max_price = (
        current.dph
        if (current.exists and current.instance_id and current.dph is not None and current.dph > 0)
        else None
    )
    offer_type = "bid" if bid_price is not None else "on-demand"

    all_offers: List[Dict[str, Any]] = []
    for p in targets:
        query, max_dph = market.build_search_query(
            config,
            profiles,
            p,
            max_price=max_price,
            bid_price=bid_price,
            unverified=config.market.allow_unverified,
            offer=None,
        )
        disk_gb = market.resolved_disk_gb(p, config)
        try:
            offers = market.search_offers(
                config,
                query,
                storage=disk_gb,
                max_dph=max_dph,
                offer=None,
                offer_type=offer_type,
                limit=10,
            )
        except Exception:
            continue
        all_offers.extend(offers)
    return all_offers


def _ranked_best_for_monitor(
    config: Config,
    profiles: Profiles,
    current: State,
    candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the cheapest offer that is an economic/performance upgrade."""
    if not candidates:
        return None

    current_gpu = current.gpu
    if current.exists and current.dph is not None:
        max_dph = current.dph
    else:
        max_dph = config.market.max_dph

    # The context/profile comparison is guaranteed by selecting the right local
    # profile(s) above.  Still enforce an exact-context check when the Vast
    # offer carries a non-zero ctx_size so a wrong-context offer cannot trigger.
    running_ctx = current.ctx_size if (current.exists and current.instance_id) else None
    matches = market.filter_eligible_offers(
        candidates,
        max_dph=max_dph,
        offer=None,
        current_gpu=current_gpu,
        profiles=profiles,
        ctx_size=running_ctx,
    )
    if not matches:
        return None

    matches.sort(key=lambda o: o.get("dph_total", float("inf")))
    return matches[0]


@cmd_monitor.command("once", help="Run a single price check.")
@click.option("--profile", help="Profile to monitor.")
@click.option("--group", help="Monitor group to search.")
@click.pass_obj
def cmd_monitor_once(config: Config, profile: Optional[str], group: Optional[str]):
    profiles = Profiles.from_file(config.root_dir / config.hostai.profiles_file)
    current = State.load(state_dir(config.root_dir) / "state.json")
    targets = _resolve_monitor_targets(config, profiles, profile, group, current)
    current_dph = current.dph if current.exists else None

    all_offers = _search_profiles(config, profiles, targets, current)
    best = _ranked_best_for_monitor(config, profiles, current, all_offers)
    if best is None:
        click.echo("no matching offers")
        return

    best_dph = best.get("dph_total", 0)
    click.echo(
        f"[monitor] best {best.get('gpu_name')} at ${best_dph:.4f}/h "
        f"(id={best.get('id') or best.get('ask_contract_id')})"
    )
    if current_dph and current_dph > 0:
        saving = (current_dph - best_dph) / current_dph * 100
        click.echo(f"[monitor] saving vs current: {saving:.1f}%")


@cmd_monitor.command("watch", help="Run a foreground price watch loop.")
@click.option("--profile", help="Profile to monitor.")
@click.option("--group", help="Monitor group to search.")
@click.option("--interval", type=int, default=None, help="Seconds between checks.")
@click.option("--threshold", type=float, default=None, help="Pct saving before alerting.")
@click.pass_obj
def cmd_monitor_watch(
    config: Config, profile: Optional[str], group: Optional[str], interval: Optional[int], threshold: Optional[float]
):
    sec = interval if interval is not None else config.monitor.interval
    pct = threshold if threshold is not None else config.monitor.threshold_pct
    profiles = Profiles.from_file(config.root_dir / config.hostai.profiles_file)
    current = State.load(state_dir(config.root_dir) / "state.json")
    targets = _resolve_monitor_targets(config, profiles, profile, group, current)
    label = group or ", ".join(p.name for p in targets)
    click.echo(f"[monitor] watching '{label}' every {sec}s (threshold {pct}%)")
    try:
        while True:
            current = State.load(state_dir(config.root_dir) / "state.json")
            current_dph = current.dph if current.exists else None
            all_offers = _search_profiles(config, profiles, targets, current)
            best = _ranked_best_for_monitor(config, profiles, current, all_offers)
            if best:
                best_dph = best.get("dph_total", 0)
                if current_dph and current_dph > 0 and current_dph > best_dph:
                    saving = (current_dph - best_dph) / current_dph * 100
                    if saving >= pct:
                        click.echo(
                            f"[monitor] ALERT: {best.get('gpu_name')} ${best_dph:.4f}/h is {saving:.1f}% cheaper"
                        )
                    else:
                        click.echo(f"[monitor] best ${best_dph:.4f}/h (saving {saving:.1f}%)")
                else:
                    click.echo(f"[monitor] best ${best_dph:.4f}/h")
            else:
                click.echo("[monitor] no cheaper equivalent offer")
            time.sleep(sec)
    except KeyboardInterrupt:
        click.echo("\n[monitor] stopped")


@cmd_monitor.command("start", help="Start the price monitor daemon.")
@click.option("--profile", help="Profile to monitor.")
@click.option("--group", help="Monitor group to search.")
@click.option("--interval", type=int, default=None, help="Seconds between checks.")
@click.option("--threshold", type=float, default=None, help="Pct saving before alerting.")
@click.pass_obj
def cmd_monitor_start(
    config: Config, profile: Optional[str], group: Optional[str], interval: Optional[int], threshold: Optional[float]
):
    pid_file = _monitor_pid_file(config)
    log_file = _monitor_log_file(config)

    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if _monitor_is_running(pid):
                click.echo(f"[monitor] already running (pid {pid})")
                return
        except (ValueError, OSError):
            pass

    sec = interval if interval is not None else config.monitor.interval
    pct = threshold if threshold is not None else config.monitor.threshold_pct
    target_label = profile or group or config.monitor.profile or config.monitor.group or config.hostai.default_profile

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n# monitor start {target_label} interval={sec}s threshold={pct}%\n")

    cmd = [_hostai_executable(), "monitor", "watch", "--interval", str(sec), "--threshold", str(pct)]
    if profile:
        cmd.extend(["--profile", profile])
    if group:
        cmd.extend(["--group", group])

    with log_file.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    pid_file.write_text(str(proc.pid))
    click.echo(f"[monitor] started daemon (pid {proc.pid}) logging to {log_file}")


@cmd_monitor.command("stop", help="Stop the price monitor daemon.")
@click.pass_obj
def cmd_monitor_stop(config: Config):
    pid_file = _monitor_pid_file(config)
    if not pid_file.exists():
        click.echo("[monitor] not running")
        return
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        pid_file.unlink(missing_ok=True)
        click.echo("[monitor] not running")
        return

    if not _monitor_is_running(pid):
        pid_file.unlink(missing_ok=True)
        click.echo("[monitor] not running")
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except Exception as exc:
        click.echo(f"[monitor] could not stop daemon: {exc}", err=True)
        return

    # Wait briefly for the process to exit.
    for _ in range(20):
        if not _monitor_is_running(pid):
            break
        time.sleep(0.2)

    if _monitor_is_running(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception as exc:
            click.echo(f"[monitor] could not kill daemon: {exc}", err=True)

    pid_file.unlink(missing_ok=True)
    click.echo("[monitor] stopped")


@cmd_monitor.command("status", help="Show monitor daemon status.")
@click.pass_obj
def cmd_monitor_status(config: Config):
    pid_file = _monitor_pid_file(config)
    if not pid_file.exists():
        click.echo("[monitor] not running")
        return
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        click.echo("[monitor] not running (stale pid file)")
        pid_file.unlink(missing_ok=True)
        return

    if _monitor_is_running(pid):
        log_file = _monitor_log_file(config)
        click.echo(f"[monitor] running (pid {pid}) log={log_file}")
    else:
        click.echo("[monitor] not running (stale pid file)")
        pid_file.unlink(missing_ok=True)


def maybe_start_monitor(config: Config, state: State) -> None:
    """Start the price monitor daemon after hostai up when auto_start is enabled."""
    if not config.monitor.auto_start:
        return
    if not state.instance_id:
        return
    pid_file = _monitor_pid_file(config)
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if _monitor_is_running(pid):
                return
        except (ValueError, OSError):
            pass
    _log_monitor(config, f"auto-starting monitor for instance {state.instance_id}")
    if cmd_monitor_start.callback is not None:
        cmd_monitor_start.callback(config, profile=None, group=None, interval=None, threshold=None)


def stop_monitor(config: Config) -> None:
    """Stop the monitor daemon if it is running."""
    pid_file = _monitor_pid_file(config)
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        pid_file.unlink(missing_ok=True)
        return
    if not _monitor_is_running(pid):
        pid_file.unlink(missing_ok=True)
        return
    if cmd_monitor_stop.callback is not None:
        cmd_monitor_stop.callback(config)


def _log_monitor(config: Config, message: str) -> None:
    log_file = _monitor_log_file(config)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n")


@cmd_monitor.command("logs", help="Tail the monitor daemon log.")
@click.option("--lines", type=int, default=50, help="Number of lines to show.")
@click.pass_obj
def cmd_monitor_logs(config: Config, lines: int):
    log_file = _monitor_log_file(config)
    if not log_file.exists():
        click.echo("[monitor] no log file")
        return
    try:
        result = subprocess.run(
            ["tail", "-n", str(lines), str(log_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        click.echo(result.stdout, nl=False)
    except Exception as exc:
        click.echo(f"[monitor] could not read log: {exc}", err=True)
