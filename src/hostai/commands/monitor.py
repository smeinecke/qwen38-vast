import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from hostai.config import Config
from hostai.profiles import Profiles
from hostai.state import State, state_dir
from hostai.vast import search_instance_offers


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
) -> List[str]:
    if profile:
        return [profile]
    if config.monitor.profile:
        return [config.monitor.profile]
    group = group or config.monitor.group
    if group:
        names = [p.name for p in profiles.profiles if p.monitor_group == group]
        if not names:
            raise click.ClickException(f"no profiles in monitor group '{group}'")
        return names
    return [config.hostai.default_profile]


def _search_profile(config: Config, profiles: Profiles, profile_name: str) -> Optional[List[Dict[str, Any]]]:
    p = profiles.resolve_profile(profile_name)
    if not p:
        return None
    query = config.hostai.gpu_query_override or p.gpu_query
    max_dph = config.market.max_dph
    if config.market.max_inet_down_cost:
        query += f" inet_down_cost<={config.market.max_inet_down_cost}"
        query += f" inet_up_cost<={config.market.max_inet_up_cost}"
    if "dph" not in query:
        query += f" dph_total <= {max_dph}"
    try:
        return search_instance_offers(config, query, limit=10, order="dph_total")
    except Exception:
        return None


def _search_targets(config: Config, profiles: Profiles, targets: List[str]) -> List[Dict[str, Any]]:
    """Search a list of profiles and merge/sort the results by dph_total."""
    all_offers: List[Dict[str, Any]] = []
    for target in targets:
        offers = _search_profile(config, profiles, target)
        if offers:
            all_offers.extend(offers)
    all_offers.sort(key=lambda o: o.get("dph_total", float("inf")))
    return all_offers


@cmd_monitor.command("once", help="Run a single price check.")
@click.option("--profile", help="Profile to monitor.")
@click.option("--group", help="Monitor group to search.")
@click.pass_obj
def cmd_monitor_once(config: Config, profile: Optional[str], group: Optional[str]):
    profiles = Profiles.from_file(config.root_dir / config.hostai.profiles_file)
    targets = _resolve_monitor_targets(config, profiles, profile, group)
    for target in targets:
        if not profiles.resolve_profile(target):
            raise click.ClickException(f"unknown profile '{target}'")
    offers = _search_targets(config, profiles, targets)
    if not offers:
        click.echo("no matching offers")
        return
    current = State.load(state_dir(config.root_dir) / "state.json")
    current_dph = current.dph if current.exists else None
    best = offers[0]
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
    targets = _resolve_monitor_targets(config, profiles, profile, group)
    for target in targets:
        if not profiles.resolve_profile(target):
            raise click.ClickException(f"unknown profile '{target}'")
    label = group or ", ".join(targets)
    click.echo(f"[monitor] watching '{label}' every {sec}s (threshold {pct}%)")
    try:
        while True:
            offers = _search_targets(config, profiles, targets)
            if offers:
                current = State.load(state_dir(config.root_dir) / "state.json")
                current_dph = current.dph if current.exists else None
                best = offers[0]
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
