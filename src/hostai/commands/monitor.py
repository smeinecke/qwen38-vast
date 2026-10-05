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
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

from hostai import market, notify
from hostai import state as state_mod
from hostai.commands import _common
from hostai.config import Config
from hostai.profiles import Profile, Profiles
from hostai.state import State


def _monitor_pid_file(config: Config, instance: Optional[str] = None) -> Path:
    return _common.daemon_pid_file(config, "monitor", instance)


def _monitor_log_file(config: Config, instance: Optional[str] = None) -> Path:
    return _common.daemon_log_file(config, "monitor", instance)


@click.group("monitor", help="Monitor Vast prices for cheaper offers.")
@click.pass_obj
def cmd_monitor(config: Config):
    pass


def _targets_for_running_instance(profiles: Profiles, current: State) -> Optional[List[Profile]]:
    """Return compatible monitor targets for the running workload, or None."""
    if not (current.exists and current.instance_id and current.profile):
        return None
    active = profiles.resolve_profile(current.profile)
    if not active:
        return None
    ctx = current.ctx_size or active.ctx_size
    active = dataclasses.replace(active, ctx_size=ctx)
    if active.monitor_group:
        targets = [p for p in profiles.all_monitor_profiles(ctx) if p.monitor_group == active.monitor_group]
        # The exact active profile is always a candidate even when it has
        # monitor_search=false, because we need to compare against the same
        # hardware class.
        if active not in targets:
            targets.insert(0, active)
    else:
        targets = [active]
    return targets or None


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
    targets = _targets_for_running_instance(profiles, current)
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


def _monitor_skips(current: State, exclusions: Optional[market.OfferExclusions] = None) -> market.OfferExclusions:
    """Merge CLI exclusions with the ones ``up`` recorded in state.json.

    Exclusions passed to ``hostai up`` (e.g. a host that failed provisioning)
    stay attached to the deployment, so the monitor does not recommend an
    offer the user already ruled out.
    """
    return market.OfferExclusions.from_state(current.data).merged(exclusions or market.OfferExclusions())


def _monitor_price_cap(current: State, max_price: Optional[float]) -> Optional[float]:
    """Merge ``--max-price`` with the running instance's own dph cap.

    A tracked instance already constrains searches to its current dph — only
    cheaper offers are interesting — so an explicit cap can only tighten it.
    ``None`` means "fall back to [market].max_dph" inside the search layer.
    """
    cap = (
        current.dph
        if (current.exists and current.instance_id and current.dph is not None and current.dph > 0)
        else None
    )
    if max_price is not None:
        cap = min(cap, max_price) if cap is not None else max_price
    return cap


def _search_profiles(
    config: Config,
    profiles: Profiles,
    targets: List[Profile],
    current: State,
    *,
    max_price: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Search a list of profiles and merge the results.

    If a running instance is active, its bid price and current dph constrain
    the search so the monitor only compares against offers that could actually
    be rented at the same economics.  ``max_price`` further tightens that cap.
    """
    bid_price = current.bid_price if (current.exists and current.bid_price is not None) else None
    max_price = _monitor_price_cap(current, max_price)
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
    *,
    exclusions: Optional[market.OfferExclusions] = None,
    max_price: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Return the cheapest offer that is an economic/performance upgrade."""
    if not candidates:
        return None

    current_gpu = current.gpu
    if current.exists and current.dph is not None:
        max_dph = current.dph
    else:
        max_dph = config.market.max_dph
    if max_price is not None:
        max_dph = min(max_dph, max_price)

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
        exclusions=exclusions,
    )
    if not matches:
        return None

    matches.sort(key=lambda o: o.get("dph_total", float("inf")))
    return matches[0]


def _skip_options(func):
    """Attach the offer-exclusion options shared by the monitor commands."""
    for args, kwargs in (
        (
            ("--skip-machine", "skip_machines"),
            {"type": int, "multiple": True, "help": "Exclude offers hosted on this Vast machine ID (repeatable)."},
        ),
        (
            ("--skip-offer", "skip_offers"),
            {"type": int, "multiple": True, "help": "Exclude this Vast offer ID (repeatable)."},
        ),
        (
            ("--skip-country", "skip_countries"),
            {"multiple": True, "help": "Exclude offers in this country (alpha-2 code or name, repeatable)."},
        ),
    ):
        func = click.option(*args, **kwargs)(func)
    return func


@cmd_monitor.command("once", help="Run a single price check.")
@click.option("--profile", help="Profile to monitor.")
@click.option("--group", help="Monitor group to search.")
@click.option("--max-price", type=float, default=None, help="Maximum all-in $/h to consider.")
@_common.instance_option
@_skip_options
@click.pass_obj
def cmd_monitor_once(
    config: Config,
    profile: Optional[str],
    group: Optional[str],
    max_price: Optional[float],
    instance_name: Optional[str],
    skip_machines: Tuple[int, ...],
    skip_offers: Tuple[int, ...],
    skip_countries: Tuple[str, ...],
):
    if max_price is not None and max_price < 0:
        raise click.ClickException("--max-price must be non-negative")
    profiles = Profiles.from_file(config.root_dir / config.hostai.profiles_file)
    _name, current = _common.resolve_state(config, instance_name, required=False)
    targets = _resolve_monitor_targets(config, profiles, profile, group, current)
    current_dph = current.dph if current.exists else None
    exclusions = _monitor_skips(
        current, market.OfferExclusions(machines=skip_machines, offers=skip_offers, countries=skip_countries)
    )

    all_offers = _search_profiles(config, profiles, targets, current, max_price=max_price)
    best = _ranked_best_for_monitor(config, profiles, current, all_offers, exclusions=exclusions, max_price=max_price)
    if best is None:
        click.echo("no matching offers")
        return

    best_dph = best.get("dph_total", 0)
    loc = market._format_country(best.get("geolocation") or best.get("location"))
    click.echo(
        f"[monitor] best {best.get('gpu_name')} at ${best_dph:.4f}/h "
        f"(id={best.get('id') or best.get('ask_contract_id')} machine={best.get('machine_id', '?')} loc={loc})"
    )
    if current_dph and current_dph > 0:
        saving = (current_dph - best_dph) / current_dph * 100
        click.echo(f"[monitor] saving vs current: {saving:.1f}%")


@cmd_monitor.command("watch", help="Run a foreground price watch loop.")
@click.option("--profile", help="Profile to monitor.")
@click.option("--group", help="Monitor group to search.")
@click.option("--interval", type=int, default=None, help="Seconds between checks.")
@click.option("--threshold", type=float, default=None, help="Pct saving before alerting.")
@click.option("--max-price", type=float, default=None, help="Maximum all-in $/h to consider.")
@_common.instance_option
@_skip_options
@click.pass_obj
def cmd_monitor_watch(
    config: Config,
    profile: Optional[str],
    group: Optional[str],
    interval: Optional[int],
    threshold: Optional[float],
    max_price: Optional[float],
    instance_name: Optional[str],
    skip_machines: Tuple[int, ...],
    skip_offers: Tuple[int, ...],
    skip_countries: Tuple[str, ...],
):
    if max_price is not None and max_price < 0:
        raise click.ClickException("--max-price must be non-negative")
    sec = interval if interval is not None else config.monitor.interval
    pct = threshold if threshold is not None else config.monitor.threshold_pct
    profiles = Profiles.from_file(config.root_dir / config.hostai.profiles_file)
    name, current = _common.resolve_state(config, instance_name, required=False)
    state_file = state_mod.instance_state_file(config.root_dir, name)
    targets = _resolve_monitor_targets(config, profiles, profile, group, current)
    label = group or ", ".join(p.name for p in targets)
    click.echo(f"[monitor] watching '{label}' every {sec}s (threshold {pct}%) instance='{name}'")
    last_alert_key: Optional[Tuple[Any, float]] = None
    try:
        while True:
            current = State.load(state_file)
            # Re-resolve targets each round: a new instance may run a
            # different profile/ctx than when the watch started.
            try:
                targets = _resolve_monitor_targets(config, profiles, profile, group, current)
            except click.ClickException:
                pass  # keep the last known-good targets
            current_dph = current.dph if current.exists else None
            # Re-merge each round: exclusions recorded by a new `up` land in
            # state.json between iterations.
            exclusions = _monitor_skips(
                current, market.OfferExclusions(machines=skip_machines, offers=skip_offers, countries=skip_countries)
            )
            all_offers = _search_profiles(config, profiles, targets, current, max_price=max_price)
            best = _ranked_best_for_monitor(
                config, profiles, current, all_offers, exclusions=exclusions, max_price=max_price
            )
            if best:
                best_dph = best.get("dph_total", 0)
                machine = best.get("machine_id", "?")
                loc = market._format_country(best.get("geolocation") or best.get("location"))
                if current_dph and current_dph > 0 and current_dph > best_dph:
                    saving = (current_dph - best_dph) / current_dph * 100
                    if saving >= pct:
                        msg = (
                            f"{best.get('gpu_name')} ${best_dph:.4f}/h is {saving:.1f}% cheaper "
                            f"(machine={machine} loc={loc})"
                        )
                        click.echo(f"[monitor] ALERT: {msg}")
                        # The daemon's stdout goes to the log file; the desktop
                        # notification is the only visible channel for alerts.
                        # Notify once per distinct offer/price, not every tick.
                        alert_key = (best.get("id") or best.get("ask_contract_id"), round(best_dph, 4))
                        if alert_key != last_alert_key:
                            notify.notify(f"hostai monitor '{label}'", msg)
                            last_alert_key = alert_key
                    else:
                        click.echo(
                            f"[monitor] best ${best_dph:.4f}/h (saving {saving:.1f}%, machine={machine} loc={loc})"
                        )
                else:
                    click.echo(f"[monitor] best ${best_dph:.4f}/h (machine={machine} loc={loc})")
            else:
                click.echo("[monitor] no cheaper equivalent offer")
            time.sleep(sec)
    except KeyboardInterrupt:
        click.echo("\n[monitor] stopped")


def _start_monitor(
    config: Config,
    profile: Optional[str],
    group: Optional[str],
    interval: Optional[int],
    threshold: Optional[float],
    exclusions: Optional[market.OfferExclusions] = None,
    max_price: Optional[float] = None,
    instance: str = state_mod.DEFAULT_INSTANCE,
) -> None:
    """Launch the monitor daemon in a detached subprocess for one instance."""
    sec = interval if interval is not None else config.monitor.interval
    pct = threshold if threshold is not None else config.monitor.threshold_pct
    target_label = profile or group or config.monitor.profile or config.monitor.group or config.hostai.default_profile

    cmd = [_common.hostai_executable(), "monitor", "watch", "--interval", str(sec), "--threshold", str(pct)]
    if profile:
        cmd.extend(["--profile", profile])
    if group:
        cmd.extend(["--group", group])
    if max_price is not None:
        cmd.extend(["--max-price", str(max_price)])
    cmd.extend((exclusions or market.OfferExclusions()).cli_args())
    # Always last: the daemon identity check matches on `--name <instance>`
    # being a trailing argv token.
    cmd.extend(["--name", instance])

    pid = _common.spawn_daemon(
        config,
        "monitor",
        cmd,
        banner=f"\n# monitor start {target_label} interval={sec}s threshold={pct}% instance={instance}\n",
        instance=instance,
    )
    click.echo(f"[monitor] started daemon (pid {pid}) logging to {_monitor_log_file(config, instance)}")


def _stop_monitor(config: Config, instance: Optional[str] = None, echo: bool = False) -> None:
    """Kill the monitor daemon and remove its pid file."""
    _common.stop_daemon(config, "monitor", instance, echo=echo)


@cmd_monitor.command("start", help="Start the price monitor daemon.")
@click.option("--profile", help="Profile to monitor.")
@click.option("--group", help="Monitor group to search.")
@click.option("--interval", type=int, default=None, help="Seconds between checks.")
@click.option("--threshold", type=float, default=None, help="Pct saving before alerting.")
@click.option("--max-price", type=float, default=None, help="Maximum all-in $/h to consider.")
@_common.instance_option
@_skip_options
@click.pass_obj
def cmd_monitor_start(
    config: Config,
    profile: Optional[str],
    group: Optional[str],
    interval: Optional[int],
    threshold: Optional[float],
    max_price: Optional[float],
    instance_name: Optional[str],
    skip_machines: Tuple[int, ...],
    skip_offers: Tuple[int, ...],
    skip_countries: Tuple[str, ...],
):
    if max_price is not None and max_price < 0:
        raise click.ClickException("--max-price must be non-negative")
    name, _state = _common.resolve_state(config, instance_name, required=False)
    if _common.daemon_running(config, "monitor", name):
        pid = _monitor_pid_file(config, name).read_text().strip()
        click.echo(f"[monitor] already running for '{name}' (pid {pid})")
        return
    _start_monitor(
        config,
        profile,
        group,
        interval,
        threshold,
        market.OfferExclusions(machines=skip_machines, offers=skip_offers, countries=skip_countries),
        max_price=max_price,
        instance=name,
    )


@cmd_monitor.command("stop", help="Stop the price monitor daemon.")
@_common.instance_option
@click.pass_obj
def cmd_monitor_stop(config: Config, instance_name: Optional[str]):
    name, _state = _common.resolve_state(config, instance_name, required=False)
    _stop_monitor(config, name, echo=True)


@cmd_monitor.command("status", help="Show monitor daemon status.")
@_common.instance_option
@click.pass_obj
def cmd_monitor_status(config: Config, instance_name: Optional[str]):
    states = state_mod.find_instance_states(config.root_dir)
    if instance_name is None and len(states) > 1:
        for name in states:
            click.echo(_common.daemon_status_line(config, "monitor", name))
        return
    name, _state = _common.resolve_state(config, instance_name, required=False)
    click.echo(_common.daemon_status_line(config, "monitor", name))


def maybe_start_monitor(config: Config, state: State) -> None:
    """Start the price monitor daemon after hostai up when auto_start is enabled."""
    if not config.monitor.auto_start:
        return
    if not state.instance_id:
        return
    instance = state.data.get("instance_name") or state_mod.instance_name_for_state_file(state.state_file)
    if _common.daemon_running(config, "monitor", instance):
        return
    _log_monitor(config, f"auto-starting monitor for instance {state.instance_id}", instance)
    _start_monitor(config, profile=None, group=None, interval=None, threshold=None, instance=instance)


def stop_monitor(config: Config, instance: Optional[str] = None) -> None:
    """Stop the monitor daemon for one instance if it is running."""
    _stop_monitor(config, instance, echo=False)


def _log_monitor(config: Config, message: str, instance: Optional[str] = None) -> None:
    _common.daemon_log(config, "monitor", message, instance)


@cmd_monitor.command("logs", help="Tail the monitor daemon log.")
@click.option("--lines", type=int, default=50, help="Number of lines to show.")
@_common.instance_option
@click.pass_obj
def cmd_monitor_logs(config: Config, lines: int, instance_name: Optional[str]):
    name, _state = _common.resolve_state(config, instance_name, required=False)
    log_file = _monitor_log_file(config, name)
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
