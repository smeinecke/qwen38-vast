"""Idle-timeout and maximum-runtime watchdog for a running instance.

The watchdog runs as a background process.  It polls the llama-server metrics
and /slots endpoint to detect active requests.  When the instance has been idle
for longer than ``vast.idle_timeout_seconds`` or has exceeded
``vast.max_runtime_seconds`` without an active request, it invokes
``hostai.commands.down.down_instance`` so the normal shutdown/cache/archive
path is reused.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import click

from hostai import api
from hostai import state as state_mod
from hostai.commands import _common
from hostai.commands.down import down_instance
from hostai.config import Config
from hostai.state import State

_WATCHDOG_METRICS = (
    "llamacpp:prompt_tokens_total",
    "llamacpp:tokens_predicted_total",
    "llamacpp:slots_idle",
    "llamacpp:slots_processing",
)


def _watchdog_pid_file(config: Config, instance: Optional[str] = None) -> Path:
    return _common.daemon_pid_file(config, "watchdog", instance)


def _watchdog_log_file(config: Config, instance: Optional[str] = None) -> Path:
    return _common.daemon_log_file(config, "watchdog", instance)


def _log(config: Config, message: str, instance: Optional[str] = None) -> None:
    _common.daemon_log(config, "watchdog", message, instance)


def _state_instance_name(state: State) -> str:
    return state.data.get("instance_name") or state_mod.instance_name_for_state_file(state.state_file)


# Number of consecutive successful "inactive" observations required before an
# idle timeout is allowed.  This protects against a single flaky read being
# misclassified as idle.
MIN_IDLE_OBSERVATIONS = 2

ActivityState = str


def _is_request_active(client: api.LlamaClient, previous: Dict[str, Any]) -> tuple[ActivityState, Dict[str, Any]]:
    """Return (state, snapshot) for the current moment.

    *state* is one of ``active``, ``inactive``, or ``unknown``.  Unknown means
    the metrics/slots endpoints could not be reached or returned unexpected
    data; the caller must not treat an unreachable server as idle.
    """
    current: Dict[str, Any] = {}
    try:
        metrics = client.get_metrics(raise_on_error=True)
        slots = client.slots(raise_on_error=True)
    except Exception:
        return "unknown", current

    if not isinstance(metrics, dict) or not isinstance(slots, list):
        return "unknown", current

    for key in _WATCHDOG_METRICS:
        current[key] = metrics.get(key, 0.0)
    current["slots"] = slots
    current["n_processing_slots"] = sum(1 for s in slots if s.get("state") == 1 or s.get("is_processing"))

    if not previous:
        # First successful observation: treat as active so the idle clock
        # starts from a known baseline.
        return "active", current

    # Counters changing means requests are still moving.  A counter reset
    # (current < previous) is also treated as activity because we cannot
    # distinguish a server restart from genuine activity.
    counters_changed = any(
        isinstance(current.get(k), (int, float))
        and isinstance(previous.get(k), (int, float))
        and current[k] != previous[k]
        for k in ("llamacpp:prompt_tokens_total", "llamacpp:tokens_predicted_total")
    )
    if counters_changed or current["n_processing_slots"] > 0:
        return "active", current

    return "inactive", current


def _decide_shutdown(
    config: Config,
    state: State,
    activity_state: ActivityState,
    consecutive_inactive: int,
    trigger: str,
    now: float,
    last_activity_epoch: float,
    instance: Optional[str] = None,
) -> Optional[str]:
    """Return a shutdown reason if the trigger should now stop the instance.

    ``trigger`` is either ``idle-timeout`` or ``max-runtime``.  Emits the
    appropriate log line and returns the trigger string when shutdown should
    proceed; otherwise returns ``None``.
    """
    idle_observations_ok = consecutive_inactive >= MIN_IDLE_OBSERVATIONS
    is_idle = activity_state == "inactive" and idle_observations_ok

    if trigger == "idle-timeout":
        if is_idle:
            _log(
                config,
                f"idle for {now - last_activity_epoch:.0f}s ({consecutive_inactive} consecutive inactive observations); destroying instance {state.instance_id}",
                instance,
            )
            return "idle-timeout"
        if activity_state == "inactive":
            _log(
                config,
                f"idle timeout reached but only {consecutive_inactive} inactive observation(s); waiting for {MIN_IDLE_OBSERVATIONS}",
                instance,
            )
        elif activity_state == "active":
            _log(config, "idle timeout reached but request still active; waiting", instance)
        else:
            _log(config, "idle timeout reached but activity state unknown; waiting", instance)
    else:  # max-runtime
        if is_idle:
            _log(
                config,
                f"max runtime reached and idle ({consecutive_inactive} observations); destroying instance {state.instance_id}",
                instance,
            )
            return "max-runtime"
        if activity_state == "inactive":
            _log(
                config,
                f"max runtime reached but only {consecutive_inactive} inactive observation(s); waiting",
                instance,
            )
        elif activity_state == "active":
            _log(config, "max runtime reached but request still active; waiting", instance)
        else:
            _log(config, "max runtime reached but activity state unknown; waiting", instance)
    return None


def _run_once(
    config: Config,
    state: State,
    previous_metrics: Dict[str, Any],
    last_activity_epoch: float,
    max_runtime_deadline: Optional[float],
    consecutive_failures: int = 0,
    consecutive_inactive: int = 0,
    instance: Optional[str] = None,
) -> tuple[Dict[str, Any], float, bool, int, int]:
    """Single watchdog iteration.

    Returns ``(updated_metrics, updated_last_activity, should_shutdown, consecutive_failures, consecutive_inactive)``.
    """
    client = api.LlamaClient(config, state)
    activity_state, current = _is_request_active(client, previous_metrics)
    now = time.time()
    reason: Optional[str] = None

    if activity_state == "active":
        last_activity_epoch = now
        consecutive_failures = 0
        consecutive_inactive = 0
    elif activity_state == "unknown":
        consecutive_failures += 1
        consecutive_inactive = 0
        # Reset the idle clock while activity is unknown so we do not destroy
        # an instance just because the metrics endpoint is unreachable.
        last_activity_epoch = now
        if consecutive_failures == 1 or consecutive_failures % 5 == 0:
            _log(
                config,
                f"activity state unknown ({consecutive_failures} consecutive failures); treating as not-idle",
                instance,
            )
        # A preempted/destroyed instance reports "unknown" forever — after a
        # few misses, confirm via the provider (twice, fail-open) and clean up
        # instead of spinning until the process is killed.
        if consecutive_failures >= 3 and state.instance_id:
            dead = _common.confirm_instance_dead(config, state.instance_id)
            if dead:
                reason = f"instance {state.instance_id} {dead}"
    else:  # inactive
        consecutive_failures = 0
        consecutive_inactive += 1

    max_runtime_elapsed = max_runtime_deadline is not None and now >= max_runtime_deadline
    idle_elapsed = (
        config.vast.idle_timeout_seconds is not None and now - last_activity_epoch >= config.vast.idle_timeout_seconds
    )

    if not reason and idle_elapsed:
        reason = _decide_shutdown(
            config, state, activity_state, consecutive_inactive, "idle-timeout", now, last_activity_epoch, instance
        )
    elif not reason and max_runtime_elapsed:
        reason = _decide_shutdown(
            config, state, activity_state, consecutive_inactive, "max-runtime", now, last_activity_epoch, instance
        )

    if reason:
        try:
            down_instance(
                config,
                state,
                pause=False,
                no_archive=False,
                no_cache=not state.slot_cache_enabled,
                reason=reason,
                skip_confirm=True,
            )
            _watchdog_pid_file(config, instance).unlink(missing_ok=True)
        except Exception as exc:
            _log(config, f"down_instance failed: {exc}", instance)
        return current, last_activity_epoch, True, consecutive_failures, consecutive_inactive

    return current, last_activity_epoch, False, consecutive_failures, consecutive_inactive


def run_watchdog(config: Config, instance: str = state_mod.DEFAULT_INSTANCE) -> None:
    """Foreground watchdog loop bound to one instance name."""
    state_file = state_mod.instance_state_file(config.root_dir, instance)
    if not state_file.exists():
        _log(config, f"no state file for instance '{instance}'; exiting", instance)
        return

    state = State.load(state_file)
    if not state.instance_id:
        _log(config, "no instance in state; exiting", instance)
        return

    # Bind to the instance id seen at startup.  If the state file now points
    # at a *different* remote instance, exit — the max-runtime deadline and
    # idle clock belong to the old one and acting on them would destroy the
    # new one.
    watched_id = state.instance_id

    interval = max(5, config.vast.idle_poll_interval_seconds or 60)
    max_runtime = config.vast.max_runtime_seconds
    max_runtime_deadline = None
    if max_runtime and state.started_epoch:
        max_runtime_deadline = state.started_epoch + max_runtime

    _log(
        config,
        f"watchdog started for instance '{instance}' (id {state.instance_id}); "
        f"idle={config.vast.idle_timeout_seconds}s max_runtime={max_runtime}s poll={interval}s",
        instance,
    )

    previous_metrics: Dict[str, Any] = {}
    last_activity_epoch = time.time()
    consecutive_failures = 0
    consecutive_inactive = 0

    try:
        while True:
            state = State.load(state_file)
            if not state.instance_id:
                _log(config, "state cleared; exiting", instance)
                break
            if state.instance_id != watched_id:
                _log(
                    config,
                    f"state now points at instance {state.instance_id} (was {watched_id}); exiting",
                    instance,
                )
                break

            previous_metrics, last_activity_epoch, done, consecutive_failures, consecutive_inactive = _run_once(
                config,
                state,
                previous_metrics,
                last_activity_epoch,
                max_runtime_deadline,
                consecutive_failures,
                consecutive_inactive,
                instance,
            )
            if done:
                break

            time.sleep(interval)
    except KeyboardInterrupt:
        _log(config, "interrupted; exiting", instance)


@click.group("watchdog", help="Idle and maximum-runtime instance safeguards.")
@click.pass_obj
def cmd_watchdog(config: Config):
    pass


@cmd_watchdog.command("run", help="Run the foreground watchdog loop.")
@_common.instance_option
@click.pass_obj
def cmd_watchdog_run(config: Config, instance_name: Optional[str]):
    name, _state = _common.resolve_state(config, instance_name, required=False)
    run_watchdog(config, name)


def _start_watchdog(config: Config, instance: str = state_mod.DEFAULT_INSTANCE) -> None:
    """Launch the watchdog daemon in a detached subprocess for one instance."""
    argv = [_common.hostai_executable(), "watchdog", "run", "--name", instance]
    pid = _common.spawn_daemon(config, "watchdog", argv, banner="\n", instance=instance)
    click.echo(f"[watchdog] started daemon (pid {pid}) logging to {_watchdog_log_file(config, instance)}")


def _stop_watchdog(config: Config, instance: Optional[str] = None, echo: bool = False) -> None:
    """Kill the watchdog daemon and remove its pid file."""
    _common.stop_daemon(config, "watchdog", instance, echo=echo)


@cmd_watchdog.command("start", help="Start the watchdog daemon.")
@_common.instance_option
@click.pass_obj
def cmd_watchdog_start(config: Config, instance_name: Optional[str]):
    name, _state = _common.resolve_state(config, instance_name, required=False)
    if _common.daemon_running(config, "watchdog", name):
        pid = _watchdog_pid_file(config, name).read_text().strip()
        click.echo(f"[watchdog] already running for '{name}' (pid {pid})")
        return
    _start_watchdog(config, name)


@cmd_watchdog.command("stop", help="Stop the watchdog daemon.")
@_common.instance_option
@click.pass_obj
def cmd_watchdog_stop(config: Config, instance_name: Optional[str]):
    name, _state = _common.resolve_state(config, instance_name, required=False)
    _stop_watchdog(config, name, echo=True)


@cmd_watchdog.command("status", help="Show watchdog daemon status.")
@_common.instance_option
@click.pass_obj
def cmd_watchdog_status(config: Config, instance_name: Optional[str]):
    states = state_mod.find_instance_states(config.root_dir)
    if instance_name is None and len(states) > 1:
        for name in states:
            click.echo(_common.daemon_status_line(config, "watchdog", name))
        return
    name, _state = _common.resolve_state(config, instance_name, required=False)
    click.echo(_common.daemon_status_line(config, "watchdog", name))


def maybe_start_watchdog(config: Config, state: State) -> None:
    """Start the watchdog if at least one safeguard is configured and auto-start is on."""
    if not config.vast.watchdog_auto_start:
        return
    if config.vast.idle_timeout_seconds is None and config.vast.max_runtime_seconds is None:
        return
    if not state.instance_id:
        return
    instance = _state_instance_name(state)
    if _common.daemon_running(config, "watchdog", instance):
        # A watchdog for this instance is already alive — spawning a second
        # would leave an unmanaged daemon.
        _log(config, f"watchdog already running; not starting another for instance {state.instance_id}", instance)
        return
    _log(config, f"auto-starting watchdog for instance {state.instance_id}", instance)
    _start_watchdog(config, instance)


def stop_watchdog(config: Config, instance: Optional[str] = None) -> None:
    """Stop the watchdog daemon for one instance, if running."""
    _stop_watchdog(config, instance, echo=False)
