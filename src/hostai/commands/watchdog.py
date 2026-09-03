"""Idle-timeout and maximum-runtime watchdog for a running instance.

The watchdog runs as a background process.  It polls the llama-server metrics
and /slots endpoint to detect active requests.  When the instance has been idle
for longer than ``vast.idle_timeout_seconds`` or has exceeded
``vast.max_runtime_seconds`` without an active request, it invokes
``hostai.commands.down.down_instance`` so the normal shutdown/cache/archive
path is reused.
"""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import click

from hostai import api
from hostai.commands.down import down_instance
from hostai.config import Config
from hostai.state import State, state_dir

_WATCHDOG_METRICS = (
    "llamacpp:prompt_tokens_total",
    "llamacpp:tokens_predicted_total",
    "llamacpp:slots_idle",
    "llamacpp:slots_processing",
)


def _watchdog_pid_file(config: Config) -> Path:
    return config.root_dir / ".hostai-cache" / "watchdog.pid"


def _watchdog_log_file(config: Config) -> Path:
    return config.root_dir / ".hostai-cache" / "watchdog.log"


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _hostai_executable() -> str:
    exe = shutil.which("hostai")
    return exe if exe else sys.argv[0]


def _log(config: Config, message: str) -> None:
    log_file = _watchdog_log_file(config)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n")


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
        isinstance(current.get(k), (int, float)) and isinstance(previous.get(k), (int, float))
        and current[k] != previous[k]
        for k in ("llamacpp:prompt_tokens_total", "llamacpp:tokens_predicted_total")
    )
    if counters_changed or current["n_processing_slots"] > 0:
        return "active", current

    return "inactive", current


def _run_once(
    config: Config,
    state: State,
    previous_metrics: Dict[str, Any],
    last_activity_epoch: float,
    max_runtime_deadline: Optional[float],
    consecutive_failures: int = 0,
    consecutive_inactive: int = 0,
) -> tuple[Dict[str, Any], float, bool, int, int]:
    """Single watchdog iteration.

    Returns ``(updated_metrics, updated_last_activity, should_shutdown, consecutive_failures, consecutive_inactive)``.
    """
    client = api.LlamaClient(config, state)
    activity_state, current = _is_request_active(client, previous_metrics)
    now = time.time()

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
            _log(config, f"activity state unknown ({consecutive_failures} consecutive failures); treating as not-idle")
    else:  # inactive
        consecutive_failures = 0
        consecutive_inactive += 1

    max_runtime_elapsed = max_runtime_deadline is not None and now >= max_runtime_deadline
    idle_elapsed = (
        config.vast.idle_timeout_seconds is not None
        and now - last_activity_epoch >= config.vast.idle_timeout_seconds
    )
    idle_observations_ok = consecutive_inactive >= MIN_IDLE_OBSERVATIONS

    reason: Optional[str] = None
    if idle_elapsed:
        if activity_state == "inactive" and idle_observations_ok:
            reason = "idle-timeout"
            _log(config, f"idle for {now - last_activity_epoch:.0f}s ({consecutive_inactive} consecutive inactive observations); destroying instance {state.instance_id}")
        elif activity_state == "inactive":
            _log(config, f"idle timeout reached but only {consecutive_inactive} inactive observation(s); waiting for {MIN_IDLE_OBSERVATIONS}")
        elif activity_state == "active":
            _log(config, "idle timeout reached but request still active; waiting")
        elif activity_state == "unknown":
            _log(config, "idle timeout reached but activity state unknown; waiting")
    elif max_runtime_elapsed:
        if activity_state == "inactive" and idle_observations_ok:
            reason = "max-runtime"
            _log(config, f"max runtime reached and idle ({consecutive_inactive} observations); destroying instance {state.instance_id}")
        elif activity_state == "inactive":
            _log(config, f"max runtime reached but only {consecutive_inactive} inactive observation(s); waiting")
        elif activity_state == "active":
            _log(config, "max runtime reached but request still active; waiting")
        elif activity_state == "unknown":
            _log(config, "max runtime reached but activity state unknown; waiting")

    if reason:
        try:
            down_instance(config, state, pause=False, no_archive=False, no_cache=False, reason=reason, skip_confirm=True)
            _watchdog_pid_file(config).unlink(missing_ok=True)
        except Exception as exc:
            _log(config, f"down_instance failed: {exc}")
        return current, last_activity_epoch, True, consecutive_failures, consecutive_inactive

    return current, last_activity_epoch, False, consecutive_failures, consecutive_inactive


def run_watchdog(config: Config) -> None:
    """Foreground watchdog loop."""
    state_file = state_dir(config.root_dir) / "state.json"
    if not state_file.exists():
        _log(config, "no state file; exiting")
        return

    state = State.load(state_file)
    if not state.instance_id:
        _log(config, "no instance in state; exiting")
        return

    interval = max(5, config.vast.idle_poll_interval_seconds or 60)
    max_runtime = config.vast.max_runtime_seconds
    max_runtime_deadline = None
    if max_runtime and state.started_epoch:
        max_runtime_deadline = state.started_epoch + max_runtime

    _log(
        config,
        f"watchdog started for instance {state.instance_id}; "
        f"idle={config.vast.idle_timeout_seconds}s max_runtime={max_runtime}s poll={interval}s",
    )

    previous_metrics: Dict[str, Any] = {}
    last_activity_epoch = time.time()
    consecutive_failures = 0
    consecutive_inactive = 0

    try:
        while True:
            state = State.load(state_file)
            if not state.instance_id:
                _log(config, "state cleared; exiting")
                break

            previous_metrics, last_activity_epoch, done, consecutive_failures, consecutive_inactive = _run_once(
                config,
                state,
                previous_metrics,
                last_activity_epoch,
                max_runtime_deadline,
                consecutive_failures,
                consecutive_inactive,
            )
            if done:
                break

            time.sleep(interval)
    except KeyboardInterrupt:
        _log(config, "interrupted; exiting")


@click.group("watchdog", help="Idle and maximum-runtime instance safeguards.")
@click.pass_obj
def cmd_watchdog(config: Config):
    pass


@cmd_watchdog.command("run", help="Run the foreground watchdog loop.")
@click.pass_obj
def cmd_watchdog_run(config: Config):
    run_watchdog(config)


@cmd_watchdog.command("start", help="Start the watchdog daemon.")
@click.pass_obj
def cmd_watchdog_start(config: Config):
    pid_file = _watchdog_pid_file(config)

    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if _is_running(pid):
                click.echo(f"[watchdog] already running (pid {pid})")
                return
        except (ValueError, OSError):
            pass

    log_file = _watchdog_log_file(config)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n")

    cmd = [_hostai_executable(), "watchdog", "run"]
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
    click.echo(f"[watchdog] started daemon (pid {proc.pid}) logging to {log_file}")


@cmd_watchdog.command("stop", help="Stop the watchdog daemon.")
@click.pass_obj
def cmd_watchdog_stop(config: Config):
    pid_file = _watchdog_pid_file(config)
    if not pid_file.exists():
        click.echo("[watchdog] not running")
        return

    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        pid_file.unlink(missing_ok=True)
        click.echo("[watchdog] not running")
        return

    if not _is_running(pid):
        pid_file.unlink(missing_ok=True)
        click.echo("[watchdog] not running")
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except Exception as exc:
        click.echo(f"[watchdog] could not stop daemon: {exc}", err=True)
        return

    for _ in range(20):
        if not _is_running(pid):
            break
        time.sleep(0.2)

    if _is_running(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception as exc:
            click.echo(f"[watchdog] could not kill daemon: {exc}", err=True)

    pid_file.unlink(missing_ok=True)
    click.echo("[watchdog] stopped")


@cmd_watchdog.command("status", help="Show watchdog daemon status.")
@click.pass_obj
def cmd_watchdog_status(config: Config):
    pid_file = _watchdog_pid_file(config)
    if not pid_file.exists():
        click.echo("[watchdog] not running")
        return

    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        click.echo("[watchdog] not running (stale pid file)")
        pid_file.unlink(missing_ok=True)
        return

    if _is_running(pid):
        log_file = _watchdog_log_file(config)
        click.echo(f"[watchdog] running (pid {pid}) log={log_file}")
    else:
        click.echo("[watchdog] not running (stale pid file)")
        pid_file.unlink(missing_ok=True)


def maybe_start_watchdog(config: Config, state: State) -> None:
    """Start the watchdog if at least one safeguard is configured and auto-start is on."""
    if not config.vast.watchdog_auto_start:
        return
    if config.vast.idle_timeout_seconds is None and config.vast.max_runtime_seconds is None:
        return
    if not state.instance_id:
        return
    _log(config, f"auto-starting watchdog for instance {state.instance_id}")
    if cmd_watchdog_start.callback is not None:
        cmd_watchdog_start.callback(config)


def stop_watchdog(config: Config) -> None:
    """Stop the watchdog daemon, if running."""
    pid_file = _watchdog_pid_file(config)
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if _is_running(pid) and cmd_watchdog_stop.callback is not None:
                cmd_watchdog_stop.callback(config)
        except (ValueError, OSError):
            pid_file.unlink(missing_ok=True)
