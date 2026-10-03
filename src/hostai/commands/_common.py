"""Helpers shared by multiple hostai CLI commands."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import click

from hostai import ssh
from hostai import state as state_mod
from hostai.config import Config
from hostai.providers import get_provider
from hostai.state import State, state_dir


def instance_option(fn):
    """Shared ``-n/--name`` selector for commands bound to one instance."""
    return click.option(
        "-n",
        "--name",
        "instance_name",
        envvar="HOSTAI_INSTANCE",
        default=None,
        help="Instance name (or instance id) to target; default when only one is tracked.",
    )(fn)


def resolve_state(
    config: Config,
    selector: Optional[str] = None,
    *,
    required: bool = True,
) -> Tuple[str, State]:
    """Resolve an instance selector to ``(instance_name, State)``.

    Without a selector, a single tracked instance wins; multiple tracked
    instances raise (the caller wants one).  With ``required=False`` an empty
    State bound to the resolved file is returned instead of raising.
    """
    states = state_mod.find_instance_states(config.root_dir)
    if selector is None:
        if not states:
            if required:
                raise click.ClickException("no hostai instances are tracked; run 'hostai up' to create one")
            return state_mod.DEFAULT_INSTANCE, State(
                state_mod.instance_state_file(config.root_dir, state_mod.DEFAULT_INSTANCE)
            )
        if len(states) > 1:
            names = ", ".join(states)
            raise click.ClickException(
                f"multiple hostai instances are tracked ({names}); use --name <name|id> to select one"
            )
        name = next(iter(states))
        return name, State.load(states[name])
    try:
        name = state_mod.resolve_instance_selector(config.root_dir, selector)
    except ValueError as exc:
        known = ", ".join(states) or "none"
        raise click.ClickException(f"{exc} (tracked instances: {known})")
    return name, State.load(state_mod.instance_state_file(config.root_dir, name))


def tracked_instance_ids(root_dir: Path) -> Dict[int, str]:
    """Return ``{provider_instance_id: instance_name}`` for all tracked instances."""
    out: Dict[int, str] = {}
    for name, sf in state_mod.find_instance_states(root_dir).items():
        try:
            iid = json.loads(sf.read_text(encoding="utf-8")).get("instance_id")
        except Exception:
            continue
        try:
            if iid is not None:
                out[int(iid)] = name
        except (TypeError, ValueError):
            continue
    return out


def claimed_local_ports(config: Config, exclude_instance: Optional[str] = None) -> Set[int]:
    """Local ports claimed by other tracked instances (live instance_id + port)."""
    exclude = state_mod.normalize_instance_name(exclude_instance)
    ports: Set[int] = set()
    for name, sf in state_mod.find_instance_states(config.root_dir).items():
        if name == exclude:
            continue
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not data.get("instance_id"):
            continue
        try:
            port = int(data.get("local_port") or 0)
        except (TypeError, ValueError):
            continue
        if port:
            ports.add(port)
    return ports


def resolve_cache_enabled(cache: bool, no_cache: bool, default: bool) -> bool:
    """Resolve mutually exclusive --cache/--no-cache flags against the config default."""
    if cache and no_cache:
        raise click.ClickException("cannot use both --cache and --no-cache")
    if cache:
        return True
    if no_cache:
        return False
    return default


def _provider(config: Config):
    return get_provider(config)


def refresh_ssh_state(config: Config, state: State, instance: Optional[Dict[str, Any]] = None) -> bool:
    """Refresh SSH endpoint fields from the provider.

    ``instance`` may carry a recently fetched provider response to avoid a
    duplicate API call; when omitted (or ``None``) the instance is fetched.
    """
    if not state.instance_id:
        return False
    if instance is None:
        try:
            instance = _provider(config).get_instance(state.instance_id)
        except Exception:
            return False
    if not instance:
        return False
    endpoint = ssh.resolve_ssh_endpoint(instance)
    if endpoint and endpoint.get("host") and endpoint.get("port"):
        state.ssh_url = str(endpoint["ssh_url"])
        state.set("ssh_host", str(endpoint["host"]))
        state.set("ssh_port", int(endpoint["port"]))
        state.set("ssh_user", str(endpoint["user"]))
        state.save()
        return True
    return False


def stop_remote_model(ssh_url: Optional[str], known_hosts: Path) -> None:
    """Stop the remote llama.cpp server gracefully."""
    if not ssh_url:
        click.echo("[down] no ssh_url; cannot stop model")
        return
    click.echo("[down] stopping llama.cpp server...")
    ssh.run_remote(
        ssh_url,
        "pkill -TERM llama-server 2>/dev/null || true; sleep 2; pkill -KILL llama-server 2>/dev/null || true",
        known_hosts=known_hosts,
        timeout=30,
    )


# Instance statuses that mean the remote can never serve again.  "stopped"
# covers interruptible instances preempted by a higher bid.
TERMINAL_INSTANCE_STATUSES = frozenset({"exited", "offline", "stopped", "unknown"})


def fetch_instance(config: Config, instance_id: int) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Return ``(instance, api_ok)`` — ``api_ok`` is False on provider errors.

    The two values are deliberately separate: ``None`` with ``api_ok`` means
    the provider confirmed the instance is gone, while ``api_ok=False`` means
    we could not ask.
    """
    try:
        return get_provider(config).get_instance(instance_id), True
    except Exception:
        return None, False


def instance_dead_reason(
    instance: Optional[Dict[str, Any]],
    api_ok: bool,
    *,
    stopped_is_dead: bool = True,
) -> Optional[str]:
    """Return a death reason, or ``None`` when the instance may still be alive.

    Only a *positive* provider answer counts as dead — API errors and
    non-terminal statuses both return ``None`` so callers never act on a
    flaky response.  ``stopped_is_dead=False`` skips the "stopped" status for
    paths where a restart may briefly report it.
    """
    if not api_ok:
        return None
    if instance is None:
        return "instance no longer exists (preempted or removed)"
    status = str(instance.get("actual_status") or instance.get("status") or "")
    dead = TERMINAL_INSTANCE_STATUSES if stopped_is_dead else TERMINAL_INSTANCE_STATUSES - {"stopped"}
    if status in dead:
        return f"instance status is '{status}'"
    return None


def confirm_instance_dead(
    config: Config,
    instance_id: int,
    *,
    stopped_is_dead: bool = True,
    confirm_delay: float = 3.0,
) -> Optional[str]:
    """Double-check provider state; return a reason only when confirmed dead.

    A single flaky "gone" response cannot trigger cleanup — death must be
    observed twice, ``confirm_delay`` seconds apart.
    """
    reason = instance_dead_reason(*fetch_instance(config, instance_id), stopped_is_dead=stopped_is_dead)
    if not reason:
        return None
    time.sleep(confirm_delay)
    return instance_dead_reason(*fetch_instance(config, instance_id), stopped_is_dead=stopped_is_dead)


def pid_cmdline_contains(pid: int, *needles: bytes) -> bool:
    """True when ``/proc/<pid>/cmdline`` contains every needle.

    Used to verify that a pid-file PID still belongs to the daemon that wrote
    it — PID reuse could otherwise SIGTERM an unrelated process.
    """
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return all(n in cmdline for n in needles)


def running_proxy_pid(state: State) -> Optional[int]:
    """Return the pid of a live ``hostai proxy`` recorded for this state.

    Checks both ``state.proxy_pid`` and the ``proxy.pid`` file the daemon
    writes next to the state file — the file survives cases where ``up``
    died before persisting ``proxy_pid``.
    """
    candidates = []
    raw = state.data.get("proxy_pid")
    if raw:
        try:
            candidates.append(int(raw))
        except (TypeError, ValueError):
            pass
    try:
        candidates.append(int((state.state_file.parent / "proxy.pid").read_text().strip()))
    except (ValueError, OSError):
        pass
    for pid in candidates:
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, 0)
        except (OSError, ValueError):
            continue
        if pid_cmdline_contains(pid, b"hostai", b"proxy"):
            return pid
    return None


def daemon_pid_running(pid_file: Path, *needles: bytes) -> bool:
    """True when *pid_file* points at a live process matching the needles.

    Checks identity, not just liveness: a reused PID must not convince the
    caller that the daemon is still running — that would silently block the
    daemon from ever starting again.
    """
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return pid_cmdline_contains(pid, *needles)


# --- background daemon helpers (watchdog, monitor) ---------------------------
#
# Both daemons share the same lifecycle: detached subprocess, pid file and log
# under .hostai-cache, identity-checked start/stop/status.  Each instance gets
# its own daemon; named instances suffix the pid/log files with the instance
# name and the daemon argv always ends in ``--name <instance>`` so identity
# checks can match on it.


def _daemon_tag(instance: Optional[str]) -> str:
    resolved = state_mod.normalize_instance_name(instance)
    return "" if resolved == state_mod.DEFAULT_INSTANCE else f"-{resolved}"


def daemon_pid_file(config: Config, name: str, instance: Optional[str] = None) -> Path:
    return config.root_dir / ".hostai-cache" / f"{name}{_daemon_tag(instance)}.pid"


def daemon_log_file(config: Config, name: str, instance: Optional[str] = None) -> Path:
    return config.root_dir / ".hostai-cache" / f"{name}{_daemon_tag(instance)}.log"


def _daemon_needles(name: str, instance: Optional[str]) -> Tuple[bytes, ...]:
    # Daemons are spawned as `hostai <name> <sub> ... --name <instance>`; the
    # NUL-delimited needle requires an exact argv token match so a stale pid
    # file can never make one instance's daemon look like another's (nor a
    # prefix-named one: 'foo' does not match 'foo2').
    resolved = state_mod.normalize_instance_name(instance)
    return (b"hostai", name.encode(), b"\x00" + resolved.encode() + b"\x00")


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def hostai_executable() -> str:
    exe = shutil.which("hostai")
    return exe if exe else sys.argv[0]


def daemon_running(config: Config, name: str, instance: Optional[str] = None) -> bool:
    """True when the pid file points at a live, identity-verified daemon."""
    return daemon_pid_running(daemon_pid_file(config, name, instance), *_daemon_needles(name, instance))


def daemon_log(config: Config, name: str, message: str, instance: Optional[str] = None) -> None:
    log_file = daemon_log_file(config, name, instance)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n")


def spawn_daemon(
    config: Config,
    name: str,
    argv: List[str],
    banner: Optional[str] = None,
    instance: Optional[str] = None,
) -> int:
    """Start a detached daemon process, record its pid file, return the pid."""
    log_file = daemon_log_file(config, name, instance)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log:
        if banner:
            log.write(banner)
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    daemon_pid_file(config, name, instance).write_text(str(proc.pid))
    return proc.pid


def stop_daemon(config: Config, name: str, instance: Optional[str] = None, *, echo: bool = False) -> None:
    """SIGTERM the recorded daemon (SIGKILL if it lingers); remove the pid file."""
    tag = _daemon_tag(instance)
    pid_file = daemon_pid_file(config, name, instance)
    if not pid_file.exists():
        if echo:
            click.echo(f"[{name}{tag}] not running")
        return
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        pid_file.unlink(missing_ok=True)
        if echo:
            click.echo(f"[{name}{tag}] not running")
        return

    if not pid_is_running(pid):
        pid_file.unlink(missing_ok=True)
        if echo:
            click.echo(f"[{name}{tag}] not running")
        return

    if not pid_cmdline_contains(pid, *_daemon_needles(name, instance)):
        # PID was reused by an unrelated process — never signal it.
        pid_file.unlink(missing_ok=True)
        if echo:
            click.echo(f"[{name}{tag}] pid file was stale (pid reused); removed")
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except Exception as exc:
        if echo:
            click.echo(f"[{name}{tag}] could not stop daemon: {exc}", err=True)
        return

    # Wait briefly for the process to exit.
    for _ in range(20):
        if not pid_is_running(pid):
            break
        time.sleep(0.2)

    if pid_is_running(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception as exc:
            if echo:
                click.echo(f"[{name}{tag}] could not kill daemon: {exc}", err=True)

    pid_file.unlink(missing_ok=True)
    if echo:
        click.echo(f"[{name}{tag}] stopped")


def daemon_status_line(config: Config, name: str, instance: Optional[str] = None) -> str:
    tag = _daemon_tag(instance)
    pid_file = daemon_pid_file(config, name, instance)
    if not pid_file.exists():
        return f"[{name}{tag}] not running"
    if daemon_running(config, name, instance):
        return (
            f"[{name}{tag}] running (pid {pid_file.read_text().strip()}) log={daemon_log_file(config, name, instance)}"
        )
    pid_file.unlink(missing_ok=True)
    return f"[{name}{tag}] not running (stale pid file)"


@contextlib.contextmanager
def lifecycle_lock(config: Config, command: str, instance: Optional[str] = None) -> Iterator[None]:
    """Advisory flock so only one mutating lifecycle command per instance runs.

    ``up``/``restart`` hold this for the whole provisioning run; a second
    invocation for the same instance fails fast instead of renting a
    duplicate.  Different instance names lock different files so parallel
    instances can be provisioned concurrently.  ``down`` intentionally does
    not take it — it must stay usable to interrupt a stuck provisioning run.
    """
    sd = state_mod.instance_state_dir(config.root_dir, instance)
    sd.mkdir(parents=True, exist_ok=True)
    fd = open(sd / ".lifecycle.lock", "a")
    try:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise click.ClickException(f"another hostai lifecycle command is in progress ({command} blocked)")
        yield
    finally:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        finally:
            fd.close()


@contextlib.contextmanager
def allocation_lock(config: Config) -> Iterator[None]:
    """Short global flock around instance creation (port claims + state write).

    Provisioning runs in parallel across instances, but selecting a local
    port, checking for an existing instance and writing the new state file
    must not interleave with another concurrent creation.
    """
    sd = state_dir(config.root_dir)
    sd.mkdir(parents=True, exist_ok=True)
    fd = open(sd / ".allocate.lock", "a")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        finally:
            fd.close()
