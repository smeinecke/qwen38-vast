"""SSH endpoint resolution, command execution, and local port tunneling.

This module uses asyncssh for all SSH operations so it can forward the local
port to a remote Unix domain socket (`direct-streamlocal`) as well as to a
remote TCP port.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import shlex
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asyncssh

from hostai import utils
from hostai.config import Config
from hostai.state import State

# Global registry of background tunnel worker threads.
# Keys are local ports; values are dicts with stop_event, thread, local_port.
_TUNNELS: Dict[int, Dict[str, Any]] = {}

# Vast hosts can accept a probe connection before sshd is responsive enough to
# complete another login. The per-stage connect timeout is derived from the
# caller's overall boot timeout so a slow second SSH handshake does not fail
# simply because an internal cap was too low.
TUNNEL_MIN_CONNECT_TIMEOUT = 10.0
TUNNEL_MAX_CONNECT_TIMEOUT = 120.0
_TUNNEL_START_TIMEOUT = 45


def resolve_ssh_endpoint(instance: dict) -> Optional[Dict[str, Any]]:
    """Extract an SSH endpoint from a Vast instance dict.

    Supports the legacy ssh_host/ssh_port fields and the newer public_ip +
    ports mapping.  Returns a dict with user, host, port, ssh_url and direct.
    """
    user = "root"
    host: Optional[str] = None
    port: Optional[int] = None

    if instance.get("ssh_host") and instance.get("ssh_port"):
        host = str(instance["ssh_host"])
        port = int(instance["ssh_port"])
        user = instance.get("ssh_user") or "root"
    else:
        host = instance.get("public_ipaddr") or instance.get("public_ip")
        ports = instance.get("ports") or {}
        tcp_22 = ports.get("22/tcp") or []
        if host and tcp_22 and len(tcp_22) > 0 and isinstance(tcp_22[0], dict):
            host_port = tcp_22[0].get("HostPort")
            if host_port:
                port = int(host_port)

    if not host or not port:
        return None

    direct = bool(instance.get("direct_port_count", 0) > 0)
    ssh_url = f"ssh://{user}@{host}:{port}"
    return {
        "user": user,
        "host": host,
        "port": port,
        "ssh_url": ssh_url,
        "direct": direct,
    }


def ssh_options(
    host: str,
    port: int,
    known_hosts: Path,
    identity: Optional[Path] = None,
) -> List[str]:
    """Kept for API compatibility.  Not used with asyncssh."""
    return []


def clear_known_hosts(host: str, port: int, known_hosts: Path) -> None:
    """Remove stale known_hosts entries for host:port."""
    known_hosts = Path(known_hosts)
    if not known_hosts.exists():
        return
    utils.run(
        ["ssh-keygen", "-R", f"[{host}]:{port}", "-f", str(known_hosts)],
        check=False,
        capture=True,
    )
    utils.run(
        ["ssh-keygen", "-R", host, "-f", str(known_hosts)],
        check=False,
        capture=True,
    )


def _connect_kwargs(known_hosts: Optional[Path] = None, identity: Optional[Path] = None) -> Dict[str, Any]:
    """Build asyncssh connect keyword args."""
    kwargs: Dict[str, Any] = {}
    # For disposable Vast instances we do not enforce known_hosts.  A future
    # improvement could record host keys in the project known_hosts file.
    if known_hosts and known_hosts.exists():
        kwargs["known_hosts"] = None
    else:
        kwargs["known_hosts"] = None
    if identity:
        kwargs["client_keys"] = [str(identity)]
    return kwargs


def _default_identity(config: Optional[Config], state: Optional[State]) -> Optional[Path]:
    """Return the SSH private key to use for the current state/config."""
    if state and state.ssh_identity:
        return state.ssh_identity
    if config and config.secrets.get("SSH_PRIVATE_KEY"):
        return Path(config.secrets["SSH_PRIVATE_KEY"])
    return None


@dataclass
class CompletedProcess:
    """Minimal replacement for subprocess.CompletedProcess."""

    args: Any
    returncode: int
    stdout: Optional[str] = None
    stderr: Optional[str] = None


def _parse_url(ssh_url: str) -> Tuple[str, str, int]:
    return utils.parse_ssh_url(ssh_url)


def _decode_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


async def _run_remote(
    host: str,
    port: int,
    user: str,
    command: Any,
    input_data: Optional[str],
    timeout: Optional[float],
    known_hosts: Optional[Path],
    identity: Optional[Path],
) -> CompletedProcess:
    if isinstance(command, (list, tuple)):
        command = " ".join(str(c) for c in command)

    async with asyncssh.connect(
        host,
        port=port,
        username=user,
        **_connect_kwargs(known_hosts, identity),
    ) as conn:
        result = await conn.run(
            command,
            input=input_data,
            timeout=timeout,
        )
        return CompletedProcess(
            args=command,
            returncode=result.returncode if result.returncode is not None else -1,
            stdout=_decode_output(result.stdout),
            stderr=_decode_output(result.stderr),
        )


def _run_coro(coro: Any) -> Any:
    """Run a coroutine, handling both sync and async (running loop) callers.

    If there is already a running event loop (e.g. we are inside `hostai proxy`),
    run the coroutine in a separate thread with its own event loop to avoid
    ``RuntimeError: asyncio.run() cannot be called from a running event loop``.
    """
    try:
        # get_running_loop raises RuntimeError if no loop is running.
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


def run_remote(
    ssh_url: Optional[str],
    command: Any,
    *,
    known_hosts: Path,
    identity: Optional[Path] = None,
    config: Optional[Config] = None,
    state: Optional[State] = None,
    timeout: Optional[float] = None,
    capture: bool = True,
    input_data: Optional[str] = None,
) -> CompletedProcess:
    """Run a command on the remote host via asyncssh."""
    if not ssh_url:
        return CompletedProcess(args=command, returncode=1, stderr="no ssh_url provided")
    if identity is None and (config or state):
        identity = _default_identity(config, state)
    user, host, port = _parse_url(ssh_url)
    try:
        return _run_coro(_run_remote(host, port, user, command, input_data, timeout, known_hosts, identity))
    except Exception as exc:
        return CompletedProcess(
            args=command,
            returncode=1,
            stdout=None,
            stderr=str(exc),
        )


async def _is_ssh_reachable(
    host: str,
    port: int,
    user: str,
    known_hosts: Optional[Path] = None,
    identity: Optional[Path] = None,
) -> bool:
    try:
        async with asyncssh.connect(host, port=port, username=user, **_connect_kwargs(known_hosts, identity)) as conn:
            result = await conn.run("echo ok")
            return result.returncode == 0 and (result.stdout or "").strip() == "ok"
    except Exception:
        return False


def is_ssh_reachable(
    ssh_url: str,
    *,
    known_hosts: Path,
    identity: Optional[Path] = None,
    config: Optional[Config] = None,
    state: Optional[State] = None,
) -> bool:
    if identity is None and (config or state):
        identity = _default_identity(config, state)
    user, host, port = _parse_url(ssh_url)
    return _run_coro(_is_ssh_reachable(host, port, user, known_hosts, identity))


def wait_for_ssh(
    ssh_url: Optional[str],
    *,
    known_hosts: Path,
    identity: Optional[Path] = None,
    config: Optional[Config] = None,
    state: Optional[State] = None,
    timeout: int = 120,
    quiet: bool = False,
) -> bool:
    """Wait up to timeout seconds for SSH to become reachable."""
    if not ssh_url:
        return False
    if identity is None and (config or state):
        identity = _default_identity(config, state)
    user, host, port = _parse_url(ssh_url)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _run_coro(_is_ssh_reachable(host, port, user, known_hosts, identity)):
            return True
        if not quiet:
            print(f"[ssh] waiting for {ssh_url} ...")
        time.sleep(2)
    return False


async def _scp(
    local_path: Path,
    host: str,
    port: int,
    user: str,
    remote_path: str,
    known_hosts: Optional[Path],
    identity: Optional[Path],
    timeout: Optional[float],
) -> CompletedProcess:
    try:
        await asyncio.wait_for(
            asyncssh.scp(
                str(local_path),
                ((host, port), remote_path),
                username=user,
                preserve=True,
                **_connect_kwargs(known_hosts, identity),
            ),
            timeout=timeout,
        )
        return CompletedProcess(args=[str(local_path), remote_path], returncode=0)
    except Exception as exc:
        return CompletedProcess(
            args=[str(local_path), remote_path],
            returncode=1,
            stderr=str(exc),
        )


def scp_to(
    ssh_url: Optional[str],
    local_path: Path,
    remote_path: str,
    *,
    known_hosts: Path,
    identity: Optional[Path] = None,
    config: Optional[Config] = None,
    state: Optional[State] = None,
    timeout: Optional[float] = None,
) -> CompletedProcess:
    """Copy a local file to the remote host via asyncssh SCP."""
    if not ssh_url:
        return CompletedProcess(args=[str(local_path), remote_path], returncode=1, stderr="no ssh_url provided")
    if identity is None and (config or state):
        identity = _default_identity(config, state)
    user, host, port = _parse_url(ssh_url)
    return _run_coro(_scp(local_path, host, port, user, remote_path, known_hosts, identity, timeout))


async def _scp_from(
    remote_path: str,
    local_path: Path,
    host: str,
    port: int,
    user: str,
    known_hosts: Optional[Path],
    identity: Optional[Path],
    timeout: Optional[float],
) -> CompletedProcess:
    try:
        await asyncio.wait_for(
            asyncssh.scp(
                ((host, port), remote_path),
                str(local_path),
                username=user,
                preserve=True,
                **_connect_kwargs(known_hosts, identity),
            ),
            timeout=timeout,
        )
        return CompletedProcess(args=[remote_path, str(local_path)], returncode=0)
    except Exception as exc:
        return CompletedProcess(
            args=[remote_path, str(local_path)],
            returncode=1,
            stderr=str(exc),
        )


def scp_from(
    ssh_url: str,
    remote_path: str,
    local_path: Path,
    *,
    known_hosts: Path,
    identity: Optional[Path] = None,
    config: Optional[Config] = None,
    state: Optional[State] = None,
    timeout: Optional[float] = None,
) -> CompletedProcess:
    """Copy a remote file to the local host via asyncssh SCP."""
    if identity is None and (config or state):
        identity = _default_identity(config, state)
    user, host, port = _parse_url(ssh_url)
    return _run_coro(_scp_from(remote_path, local_path, host, port, user, known_hosts, identity, timeout))


def _default_remote_dest(state: State) -> str:
    return "127.0.0.1:8080" if state.unsecure else "/dev/shm/qwen38/llama.sock"


def _connect_timeout_for_stage(total_timeout: float) -> float:
    """Return a per-stage SSH connect timeout that scales with the global deadline.

    The handshake must be allowed at least ``TUNNEL_MIN_CONNECT_TIMEOUT``
    seconds, but never more than ``TUNNEL_MAX_CONNECT_TIMEOUT`` so a stuck
    connection is reported before the overall boot deadline expires.
    """
    return max(TUNNEL_MIN_CONNECT_TIMEOUT, min(total_timeout, TUNNEL_MAX_CONNECT_TIMEOUT))


def is_remote_socket_present(
    ssh_url: str,
    socket_path: str,
    *,
    known_hosts: Path,
    identity: Optional[Path] = None,
    config: Optional[Config] = None,
    state: Optional[State] = None,
    timeout: float = 10.0,
) -> bool:
    """Return True when *socket_path* exists as a socket on the remote host."""
    if not ssh_url:
        return False
    if identity is None and (config or state):
        identity = _default_identity(config, state)
    res = run_remote(
        ssh_url,
        f"test -S {shlex.quote(socket_path)} && echo present",
        known_hosts=known_hosts,
        identity=identity,
        config=config,
        state=state,
        timeout=timeout,
    )
    return res.returncode == 0 and "present" in (res.stdout or "")


async def _start_tunnel_worker(
    user: str,
    host: str,
    port: int,
    local_port: int,
    remote_dest: str,
    identity: Optional[Path],
    connect_timeout: float,
    ready: threading.Event,
    stop: threading.Event,
    outcome: Dict[str, Any],
) -> None:
    try:
        known_hosts = outcome.get("known_hosts")
        kwargs = _connect_kwargs(known_hosts, identity)
        kwargs["connect_timeout"] = connect_timeout
        kwargs["login_timeout"] = connect_timeout
        async with asyncssh.connect(host, port=port, username=user, **kwargs) as conn:
            if remote_dest.startswith("/"):
                # Forward local TCP port to remote Unix domain socket.
                forwarder = conn.forward_local_port_to_path("127.0.0.1", local_port, remote_dest)
            else:
                dest_host, _, dest_port = remote_dest.rpartition(":")
                dest_port = int(dest_port)
                forwarder = conn.forward_local_port("127.0.0.1", local_port, dest_host, dest_port)
            async with forwarder as listener:
                _TUNNELS[local_port] = {
                    "conn": conn,
                    "listener": listener,
                    "stop": stop,
                    "remote_dest": remote_dest,
                }
                ready.set()
                while not stop.is_set():
                    await asyncio.sleep(0.5)
    except Exception as exc:
        outcome["error"] = exc
        ready.set()
    finally:
        _TUNNELS.pop(local_port, None)


def _tunnel_thread_runner(
    loop: asyncio.AbstractEventLoop,
    user: str,
    host: str,
    port: int,
    local_port: int,
    remote_dest: str,
    identity: Optional[Path],
    connect_timeout: float,
    ready: threading.Event,
    stop: threading.Event,
    outcome: Dict[str, Any],
) -> None:
    asyncio.set_event_loop(loop)
    loop.run_until_complete(
        _start_tunnel_worker(user, host, port, local_port, remote_dest, identity, connect_timeout, ready, stop, outcome)
    )
    loop.close()


def _tunnel_is_running(state: State) -> bool:
    if state.local_port and state.local_port in _TUNNELS:
        return True
    return False


def _boot_log(stage: str, elapsed: float, message: str) -> None:
    """Emit a structured boot-stage diagnostic line."""
    ts = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{ts}] [boot:{stage}] {message} after {elapsed:.1f}s", flush=True)


def _wait_for_remote_socket(
    ssh_url: str,
    socket_path: str,
    *,
    known_hosts: Path,
    identity: Optional[Path] = None,
    config: Optional[Config] = None,
    state: Optional[State] = None,
    timeout: float = 30.0,
) -> bool:
    """Poll until *socket_path* exists on the remote host, logging progress."""
    if not ssh_url:
        return False
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if is_remote_socket_present(
            ssh_url, socket_path, known_hosts=known_hosts, identity=identity, config=config, state=state, timeout=5
        ):
            return True
        time.sleep(1)
    return False


def ensure_tunnel(
    config: Config,
    state: State,
    remote_dest: Optional[str] = None,
    timeout: Optional[float] = None,
) -> int:
    """Start or reuse an SSH -L tunnel and persist its pid/port in state."""
    if not state.ssh_url:
        raise RuntimeError("no ssh_url in state; cannot establish tunnel")

    remote_dest = remote_dest or _default_remote_dest(state)
    timeout = timeout if timeout is not None else config.ssh.start_timeout
    if timeout is None or timeout <= 0:
        timeout = _TUNNEL_START_TIMEOUT

    # Reuse an existing healthy tunnel.
    if _tunnel_is_running(state) and is_tunnel_healthy(config, state, timeout=3):
        return state.local_port

    # Stop a stale tunnel before starting a new one.
    if _tunnel_is_running(state):
        stop_tunnel(state)

    user, host, port = _parse_url(state.ssh_url)
    local_port = state.local_port or config.ssh.local_port
    if not utils.port_is_free(local_port):
        if not config.ssh.local_port_auto:
            raise RuntimeError(f"local port {local_port} is in use and local_port_auto is disabled")
        local_port = utils.find_free_port(start=local_port + 1)

    known_hosts = state.state_file.parent / "known_hosts"
    known_hosts.parent.mkdir(parents=True, exist_ok=True)
    clear_known_hosts(host, port, known_hosts)

    identity = _default_identity(config, state)
    connect_timeout = _connect_timeout_for_stage(timeout)

    start = time.monotonic()
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    stop = threading.Event()
    outcome: Dict[str, Any] = {"known_hosts": known_hosts}
    thread = threading.Thread(
        target=_tunnel_thread_runner,
        args=(loop, user, host, port, local_port, remote_dest, identity, connect_timeout, ready, stop, outcome),
        daemon=True,
    )
    thread.start()
    ready.wait(timeout=connect_timeout + 5)
    ssh_elapsed = time.monotonic() - start
    if not ready.is_set():
        stop.set()
        thread.join(timeout=5)
        raise RuntimeError(f"[boot:tunnel-ssh] timeout after {ssh_elapsed:.1f}s")

    if "error" in outcome:
        err = outcome["error"]
        _boot_log("tunnel-ssh", ssh_elapsed, f"failed: {err}")
        stop.set()
        thread.join(timeout=5)
        raise RuntimeError(f"[boot:tunnel-ssh] failed: {err}")
    _boot_log("tunnel-ssh", ssh_elapsed, "connected")

    # Wait for the local port to accept connections.
    if not _local_port_is_open(local_port, timeout=5):
        forward_elapsed = time.monotonic() - start
        _boot_log("tunnel-forward", forward_elapsed, f"local port {local_port} is not accepting connections")
        stop.set()
        thread.join(timeout=5)
        raise RuntimeError("[boot:tunnel-forward] local port is not accepting connections")
    forward_elapsed = time.monotonic() - start
    _boot_log("tunnel-forward", forward_elapsed, f"local port {local_port} accepting connections")

    # For Unix-socket forwarding, verify the remote socket actually exists.
    if remote_dest.startswith("/"):
        if not _wait_for_remote_socket(
            state.ssh_url,
            remote_dest,
            known_hosts=known_hosts,
            identity=identity,
            config=config,
            state=state,
            timeout=max(5.0, timeout - (time.monotonic() - start) - 2),
        ):
            socket_elapsed = time.monotonic() - start
            _boot_log("remote-socket", socket_elapsed, f"{remote_dest} did not appear")
            stop.set()
            thread.join(timeout=5)
            raise RuntimeError(f"[boot:remote-socket] {remote_dest} did not appear")
        socket_elapsed = time.monotonic() - start
        _boot_log("remote-socket", socket_elapsed, f"{remote_dest} present")

    _TUNNELS[local_port]["state"] = state
    _TUNNELS[local_port]["thread"] = thread
    _TUNNELS[local_port]["stop"] = stop
    _TUNNELS[local_port]["identity"] = identity
    state.tunnel_pid = 0
    state.local_port = local_port
    state.save()
    return local_port


def stop_tunnel(state: State) -> None:
    """Stop the managed SSH tunnel and clear its pid from state."""
    if not state.local_port:
        return
    info = _TUNNELS.pop(state.local_port, None)
    if info and "stop" in info:
        info["stop"].set()
        thread = info.get("thread")
        if thread and thread.is_alive():
            thread.join(timeout=5)
    state.tunnel_pid = None
    state.save()


def is_tunnel_healthy(config: Config, state: State, timeout: int = 3) -> bool:
    """Return True if a TCP connection to the local tunnel port succeeds."""
    _ = config
    local_port = state.local_port
    if not local_port:
        return False
    return _local_port_is_open(local_port, timeout)


_UNI_TUNNELS: Dict[str, Dict[str, Any]] = {}


def _unix_tunnel_is_running(local_path: str) -> bool:
    return local_path in _UNI_TUNNELS


async def _start_unix_tunnel_worker(
    local_path: str,
    user: str,
    host: str,
    port: int,
    remote_dest: str,
    identity: Optional[Path],
    connect_timeout: float,
    ready: threading.Event,
    stop: threading.Event,
    outcome: Dict[str, Any],
) -> None:
    try:
        known_hosts = outcome.get("known_hosts")
        kwargs = _connect_kwargs(known_hosts, identity)
        kwargs["connect_timeout"] = connect_timeout
        kwargs["login_timeout"] = connect_timeout
        async with asyncssh.connect(host, port=port, username=user, **kwargs) as conn:
            forwarder = conn.forward_local_path(local_path, remote_dest)
            async with forwarder as listener:
                _UNI_TUNNELS[local_path] = {
                    "conn": conn,
                    "listener": listener,
                    "stop": stop,
                    "remote_dest": remote_dest,
                }
                ready.set()
                while not stop.is_set():
                    await asyncio.sleep(0.5)
    except Exception as exc:
        outcome["error"] = exc
        ready.set()
    finally:
        _UNI_TUNNELS.pop(local_path, None)
        try:
            Path(local_path).unlink(missing_ok=True)
        except OSError:
            pass


def _unix_tunnel_thread_runner(
    loop: asyncio.AbstractEventLoop,
    local_path: str,
    user: str,
    host: str,
    port: int,
    remote_dest: str,
    identity: Optional[Path],
    connect_timeout: float,
    ready: threading.Event,
    stop: threading.Event,
    outcome: Dict[str, Any],
) -> None:
    asyncio.set_event_loop(loop)
    loop.run_until_complete(
        _start_unix_tunnel_worker(
            local_path, user, host, port, remote_dest, identity, connect_timeout, ready, stop, outcome
        )
    )
    loop.close()


def _unix_socket_is_open(local_path: str, timeout: int = 3) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(local_path)
            return True
    except OSError:
        return False


def ensure_unix_tunnel(
    config: Config,
    state: State,
    remote_dest: Optional[str] = None,
    timeout: Optional[float] = None,
) -> str:
    """Start or reuse an SSH local Unix socket forward to a remote Unix socket.

    Returns the local Unix socket path. The path is stored in
    ``state.data["upstream_socket"]``.
    """
    if not state.ssh_url:
        raise RuntimeError("no ssh_url in state; cannot establish tunnel")

    remote_dest = remote_dest or _default_remote_dest(state)
    if not remote_dest.startswith("/"):
        raise RuntimeError(f"ensure_unix_tunnel only supports remote Unix sockets, got {remote_dest}")

    timeout = timeout if timeout is not None else config.ssh.start_timeout
    if timeout is None or timeout <= 0:
        timeout = _TUNNEL_START_TIMEOUT

    local_path = state.data.get("upstream_socket") or str(state.state_file.parent / "upstream.sock")

    # Reuse an existing healthy tunnel.
    if _unix_tunnel_is_running(local_path) and _unix_socket_is_open(local_path, timeout=3):
        state.data["upstream_socket"] = local_path
        state.local_port = 0
        state.tunnel_pid = 0
        state.save()
        return local_path

    # Stop a stale tunnel before starting a new one.
    if _unix_tunnel_is_running(local_path):
        stop_unix_tunnel(local_path)

    Path(local_path).unlink(missing_ok=True)

    user, host, port = _parse_url(state.ssh_url)
    known_hosts = state.state_file.parent / "known_hosts"
    known_hosts.parent.mkdir(parents=True, exist_ok=True)
    clear_known_hosts(host, port, known_hosts)

    identity = _default_identity(config, state)
    connect_timeout = _connect_timeout_for_stage(timeout)

    start = time.monotonic()
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    stop = threading.Event()
    outcome: Dict[str, Any] = {"known_hosts": known_hosts}
    thread = threading.Thread(
        target=_unix_tunnel_thread_runner,
        args=(loop, local_path, user, host, port, remote_dest, identity, connect_timeout, ready, stop, outcome),
        daemon=True,
    )
    thread.start()
    ready.wait(timeout=connect_timeout + 5)
    ssh_elapsed = time.monotonic() - start
    if not ready.is_set():
        stop.set()
        thread.join(timeout=5)
        raise RuntimeError(f"[boot:unix-tunnel-ssh] timeout after {ssh_elapsed:.1f}s")

    if "error" in outcome:
        err = outcome["error"]
        _boot_log("unix-tunnel-ssh", ssh_elapsed, f"failed: {err}")
        stop.set()
        thread.join(timeout=5)
        raise RuntimeError(f"[boot:unix-tunnel-ssh] failed: {err}")
    _boot_log("unix-tunnel-ssh", ssh_elapsed, "connected")

    if not _unix_socket_is_open(local_path, timeout=5):
        forward_elapsed = time.monotonic() - start
        _boot_log("unix-tunnel-forward", forward_elapsed, f"local socket {local_path} is not accepting connections")
        stop.set()
        thread.join(timeout=5)
        raise RuntimeError("[boot:unix-tunnel-forward] local socket is not accepting connections")
    forward_elapsed = time.monotonic() - start
    _boot_log("unix-tunnel-forward", forward_elapsed, f"local socket {local_path} accepting connections")

    if not _wait_for_remote_socket(
        state.ssh_url,
        remote_dest,
        known_hosts=known_hosts,
        identity=identity,
        config=config,
        state=state,
        timeout=max(5.0, timeout - (time.monotonic() - start) - 2),
    ):
        socket_elapsed = time.monotonic() - start
        _boot_log("remote-socket", socket_elapsed, f"{remote_dest} did not appear")
        stop.set()
        thread.join(timeout=5)
        raise RuntimeError(f"[boot:remote-socket] {remote_dest} did not appear")
    socket_elapsed = time.monotonic() - start
    _boot_log("remote-socket", socket_elapsed, f"{remote_dest} present")

    _UNI_TUNNELS[local_path]["state"] = state
    _UNI_TUNNELS[local_path]["thread"] = thread
    _UNI_TUNNELS[local_path]["stop"] = stop
    _UNI_TUNNELS[local_path]["identity"] = identity
    state.data["upstream_socket"] = local_path
    state.local_port = 0
    state.tunnel_pid = 0
    state.save()
    return local_path


def stop_unix_tunnel(local_path: str) -> None:
    """Stop a managed Unix socket SSH tunnel."""
    info = _UNI_TUNNELS.pop(local_path, None)
    if info and "stop" in info:
        info["stop"].set()
        thread = info.get("thread")
        if thread and thread.is_alive():
            thread.join(timeout=5)
    try:
        Path(local_path).unlink(missing_ok=True)
    except OSError:
        pass


def _local_port_is_open(local_port: int, timeout: int) -> bool:
    """Return whether a specific local tunnel port accepts TCP connections."""
    try:
        with socket.create_connection(("127.0.0.1", local_port), timeout=timeout):
            return True
    except OSError:
        return False
