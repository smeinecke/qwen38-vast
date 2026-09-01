"""Shared helpers for hostai."""

from __future__ import annotations

import datetime
import os
import re
import secrets
import shlex
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def now_rfc3339() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_epoch() -> int:
    return int(time.time())


def format_cost(seconds: float, dph: float) -> float:
    return seconds / 3600.0 * dph


def format_dph(value: float) -> str:
    return f"{value:.4f}"


def safe_label(label: str) -> str:
    """Turn a user label into a filesystem-safe string."""
    label = re.sub(r"[\s/]+", "-", label)
    label = re.sub(r"[^A-Za-z0-9_.-]+", "", label)
    return label or "bench"


def make_api_key() -> str:
    return f"sk-local-{secrets.token_hex(24)}"


def make_run_id(profile: str) -> str:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{profile}-{os.getpid()}"


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def find_free_port(start: int = 18080, count: int = 100, host: str = "127.0.0.1") -> int:
    for port in range(start, min(65536, start + count)):
        if port_is_free(port, host=host):
            return port
    raise RuntimeError(f"no free localhost port found in range {start}-{start + count - 1}")


def run(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    timeout: Optional[float] = None,
    check: bool = True,
    capture: bool = True,
    input_data: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    text: bool = True,
) -> subprocess.CompletedProcess:
    """Run a subprocess with sensible defaults."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    kwargs: Dict[str, Any] = {
        "cwd": str(cwd) if cwd else None,
        "env": merged_env,
        "text": text,
    }
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT
    if input_data is not None:
        kwargs["input"] = input_data
    return subprocess.run(cmd, timeout=timeout, check=check, **kwargs)


def run_output(cmd: List[str], **kwargs: Any) -> str:
    """Run a command and return its stdout as a string."""
    result = run(cmd, capture=True, check=False, **kwargs)
    return (result.stdout or "").strip()


def sanitize_for_shell(value: str) -> str:
    """Return a string that is safe to pass through shell-like interpolation."""
    return shlex.quote(value)


def parse_ssh_url(url: str) -> Tuple[str, str, int]:
    """Parse ssh://user@host:port and return (user, host, port)."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    user = parsed.username or "root"
    host = parsed.hostname or ""
    port = parsed.port or 22
    return user, host, port


def mkdir_private(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)
    return path
