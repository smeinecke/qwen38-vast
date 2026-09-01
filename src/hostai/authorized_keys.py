"""Generate the runtime SSH authorized_keys file.

Replaces `scripts/prepare-authorized-keys`.
"""

from __future__ import annotations

import configparser
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Prefer requests if available (declared dependency), otherwise fall back to
# the standard library.
try:
    import requests
except ModuleNotFoundError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

try:
    from dotenv import dotenv_values
except ModuleNotFoundError:  # pragma: no cover
    dotenv_values = None  # type: ignore[assignment]


SSH_PUBLIC_KEY_ENV = "SSH_PUBLIC_KEY"
GITHUB_SSH_KEY_USER_ENV = "GITHUB_SSH_KEY_USER"

# OpenSSH public-key line: type + base64 blob, optional comment.
# Matches the prepare-authorized-keys grep filter.
_AUTHORIZED_KEY_RE = re.compile(
    r"^\s*(?:ssh-|ecdsa-|sk-)\S+\s+\S+",
    re.MULTILINE,
)


def _load_dotenv(path: Path) -> Dict[str, Optional[str]]:
    """Read a .env file without overwriting the running environment."""
    if dotenv_values is not None:
        return dotenv_values(str(path))  # type: ignore[no-any-return]

    # Minimal fallback parser for the same simple KEY=value format.
    values: Dict[str, Optional[str]] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        values[key] = val or None
    return values


def _public_key_lines(text: str) -> List[str]:
    """Return deduplicated, plausible OpenSSH public-key lines."""
    seen: set = set()
    out: List[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if _AUTHORIZED_KEY_RE.match(line):
            clean = line.strip()
            if clean not in seen:
                seen.add(clean)
                out.append(clean)
    return out


def _http_get(url: str, timeout: float = 15.0, retries: int = 3) -> Optional[str]:
    """Fetch a URL, returning the decoded body on success."""
    for _ in range(retries):
        try:
            if requests is not None:
                resp = requests.get(url, timeout=timeout)
                resp.raise_for_status()
                return resp.text

            # Standard-library fallback.
            import urllib.request

            with urllib.request.urlopen(url, timeout=timeout) as resp:  # type: ignore[attr-defined]
                data = resp.read()
                return data.decode("utf-8")
        except Exception as exc:  # pragma: no cover
            _ = exc
    return None


def _github_user_from_remote_url(url: str) -> Optional[str]:
    """Extract a GitHub username/organisation from a remote URL."""
    url = url.strip()
    # https://github.com/USER/REPO.git or http://github.com/USER/REPO
    m = re.match(r"https?://github\.com/([^/]+)/[^/]+/?(?:\.git)?", url)
    if m:
        return m.group(1)
    # git@github.com:USER/REPO.git
    m = re.match(r"git@github\.com:([^/]+)/[^/]+(?:\.git)?", url)
    if m:
        return m.group(1)
    # ssh://git@github.com/USER/REPO.git
    m = re.match(r"ssh://[^@]+@github\.com/([^/]+)/[^/]+(?:\.git)?", url)
    if m:
        return m.group(1)
    # git://github.com/USER/REPO.git
    m = re.match(r"git://github\.com/([^/]+)/[^/]+(?:\.git)?", url)
    if m:
        return m.group(1)
    return None


def _github_user_from_git_config(root_dir: Path) -> Optional[str]:
    """Try to read a GitHub username from the repo's git remotes."""
    git_dir = root_dir / ".git"
    config_file = git_dir / "config"

    # Try git config directly first.
    if shutil.which("git") and git_dir.is_dir():
        try:
            proc = subprocess.run(
                ["git", "-C", str(root_dir), "config", "--get", "remote.origin.url"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if proc.returncode == 0 and proc.stdout:
                user = _github_user_from_remote_url(proc.stdout)
                if user:
                    return user
        except (subprocess.SubprocessError, OSError):
            pass

    # Fallback to parsing .git/config.
    if config_file.exists():
        try:
            cfg = configparser.ConfigParser(strict=False)
            cfg.read(config_file, encoding="utf-8")
            for section in cfg.sections():
                if section.startswith("remote"):
                    url = cfg.get(section, "url", fallback=None)
                    if url:
                        user = _github_user_from_remote_url(url)
                        if user:
                            return user
        except (configparser.Error, OSError):
            pass

    return None


def _github_keys(user: str) -> str:
    """Fetch public keys for a GitHub user."""
    url = f"https://github.com/{user}.keys"
    text = _http_get(url)
    if text is None:
        raise RuntimeError(f"could not fetch https://github.com/{user}.keys")
    return text


def prepare_authorized_keys(
    *,
    root_dir: Path,
    env_file: Optional[Path] = None,
    strict: bool = False,
    from_github: bool = True,
) -> Path:
    """Prepare ``ssh/authorized_keys.generated`` for the runtime image.

    Resolution order:
      1. ``SSH_PUBLIC_KEY`` from the running environment or ``.env``.
      2. ``GITHUB_SSH_KEY_USER`` from the running environment or ``.env``.
      3. If ``from_github`` is True, a GitHub username extracted from the
         repository's git remote (``.git/config`` or ``git remote -v``).

    The function also appends any existing ``ssh/authorized_keys`` file and
    deduplicates the result.  Directory permissions are set to 0700 and file
    permissions to 0600.

    Args:
        root_dir: repository root.
        env_file: optional dotenv file.  Defaults to ``root_dir / ".env"``.
        strict: if True, raise when no public keys can be found.
        from_github: allow inferring the GitHub user from the repo remotes.

    Returns:
        Path to ``ssh/authorized_keys.generated``.
    """
    env_file = env_file or root_dir / ".env"
    env: Dict[str, Optional[str]] = {}
    if env_file.exists():
        env = _load_dotenv(env_file)
    # Current process environment always wins.
    for key in (SSH_PUBLIC_KEY_ENV, GITHUB_SSH_KEY_USER_ENV):
        if os.environ.get(key):
            env[key] = os.environ[key]

    ssh_dir = root_dir / "ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True)
    ssh_dir.chmod(0o700)

    raw_keys: List[str] = []

    public_key = env.get(SSH_PUBLIC_KEY_ENV)
    if public_key:
        raw_keys.append(public_key)
    else:
        github_user = env.get(GITHUB_SSH_KEY_USER_ENV)
        if not github_user and from_github:
            github_user = _github_user_from_git_config(root_dir)

        if github_user:
            print(
                f"[ssh-keys] fetching public SSH keys for GitHub user: {github_user}",
                file=sys.stderr,
            )
            try:
                raw_keys.append(_github_keys(github_user))
            except RuntimeError as exc:
                if strict:
                    raise
                print(f"[ssh-keys] {exc}", file=sys.stderr)

    # Also append any committed/explicit authorized_keys file.
    committed = ssh_dir / "authorized_keys"
    if committed.exists():
        raw_keys.append(committed.read_text())

    all_keys = "\n".join(raw_keys)
    out = ssh_dir / "authorized_keys.generated"

    lines = _public_key_lines(all_keys)

    if not lines:
        msg = (
            "no SSH public key is available for the runtime image.\n"
            "Set one of:\n"
            f"  - {SSH_PUBLIC_KEY_ENV}\n"
            f"  - {GITHUB_SSH_KEY_USER_ENV}\n"
            "  - ssh/authorized_keys in the repository\n"
            "For GitHub Actions, set the repository owner as the GitHub user."
        )
        if strict:
            raise RuntimeError(msg)
        print(f"[ssh-keys] WARNING: {msg}", file=sys.stderr)

    content = "\n".join(lines) + ("\n" if lines else "")
    tmp = out.with_suffix(".generated.tmp")
    tmp.write_text(content)
    tmp.chmod(0o600)
    tmp.replace(out)
    out.chmod(0o600)

    if lines:
        print(
            f"[ssh-keys] prepared {len(lines)} public key(s) for image build",
            file=sys.stderr,
        )

    if shutil.which("ssh-keygen") and lines:
        for key in lines:
            try:
                proc = subprocess.run(
                    ["ssh-keygen", "-lf", "-"],
                    input=key,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=5,
                )
                if proc.returncode == 0 and proc.stdout:
                    for line in proc.stdout.strip().splitlines():
                        print(f"[ssh-keys] {line}", file=sys.stderr)
            except (subprocess.SubprocessError, OSError):
                pass

    return out
