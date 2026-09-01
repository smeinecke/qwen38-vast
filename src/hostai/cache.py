"""Persistent llama.cpp slot/KV cache helpers.

The cache server only needs SSH + rsync. A dedicated Ed25519 key lives locally
under ``.hostai-cache/`` and is authorized only for the cache account on the
persistent cache server.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import click

from hostai import ssh, utils
from hostai.config import Config
from hostai.state import State


def _cache_dir(root_dir: Path) -> Path:
    return utils.mkdir_private(root_dir / ".hostai-cache")


def _cache_key_path(root_dir: Path) -> Path:
    return _cache_dir(root_dir) / "cache_ed25519"


def _known_hosts_path(root_dir: Path) -> Path:
    return _cache_dir(root_dir) / "known_hosts"


def _default_local_dir(config: Config) -> str:
    if config.cache.local_dir:
        return config.cache.local_dir
    if config.cache.use_shm:
        return "/dev/shm/qwen38/slots"
    return "/var/lib/qwen38/slots"


def cache_signature(
    llama_commit: str,
    model: str,
    hf_revision: str,
    ctx_size: int,
    use_fastmtp: int,
    cache_type_k: str = "default",
    cache_type_v: str = "default",
) -> str:
    """SHA256 cache signature for the remote slot cache."""
    obj = {
        "llama_cpp_commit": llama_commit,
        "model": model,
        "hf_revision": hf_revision,
        "ctx_size": int(ctx_size),
        "use_fastmtp": int(use_fastmtp),
        "cache_type_k": cache_type_k,
        "cache_type_v": cache_type_v,
    }
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def _cache_key_path_for_config(config: Config) -> Path:
    """Return the configured or default cache key path."""
    if config.cache.key:
        return Path(config.cache.key).expanduser()
    return _cache_key_path(config.root_dir)


def cache_config(config: Config) -> SimpleNamespace:
    """Return a namespace with the common cache settings."""
    return SimpleNamespace(
        enabled=config.cache.enabled,
        host=config.cache.host,
        port=config.cache.port,
        user=config.cache.user,
        root=config.cache.root,
        session=config.cache.session,
        max_gb=config.cache.max_gb,
        use_shm=config.cache.use_shm,
        shm_min_gb=config.cache.shm_min_gb,
        shm_require=config.cache.shm_require,
        local_dir=_default_local_dir(config),
        key=_cache_key_path_for_config(config),
        slot_id=config.cache.slot_id,
        require_save=config.cache.require_save,
        rclone=config.cache.rclone,
        rclone_remote=config.cache.rclone_remote,
        rclone_type=config.cache.rclone_type,
        rclone_url=config.cache.rclone_url,
        rclone_user=config.cache.rclone_user,
        rclone_password=config.cache.rclone_password,
    )


def rclone_enabled(config: Config) -> bool:
    """Return True when the rclone cache backend is enabled."""
    return config.cache.rclone


def rclone_remote_name(config: Config) -> str:
    """Return the configured rclone remote name, or the built-in default."""
    if config.cache.rclone_remote:
        return config.cache.rclone_remote
    return "hostai"


def _rclone_type(config: Config) -> str:
    """Return the configured rclone backend type, defaulting to webdav."""
    return config.cache.rclone_type or "webdav"


def _rclone_user(config: Config) -> str:
    """Return the rclone backend user, falling back to the cache user."""
    return config.cache.rclone_user or config.cache.user


def _rclone_env_script(config: Config) -> str:
    """Return shell export statements for a non-preconfigured rclone backend.

    The password is obfuscated at runtime with ``rclone obscure`` so the raw
    password only lives in the shell script for a single moment.  When a
    preconfigured remote is used, no extra environment is generated.
    """
    if config.cache.rclone_remote:
        return ""

    name = rclone_remote_name(config).upper()
    rtype = _rclone_type(config)
    url = config.cache.rclone_url or config.cache.host
    user = _rclone_user(config)
    password = config.cache.rclone_password

    lines = [
        f"export RCLONE_CONFIG_{name}_TYPE={shlex.quote(rtype)}",
        f"export RCLONE_CONFIG_{name}_URL={shlex.quote(url)}",
        f"export RCLONE_CONFIG_{name}_USER={shlex.quote(user)}",
    ]
    if password:
        pass_quoted = shlex.quote(password)
        lines.extend(
            [
                f"pass_plain={pass_quoted}",
                f'export RCLONE_CONFIG_{name}_PASS="$(printf \'%s\\n\' "$pass_plain" | rclone obscure)"',
                "unset pass_plain",
            ]
        )
    return "\n".join(lines)


def rclone_prefetch_script(config: Config, slot_dir: str, remote_dir: str) -> str:
    """Return a bash script that downloads current.bin/json via rclone."""
    env = _rclone_env_script(config)
    remote_name = rclone_remote_name(config)
    return f"""set -Eeuo pipefail
umask 077
slot_dir={shlex.quote(slot_dir)}
remote_dir={shlex.quote(remote_dir)}
mkdir -p "$slot_dir"
chmod 700 "$slot_dir"
{env}
remote_name={shlex.quote(remote_name)}
rclone copyto "$remote_name:$remote_dir/current.bin" "$slot_dir/current.bin"
rclone copyto "$remote_name:$remote_dir/current.json" "$slot_dir/current.json" || true
chmod 600 "$slot_dir/current.bin" "$slot_dir/current.json" 2>/dev/null || true
echo ok
"""


def rclone_upload_script(config: Config, slot_dir: str, remote_dir: str) -> str:
    """Return a bash script that uploads current.bin/json via rclone."""
    env = _rclone_env_script(config)
    remote_name = rclone_remote_name(config)
    return f"""set -Eeuo pipefail
umask 077
slot_dir={shlex.quote(slot_dir)}
remote_dir={shlex.quote(remote_dir)}
chmod 600 "$slot_dir/current.bin" "$slot_dir/current.json" 2>/dev/null || true
{env}
remote_name={shlex.quote(remote_name)}
if ! command -v rclone >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y --no-install-recommends rclone
fi
rclone copyto "$slot_dir/current.bin" "$remote_name:$remote_dir/current.bin"
rclone copyto "$slot_dir/current.json" "$remote_name:$remote_dir/current.json"
echo ok
"""


def validate_cache_config(config: Config) -> bool:
    """Validate cache settings; returns False and warns when invalid."""
    if not config.cache.enabled:
        return False

    if not config.cache.root or ".." in config.cache.root:
        click.echo("[cache] ERROR: invalid cache.root", err=True)
        return False
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$", config.cache.session):
        click.echo("[cache] ERROR: cache.session must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}", err=True)
        return False
    if config.cache.max_gb <= 0:
        click.echo("[cache] ERROR: cache.max_gb must be positive", err=True)
        return False

    if config.cache.rclone:
        if config.cache.rclone_remote:
            return True
        if not config.cache.rclone_url and not config.cache.host:
            click.echo("[cache] ERROR: rclone enabled but no rclone_url or host configured", err=True)
            return False
        if not _rclone_user(config):
            click.echo("[cache] ERROR: rclone enabled but no user configured", err=True)
            return False
        return True

    if not config.cache.host or not re.match(r"^[A-Za-z0-9._:-]+$", config.cache.host):
        click.echo("[cache] ERROR: invalid or missing cache.host", err=True)
        return False
    if not (1 <= config.cache.port <= 65535):
        click.echo(f"[cache] ERROR: invalid cache.port {config.cache.port}", err=True)
        return False
    if not config.cache.user or not re.match(r"^[A-Za-z0-9._-]+$", config.cache.user):
        click.echo("[cache] ERROR: invalid cache.user", err=True)
        return False
    return True


def cache_ssh_url(config: Config) -> str:
    """Return the cache server ssh:// URL, e.g. ssh://user@host:port."""
    if not config.cache.host:
        return ""
    return f"ssh://{config.cache.user}@{config.cache.host}:{config.cache.port}"


def _parse_ssh_url(url: str, config: Config) -> Tuple[str, str, int]:
    """Parse ssh://user@host:port or user@host:port."""
    if url.startswith("ssh://"):
        user, host, port = utils.parse_ssh_url(url)
        return user, host, port

    m = re.match(r"^(?:([^@]+)@)?([^:@\s]+)(?::(\d+))?$", url)
    if not m:
        return "", "", 0

    user = m.group(1) or config.cache.user
    host = m.group(2) or ""
    port = int(m.group(3)) if m.group(3) else config.cache.port
    return user, host, port


def ensure_cache_key(config: Config, root_dir: Path) -> Path:
    """Create a dedicated Ed25519 key for the cache server if missing."""
    key_path = _cache_key_path_for_config(config)
    pub_path = key_path.with_suffix(".pub")

    if not key_path.is_file() or not pub_path.is_file():
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.parent.chmod(0o700)
        # Remove the public half only if the private half is also missing,
        # then generate a fresh pair. Use -y to avoid interactive prompts.
        if key_path.is_file() and not pub_path.is_file():
            pub_path.unlink(missing_ok=True)
        if not key_path.is_file():
            utils.run(
                [
                    "ssh-keygen",
                    "-q",
                    "-t",
                    "ed25519",
                    "-N",
                    "",
                    "-C",
                    "hostai-slot-cache",
                    "-f",
                    str(key_path),
                ],
                check=True,
                timeout=60,
            )
        else:
            # Private exists, regenerate public non-interactively.
            result = utils.run(
                ["ssh-keygen", "-q", "-y", "-f", str(key_path)],
                capture=True,
                check=True,
                timeout=30,
            )
            pub_path.write_text(result.stdout.strip() + "\n")
        key_path.chmod(0o600)
        if pub_path.exists():
            pub_path.chmod(0o644)

    return key_path


def _public_key(config: Config, root_dir: Path) -> str:
    key_path = ensure_cache_key(config, root_dir)
    pub_path = key_path.with_suffix(".pub")
    return pub_path.read_text().strip()


def copy_cache_key(config: Config, ssh_url: Optional[str] = None) -> bool:
    """Copy the cache public key to the cache server's ~/.ssh/authorized_keys.

    Uses the equivalent of ``ssh ... 'cat >> .ssh/authorized_keys'`` so it can
    be used interactively the first time the cache key is installed.
    """
    target = ssh_url or cache_ssh_url(config)
    if not target:
        return False

    user, host, port = _parse_ssh_url(target, config)
    if not host:
        return False

    pub = _public_key(config, config.root_dir)
    if not pub:
        return False

    pub = pub.strip()
    if not pub.endswith("\n"):
        pub += "\n"
    pub_quoted = shlex.quote(pub.rstrip("\n"))
    parts = pub.split()
    key_blob = parts[1] if len(parts) >= 2 else pub.rstrip("\n")
    blob_quoted = shlex.quote(key_blob)

    remote = (
        "umask 077; "
        "mkdir -p ~/.ssh; "
        "chmod 700 ~/.ssh; "
        "touch ~/.ssh/authorized_keys; "
        "chmod 600 ~/.ssh/authorized_keys; "
        f"pub_line={pub_quoted}; "
        f"key_blob={blob_quoted}; "
        'if [ -n "$key_blob" ]; then '
        'grep -vF "$key_blob" ~/.ssh/authorized_keys > ~/.ssh/authorized_keys.tmp || true; '
        "mv ~/.ssh/authorized_keys.tmp ~/.ssh/authorized_keys 2>/dev/null || true; "
        "fi; "
        "printf 'restrict %s\\n' \"$pub_line\" >> ~/.ssh/authorized_keys"
    )

    try:
        result = utils.run(
            [
                "ssh",
                "-p",
                str(port),
                "-o",
                "StrictHostKeyChecking=accept-new",
                f"{user}@{host}",
                remote,
            ],
            check=False,
            timeout=120,
        )
    except Exception:
        return False

    return result.returncode == 0


def install_cache_key_on_vast(state: State, config: Config) -> bool:
    """Copy the cache private key to a running Vast host so it can upload.

    The key is placed at ``/root/.ssh/qwen-slot-cache`` with mode 0600.
    Returns True on success.
    """
    if not state.ssh_url:
        return False
    key_path = ensure_cache_key(config, config.root_dir)
    if not key_path.exists():
        return False

    known_hosts = state.state_file.parent / "known_hosts"
    remote_key_path = "/root/.ssh/qwen-slot-cache"
    remote_dir = Path("/root/.ssh")
    try:
        mkdir_res = ssh.run_remote(
            state.ssh_url,
            f"mkdir -p {remote_dir} && chmod 700 {remote_dir}",
            known_hosts=known_hosts,
            timeout=30,
        )
        if mkdir_res.returncode != 0:
            return False
    except Exception:
        return False

    try:
        scp_res = ssh.scp_to(
            state.ssh_url,
            key_path,
            remote_key_path,
            known_hosts=known_hosts,
            timeout=60,
        )
        if scp_res.returncode != 0:
            return False
    except Exception:
        return False

    return True


def _cache_ssh_cmd(config: Config, root_dir: Path) -> str:
    """Build the SSH command string used by rsync's ``-e`` option."""
    key_path = ensure_cache_key(config, root_dir)
    known_hosts = _known_hosts_path(root_dir)
    return shlex.join(
        [
            "ssh",
            "-i",
            str(key_path),
            "-p",
            str(config.cache.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={known_hosts}",
        ]
    )


def preflight_remote(config: Config) -> bool:
    """SSH to the cache server and verify the root dir exists and is writable."""
    if not config.cache.host:
        return False

    root_dir = config.root_dir
    qroot = utils.sanitize_for_shell(config.cache.root)
    known_hosts = _known_hosts_path(root_dir)
    key_path = ensure_cache_key(config, root_dir)

    try:
        result = utils.run(
            [
                "ssh",
                "-i",
                str(key_path),
                "-p",
                str(config.cache.port),
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=3",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                f"UserKnownHostsFile={known_hosts}",
                f"{config.cache.user}@{config.cache.host}",
                f"command -v rsync >/dev/null 2>&1 && mkdir -p {qroot} && chmod 700 {qroot} && test -w {qroot}",
            ],
            check=False,
            timeout=60,
        )
    except Exception:
        return False

    return result.returncode == 0


def remote_cache_dir(
    config: Config,
    signature: str,
    session: Optional[str] = None,
) -> str:
    """Return the remote cache directory path for a signature.

    Example: ``<root>/<session>/<signature>``.
    """
    root = config.cache.root.rstrip("/")
    session = session or config.cache.session
    return f"{root}/{session}/{signature}"


def _signature_for_state(config: Config, state: State, llama_commit: str = "unknown") -> str:
    """Compute the cache signature from a runtime state object."""
    use_fastmtp = 1 if state.data.get("use_fastmtp", config.model.use_fastmtp) else 0
    cache_k = state.data.get("cache_type_k") or config.model.cache_type_k or "default"
    cache_v = state.data.get("cache_type_v") or config.model.cache_type_v or "default"
    return cache_signature(
        llama_commit,
        state.data.get("model", config.model.model),
        state.data.get("hf_revision", config.model.hf_revision),
        state.ctx_size,
        use_fastmtp,
        cache_k,
        cache_v,
    )


def _remote_cache_dir_for_state(
    config: Config,
    state: State,
    llama_commit: str = "unknown",
) -> str:
    """Compute the remote cache directory from a runtime state object."""
    signature = _signature_for_state(config, state, llama_commit)
    return remote_cache_dir(config, signature, state.slot_cache_session)


def _ensure_local_dir(local_dir: Path) -> None:
    """Create a local cache directory with private permissions."""
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        local_dir.chmod(0o700)
    except OSError:
        pass


def prefetch_cache(
    config: Config,
    state: State,
    local_dir: Path,
    llama_commit: str = "unknown",
) -> bool:
    """Rsync the remote cache directory to ``local_dir``."""
    if not config.cache.enabled or not config.cache.host:
        return False

    _ensure_local_dir(local_dir)

    remote_dir = _remote_cache_dir_for_state(config, state, llama_commit)
    src = f"{config.cache.user}@{config.cache.host}:{remote_dir}/"
    dest = str(local_dir)
    ssh_cmd = _cache_ssh_cmd(config, config.root_dir)

    try:
        result = utils.run(
            ["rsync", "-av", "-e", ssh_cmd, src, dest],
            check=False,
            timeout=1800,
        )
    except Exception:
        return False

    return result.returncode == 0


def upload_cache(
    config: Config,
    state: State,
    local_dir: Path,
    llama_commit: str = "unknown",
) -> bool:
    """Rsync ``local_dir`` to the remote cache directory atomically.

    Each file is first uploaded to ``.<name>.part`` and then ``chmod 600`` +
    ``mv``-ed into place so a partial upload is never visible as ``current.*``.
    """
    if not config.cache.enabled or not config.cache.host:
        return False

    if not local_dir.exists():
        return False

    remote_dir = _remote_cache_dir_for_state(config, state, llama_commit)
    key_path = ensure_cache_key(config, config.root_dir)
    known_hosts = _known_hosts_path(config.root_dir)
    ssh_cmd = _cache_ssh_cmd(config, config.root_dir)

    # Ensure the remote directory exists and is private.
    qremote = utils.sanitize_for_shell(remote_dir)
    try:
        mkdir_result = utils.run(
            [
                "ssh",
                "-i",
                str(key_path),
                "-p",
                str(config.cache.port),
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                f"UserKnownHostsFile={known_hosts}",
                f"{config.cache.user}@{config.cache.host}",
                f"mkdir -p {qremote} && chmod 700 {qremote}",
            ],
            check=False,
            timeout=60,
        )
        if mkdir_result.returncode != 0:
            return False
    except Exception:
        return False

    files = sorted(p for p in local_dir.iterdir() if p.is_file())
    if not files:
        return False

    for attempt in range(1, 4):
        for src_file in files:
            part_name = f".{src_file.name}.part"
            dest = f"{config.cache.user}@{config.cache.host}:{remote_dir}/{part_name}"
            try:
                res = utils.run(
                    ["rsync", "-a", "--inplace", "-e", ssh_cmd, str(src_file), dest],
                    check=False,
                    timeout=1800,
                )
            except Exception:
                return False
            if res.returncode != 0:
                if attempt == 3:
                    return False
                time.sleep(3)
                break
        else:
            # All files staged; atomically rename them and set permissions.
            qparts = " ".join(utils.sanitize_for_shell(f".{p.name}.part") for p in files)
            qfinals = " ".join(
                f"mv {utils.sanitize_for_shell(f'.{p.name}.part')} {utils.sanitize_for_shell(p.name)}" for p in files
            )
            finalize = utils.run(
                [
                    "ssh",
                    "-i",
                    str(key_path),
                    "-p",
                    str(config.cache.port),
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=8",
                    "-o",
                    "StrictHostKeyChecking=accept-new",
                    "-o",
                    f"UserKnownHostsFile={known_hosts}",
                    f"{config.cache.user}@{config.cache.host}",
                    f"cd {qremote} && chmod 600 {qparts} && {qfinals}",
                ],
                check=False,
                timeout=60,
            )
            return finalize.returncode == 0

    return False


def validate_cache(local_dir: Path) -> bool:
    """Check that a cache snapshot has the expected files.

    If ``local_dir`` is a directory, look for ``current.bin`` (and prefer a
    ``current.json`` metadata file). If it is a file, treat it as a sentinel.
    """
    if local_dir.is_file():
        try:
            text = local_dir.read_text()
            return "current.bin" in text or "sentinel" in text
        except OSError:
            return False

    if not local_dir.is_dir():
        return False

    bin_file = local_dir / "current.bin"
    if not bin_file.is_file() or bin_file.stat().st_size == 0:
        return False

    # Metadata is optional but strongly encouraged.
    json_file = local_dir / "current.json"
    if json_file.is_file() and json_file.stat().st_size > 0:
        return True

    return True
