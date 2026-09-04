"""Start a Vast instance for a profile."""

from __future__ import annotations

import base64
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import click

from hostai import cache, market, ssh, tls, utils
from hostai.api import LlamaClient
from hostai.commands import _common
from hostai.commands.monitor import maybe_start_monitor
from hostai.commands.watchdog import maybe_start_watchdog
from hostai.config import Config, image_for_profile
from hostai.profiles import Profiles
from hostai.providers import get_provider
from hostai.state import State, init_run_dir, runs_dir, state_dir
from hostai.validate import ValidationRecord, _image_info, compare_validations, load_last_validation


def _provider(config: Config):
    return get_provider(config)


def _log(message: str, err: bool = False) -> None:
    """Emit a console line with a wall-clock timestamp."""
    ts = time.strftime("%H:%M:%S", time.localtime())
    click.echo(f"[{ts}] {message}", err=err)


@click.command("up", help="Start a Vast instance for a profile.")
@click.argument("profile", required=False)
@click.option("-p", "--profile", "profile_opt", help="Profile to run.")
@click.option("-s", "--session", "cache_session", help="Slot-cache session name.")
@click.option("-l", "--local-port", type=int, help="Local tunnel port.")
@click.option("--max-price", type=float, help="Maximum all-in $/h.")
@click.option("--unverified", is_flag=True, help="Also consider unverified/unknown hosts.")
@click.option("--unsecure", is_flag=True, help="Use legacy TCP/no-TLS mode.")
@click.option("--no-cache", is_flag=True, help="Disable slot cache.")
@click.option("--keep-on-failure", is_flag=True, help="Do not destroy the instance if provisioning fails.")
@click.option("--abort-if-shm-too-small", is_flag=True, help="Fail if /dev/shm is too small.")
@click.option("--offer", type=int, help="Use a specific offer ID.")
@click.option("--restart", is_flag=True, help="Restart an existing paused instance.")
@click.option("--interruptible", is_flag=True, help="Use an interruptible/bid instance.")
@click.option("--bid", type=float, help="Bid price (max $/h) for interruptible instances.")
@click.option("--expected-session", help="Expected session duration, e.g. 30m, 2h.")
@click.option("--dry-run", is_flag=True, help="Search and print the chosen offer without renting.")
@click.option("--scoring-mode", help="Override market scoring mode (dph, perf, session).")
@click.option(
    "--allow-unvalidated",
    is_flag=True,
    help="Skip the production-validation gate when [vast].require_production_validation is true.",
)
@click.pass_obj
def cmd_up(
    config: Config,
    profile: Optional[str],
    profile_opt: Optional[str],
    cache_session: Optional[str],
    local_port: Optional[int],
    max_price: Optional[float],
    unverified: bool,
    unsecure: bool,
    no_cache: bool,
    keep_on_failure: bool,
    abort_if_shm_too_small: bool,
    offer: Optional[int],
    restart: bool,
    interruptible: bool,
    bid: Optional[float],
    expected_session: Optional[str],
    dry_run: bool,
    scoring_mode: Optional[str],
    allow_unvalidated: bool,
):
    chosen_profile = profile or profile_opt or config.hostai.default_profile
    if not chosen_profile:
        raise click.ClickException("no profile specified and no default profile configured")
    if local_port is not None and not (1 <= local_port <= 65535):
        raise click.ClickException("--local-port must be between 1 and 65535")
    if max_price is not None and max_price < 0:
        raise click.ClickException("--max-price must be non-negative")
    if bid is not None and bid <= 0:
        raise click.ClickException("--bid must be positive")
    if scoring_mode is not None and scoring_mode not in ("dph", "perf", "session"):
        raise click.ClickException("--scoring-mode must be dph, perf, or session")

    if scoring_mode:
        config.market.scoring_mode = scoring_mode

    if keep_on_failure:
        config.vast.keep_on_failure = True

    # Interruptible mode is enabled by CLI flag, config flag, or an explicit bid.
    use_interruptible = (
        interruptible or config.vast.interruptible or bid is not None or config.vast.bid_price is not None
    )
    if use_interruptible:
        bid_price = bid if bid is not None else config.vast.bid_price
        if bid_price is None:
            bid_price = max_price if max_price is not None else config.market.max_dph
        if bid_price is None or bid_price <= 0:
            raise click.ClickException(
                "interruptible mode requires a positive bid price: set --bid, [vast].bid_price, or [market].max_dph"
            )
    else:
        bid_price = None

    session_seconds = None
    if expected_session is not None:
        try:
            session_seconds = utils.parse_duration_to_seconds(expected_session)
        except ValueError as exc:
            raise click.ClickException(str(exc))
    if session_seconds is None and config.vast.expected_session_seconds is not None:
        session_seconds = config.vast.expected_session_seconds

    if restart:
        _do_restart(config, chosen_profile, local_port, unsecure, no_cache, allow_unvalidated=allow_unvalidated)
    else:
        _do_fresh(
            config,
            chosen_profile,
            cache_session,
            local_port,
            max_price,
            unverified,
            unsecure,
            no_cache,
            abort_if_shm_too_small,
            offer,
            bid_price=bid_price,
            session_seconds=session_seconds,
            dry_run=dry_run,
            allow_unvalidated=allow_unvalidated,
        )


def _check_production_validation(config: Config, allow_unvalidated: bool) -> Optional[ValidationRecord]:
    """Block a Vast launch unless a current production validation exists.

    When `[vast].require_production_validation` is true, this compares the
    working tree against the last successful `hostai validate --production` run.
    If the git commit, dirty state, integration image, or profiles have drifted,
    we refuse to call Vast unless the user passes --allow-unvalidated.
    """
    if config.provider.backend != "vast":
        return
    if not config.vast.require_production_validation:
        return
    if allow_unvalidated:
        _log("[validation] --allow-unvalidated set; skipping production validation gate", err=True)
        return

    previous = load_last_validation(config.root_dir, success=True)
    if previous is None or previous.result != "ok":
        raise click.ClickException(
            "[validation] no successful production validation on record. "
            "Run 'hostai validate --production' before a real Vast rental, "
            "or pass --allow-unvalidated."
        )
    if previous.level != "production" or "integration-tests" not in previous.checks_run:
        raise click.ClickException(
            "[validation] the last successful validation was not a production validation. "
            "Run 'hostai validate --production', or pass --allow-unvalidated."
        )

    current_image_id, current_image_digest = _image_info(previous.image)
    current_record = ValidationRecord(
        timestamp="",
        result="ok",
        duration_seconds=0.0,
        git_commit=utils.git_commit(config.root_dir) or "",
        dirty=utils.is_dirty_tree(config.root_dir),
        image=previous.image,
        image_id=current_image_id,
        image_digest=current_image_digest,
        profile_hash=utils.file_hash(config.root_dir / "profiles.json"),
        errors=[],
        level="production",
        checks_run=["integration-tests"],
    )
    diffs = compare_validations(current_record, previous)
    if diffs:
        for d in diffs:
            _log(f"[validation] DRIFT: {d}", err=True)
        raise click.ClickException(
            "[validation] current state does not match the last successful production validation. "
            "Run 'hostai validate --production' again, or pass --allow-unvalidated."
        )

    _log("[validation] production validation gate passed")
    return previous


def _now_rfc() -> str:
    return utils.now_rfc3339()


def _now_epoch() -> int:
    return int(time.time())


def _cleanup_instance(config: Config, state: State, reason: str) -> None:
    """Destroy the instance if something went wrong during provisioning."""
    if not state.instance_id:
        return
    if config.vast.keep_on_failure:
        _log(f"[cleanup] {reason}; keep_on_failure is set, not destroying {state.instance_id}", err=True)
        state.status = "failed"
        state.set("failure_reason", reason)
        state.save()
        return
    _log(f"[cleanup] {reason}; destroying instance {state.instance_id}...", err=True)
    try:
        _provider(config).destroy_instance(state.instance_id)
        _log(f"[cleanup] instance {state.instance_id} destroyed", err=True)
    except Exception as exc:
        _log(f"[cleanup] destroy failed: {exc}; please remove it manually", err=True)
    state.status = "failed"
    state.set("failure_reason", reason)
    state.save()


def _shm_preflight(
    ssh_url: Optional[str],
    config: Config,
    known_hosts: Path,
    min_gb: int,
) -> int:
    """Check /dev/shm on the Vast host. Returns 0=ok, 1=too-small, 2=error."""
    if not ssh_url:
        return 2
    if not config.cache.use_shm:
        return 0
    min_bytes = min_gb * 1024 * 1024 * 1024
    if min_bytes == 0:
        return 0
    res = ssh.run_remote(
        ssh_url,
        "df -P -B1 /dev/shm | awk 'NR==2 {print $4}'",
        known_hosts=known_hosts,
        config=config,
        timeout=30,
    )
    if res.returncode != 0:
        return 2
    try:
        free = int((res.stdout or "").strip() or 0)
    except (TypeError, ValueError):
        return 2
    if free < min_bytes:
        _log(
            f"[cache] /dev/shm free={free / 1024 / 1024 / 1024:.2f}GB is below shm_min_gb={min_gb}GB",
            err=True,
        )
        return 1
    _log(f"[cache] /dev/shm free={free / 1024 / 1024 / 1024:.2f}GB OK")
    return 0


def _resolve_profile(config: Config, name: str) -> Tuple[Profiles, Any, Any]:
    profiles = Profiles.from_file(config.root_dir / config.hostai.profiles_file)
    p = profiles.resolve_profile(name)
    if not p:
        raise click.ClickException(f"unknown profile '{name}'")
    image = profiles.image_by_name(p.image)
    if not image:
        raise click.ClickException(f"profile '{p.name}' references unknown image '{p.image}'")
    return profiles, p, image


def _resolve_client_port(
    config: Config,
    *,
    user_port: Optional[int] = None,
    state_port: Optional[int] = None,
) -> int:
    """Return a local client port that is free on 127.0.0.1.

    Honors the user-specified port (``--local-port`` or ``[proxy].port``) and
    fails early if that exact port is in use.  For the default port the helper
    searches for the next free port instead of starting a rental that will
    later fail to bind.
    """
    user_set = False
    if user_port is not None:
        desired = user_port
        user_set = True
    elif config.proxy.port:
        desired = config.proxy.port
        user_set = True
    elif state_port is not None:
        desired = state_port
    else:
        desired = config.ssh.local_port or 18081

    if not (1 <= desired <= 65535):
        raise click.ClickException("client port must be between 1 and 65535")

    if utils.port_is_free(desired, host="127.0.0.1"):
        config.ssh.local_port = desired
        return desired

    if user_set:
        raise click.ClickException(f"client port {desired} is already in use; choose another with --local-port")

    _log(f"[up] client port {desired} is in use; searching for a free port", err=True)
    try:
        free_port = utils.find_free_port(start=desired)
    except RuntimeError:
        raise click.ClickException("no free localhost port found")

    _log(f"[up] using client port {free_port}", err=True)
    config.ssh.local_port = free_port
    return free_port


def _env_dict(
    config: Config,
    profile: Any,
    image: Any,
    model: str,
    ctx_size: int,
    api_key: str,
    unsecure: bool,
    no_cache: bool,
    session: str,
) -> Dict[str, str]:
    slot_dir = cache._default_local_dir(config)
    env: Dict[str, str] = {
        "HOSTAI_PROFILE": profile.name,
        "LLAMA_API_KEY": api_key,
        "MODEL": model,
        "CTX_SIZE": str(ctx_size),
        "USE_FASTMTP": str(int(config.model.use_fastmtp)),
        "REASONING_EFFORT": config.model.reasoning_effort,
        "HF_REVISION": config.model.hf_revision,
        "HOSTAI_UNSECURE": "1" if unsecure else "0",
        "HOSTAI_TOKENIZED_ONLY": "1" if config.proxy.tokenized_only else "0",
        "SLOT_SAVE_PATH": slot_dir,
    }
    hf_token = config.secrets.get("HF_TOKEN") or config.secrets.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        env["HF_TOKEN"] = hf_token

    cache_ram = config.model.cache_ram if config.model.cache_ram is not None else profile.cache_ram
    ctx_checkpoints = (
        config.model.ctx_checkpoints if config.model.ctx_checkpoints is not None else profile.ctx_checkpoints
    )
    if cache_ram:
        env["CACHE_RAM"] = str(cache_ram)
    if ctx_checkpoints:
        env["CTX_CHECKPOINTS"] = str(ctx_checkpoints)

    ssh_public_key = config.secrets.get("SSH_PUBLIC_KEY")
    if ssh_public_key:
        env["HOSTAI_SSH_PUBLIC_KEY_B64"] = base64.b64encode(ssh_public_key.encode()).decode()

    if config.model.cache_type_k and config.model.cache_type_k != "default":
        env["CACHE_TYPE_K"] = config.model.cache_type_k
    if config.model.cache_type_v and config.model.cache_type_v != "default":
        env["CACHE_TYPE_V"] = config.model.cache_type_v

    # Forward deterministic fault-injection variables so integration tests can
    # exercise boot deadlines without changing production start.sh.
    for key, value in os.environ.items():
        if key.startswith("HOSTAI_FAULT_") or key.startswith("HOSTAI_TEST_"):
            env[key] = value

    # Vast maps container port 22 to a public host port in args/entrypoint mode.
    env["-p 22:22"] = "1"

    cache_configured = config.cache.host or config.cache.rclone_url or config.cache.rclone_remote
    if not no_cache and config.cache.enabled and cache_configured:
        env["HOSTAI_SLOT_CACHE_ENABLED"] = "1"
        env["HOSTAI_SLOT_CACHE_HOST"] = config.cache.host
        env["HOSTAI_SLOT_CACHE_PORT"] = str(config.cache.port)
        env["HOSTAI_SLOT_CACHE_USER"] = config.cache.user
        env["HOSTAI_SLOT_CACHE_ROOT"] = config.cache.root
        env["HOSTAI_SLOT_CACHE_SESSION"] = session
        env["HOSTAI_SLOT_CACHE_MAX_GB"] = str(config.cache.max_gb)
        env["HOSTAI_SLOT_CACHE_USE_SHM"] = "1" if config.cache.use_shm else "0"
        env["HOSTAI_SLOT_CACHE_MIN_GB"] = str(config.cache.shm_min_gb)
        env["HOSTAI_SLOT_CACHE_LOCAL_DIR"] = slot_dir
        if config.cache.rclone:
            env["HOSTAI_SLOT_CACHE_RCLONE"] = "1"
            if config.cache.rclone_remote:
                env["HOSTAI_SLOT_CACHE_RCLONE_REMOTE"] = config.cache.rclone_remote
            if config.cache.rclone_type:
                env["HOSTAI_SLOT_CACHE_RCLONE_TYPE"] = config.cache.rclone_type
            if config.cache.rclone_url:
                env["HOSTAI_SLOT_CACHE_RCLONE_URL"] = config.cache.rclone_url
            if config.cache.rclone_user:
                env["HOSTAI_SLOT_CACHE_RCLONE_USER"] = config.cache.rclone_user
            if config.cache.rclone_password:
                env["HOSTAI_SLOT_CACHE_RCLONE_PASSWORD"] = config.cache.rclone_password
    return env


def _extra_args(config: Config, no_cache: bool = False) -> str:
    parts = []
    shm_size_gb = config.vast.shm_size_gb
    if shm_size_gb is None and config.cache.enabled and config.cache.use_shm and not no_cache:
        shm_size_gb = config.cache.shm_min_gb
    if shm_size_gb:
        parts.append(f"--shm-size={shm_size_gb}g")
    return " ".join(parts)


def _emit_instance_logs(config: Config, instance_id: int, seen: Dict[str, Set[str]]) -> None:
    """Fetch and emit any new container/daemon log lines from the provider."""
    for kind, daemon in (("container", False), ("daemon", True)):
        try:
            text = _provider(config).get_logs(
                instance_id,
                tail=50,
                daemon_logs=daemon,
                timeout=15.0,
            )
        except Exception:
            # Logs may not be ready while the container is still starting.
            continue
        if not text or not isinstance(text, str):
            continue
        for line in text.splitlines():
            line = line.rstrip()
            if not line or line in seen[kind]:
                continue
            seen[kind].add(line)
            _log(f"[logs:{kind}] {line}")


def _wait_for_ssh_endpoint(config: Config, state: State, timeout: int) -> None:
    if not state.instance_id:
        raise click.ClickException("no instance id in state")
    start = time.monotonic()
    last_status = 0
    seen_log_lines: Dict[str, Set[str]] = {"container": set(), "daemon": set()}
    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout:
            raise click.ClickException(f"[boot:ssh-endpoint] timeout after {elapsed:.1f}s")
        inst = _provider(config).get_instance(state.instance_id)
        if not inst:
            _log("[boot] waiting for instance to appear...")
            time.sleep(5)
            continue
        status = inst.get("actual_status") or inst.get("status") or "loading"
        if status in ("exited", "offline", "unknown"):
            raise click.ClickException(f"instance entered status '{status}'")
        endpoint = ssh.resolve_ssh_endpoint(inst)
        if endpoint:
            state.ssh_url = endpoint["ssh_url"]
            state.status = "ssh-ready"
            state.save()
            _log(f"[boot:ssh-endpoint] {state.ssh_url} ready after {elapsed:.1f}s")
            return
        if elapsed - last_status >= 15:
            last_status = elapsed
            host = inst.get("public_ipaddr") or inst.get("public_ip") or "?"
            ports = inst.get("ports") or {}
            tcp = ports.get("22/tcp") or []
            port = tcp[0].get("HostPort") if tcp and isinstance(tcp[0], dict) else "?"
            _log(f"[boot] status={status} | ssh={host}:{port} | waiting...")
            _emit_instance_logs(config, state.instance_id, seen_log_lines)
        time.sleep(5)


def _wait_for_api(
    config: Config,
    state: State,
    timeout: int,
    client: LlamaClient,
    *,
    stage_label: str = "end-to-end",
) -> None:
    start = _now_epoch()
    interval = 1.0
    last_log = start
    known_hosts = state.state_file.parent / "known_hosts"
    seen_logs: Dict[str, Set[str]] = {"container": set(), "daemon": set(), "server": set()}
    while True:
        now = _now_epoch()
        if now - start > timeout:
            raise click.ClickException(f"timeout waiting for llama-server /health ({stage_label})")
        if client.health():
            _log(f"[boot:{stage_label}] /health OK after {now - start:.1f}s")
            return
        if now - last_log >= 15:
            last_log = now
            _log(f"[api] waiting for llama-server ({now - start}s / {timeout}s)")
            if state.instance_id:
                _emit_instance_logs(config, state.instance_id, seen_logs)
            # best-effort server log tail from inside the container
            try:
                result = ssh.run_remote(
                    state.ssh_url,
                    "tail -n 50 /var/log/qwen38/server.log 2>/dev/null || true",
                    known_hosts=known_hosts,
                    config=config,
                    state=state,
                    timeout=10,
                )
                if result.stdout:
                    for line in result.stdout.splitlines():
                        line = line.rstrip()
                        if not line or line in seen_logs["server"]:
                            continue
                        seen_logs["server"].add(line)
                        _log(f"[logs:server] {line}")
            except Exception:
                pass
        time.sleep(interval)
        interval = min(interval * 2, 5.0)


def _capture_disk_telemetry(config: Config, state: State, known_hosts: Path) -> Optional[Dict[str, Any]]:
    """Capture container disk usage after a successful cold start.

    Reads the disk-usage log written by ``start.sh`` and appends a final
    snapshot so operators can verify that the allocated disk is realistically
    sized.  Returns the combined telemetry or ``None`` if it could not be
    gathered.
    """
    if not state.ssh_url:
        return None

    run_dir = state.run_dir
    if not run_dir:
        return None

    log_path = "/var/log/qwen38/disk-usage.log"
    res = ssh.run_remote(
        state.ssh_url,
        f"cat {log_path} 2>/dev/null || true",
        known_hosts=known_hosts,
        config=config,
        state=state,
        timeout=30,
    )
    records: List[Dict[str, Any]] = []
    if res.stdout:
        for line in res.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    # Always capture a final snapshot in case the start.sh log is missing
    # (older images) or the container's log path differs.
    final_script = r"""python3 - <<'PY'
import json, os, subprocess

def du(path):
    if not os.path.exists(path):
        return 0
    try:
        return int(subprocess.check_output(["du", "-sb", path], text=True).split()[0])
    except Exception:
        return 0

df = subprocess.check_output(["df", "-B1", "/"], text=True).strip().splitlines()[1].split()
record = {
    "stage": "up-final",
    "total_bytes": int(df[1]),
    "used_bytes": int(df[2]),
    "free_bytes": int(df[3]),
    "models_bytes": du("/models"),
    "slots_disk_bytes": du("/var/lib/qwen38/slots"),
    "slots_shm_bytes": du("/dev/shm/qwen38/slots"),
    "log_bytes": du("/var/log/qwen38"),
    "run_bytes": du("/run/qwen38"),
    "tmp_bytes": du("/dev/shm/qwen38/tmp"),
}
print(json.dumps(record))
PY"""
    res2 = ssh.run_remote(state.ssh_url, final_script, known_hosts=known_hosts, config=config, state=state, timeout=60)
    if res2.returncode == 0 and res2.stdout:
        try:
            record = json.loads(res2.stdout.strip().splitlines()[-1])
            records.append(record)
        except (json.JSONDecodeError, IndexError):
            pass

    if not records:
        return None

    telemetry = {
        "captured_at": utils.now_rfc3339(),
        "disk_gb": state.disk_gb,
        "records": records,
    }
    out = run_dir / "disk-telemetry.json"
    out.write_text(json.dumps(telemetry, indent=2, default=str) + "\n")
    out.chmod(0o600)
    return telemetry


def _write_env_file(config: Config, state: State, api_url: str, base_url: str) -> None:
    env_path = state_dir(config.root_dir) / "env"

    client_base: Optional[str]
    if config.proxy.tokenized_only and config.proxy.port:
        # The proxy is the OpenAI-compatible endpoint for clients.
        client_base = f"http://127.0.0.1:{config.proxy.port}/v1"
    else:
        client_base = base_url

    lines = [
        f"export OPENAI_API_KEY='{state.api_key}'",
        f"export HOSTAI_MODEL='{state.data.get('model')}'",
        f"export HOSTAI_PROFILE='{state.data.get('profile')}'",
        f"export HOSTAI_VAST_INSTANCE_ID='{state.instance_id}'",
        f"export HOSTAI_BASE_URL='{base_url}'",
        f"export HOSTAI_API_URL='{api_url}'",
    ]
    if client_base:
        lines += [
            f"export OPENAI_BASE_URL='{client_base}'",
            f"export OPENAI_API_BASE='{client_base}'",
        ]

    if config.proxy.tokenized_only:
        socket_path = Path(config.proxy.socket_path) if config.proxy.socket_path else _default_proxy_socket(config)
        lines += [
            f"export HOSTAI_PROXY_SOCKET='{socket_path}'",
            "export HOSTAI_TOKENIZED_ONLY=1",
        ]
        if not config.proxy.port:
            lines += [
                "# tokenized-only is enabled; run 'hostai proxy' and connect your OpenAI client to HOSTAI_PROXY_SOCKET",
                "# or set HOSTAI_PROXY_PORT to expose the proxy on a local TCP port as well.",
            ]
        lines += [
            "# In tokenized-only mode the raw HOSTAI_API_URL is for diagnostics only.",
            "# All chat traffic must go through the hostai proxy so prompts are tokenized locally.",
        ]

    if not state.unsecure and state.tls_ca:
        ca = str(state.tls_ca)
        lines += [
            f"export HOSTAI_CA_CERT='{ca}'",
            f"export SSL_CERT_FILE='{ca}'",
            f"export CURL_CA_BUNDLE='{ca}'",
            f"export REQUESTS_CA_BUNDLE='{ca}'",
        ]
    env_path.write_text("\n".join(lines) + "\n")
    env_path.chmod(0o600)


def _default_proxy_socket(config: Config) -> Path:
    return state_dir(config.root_dir) / "proxy.sock"


def _hostai_binary() -> Path:
    """Return the path to the hostai executable for spawning child processes."""
    candidate = Path(sys.executable).parent / "hostai"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    path = shutil.which("hostai")
    if path:
        return Path(path)
    raise RuntimeError("could not find hostai executable")


def _start_proxy(config: Config, state: State, client_api_scheme: str = "http") -> Optional[int]:
    """Auto-start the local OpenAI proxy.

    In tokenized-only mode it tokenizes /v1/chat/completions. In normal mode it
    passes traffic through. The proxy owns the SSH connection to the remote
    (Unix-socket upstream if secure) and stays up after `up` exits.

    Returns the TCP port the proxy listens on for clients, or None when the
    proxy is not needed (tokenized-only is false and there is no pass-through
    configured).
    """
    if not config.proxy.tokenized_only:
        return None

    port = config.proxy.port or config.ssh.local_port or 0
    if port and not utils.port_is_free(port, host="127.0.0.1"):
        _log(f"[proxy] WARNING: configured port {port} is in use; skipping", err=True)
        return None
    if not port:
        try:
            port = utils.find_free_port(start=18083, host="127.0.0.1")
        except RuntimeError as exc:
            _log(f"[proxy] WARNING: no free TCP port: {exc}; skipping", err=True)
            return None
    config.proxy.port = port

    env = os.environ.copy()
    env["HOSTAI_PROXY_PORT"] = str(port)
    env["PYTHONUNBUFFERED"] = "1"
    log_path = state.state_file.parent / "proxy.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        proxy_bin = _hostai_binary()
    except RuntimeError as exc:
        _log(f"[proxy] WARNING: {exc}; skipping", err=True)
        return None

    _log(f"[proxy] auto-starting on {client_api_scheme}://127.0.0.1:{port}")
    try:
        with open(log_path, "a", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                [str(proxy_bin), "proxy"],
                env=env,
                start_new_session=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
    except OSError as exc:
        _log(f"[proxy] WARNING: failed to start: {exc}; skipping", err=True)
        return None

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if not utils.port_is_free(port, host="127.0.0.1"):
            break
        time.sleep(0.2)
    else:
        _log(f"[proxy] WARNING: did not see proxy on port {port}; check {log_path}", err=True)
        try:
            tail = log_path.read_text(encoding="utf-8").splitlines()[-50:]
            for line in tail:
                _log(f"[proxy] log: {line}", err=True)
        except Exception:
            pass
        return None

    # Record the proxy pid/port in the same state object the caller will keep
    # writing.  Reloading would create a fresh copy that the next state.save()
    # in the caller would overwrite.
    state.data["proxy_pid"] = proc.pid
    state.local_port = port
    state.save()

    api_url = f"{client_api_scheme}://127.0.0.1:{port}"
    base_url = f"{api_url}/v1"
    _write_env_file(config, state, api_url, base_url)
    _log(f"[proxy] ready (pid {proc.pid}); OPENAI_BASE_URL={base_url}")
    return port


def _prefetch_slot_cache_to_vast(
    ssh_url: Optional[str],
    config: Config,
    slot_dir: str,
    remote_dir: str,
    known_hosts: Path,
) -> bool:
    """Pull current.bin/json from the cache server (rsync or rclone)."""
    if not ssh_url:
        return False
    cache_configured = config.cache.host or config.cache.rclone_url or config.cache.rclone_remote
    if not cache_configured:
        return False

    if config.cache.rclone:
        script = cache.rclone_prefetch_script(config, slot_dir, remote_dir)
    else:
        script = cache.rsync_prefetch_script(config, slot_dir, remote_dir)
    res = ssh.run_remote(ssh_url, "bash -s", input_data=script, known_hosts=known_hosts, config=config, timeout=330)
    return res.returncode == 0 and "ok" in (res.stdout or "")


def _do_fresh(
    config: Config,
    profile_name: str,
    cache_session: Optional[str],
    local_port: Optional[int],
    max_price: Optional[float],
    unverified: bool,
    unsecure: bool,
    no_cache: bool,
    abort_if_shm_too_small: bool,
    offer: Optional[int],
    *,
    bid_price: Optional[float] = None,
    session_seconds: Optional[int] = None,
    dry_run: bool = False,
    allow_unvalidated: bool = False,
) -> None:
    sdir = state_dir(config.root_dir)
    existing = sdir / "state.json"
    if existing.exists():
        old = State.load(existing)
        if old.instance_id:
            try:
                inst = _provider(config).get_instance(old.instance_id)
            except Exception:
                # Could not reach provider to verify; proceed rather than hard-block.
                inst = None
            if inst and (inst.get("actual_status") or inst.get("status")) not in ("exited", "offline"):
                raise click.ClickException(
                    f"state file already references instance {old.instance_id}; run hostai down first"
                )

    # Resolve the client-side port before spending money on a rental.  Fail early
    # when the user explicitly asked for a port that is already in use; otherwise
    # search for a free port near the configured default.
    local_port = _resolve_client_port(config, user_port=local_port)

    profiles, profile, image = _resolve_profile(config, profile_name)
    ctx_size = config.hostai.ctx_size_override if config.hostai.ctx_size_override else profile.ctx_size
    model = config.model.model
    selected_image = image_for_profile(config, image.image_tag)
    disk_gb = market.resolved_disk_gb(profile, config)
    interruptible = bid_price is not None

    query, max_dph = market.build_search_query(
        config, profiles, profile, max_price=max_price, unverified=unverified, offer=offer, bid_price=bid_price
    )
    _log(
        f"[profile] {profile.name} | sm_{image.cuda_arch} | ctx={ctx_size} | image={selected_image} | disk={disk_gb}GB"
    )
    _log(f"[search]  {query}")

    offer_type = "bid" if interruptible else "on-demand"
    configured_max_dph = max_price if max_price is not None else config.market.max_dph
    if interruptible and bid_price is not None:
        effective_cap = min(configured_max_dph, bid_price)
        _log(
            f"[search]  mode={offer_type} configured_max_dph=${configured_max_dph:.4f}/h "
            f"bid=${bid_price:.4f}/h effective_cap=${effective_cap:.4f}/h"
        )
    else:
        _log(f"[search]  mode={offer_type} max_dph=${configured_max_dph:.4f}/h")
    provider = _provider(config)
    _log(f"[provider] {provider.name}")
    previous = _check_production_validation(config, allow_unvalidated)
    if config.vast.require_production_validation and previous is not None and not allow_unvalidated:
        # Use the immutable SHA-tagged image that the CI published for the
        # validated Git commit instead of the mutable profile tag.
        commit = previous.git_commit[:7]
        selected_image = image_for_profile(config, f"{image.image_tag}-sha-{commit}")
        _log(f"[image] gated to validated image: {selected_image}")

    offer_data = market.select_offer(
        config,
        profiles,
        query,
        max_dph=max_dph,
        unverified=unverified,
        offer=offer,
        storage=disk_gb,
        offer_type=offer_type,
        session_seconds=session_seconds,
        verbose=True,
    )
    offer_id_raw = offer_data.get("id") or offer_data.get("ask_contract_id")
    if offer_id_raw is None:
        raise click.ClickException("selected offer has no id")
    offer_id = int(offer_id_raw)
    gpu_name = offer_data.get("gpu_name", "unknown")
    dph = offer_data.get("dph_total", 0.0)
    _log(f"[rent] {market.offer_summary(offer_data)}")

    if dry_run:
        _log("\nDRY RUN: not creating an instance")
        return

    run_id = utils.make_run_id(profile.name)
    run_dir = init_run_dir(runs_dir(config.root_dir), profile.name, run_id)
    run_started = _now_rfc()
    run_epoch = _now_epoch()

    api_key = "sk-local-" + secrets.token_hex(24)
    session = cache_session or config.cache.session

    # metadata
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "provisioning",
        "started_at": _now_rfc(),
        "profile": profile.name,
        "monitor_group": profile.monitor_group or "",
        "gpu_query": query,
        "disk_gb": disk_gb,
        "interruptible": interruptible,
        "bid_price": bid_price if interruptible else None,
        "expected_session_seconds": session_seconds,
        "scoring_mode": config.market.scoring_mode,
        "cuda_arch": image.cuda_arch,
        "image": selected_image,
        "ctx_size": ctx_size,
        "model": model,
        "hf_revision": config.model.hf_revision,
        "use_fastmtp": config.model.use_fastmtp,
        "cache_type_k": config.model.cache_type_k or "default",
        "cache_type_v": config.model.cache_type_v or "default",
        "slot_cache_enabled": config.cache.enabled and not no_cache,
        "slot_cache_host": config.cache.host,
        "slot_cache_port": config.cache.port,
        "slot_cache_user": config.cache.user,
        "slot_cache_root": config.cache.root,
        "slot_cache_session": session,
        "slot_cache_max_gb": config.cache.max_gb,
        "slot_cache_local_dir": cache._default_local_dir(config),
        "slot_cache_use_shm": config.cache.use_shm,
        "unsecure": unsecure,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (run_dir / "metadata.json").chmod(0o600)

    env = _env_dict(config, profile, image, model, ctx_size, api_key, unsecure, no_cache, session)
    extra = _extra_args(config, no_cache=no_cache)
    label = f"hostai-{profile.name}-{_now_epoch()}"

    volume_info = None
    if config.vast.volume_id and config.vast.volume_mount_path:
        try:
            volume_info = {"volume_id": int(config.vast.volume_id), "mount_path": config.vast.volume_mount_path}
        except ValueError:
            volume_info = None

    try:
        create_kwargs: Dict[str, Any] = {
            "image": selected_image,
            "disk": disk_gb,
            "env": env,
            "label": label,
            "extra": extra,
            "runtype": "args",
            "args": None,
            "volume_info": volume_info,
        }
        if interruptible:
            create_kwargs["bid_price"] = bid_price
        create_raw = _provider(config).create_instance(offer_id, **create_kwargs)
    except Exception as exc:
        raise click.ClickException(f"create instance failed: {exc}")

    instance_id = create_raw.get("new_contract") or create_raw.get("instance_id") or create_raw.get("id")
    if not instance_id:
        raise click.ClickException(f"create response did not contain an instance ID: {create_raw}")

    started_at = _now_rfc()
    started_epoch = _now_epoch()

    state = State.load(sdir / "state.json")
    state.instance_id = int(instance_id)
    state.status = "provisioning"
    state.offer_id = offer_id
    state.gpu = gpu_name
    state.dph = float(dph)
    state.local_port = local_port if local_port is not None else config.ssh.local_port
    state.location = offer_data.get("geolocation", "") or offer_data.get("location", "")
    state.inet_down = offer_data.get("inet_down", 0.0)
    state.inet_down_cost = offer_data.get("inet_down_cost", 0.0)
    state.inet_up = offer_data.get("inet_up", 0.0)
    state.inet_up_cost = offer_data.get("inet_up_cost", 0.0)
    state.disk_bw = offer_data.get("disk_bw", 0.0)
    state.reliability = offer_data.get("reliability", 0.0)
    state.model = model
    state.hf_revision = config.model.hf_revision
    state.ctx_size = ctx_size
    state.api_key = api_key
    state.image = selected_image
    state.label = label
    state.profile = profile.name
    state.monitor_group = profile.monitor_group or ""
    state.disk_gb = disk_gb
    state.interruptible = interruptible
    state.bid_price = bid_price if interruptible else None
    state.expected_session_seconds = session_seconds
    state.run_id = run_id
    state.run_dir = run_dir
    state.started_at = started_at
    state.started_epoch = started_epoch
    state.run_started_at = run_started
    state.run_started_epoch = run_epoch
    state.unsecure = unsecure
    state.slot_cache_enabled = config.cache.enabled and not no_cache
    state.slot_cache_host = config.cache.host
    state.slot_cache_port = config.cache.port
    state.slot_cache_user = config.cache.user
    state.slot_cache_root = config.cache.root
    state.slot_cache_session = session
    state.slot_cache_max_gb = config.cache.max_gb
    state.slot_cache_local_dir = cache._default_local_dir(config)
    state.slot_cache_use_shm = config.cache.use_shm

    # LocalProvider generates an SSH key pair for the container.
    provider = _provider(config)
    private_key: Optional[Path] = getattr(provider, "ssh_private_key", None)
    if private_key:
        state.ssh_identity = private_key

    state.save()
    state.save_metadata(run_dir, status="provisioning")

    try:
        _do_fresh_core(config, state, image, no_cache, abort_if_shm_too_small)
    except click.ClickException:
        _cleanup_instance(config, state, "provisioning failed")
        raise
    except Exception as exc:
        _cleanup_instance(config, state, f"provisioning error: {exc}")
        raise click.ClickException(str(exc)) from exc


def _do_fresh_core(
    config: Config,
    state: State,
    image: Any,
    no_cache: bool,
    abort_if_shm_too_small: bool,
) -> None:
    """Provision a freshly created instance (SSH, cache, tunnel, TLS, API)."""
    _log(f"[boot] instance {state.instance_id} created; waiting for SSH...")
    _wait_for_ssh_endpoint(config, state, config.ssh.start_timeout)

    known_hosts = state.state_file.parent / "known_hosts"
    ssh_start = time.monotonic()
    if not ssh.wait_for_ssh(state.ssh_url, known_hosts=known_hosts, config=config, state=state, timeout=300):
        raise click.ClickException("[boot:ssh-command] timeout")
    _log(f"[boot:ssh-command] ready after {time.monotonic() - ssh_start:.1f}s")

    # runtime preflight
    result = ssh.run_remote(
        state.ssh_url,
        "/usr/local/bin/llama-server --version",
        known_hosts=known_hosts,
        config=config,
        state=state,
        timeout=30,
    )
    if result.returncode != 0:
        raise click.ClickException(f"remote llama-server preflight failed: {result.stderr}")

    # TLS: deliver certificates as soon as SSH is ready, before the slower cache
    # and slot-cache steps, so the remote start.sh does not time out waiting.
    if not state.unsecure:
        tls_dir = tls.ensure_local_tls_dir(config.root_dir)
        tls.generate_cert(tls_dir)
        _log("[tls] delivering certificates to container")
        tls_deadline = time.monotonic() + min(120.0, float(config.ssh.start_timeout or 1200))
        tls_delivered = False
        while not tls_delivered:
            if tls.deliver_cert(
                state.ssh_url,
                tls_dir,
                known_hosts=known_hosts,
                config=config,
                state=state,
                timeout=60,
            ):
                tls_delivered = True
                break
            if time.monotonic() >= tls_deadline:
                raise click.ClickException("TLS certificate delivery failed")
            _log("[tls] delivery attempt failed; retrying in 5s", err=True)
            time.sleep(5)
        state.tls_ca = tls_dir / "ca.crt"
        state.save()
        _log("[tls] certificates delivered")
        api_scheme = "https"
    else:
        api_scheme = "http"

    # cache setup
    cache_enabled = state.slot_cache_enabled
    if cache_enabled and not cache.validate_cache_config(config):
        _log("[cache] WARNING: cache config invalid; continuing cold", err=True)
        cache_enabled = False
        state.slot_cache_enabled = False

    if cache_enabled and not config.cache.rclone and not cache.install_cache_key_on_vast(state, config):
        _log("[cache] WARNING: could not install cache private key; continuing cold", err=True)
        cache_enabled = False
        state.slot_cache_enabled = False

    # slot cache restore (best effort)
    cache_remote = ""
    session = state.slot_cache_session
    llama_commit = "unknown"
    if cache_enabled:
        llama_commit = _common.fetch_llama_commit(state.ssh_url, known_hosts)
        state.data["llama_cpp_commit"] = llama_commit

        if config.cache.use_shm:
            shm_rc = _shm_preflight(
                state.ssh_url,
                config,
                known_hosts,
                config.cache.shm_min_gb,
            )
            if shm_rc == 1:
                if abort_if_shm_too_small or config.cache.shm_require:
                    raise click.ClickException("[cache] /dev/shm is too small and abort-if-shm-too-small is set")
                _log(
                    "[cache] /dev/shm is too small; falling back to disk slot cache",
                    err=True,
                )
                state.slot_cache_use_shm = False
                state.slot_cache_local_dir = "/var/lib/qwen38/slots"
            elif shm_rc == 2:
                raise click.ClickException("[cache] slot cache /dev/shm preflight failed")

    if cache_enabled:
        remote_local_dir = state.slot_cache_local_dir
        signature = cache._signature_for_state(config, state, llama_commit)
        cache_remote = cache.remote_cache_dir(config, signature, session)
        state.data["cuda_arch"] = image.cuda_arch
        state.data["slot_cache_signature"] = signature
        state.data["slot_cache_remote_dir"] = cache_remote
        state.data["slot_cache_restore"] = "pending"
        state.save()

        if _prefetch_slot_cache_to_vast(state.ssh_url, config, remote_local_dir, cache_remote, known_hosts):
            _log("[cache] prefetched slot from cache server")
            state.data["slot_cache_prefetch"] = "ok"
        else:
            _log("[cache] no slot cache on server; will start cold", err=True)
            state.data["slot_cache_prefetch"] = "empty"
        state.save()

    # Start the local proxy if tokenized-only is enabled. The proxy owns the
    # SSH tunnel to the remote Unix socket and provides the client-facing
    # OpenAI endpoint on LOCAL_PORT (or [proxy].port).
    if config.proxy.tokenized_only:
        proxy_port = _start_proxy(config, state, client_api_scheme="http")
        if not proxy_port:
            raise click.ClickException("[proxy] failed to start")
        _log(f"[tunnel] proxy on localhost:{proxy_port}")
    else:
        proxy_port = ssh.ensure_tunnel(config, state)
        _log(f"[tunnel] localhost:{proxy_port}")

    # Reload state after the proxy/tunnel process has written its metadata
    # (proxy_pid, upstream_socket, etc.) so the rest of provisioning and the
    # final ready state are consistent with the running proxy.
    state = State.load(state.state_file)

    # Build client API URL. When the proxy is active it is local HTTP; in the
    # non-tokenized legacy path we still speak directly to the remote TLS port.
    client_scheme = "http" if config.proxy.tokenized_only else api_scheme
    client_api_url = f"{client_scheme}://127.0.0.1:{proxy_port}"
    client_base_url = f"{client_api_url}/v1"
    _write_env_file(config, state, client_api_url, client_base_url)

    # Wait for the API to be reachable. In tokenized mode the proxy is the
    # endpoint and only speaks HTTP, so use a temporary client state with
    # unsecure=True to avoid requiring TLS for the local hop.
    if config.proxy.tokenized_only:
        client_state = State.load(state.state_file)
        client_state.local_port = proxy_port
        client_state.unsecure = True
        client = LlamaClient(config, client_state)
        _wait_for_api(config, client_state, config.ssh.start_timeout, client, stage_label="end-to-end via proxy")
    else:
        client = LlamaClient(config, state)
        _wait_for_api(config, state, config.ssh.start_timeout, client, stage_label="end-to-end")

    # restore slot cache
    if cache_enabled:
        if client.slot_restore(config.cache.slot_id):
            _log("[cache] slot restore requested")
            state.data["slot_cache_restore"] = "restored"
        else:
            _log("[cache] WARNING: slot restore failed; continuing cold", err=True)
            state.data["slot_cache_restore"] = "failed"
        state.save()

    ready_at = _now_rfc()
    ready_epoch = _now_epoch()
    state.status = "running"
    state.data["ready_at"] = ready_at
    state.data["ready_epoch"] = ready_epoch
    state.data["startup_seconds"] = ready_epoch - (state.started_epoch or 0)
    state.save()
    run_dir = state.run_dir
    if run_dir:
        state.save_metadata(run_dir, status="ready")

    if run_dir and state.ssh_url:
        telemetry = _capture_disk_telemetry(config, state, known_hosts)
        if telemetry:
            _log(
                f"[disk] telemetry: {len(telemetry['records'])} stages, free={telemetry['records'][-1]['free_bytes'] / 1e9:.2f}GB"
            )

    _log("\nREADY")
    _log(f"  Profile:   {state.profile} (sm_{image.cuda_arch})")
    _log(f"  Image:     {state.image}")
    _log(f"  GPU:       {state.gpu}")
    _log(f"  Cost:      ${float(state.dph):.4f}/h")
    _log(f"  Context:   {state.ctx_size}")
    _log(f"  API:       {client_base_url}")
    _log(f"  Instance:  {state.instance_id}")
    _log(f"  Run log:   {run_dir}")
    if cache_enabled:
        _log(f"  Slot cache: session={session} remote={cache_remote}")
    _log("\nRun: source .hostai-vast/env")
    _log("Stop: hostai down")

    maybe_start_watchdog(config, state)
    maybe_start_monitor(config, state)


def _do_restart(
    config: Config,
    profile_name: str,
    local_port: Optional[int],
    unsecure: bool,
    no_cache: bool,
    *,
    allow_unvalidated: bool = False,
) -> None:
    _check_production_validation(config, allow_unvalidated)
    sdir = state_dir(config.root_dir)
    state = State.load(sdir / "state.json")
    if not state.instance_id:
        raise click.ClickException("no state to restart; run hostai up first")

    state.unsecure = unsecure
    resolved = _resolve_client_port(config, user_port=local_port, state_port=state.local_port)
    if resolved != state.local_port:
        state.local_port = resolved
        state.save()

    inst = _provider(config).get_instance(state.instance_id)
    status = inst.get("actual_status") or inst.get("status") if inst else None
    if status != "running":
        try:
            _provider(config).start_instance(state.instance_id)
            _log(f"[restart] started instance {state.instance_id}")
        except Exception as exc:
            raise click.ClickException(f"failed to start instance: {exc}")

    _wait_for_ssh_endpoint(config, state, config.ssh.start_timeout)
    known_hosts = state.state_file.parent / "known_hosts"
    if not ssh.wait_for_ssh(state.ssh_url, known_hosts=known_hosts, config=config, state=state, timeout=300):
        raise click.ClickException("SSH daemon did not become reachable")

    # TLS: deliver certificates as soon as SSH is ready, before slower cache setup.
    if not state.unsecure:
        tls_dir = tls.ensure_local_tls_dir(config.root_dir)
        if not (tls_dir / "server.crt").exists():
            tls.generate_cert(tls_dir)
        _log("[tls] delivering certificates to container")
        tls_deadline = time.monotonic() + min(120.0, float(config.ssh.start_timeout or 1200))
        tls_delivered = False
        while not tls_delivered:
            if tls.deliver_cert(
                state.ssh_url,
                tls_dir,
                known_hosts=known_hosts,
                config=config,
                state=state,
                timeout=60,
            ):
                tls_delivered = True
                break
            if time.monotonic() >= tls_deadline:
                raise click.ClickException("TLS certificate delivery failed")
            _log("[tls] delivery attempt failed; retrying in 5s", err=True)
            time.sleep(5)
        state.tls_ca = tls_dir / "ca.crt"
        state.save()
        _log("[tls] certificates delivered")
        api_scheme = "https"
    else:
        api_scheme = "http"

    # cache setup for restart
    cache_enabled = state.slot_cache_enabled and not no_cache
    if cache_enabled and not cache.validate_cache_config(config):
        _log("[cache] WARNING: cache config invalid; continuing cold", err=True)
        cache_enabled = False
        state.slot_cache_enabled = False

    if cache_enabled:
        if not config.cache.rclone and not cache.install_cache_key_on_vast(state, config):
            _log("[cache] WARNING: could not install cache key; continuing cold", err=True)
            cache_enabled = False
            state.slot_cache_enabled = False
        else:
            if config.cache.use_shm:
                shm_rc = _shm_preflight(
                    state.ssh_url,
                    config,
                    known_hosts,
                    config.cache.shm_min_gb,
                )
                if shm_rc == 1:
                    if config.cache.shm_require:
                        raise click.ClickException("[cache] /dev/shm is too small and [cache].shm_require is set")
                    _log(
                        "[cache] /dev/shm is too small; falling back to disk slot cache",
                        err=True,
                    )
                    state.slot_cache_use_shm = False
                    state.slot_cache_local_dir = "/var/lib/qwen38/slots"
                elif shm_rc == 2:
                    raise click.ClickException("[cache] slot cache /dev/shm preflight failed")

            llama_commit = state.data.get("llama_cpp_commit") or _common.fetch_llama_commit(state.ssh_url, known_hosts)
            state.data["llama_cpp_commit"] = llama_commit
            signature = cache._signature_for_state(config, state, llama_commit)
            remote_local_dir = state.slot_cache_local_dir
            cache_remote = cache.remote_cache_dir(config, signature, state.slot_cache_session)
            state.data["slot_cache_signature"] = signature
            state.data["slot_cache_remote_dir"] = cache_remote
            state.data["slot_cache_restore"] = "pending"
            if _prefetch_slot_cache_to_vast(state.ssh_url, config, remote_local_dir, cache_remote, known_hosts):
                _log("[cache] prefetched slot from cache server")
                state.data["slot_cache_prefetch"] = "ok"
            else:
                _log("[cache] no slot cache on server; will start cold", err=True)
                state.data["slot_cache_prefetch"] = "empty"
            state.save()

    # Start the local proxy if tokenized-only is enabled; otherwise open a
    # direct SSH tunnel to the remote.
    if config.proxy.tokenized_only:
        proxy_port = _start_proxy(config, state, client_api_scheme="http")
        if not proxy_port:
            raise click.ClickException("[proxy] failed to start")
        _log(f"[tunnel] proxy on localhost:{proxy_port}")
    else:
        proxy_port = ssh.ensure_tunnel(config, state)
        _log(f"[tunnel] localhost:{proxy_port}")

    # Reload state after the proxy/tunnel process has updated its metadata.
    state = State.load(state.state_file)

    state.save()
    client_scheme = "http" if config.proxy.tokenized_only else api_scheme
    client_api_url = f"{client_scheme}://127.0.0.1:{proxy_port}"
    client_base_url = f"{client_api_url}/v1"
    _write_env_file(config, state, client_api_url, client_base_url)

    if config.proxy.tokenized_only:
        client_state = State.load(state.state_file)
        client_state.local_port = proxy_port
        client_state.unsecure = True
        client = LlamaClient(config, client_state)
        _wait_for_api(config, client_state, config.ssh.start_timeout, client, stage_label="end-to-end via proxy")
    else:
        client = LlamaClient(config, state)
        _wait_for_api(config, state, config.ssh.start_timeout, client, stage_label="end-to-end")

    # restore slot cache on restart
    if cache_enabled:
        if client.slot_restore(config.cache.slot_id):
            _log("[cache] slot restored")
            state.data["slot_cache_restore"] = "restored"
        else:
            _log("[cache] WARNING: slot restore failed; continuing cold", err=True)
            state.data["slot_cache_restore"] = "failed"
        state.save()

    state.status = "running"
    state.data["ready_at"] = _now_rfc()
    state.data["ready_epoch"] = _now_epoch()
    state.save()
    if state.run_dir:
        state.save_metadata(state.run_dir, status="restarted")

    _log("\nREADY")
    _log(f"  Profile:   {state.profile}")
    _log(f"  API:       {client_base_url}")
    _log(f"  Instance:  {state.instance_id}")

    maybe_start_watchdog(config, state)
    maybe_start_monitor(config, state)
