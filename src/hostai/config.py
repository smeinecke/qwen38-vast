"""Configuration loading for hostai.

Settings are read from `hostai.toml` in the project root.  Secrets
(`VAST_API_KEY`, `HF_TOKEN`, `SSH_PUBLIC_KEY`) are read from `.env`.  On first
run, if `hostai.toml` is missing but `.env` exists, the tool imports non-secret
values from `.env` into `hostai.toml` and warns about the deprecated pattern.
"""

import dataclasses
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union, cast, get_args, get_origin

try:
    import tomllib  # type: ignore
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore
import tomli_w
from dotenv import dotenv_values


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    return int(value)


def _as_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _coerce(value: Any, field_type: Any) -> Any:
    if value is None:
        return None
    # Strip Optional / Union-with-None down to the concrete type.
    origin = get_origin(field_type)
    if origin is Union:
        inner = [a for a in get_args(field_type) if a is not type(None)]
        field_type = inner[0] if inner else str
    if field_type is bool:
        return _as_bool(value)
    if field_type is int:
        return _as_int(value)
    if field_type is float:
        return _as_float(value)
    if field_type is str:
        return _as_str(value)
    return value


T = TypeVar("T")


def _from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """Recursively instantiate a dataclass from a dict, ignoring extra keys."""
    if not dataclasses.is_dataclass(cls):
        return data  # type: ignore
    kwargs: Dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            value = data[f.name]
            if dataclasses.is_dataclass(f.type) and isinstance(value, dict):
                kwargs[f.name] = _from_dict(cast(Type[Any], f.type), value)
            else:
                kwargs[f.name] = _coerce(value, f.type)
    return cls(**kwargs)


def _to_dict(obj: Any) -> Any:
    """Recursively turn dataclasses into plain dicts for TOML/JSON output."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj) if getattr(obj, f.name) is not None
        }
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj


@dataclass
class HostaiSection:
    default_profile: str = "a6000"
    profiles_file: str = "profiles.json"
    ctx_size_override: Optional[int] = None
    gpu_query_override: str = ""


@dataclass
class MarketSection:
    max_dph: float = 0.80
    disk_gb: int = 100
    allow_paid_traffic: bool = False
    allow_unverified: bool = False
    max_inet_down_cost: float = 0.001
    max_inet_up_cost: float = 0.001


@dataclass
class ModelSection:
    hf_repo: str = "HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF"
    hf_revision: str = "993a5971fda8f30dd1b7eb2654792ba4415c7460"
    model: str = "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf"
    draft: str = "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf"
    use_fastmtp: bool = True
    reasoning_effort: str = "xhigh"
    cache_ram: Optional[int] = None
    ctx_checkpoints: Optional[int] = None
    cache_type_k: str = ""
    cache_type_v: str = ""


@dataclass
class ImageSection:
    base: str = ""
    unsecure: bool = False


@dataclass
class VastSection:
    volume_id: str = ""
    volume_mount_path: str = ""
    shm_size_gb: Optional[int] = None
    keep_on_failure: bool = False
    destroy_timeout_seconds: int = 45
    pause_timeout_seconds: int = 60


@dataclass
class CacheSection:
    enabled: bool = True
    host: str = ""
    port: int = 22
    user: str = "qwen-cache"
    root: str = "qwen-slot-cache"
    max_gb: int = 80
    session: str = "default"
    use_shm: bool = False
    shm_min_gb: int = 0
    shm_require: bool = False
    key: str = ""
    slot_id: int = 0
    local_dir: str = ""
    require_save: bool = False
    rclone: bool = False
    rclone_remote: str = ""
    rclone_type: str = ""
    rclone_url: str = ""
    rclone_user: str = ""
    rclone_password: str = ""


@dataclass
class MonitorSection:
    threshold_pct: float = 10.0
    interval: int = 180
    max_results: int = 5
    auto_start: bool = False
    profile: str = ""
    group: str = ""


@dataclass
class SSHSection:
    local_port: int = 18080
    local_port_auto: bool = True
    start_timeout: int = 1200


@dataclass
class BenchSection:
    max_tokens: int = 512
    timeout: int = 900
    temperature: float = 0.7


@dataclass
class Config:
    hostai: HostaiSection = field(default_factory=HostaiSection)
    market: MarketSection = field(default_factory=MarketSection)
    model: ModelSection = field(default_factory=ModelSection)
    image: ImageSection = field(default_factory=ImageSection)
    vast: VastSection = field(default_factory=VastSection)
    cache: CacheSection = field(default_factory=CacheSection)
    monitor: MonitorSection = field(default_factory=MonitorSection)
    ssh: SSHSection = field(default_factory=SSHSection)
    bench: BenchSection = field(default_factory=BenchSection)

    # Filled by loader
    root_dir: Path = field(default_factory=Path)
    config_file: Path = field(default_factory=Path)
    env_file: Path = field(default_factory=Path)
    secrets: Dict[str, str] = field(default_factory=dict)


# Mapping from legacy .env / environment variable name to (section, attribute, converter).
ENV_MAP: Dict[str, tuple[str, str, Optional[Type[Any]]]] = {
    "QWEN_PROFILE": ("hostai", "default_profile", str),
    "QWEN_PROFILES_FILE": ("hostai", "profiles_file", str),
    "CTX_SIZE_OVERRIDE": ("hostai", "ctx_size_override", int),
    "GPU_QUERY_OVERRIDE": ("hostai", "gpu_query_override", str),
    "MAX_DPH": ("market", "max_dph", float),
    "DISK_GB": ("market", "disk_gb", int),
    "QWEN_ALLOW_PAID_TRAFFIC": ("market", "allow_paid_traffic", bool),
    "QWEN_ALLOW_UNVERIFIED": ("market", "allow_unverified", bool),
    "QWEN_MAX_INET_DOWN_COST": ("market", "max_inet_down_cost", float),
    "QWEN_MAX_INET_UP_COST": ("market", "max_inet_up_cost", float),
    "HF_REPO": ("model", "hf_repo", str),
    "HF_REVISION": ("model", "hf_revision", str),
    "MODEL": ("model", "model", str),
    "DRAFT": ("model", "draft", str),
    "USE_FASTMTP": ("model", "use_fastmtp", bool),
    "REASONING_EFFORT": ("model", "reasoning_effort", str),
    "CACHE_RAM": ("model", "cache_ram", int),
    "CTX_CHECKPOINTS": ("model", "ctx_checkpoints", int),
    "CACHE_TYPE_K": ("model", "cache_type_k", str),
    "CACHE_TYPE_V": ("model", "cache_type_v", str),
    "GHCR_IMAGE_BASE": ("image", "base", str),
    "GHCR_IMAGE": ("image", "base", str),
    "QWEN_UNSECURE": ("image", "unsecure", bool),
    "QWEN_VOLUME_ID": ("vast", "volume_id", str),
    "QWEN_VOLUME_MOUNT_PATH": ("vast", "volume_mount_path", str),
    "QWEN_SHM_SIZE_GB": ("vast", "shm_size_gb", int),
    "KEEP_ON_FAILURE": ("vast", "keep_on_failure", bool),
    "QWEN_DESTROY_TIMEOUT_SECONDS": ("vast", "destroy_timeout_seconds", int),
    "QWEN_PAUSE_TIMEOUT_SECONDS": ("vast", "pause_timeout_seconds", int),
    "QWEN_SLOT_CACHE_ENABLED": ("cache", "enabled", bool),
    "QWEN_SLOT_CACHE_HOST": ("cache", "host", str),
    "QWEN_SLOT_CACHE_PORT": ("cache", "port", int),
    "QWEN_SLOT_CACHE_USER": ("cache", "user", str),
    "QWEN_SLOT_CACHE_ROOT": ("cache", "root", str),
    "QWEN_SLOT_CACHE_MAX_GB": ("cache", "max_gb", int),
    "QWEN_SLOT_CACHE_SESSION": ("cache", "session", str),
    "QWEN_SLOT_CACHE_USE_SHM": ("cache", "use_shm", bool),
    "QWEN_SHM_MIN_GB": ("cache", "shm_min_gb", int),
    "QWEN_SLOT_CACHE_SHM_REQUIRE": ("cache", "shm_require", bool),
    "QWEN_SLOT_CACHE_KEY": ("cache", "key", str),
    "QWEN_SLOT_CACHE_SLOT_ID": ("cache", "slot_id", int),
    "QWEN_SLOT_CACHE_LOCAL_DIR": ("cache", "local_dir", str),
    "QWEN_SLOT_CACHE_REQUIRE_SAVE": ("cache", "require_save", bool),
    "QWEN_SLOT_CACHE_RCLONE": ("cache", "rclone", bool),
    "QWEN_SLOT_CACHE_RCLONE_REMOTE": ("cache", "rclone_remote", str),
    "QWEN_SLOT_CACHE_RCLONE_TYPE": ("cache", "rclone_type", str),
    "QWEN_SLOT_CACHE_RCLONE_URL": ("cache", "rclone_url", str),
    "QWEN_SLOT_CACHE_RCLONE_USER": ("cache", "rclone_user", str),
    "QWEN_SLOT_CACHE_RCLONE_PASSWORD": ("cache", "rclone_password", str),
    "QWEN_MONITOR_THRESHOLD_PCT": ("monitor", "threshold_pct", float),
    "QWEN_MONITOR_INTERVAL": ("monitor", "interval", int),
    "QWEN_MONITOR_MAX_RESULTS": ("monitor", "max_results", int),
    "QWEN_MONITOR_AUTO_START": ("monitor", "auto_start", bool),
    "QWEN_MONITOR_PROFILE": ("monitor", "profile", str),
    "QWEN_MONITOR_GROUP": ("monitor", "group", str),
    "LOCAL_PORT": ("ssh", "local_port", int),
    "LOCAL_PORT_AUTO": ("ssh", "local_port_auto", bool),
    "START_TIMEOUT": ("ssh", "start_timeout", int),
    "BENCH_MAX_TOKENS": ("bench", "max_tokens", int),
    "BENCH_TIMEOUT": ("bench", "timeout", int),
}

SECRETS = {"VAST_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "SSH_PUBLIC_KEY"}


CTX_SIZE_OVERRIDE = "CTX_SIZE_OVERRIDE"
GPU_QUERY_OVERRIDE = "GPU_QUERY_OVERRIDE"
GHCR_IMAGE_OVERRIDE = "GHCR_IMAGE_OVERRIDE"


def _apply_env_to_config(config: Config, env: Dict[str, Optional[str]]) -> None:
    for key, raw in env.items():
        if not raw:
            continue
        if key in ENV_MAP:
            section_name, attr, converter = ENV_MAP[key]
            section = getattr(config, section_name)
            converted = _coerce(raw, converter) if converter else raw
            setattr(section, attr, converted)


def _apply_os_env_overrides(config: Config) -> None:
    """Apply environment variables that should override TOML/.env for one-offs."""
    for key, (section_name, attr, converter) in ENV_MAP.items():
        raw = os.environ.get(key)
        if raw is not None:
            section = getattr(config, section_name)
            converted = _coerce(raw, converter) if converter else raw
            setattr(section, attr, converted)


def _set_secret_env(env: Dict[str, Optional[str]]) -> None:
    """Put secrets from .env into os.environ for tools that read them (vastai, hf)."""
    for key in SECRETS:
        raw = env.get(key)
        if raw and not os.environ.get(key):
            os.environ[key] = raw


def _write_hostai_toml(config: Config) -> None:
    """Write the current settings (minus metadata) to hostai.toml."""
    data = _to_dict(config)
    # Drop loader-only fields
    data.pop("root_dir", None)
    data.pop("config_file", None)
    data.pop("env_file", None)
    data.pop("secrets", None)
    config.config_file.write_bytes(tomli_w.dumps(data).encode("utf-8"))
    config.config_file.chmod(0o600)


def find_project_root(start: Optional[Path] = None) -> Path:
    path = start or Path.cwd()
    for candidate in [path, *path.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / "hostai.toml").exists():
            return candidate
    return path


def load_config(project_root: Optional[Path] = None) -> Config:
    root = project_root or find_project_root()
    config_file = root / "hostai.toml"
    env_file = root / ".env"

    config = Config(root_dir=root, config_file=config_file, env_file=env_file)

    if config_file.exists():
        with config_file.open("rb") as f:
            data = tomllib.load(f)
        if data:
            for section_name in data:
                if hasattr(config, section_name):
                    section = getattr(config, section_name)
                    if dataclasses.is_dataclass(section) and not isinstance(section, type):
                        for f in dataclasses.fields(section):
                            if f.name in data[section_name]:
                                value = data[section_name][f.name]
                                if dataclasses.is_dataclass(f.type) and isinstance(value, dict):
                                    setattr(section, f.name, _from_dict(cast(Type[Any], f.type), value))
                                else:
                                    setattr(section, f.name, _coerce(value, f.type))

    # Handle .env: secrets and (on first run) migration to hostai.toml
    env: Dict[str, Optional[str]] = {}
    if env_file.exists():
        env = dotenv_values(env_file)

    if not config_file.exists() and env:
        warnings.warn(
            f"{config_file.name} not found; importing non-secret settings from {env_file.name}. "
            "Please move non-secret settings into hostai.toml and keep .env only for secrets.",
            stacklevel=2,
        )
        _apply_env_to_config(config, env)
        _write_hostai_toml(config)
        print(f"[config] created {config_file}", file=sys.stderr)

    if env:
        # Always make secrets available in the environment for subprocesses/SKDs.
        _set_secret_env(env)
        # If hostai.toml exists, collect only secret values; warn about ignored non-secrets.
        if config_file.exists():
            for key, raw in env.items():
                if not raw or key in SECRETS or key not in ENV_MAP:
                    continue
                section_name, attr, _ = ENV_MAP[key]
                section = getattr(config, section_name)
                current = getattr(section, attr)
                if _coerce(raw, type(current)) != current:
                    warnings.warn(
                        f"{env_file.name} contains non-secret '{key}'; "
                        f"ignoring it because {config_file.name} takes precedence.",
                        stacklevel=2,
                    )

    # Apply current environment overrides (CLI one-offs take precedence).
    _apply_os_env_overrides(config)

    # Expose secrets in the config object without printing them.
    for key in SECRETS:
        value = os.environ.get(key) or env.get(key) or ""
        if value:
            config.secrets[key] = value

    # Empty optional ints from env can become 0; treat 0 as None where empty means unset.
    for field_name in ("cache_ram", "ctx_checkpoints", "shm_size_gb"):
        section = getattr(config, "model" if field_name in ("cache_ram", "ctx_checkpoints") else "vast")
        current = getattr(section, field_name)
        if current == 0:
            setattr(section, field_name, None)

    return config


def image_for_profile(config: Config, image_tag: str) -> str:
    base = config.image.base or ""
    base = base.rstrip("/")
    if not base or "OWNER/REPO" in base:
        raise ValueError("image.base is not configured; set it in hostai.toml or GHCR_IMAGE_BASE in .env")
    return f"{base}:{image_tag}"
