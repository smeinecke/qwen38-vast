"""Runtime state persistence (.hostai-vast/state.json and run metadata)."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from hostai import utils

SENSITIVE_KEYS = ("api_key", "tunnel_pid")


class State:
    """Mutable wrapper around state.json."""

    def __init__(self, state_file: Path, data: Optional[Dict[str, Any]] = None):
        self.state_file = state_file
        self._data: Dict[str, Any] = data or {}

    def __getattr__(self, name: str) -> Any:
        return self._data.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("state_file", "_data"):
            super().__setattr__(name, value)
            return
        prop = getattr(type(self), name, None)
        if isinstance(prop, property) and prop.fset is not None:
            prop.fset(self, value)
        else:
            self._data[name] = value

    @classmethod
    def load(cls, state_file: Path) -> "State":
        if state_file.exists():
            data = json.loads(state_file.read_text())
        else:
            data = {}
        return cls(state_file, data)

    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.parent.chmod(0o700)
        tmp = self.state_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False))
        tmp.chmod(0o600)
        tmp.replace(self.state_file)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def update(self, values: Dict[str, Any]) -> None:
        self._data.update(values)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    @property
    def exists(self) -> bool:
        return self.state_file.exists()

    @property
    def instance_id(self) -> Optional[int]:
        value = self._data.get("instance_id")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @instance_id.setter
    def instance_id(self, value: Optional[int]) -> None:
        self._data["instance_id"] = value

    @property
    def status(self) -> str:
        return str(self._data.get("status", "unknown"))

    @status.setter
    def status(self, value: str) -> None:
        self._data["status"] = value

    @property
    def run_dir(self) -> Optional[Path]:
        value = self._data.get("run_dir")
        return Path(value) if value else None

    @run_dir.setter
    def run_dir(self, value: Path) -> None:
        self._data["run_dir"] = str(value)

    @property
    def api_key(self) -> Optional[str]:
        return self._data.get("api_key")

    @api_key.setter
    def api_key(self, value: Optional[str]) -> None:
        self._data["api_key"] = value

    @property
    def local_port(self) -> int:
        try:
            return int(self._data.get("local_port", 18080))
        except (TypeError, ValueError):
            return 18080

    @local_port.setter
    def local_port(self, value: int) -> None:
        self._data["local_port"] = value

    @property
    def tunnel_pid(self) -> Optional[int]:
        value = self._data.get("tunnel_pid")
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @tunnel_pid.setter
    def tunnel_pid(self, value: Optional[int]) -> None:
        self._data["tunnel_pid"] = value

    @property
    def ssh_url(self) -> Optional[str]:
        return self._data.get("ssh_url")

    @ssh_url.setter
    def ssh_url(self, value: Optional[str]) -> None:
        self._data["ssh_url"] = value

    @property
    def ssh_identity(self) -> Optional[Path]:
        value = self._data.get("ssh_identity")
        return Path(value) if value else None

    @ssh_identity.setter
    def ssh_identity(self, value: Optional[Path]) -> None:
        self._data["ssh_identity"] = str(value) if value else None

    @property
    def started_epoch(self) -> Optional[int]:
        value = self._data.get("started_epoch")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @started_epoch.setter
    def started_epoch(self, value: Optional[int]) -> None:
        self._data["started_epoch"] = value

    @property
    def run_started_epoch(self) -> Optional[int]:
        value = self._data.get("run_started_epoch")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @run_started_epoch.setter
    def run_started_epoch(self, value: Optional[int]) -> None:
        self._data["run_started_epoch"] = value

    @property
    def dph(self) -> float:
        try:
            return float(self._data.get("dph", 0))
        except (TypeError, ValueError):
            return 0.0

    @dph.setter
    def dph(self, value: float) -> None:
        self._data["dph"] = value

    @property
    def profile(self) -> str:
        return str(self._data.get("profile", "unknown"))

    @profile.setter
    def profile(self, value: str) -> None:
        self._data["profile"] = value

    @property
    def ctx_size(self) -> int:
        try:
            return int(self._data.get("ctx_size", 0))
        except (TypeError, ValueError):
            return 0

    @ctx_size.setter
    def ctx_size(self, value: int) -> None:
        self._data["ctx_size"] = value

    @property
    def image(self) -> str:
        return str(self._data.get("image", "unknown"))

    @image.setter
    def image(self, value: str) -> None:
        self._data["image"] = value

    @property
    def gpu(self) -> str:
        return str(self._data.get("gpu", "unknown"))

    @gpu.setter
    def gpu(self, value: str) -> None:
        self._data["gpu"] = value

    @property
    def slot_cache_enabled(self) -> bool:
        return bool(self._data.get("slot_cache_enabled", False))

    @slot_cache_enabled.setter
    def slot_cache_enabled(self, value: bool) -> None:
        self._data["slot_cache_enabled"] = value

    @property
    def unsecure(self) -> bool:
        return bool(self._data.get("unsecure", False))

    @unsecure.setter
    def unsecure(self, value: bool) -> None:
        self._data["unsecure"] = value

    @property
    def tls_ca(self) -> Optional[Path]:
        value = self._data.get("tls_ca")
        return Path(value) if value else None

    @tls_ca.setter
    def tls_ca(self, value: Optional[Path]) -> None:
        self._data["tls_ca"] = str(value) if value else ""

    @property
    def slot_cache_session(self) -> str:
        return str(self._data.get("slot_cache_session", "default"))

    @slot_cache_session.setter
    def slot_cache_session(self, value: str) -> None:
        self._data["slot_cache_session"] = value

    @property
    def slot_cache_signature(self) -> str:
        return str(self._data.get("slot_cache_signature", ""))

    def public_dict(self) -> Dict[str, Any]:
        """Return a copy with sensitive/ transient keys removed."""
        out = deepcopy(self._data)
        for key in SENSITIVE_KEYS:
            out.pop(key, None)
        return out

    def save_metadata(self, run_dir: Path, status: Optional[str] = None) -> None:
        metadata = self.public_dict()
        if status:
            metadata["status"] = status
        metadata["metadata_updated_at"] = utils.now_rfc3339()
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dir.chmod(0o700)
        path = run_dir / "metadata.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        tmp.chmod(0o600)
        tmp.replace(path)


def init_run_dir(runs_dir: Path, profile: str, run_id: Optional[str] = None) -> Path:
    run_id = run_id or utils.make_run_id(profile)
    run_dir = runs_dir / run_id
    utils.mkdir_private(run_dir)
    return run_dir


def state_dir(root_dir: Path) -> Path:
    return utils.mkdir_private(root_dir / ".hostai-vast")


def runs_dir(root_dir: Path) -> Path:
    return utils.mkdir_private(root_dir / ".hostai-runs")


def cache_dir(root_dir: Path) -> Path:
    return utils.mkdir_private(root_dir / ".hostai-cache")
