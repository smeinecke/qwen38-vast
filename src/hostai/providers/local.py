"""Local Docker provider for integration testing without real Vast.ai costs.

LocalProvider launches real local containers from the same runtime image used in
production (or a test derivative), maps container port 22 to a random host port,
and exposes Vast-like instance metadata so the rest of hostai can treat a local
container as a synthetic Vast host.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hostai import utils
from hostai.config import Config
from hostai.state import state_dir

from .base import Provider, ProviderError

# Deterministic local offers.  They are intentionally cheap ($0/h) and sized so
# the normal market filter/sort logic accepts them for common profiles.
_LOCAL_OFFERS = [
    {
        "id": 100001,
        "ask_contract_id": 100001,
        "gpu_name": "Tesla V100",
        "num_gpus": 1,
        "gpu_ram": 32,
        "cpu_ram": 32,
        "cuda_vers": 12.2,
        "disk_bw": 1000,
        "inet_down": 1000,
        "inet_up": 1000,
        "inet_down_cost": 0.0,
        "inet_up_cost": 0.0,
        "reliability": 0.99,
        "reliability2": 0.99,
        "dph_total": 0.0,
        "disk_space": 64,
        "ctx_size": 131072,
        "geolocation": "local",
        "rented": False,
        "rentable": True,
        "direct_port_count": 1,
    },
    {
        "id": 100002,
        "ask_contract_id": 100002,
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "gpu_ram": 24,
        "cpu_ram": 32,
        "cuda_vers": 12.8,
        "disk_bw": 1000,
        "inet_down": 1000,
        "inet_up": 1000,
        "inet_down_cost": 0.0,
        "inet_up_cost": 0.0,
        "reliability": 0.99,
        "reliability2": 0.99,
        "dph_total": 0.0,
        "disk_space": 64,
        "ctx_size": 32768,
        "geolocation": "local",
        "rented": False,
        "rentable": True,
        "direct_port_count": 1,
    },
    {
        "id": 100003,
        "ask_contract_id": 100003,
        "gpu_name": "RTX 5090",
        "num_gpus": 1,
        "gpu_ram": 32,
        "cpu_ram": 64,
        "cuda_vers": 12.8,
        "disk_bw": 1000,
        "inet_down": 1000,
        "inet_up": 1000,
        "inet_down_cost": 0.0,
        "inet_up_cost": 0.0,
        "reliability": 0.99,
        "reliability2": 0.99,
        "dph_total": 0.0,
        "disk_space": 64,
        "ctx_size": 131072,
        "geolocation": "local",
        "rented": False,
        "rentable": True,
        "direct_port_count": 1,
    },
]

_DOCKER_LABEL = "hostai.provider=local"
_DOCKER_TEST_LABEL = "hostai.test"


class LocalProviderError(ProviderError):
    pass


class LocalProvider(Provider):
    """Docker-based local provider."""

    name = "local"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if not shutil.which("docker"):
            raise LocalProviderError("docker command not found; is Docker installed?")
        self._docker = "docker"
        self._ssh_private_key, self._ssh_public_key = self._ensure_ssh_key()
        self.config.secrets["SSH_PUBLIC_KEY"] = self._ssh_public_key
        self.config.secrets["SSH_PRIVATE_KEY"] = str(self._ssh_private_key)
        self._state_file = state_dir(config.root_dir) / "local-provider.json"
        self._state = self._load_state()

    @property
    def ssh_private_key(self) -> Path:
        return self._ssh_private_key

    # ------------------------------------------------------------------
    # offers
    # ------------------------------------------------------------------
    def search_offers(
        self,
        query: str,
        *,
        limit: int = 25,
        order: Optional[str] = "dph_total",
        storage: float = 5.0,
        no_default: bool = False,
        offer_type: str = "on-demand",
    ) -> List[Dict[str, Any]]:
        _ = order, storage, no_default, offer_type
        normalized_query = re.sub(r"[^a-z0-9]", " ", query.lower())
        selected = []
        for offer in _LOCAL_OFFERS:
            if self._offer_matches_query(offer, normalized_query):
                selected.append(dict(offer))
        if not selected:
            selected = [dict(o) for o in _LOCAL_OFFERS]
        if limit:
            selected = selected[:limit]
        return selected

    @staticmethod
    def _offer_matches_query(offer: Dict[str, Any], query: str) -> bool:
        """Best-effort matching of an offer to a Vast query string."""
        gpu = re.sub(r"[^a-z0-9]", " ", str(offer.get("gpu_name", "")).lower())
        if gpu in query:
            return True
        # V100/Volta shorthand
        if "v100" in query or "tesla" in query or "volta" in query:
            return "tesla" in gpu or "v100" in gpu
        if "4090" in query:
            return "4090" in gpu
        if "5090" in query:
            return "5090" in gpu
        if "blackwell" in query:
            return "5090" in gpu or "pro" in gpu
        if "ada" in query:
            return "4090" in gpu or "6000" in gpu or "l40" in gpu
        return False

    # ------------------------------------------------------------------
    # instance lifecycle
    # ------------------------------------------------------------------
    def create_instance(
        self,
        offer_id: int,
        *,
        image: str,
        disk: float,
        env: Dict[str, str],
        price: Optional[float] = None,
        bid_price: Optional[float] = None,
        label: Optional[str] = None,
        extra: Optional[str] = None,
        runtype: Optional[str] = None,
        args: Optional[str] = None,
        force: bool = False,
        cancel_unavail: bool = False,
        template_hash: Optional[str] = None,
        volume_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        _ = price, bid_price, runtype, args, force, cancel_unavail, template_hash, volume_info
        offer = next((o for o in _LOCAL_OFFERS if o["id"] == offer_id or o["ask_contract_id"] == offer_id), None)
        if not offer:
            raise LocalProviderError(f"unknown local offer {offer_id}")

        instance_id = self._next_instance_id()
        test_label = self._test_label()

        run_image = self._resolve_image(image)
        run_extra = self._build_extra_args(extra, disk)
        run_env = {k: v for k, v in env.items() if self._is_valid_env_key(k)}
        run_env.setdefault("HOSTAI_PROVIDER", "local")
        run_env.setdefault("HOSTAI_LOCAL_INSTANCE", str(instance_id))
        run_env["HOSTAI_SSH_PUBLIC_KEY_B64"] = base64_if_needed(self._ssh_public_key)

        container_name = f"hostai-test-{instance_id}-{secrets.token_hex(4)}"

        cmd = [
            self._docker,
            "run",
            "-d",
            "--rm",
            "--name",
            container_name,
            "--label",
            _DOCKER_LABEL,
            "--label",
            f"{_DOCKER_TEST_LABEL}={test_label}",
            "--label",
            f"hostai.instance_id={instance_id}",
            "-p",
            "127.0.0.1::22/tcp",
            *run_extra,
            *(e for k, v in run_env.items() for e in ("-e", f"{k}={v}")),
            run_image,
        ]

        try:
            result = utils.run(cmd, capture=True, check=True, timeout=120)
        except Exception as exc:
            raise LocalProviderError(f"docker run failed: {exc}") from exc

        container_id = result.stdout.strip().splitlines()[-1].strip()
        if not container_id:
            raise LocalProviderError("docker run did not return a container id")

        ssh_port = self._container_ssh_port(container_id)
        if not ssh_port:
            raise LocalProviderError(f"could not find mapped SSH port for {container_id}")

        self._state["containers"][str(instance_id)] = {
            "container_id": container_id,
            "container_name": container_name,
            "created_at": time.time(),
            "offer_id": offer_id,
            "test_label": test_label,
        }
        self._save_state()

        return {
            "success": True,
            "new_contract": instance_id,
            "instance_id": instance_id,
            "id": instance_id,
            "actual_status": "loading",
            "status": "loading",
            "ssh_host": "127.0.0.1",
            "ssh_port": ssh_port,
            "ssh_user": "root",
            "public_ipaddr": "127.0.0.1",
            "ports": {"22/tcp": [{"HostPort": ssh_port}]},
            "gpu_name": offer["gpu_name"],
            "dph_total": 0.0,
            "disk_gb": int(disk),
            "image": run_image,
            "direct_port_count": 1,
        }

    def get_instance(self, instance_id: int) -> Optional[Dict[str, Any]]:
        info = self._state["containers"].get(str(instance_id))
        if not info:
            return None
        container_id = info["container_id"]
        return self._inspect_container(instance_id, container_id, info)

    def list_instances(self) -> List[Dict[str, Any]]:
        out = []
        for instance_id_str, info in self._state["containers"].items():
            inst = self._inspect_container(int(instance_id_str), info["container_id"], info)
            if inst:
                out.append(inst)
        return out

    def start_instance(self, instance_id: int) -> Dict[str, Any]:
        container_id = self._container_id(instance_id)
        self._docker_cmd(["start", container_id], timeout=60)
        return {"success": True, "instance_id": instance_id}

    def stop_instance(self, instance_id: int) -> Dict[str, Any]:
        container_id = self._container_id(instance_id)
        self._docker_cmd(["stop", "-t", "30", container_id], timeout=60)
        return {"success": True, "instance_id": instance_id}

    def destroy_instance(self, instance_id: int) -> Dict[str, Any]:
        container_id = self._container_id(instance_id)
        try:
            self._docker_cmd(["rm", "-f", "-v", container_id], timeout=60)
        except LocalProviderError:
            pass
        self._state["containers"].pop(str(instance_id), None)
        self._save_state()
        return {"success": True, "instance_id": instance_id, "destroy_outcome": "destroyed"}

    def get_logs(
        self,
        instance_id: int,
        *,
        tail: Optional[int] = 100,
        daemon_logs: bool = False,
        timeout: float = 30.0,
    ) -> Optional[str]:
        _ = daemon_logs
        container_id = self._container_id(instance_id)
        try:
            result = self._docker_cmd(["logs", f"--tail={tail}", container_id], timeout=timeout)
            return result.stdout
        except LocalProviderError:
            return None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _resolve_image(self, image: str) -> str:
        """Return the image to run locally, allowing provider overrides."""
        configured = self.config.provider.local_image or self.config.image.base
        if configured:
            configured = configured.rstrip("/")
        if configured and "OWNER/REPO" not in configured:
            # If the requested image tag is a suffix and the configured image
            # does not already have one, try to honour it.
            if ":" in configured:
                return configured
            if ":" in image:
                tag = image.rsplit(":", 1)[-1]
                return f"{configured}:{tag}"
            return configured
        if ":" in image and shutil.which("docker"):
            return image
        return os.environ.get("HOSTAI_LOCAL_IMAGE", "ghcr.io/smeinecke/qwen38-vast:v100")

    def _build_extra_args(self, extra: Optional[str], disk: float) -> List[str]:
        """Build docker run extra args (shm-size, labels, disk bound)."""
        parts: List[str] = []

        # Honour the configured / requested shm size.
        shm_gb = self.config.vast.shm_size_gb or self.config.provider.local_shm_size_gb
        if shm_gb is None and self.config.cache.enabled and self.config.cache.use_shm:
            shm_gb = self.config.cache.shm_min_gb or 1
        if shm_gb:
            parts.extend(["--shm-size", f"{shm_gb}g"])

        # docker run does not enforce a disk bound, but the test image can mount
        # a tmpfs /models of this size if requested.
        if extra:
            for token in extra.split():
                if token.startswith("--shm-size="):
                    # replace an explicit token with the resolved size above
                    continue
                parts.append(token)

        # Bind-mount a local project cache if an integration image wants it.
        if self.config.provider.local_volume:
            parts.extend(["-v", f"{self.config.provider.local_volume}:/data"])

        # Use the requested disk allocation as a generous tmpfs limit for /models.
        parts.extend(["--tmpfs", f"/models:size={int(disk)}g"])

        return parts

    def _container_id(self, instance_id: int) -> str:
        info = self._state["containers"].get(str(instance_id))
        if not info:
            raise LocalProviderError(f"instance {instance_id} not found")
        return info["container_id"]

    def _container_ssh_port(self, container_id: str) -> Optional[int]:
        try:
            result = utils.run(
                [self._docker, "port", container_id, "22/tcp"],
                capture=True,
                check=True,
                timeout=30,
            )
        except Exception:
            return None
        for line in result.stdout.strip().splitlines():
            m = re.search(r"(\d+)$", line)
            if m:
                return int(m.group(1))
        return None

    def _inspect_container(
        self,
        instance_id: int,
        container_id: str,
        info: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        try:
            result = utils.run(
                [self._docker, "inspect", "--format={{json .}}", container_id],
                capture=True,
                check=True,
                timeout=30,
            )
            data = json.loads(result.stdout)
        except Exception:
            # Container no longer exists.
            return {
                "id": instance_id,
                "actual_status": "exited",
                "status": "exited",
                "gpu_name": "Local Test GPU",
                "dph_total": 0.0,
                "container_id": container_id,
            }

        state = data.get("State", {}).get("Status", "unknown")
        status = self._map_status(state)
        config = data.get("Config", {})
        labels = config.get("Labels", {})
        offer_id = info.get("offer_id", 100001)
        offer = next((o for o in _LOCAL_OFFERS if o["id"] == offer_id), _LOCAL_OFFERS[0])

        ssh_port = self._container_ssh_port(container_id) or info.get("ssh_port")
        if ssh_port:
            info["ssh_port"] = ssh_port
            self._save_state()

        return {
            "id": instance_id,
            "actual_status": status,
            "status": status,
            "ssh_host": "127.0.0.1",
            "ssh_port": ssh_port,
            "ssh_user": "root",
            "public_ipaddr": "127.0.0.1",
            "ports": {"22/tcp": [{"HostPort": ssh_port}]},
            "gpu_name": offer["gpu_name"],
            "dph_total": 0.0,
            "image": config.get("Image", ""),
            "container_id": container_id,
            "direct_port_count": int(labels.get("hostai.direct_port_count", 1)),
        }

    @staticmethod
    def _map_status(docker_state: str) -> str:
        mapping = {
            "created": "loading",
            "restarting": "loading",
            "running": "running",
            "removing": "loading",
            "paused": "stopped",
            "exited": "exited",
            "dead": "offline",
        }
        return mapping.get(docker_state, "unknown")

    def _docker_cmd(self, subcmd: List[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess:
        result = utils.run([self._docker, *subcmd], capture=True, check=False, timeout=timeout)
        if result.returncode != 0:
            raise LocalProviderError(f"docker {' '.join(subcmd)} failed: {result.stderr or result.stdout}")
        return result

    def _next_instance_id(self) -> int:
        self._state["next_id"] = self._state.get("next_id", 10001) + 1
        return self._state["next_id"]

    def _load_state(self) -> Dict[str, Any]:
        if self._state_file.exists():
            try:
                return json.loads(self._state_file.read_text())
            except Exception:
                pass
        return {"containers": {}, "next_id": 10000}

    def _save_state(self) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(json.dumps(self._state, indent=2, default=str))

    def _ensure_ssh_key(self) -> tuple[Path, str]:
        key_dir = utils.mkdir_private(self.config.root_dir / ".hostai-vast" / "ssh")
        key_path = key_dir / "id_ed25519"
        pub_path = key_path.with_suffix(".pub")
        if not key_path.is_file() or not pub_path.is_file():
            if key_path.is_file() and not pub_path.is_file():
                pub_path.unlink(missing_ok=True)
            utils.run(
                [
                    "ssh-keygen",
                    "-q",
                    "-t",
                    "ed25519",
                    "-N",
                    "",
                    "-C",
                    "hostai-local-test",
                    "-f",
                    str(key_path),
                ],
                check=True,
                timeout=60,
            )
        key_path.chmod(0o600)
        if pub_path.exists():
            pub_path.chmod(0o644)
        return key_path, pub_path.read_text().strip()

    @staticmethod
    def _is_valid_env_key(key: str) -> bool:
        """Return True when *key* is a legal Docker environment variable name."""
        if not key:
            return False
        if key[0].isdigit():
            return False
        return all(c.isalnum() or c == "_" for c in key)

    def _test_label(self) -> str:
        label = self.config.provider.local_test_label or "hostai-test"
        return re.sub(r"[^a-z0-9_.-]", "-", label.lower()).strip("-.") or "hostai-test"


def base64_if_needed(value: str) -> str:
    """Return a base64-encoded string if the input is not already base64."""
    import base64

    try:
        decoded = base64.b64decode(value, validate=True)
        if base64.b64encode(decoded).decode() == value:
            return value
    except Exception:
        pass
    return base64.b64encode(value.encode()).decode()
