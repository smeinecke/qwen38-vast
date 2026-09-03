"""Deterministic in-memory provider for fast unit tests.

FakeProvider never makes network calls.  It is useful for exercising market
logic, the up/down state machine, and watchdog behavior without Vast credentials.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from hostai.config import Config

from .base import Provider, ProviderError

_LOCAL_OFFERS = [
    {
        "id": 90001,
        "ask_contract_id": 90001,
        "gpu_name": "Tesla V100",
        "gpu_ram": 32,
        "num_gpus": 1,
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
        "id": 90002,
        "ask_contract_id": 90002,
        "gpu_name": "RTX 4090",
        "gpu_ram": 24,
        "num_gpus": 1,
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
        "id": 90003,
        "ask_contract_id": 90003,
        "gpu_name": "RTX 5090",
        "gpu_ram": 32,
        "num_gpus": 1,
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


class FakeProvider(Provider):
    """In-memory provider that simulates offers and instances."""

    name = "fake"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._offers = {o["id"]: o for o in _LOCAL_OFFERS}
        self._instances: Dict[int, Dict[str, Any]] = {}
        self._next_id = 80001

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
        _ = query, order, storage, no_default, offer_type
        offers = list(self._offers.values())
        if limit:
            offers = offers[:limit]
        return offers

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
        _ = price, bid_price, label, extra, runtype, args, force, cancel_unavail, template_hash, volume_info
        offer = self._offers.get(offer_id)
        if not offer:
            raise ProviderError(f"unknown offer {offer_id}")

        instance_id = self._next_id
        self._next_id += 1

        instance = {
            "id": instance_id,
            "actual_status": "loading",
            "status": "loading",
            "ssh_host": "127.0.0.1",
            "ssh_port": 2222,
            "ssh_user": "root",
            "public_ipaddr": "127.0.0.1",
            "ports": {"22/tcp": [{"HostPort": 2222}]},
            "gpu_name": offer["gpu_name"],
            "dph_total": 0.0,
            "disk_gb": int(disk),
            "image": image,
            "env": dict(env),
            "direct_port_count": 1,
        }
        self._instances[instance_id] = instance
        return {
            "success": True,
            "new_contract": instance_id,
            "instance_id": instance_id,
            "id": instance_id,
            "image": image,
            "disk": disk,
        }

    def get_instance(self, instance_id: int) -> Optional[Dict[str, Any]]:
        return self._instances.get(instance_id)

    def list_instances(self) -> List[Dict[str, Any]]:
        return list(self._instances.values())

    def start_instance(self, instance_id: int) -> Dict[str, Any]:
        inst = self._get_or_raise(instance_id)
        inst["actual_status"] = "running"
        inst["status"] = "running"
        return {"success": True, "instance_id": instance_id}

    def stop_instance(self, instance_id: int) -> Dict[str, Any]:
        inst = self._get_or_raise(instance_id)
        inst["actual_status"] = "stopped"
        inst["status"] = "stopped"
        return {"success": True, "instance_id": instance_id}

    def destroy_instance(self, instance_id: int) -> Dict[str, Any]:
        inst = self._instances.pop(instance_id, None)
        if inst is None:
            raise ProviderError(f"instance {instance_id} not found")
        inst["actual_status"] = "offline"
        inst["status"] = "offline"
        return {"success": True, "instance_id": instance_id}

    def get_logs(
        self,
        instance_id: int,
        *,
        tail: Optional[int] = 100,
        daemon_logs: bool = False,
        timeout: float = 30.0,
    ) -> Optional[str]:
        _ = tail, daemon_logs, timeout
        inst = self._instances.get(instance_id)
        if not inst:
            return None
        return f"[fake] instance {instance_id} logs at {time.time()}"

    def _get_or_raise(self, instance_id: int) -> Dict[str, Any]:
        inst = self._instances.get(instance_id)
        if not inst:
            raise ProviderError(f"instance {instance_id} not found")
        return inst
