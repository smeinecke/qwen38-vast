"""Vast.ai provider implementation.

This provider is a thin wrapper around the official ``vastai`` Python SDK.  The
old module-level functions in ``hostai.vast`` are preserved by re-exporting them
from this module for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from vastai.api.client import VastClient
from vastai.api.instances import (
    create_instance,
    destroy_instance,
    show_instance,
    show_instances,
    start_instance,
    stop_instance,
)
from vastai.api.instances import (
    logs as fetch_logs,
)
from vastai.api.offers import search_offers
from vastai.api.query import offers_alias, offers_fields, offers_mult, parse_order, parse_query

from hostai.config import Config

from .base import Provider, ProviderError


class VastProvider(Provider):
    """Production Vast.ai backend."""

    name = "vast"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if config.provider.backend != "vast":
            raise ProviderError(f"VastProvider construction blocked: provider.backend is '{config.provider.backend}'")
        self._vast_api_key = self._api_key_value()

    def _client(self, timeout: float = 120.0) -> VastClient:
        return VastClient(api_key=self._vast_api_key, timeout=int(timeout))

    def _api_key_value(self) -> str:
        key = self.config.secrets.get("VAST_API_KEY") or ""
        if not key:
            key = self.config.secrets.get("VASTAI_API_KEY") or ""
        if not key:
            raise ProviderError("VAST_API_KEY is not set in .env or environment")
        return key

    @staticmethod
    def _parse_query_string(query: str) -> Dict[str, Any]:
        return parse_query(query, {}, offers_fields, offers_alias, offers_mult)

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
        parsed_query = self._parse_query_string(query) if not no_default else {}
        if not no_default:
            parsed_query = parse_query(query, parsed_query, offers_fields, offers_alias, offers_mult)
        parsed_order: Optional[List[List[str]]] = parse_order(order) if order else None
        client = self._client(timeout=120.0)
        return search_offers(
            client,
            query=parsed_query,
            order=parsed_order or [],
            limit=limit,
            storage=storage,
            no_default=no_default,
            offer_type=offer_type,
        )

    def list_instances(self) -> List[Dict[str, Any]]:
        client = self._client(timeout=120.0)
        return show_instances(client)

    def get_instance(self, instance_id: int) -> Optional[Dict[str, Any]]:
        client = self._client(timeout=120.0)
        return show_instance(client, instance_id)

    def get_logs(
        self,
        instance_id: int,
        *,
        tail: Optional[int] = 100,
        daemon_logs: bool = False,
        timeout: float = 30.0,
    ) -> Optional[str]:
        client = self._client(timeout=timeout)
        try:
            result = fetch_logs(client, instance_id, tail=tail, daemon_logs=daemon_logs)
        except TimeoutError:
            return None
        if isinstance(result, str):
            return result
        return None

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
        if price is not None and bid_price is not None:
            raise ProviderError("pass either price or bid_price, not both")
        client = self._client(timeout=120.0)
        if bid_price is not None:
            return create_instance(
                client,
                offer_id,
                image=image,
                disk=int(disk),
                env=env,
                bid_price=bid_price,
                label=label,
                extra=extra,
                runtype=runtype,
                args=args,
                force=force,
                cancel_unavail=cancel_unavail,
                template_hash=template_hash,
                volume_info=volume_info,
            )
        return create_instance(
            client,
            offer_id,
            image=image,
            disk=int(disk),
            env=env,
            price=price,
            label=label,
            extra=extra,
            runtype=runtype,
            args=args,
            force=force,
            cancel_unavail=cancel_unavail,
            template_hash=template_hash,
            volume_info=volume_info,
        )

    def destroy_instance(self, instance_id: int) -> Dict[str, Any]:
        client = self._client(timeout=self.config.vast.destroy_timeout_seconds)
        return destroy_instance(client, instance_id)

    def stop_instance(self, instance_id: int) -> Dict[str, Any]:
        client = self._client(timeout=self.config.vast.pause_timeout_seconds)
        return stop_instance(client, instance_id)

    def start_instance(self, instance_id: int) -> Dict[str, Any]:
        client = self._client(timeout=self.config.vast.pause_timeout_seconds)
        return start_instance(client, instance_id)


# Backward-compatible module-level aliases used by older code/tests.
VastError = ProviderError


def _client(api_key: str, timeout: float = 120.0) -> VastClient:
    return VastClient(api_key=api_key, timeout=int(timeout))
