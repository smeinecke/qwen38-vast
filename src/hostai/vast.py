"""Backward-compatible re-exports for the Vast.ai provider.

New code should use ``hostai.providers.get_provider`` and the ``Provider``
abstraction.  This module keeps the old function names working by delegating to
``VastProvider``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from vastai.api.client import VastClient
from vastai.api.query import offers_alias, offers_fields, offers_mult, parse_query

from hostai.config import Config

from .providers.base import ProviderError
from .providers.vast import VastProvider

VastError = ProviderError


def _provider(config: Config) -> VastProvider:
    return VastProvider(config)


def _api_key(config: Config) -> str:
    return _provider(config)._api_key_value()


def _client(api_key: str, timeout: float = 120.0) -> VastClient:
    return VastClient(api_key=api_key, timeout=int(timeout))


def _parse_query_string(query: str) -> Dict[str, Any]:
    return parse_query(query, {}, offers_fields, offers_alias, offers_mult)


def search_instance_offers(
    config: Config,
    query: str,
    *,
    limit: int = 25,
    order: Optional[str] = "dph_total",
    storage: float = 5.0,
    no_default: bool = False,
    offer_type: str = "on-demand",
) -> List[Dict[str, Any]]:
    return _provider(config).search_offers(
        query,
        limit=limit,
        order=order,
        storage=storage,
        no_default=no_default,
        offer_type=offer_type,
    )


def get_instances(config: Config) -> List[Dict[str, Any]]:
    return _provider(config).list_instances()


def get_instance(config: Config, instance_id: int) -> Optional[Dict[str, Any]]:
    return _provider(config).get_instance(instance_id)


def get_instance_logs(
    config: Config,
    instance_id: int,
    *,
    tail: Optional[int] = 100,
    daemon_logs: bool = False,
    timeout: float = 30.0,
) -> Optional[str]:
    return _provider(config).get_logs(
        instance_id,
        tail=tail,
        daemon_logs=daemon_logs,
        timeout=timeout,
    )


def create_instance_from_offer(
    config: Config,
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
    return _provider(config).create_instance(
        offer_id,
        image=image,
        disk=disk,
        env=env,
        price=price,
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


def destroy(config: Config, instance_id: int, timeout: float = 45.0) -> Dict[str, Any]:
    _ = timeout
    return _provider(config).destroy_instance(instance_id)


def pause(config: Config, instance_id: int, timeout: float = 60.0) -> Dict[str, Any]:
    _ = timeout
    return _provider(config).stop_instance(instance_id)


def start(config: Config, instance_id: int, timeout: float = 60.0) -> Dict[str, Any]:
    _ = timeout
    return _provider(config).start_instance(instance_id)
