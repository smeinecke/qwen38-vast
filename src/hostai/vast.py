"""Low-level Vast.ai SDK wrapper.

This module uses the official `vastai` Python package.  If the SDK cannot
reproduce the old CLI behaviour for an operation, it raises a clear
`NotImplementedError` rather than silently falling back to the `vastai` CLI.
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


class VastError(Exception):
    pass


def _client(api_key: str, timeout: float = 120.0) -> VastClient:
    return VastClient(api_key=api_key, timeout=int(timeout))


def _api_key(config: Config) -> str:
    key = config.secrets.get("VAST_API_KEY") or ""
    if not key:
        key = config.secrets.get("VASTAI_API_KEY") or ""
    if not key:
        raise VastError("VAST_API_KEY is not set in .env or environment")
    return key


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
    """Search for offers matching a Vast query string."""
    parsed_query = _parse_query_string(query) if not no_default else {}
    if not no_default:
        parsed_query = parse_query(query, parsed_query, offers_fields, offers_alias, offers_mult)
    parsed_order: Optional[List[List[str]]] = parse_order(order) if order else None
    client = _client(_api_key(config), timeout=120.0)
    return search_offers(
        client,
        query=parsed_query,
        order=parsed_order or [],
        limit=limit,
        storage=storage,
        no_default=no_default,
        offer_type=offer_type,
    )


def get_instances(config: Config) -> List[Dict[str, Any]]:
    """Return all instances owned by the user."""
    client = _client(_api_key(config), timeout=120.0)
    return show_instances(client)


def get_instance(config: Config, instance_id: int) -> Optional[Dict[str, Any]]:
    """Return a single instance by ID."""
    client = _client(_api_key(config), timeout=120.0)
    return show_instance(client, instance_id)


def get_instance_logs(
    config: Config,
    instance_id: int,
    *,
    tail: Optional[int] = 100,
    daemon_logs: bool = False,
    timeout: float = 30.0,
) -> Optional[str]:
    """Fetch container or daemon logs for an instance.

    Returns the log text, or None if the logs are not yet available."""
    client = _client(_api_key(config), timeout=timeout)
    try:
        result = fetch_logs(client, instance_id, tail=tail, daemon_logs=daemon_logs)
    except TimeoutError:
        return None
    if isinstance(result, str):
        return result
    return None


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
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Create an instance from an offer ID.

    Pass either ``price`` for on-demand instances or ``bid_price`` for
    interruptible/bid instances, never both.
    """
    if price is not None and bid_price is not None:
        raise VastError("pass either price or bid_price, not both")
    client = _client(_api_key(config), timeout=timeout)
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


def destroy(config: Config, instance_id: int, timeout: float = 45.0) -> Dict[str, Any]:
    client = _client(_api_key(config), timeout=timeout)
    return destroy_instance(client, instance_id)


def pause(config: Config, instance_id: int, timeout: float = 60.0) -> Dict[str, Any]:
    client = _client(_api_key(config), timeout=timeout)
    return stop_instance(client, instance_id)


def start(config: Config, instance_id: int, timeout: float = 60.0) -> Dict[str, Any]:
    client = _client(_api_key(config), timeout=timeout)
    return start_instance(client, instance_id)
