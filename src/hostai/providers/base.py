"""Abstract provider interface for hostai backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from hostai.config import Config


class ProviderError(Exception):
    pass


class Provider(ABC):
    """A backend that can offer, create, and manage hostai compute instances.

    The lifecycle code in ``hostai.commands.up``/``down``/``status`` should not
    need to know whether the active backend is Vast, a local Docker container,
    or an in-memory fake: all provider-specific control-plane logic lives here.
    """

    name: str = ""

    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
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
        """Search for offers matching a query string."""

    @abstractmethod
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
        """Create an instance from an offer ID."""

    @abstractmethod
    def get_instance(self, instance_id: int) -> Optional[Dict[str, Any]]:
        """Return a single instance by ID."""

    @abstractmethod
    def list_instances(self) -> List[Dict[str, Any]]:
        """Return all instances known to this provider."""

    @abstractmethod
    def start_instance(self, instance_id: int) -> Dict[str, Any]:
        """Start/resume a stopped instance."""

    @abstractmethod
    def stop_instance(self, instance_id: int) -> Dict[str, Any]:
        """Stop/pause a running instance."""

    @abstractmethod
    def destroy_instance(self, instance_id: int) -> Dict[str, Any]:
        """Destroy an instance permanently."""

    def get_logs(
        self,
        instance_id: int,
        *,
        tail: Optional[int] = 100,
        daemon_logs: bool = False,
        timeout: float = 30.0,
    ) -> Optional[str]:
        """Fetch container/daemon logs for an instance."""
        return None

    def instance_to_endpoint(self, instance: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert an instance dict to an SSH endpoint dict.

        The default implementation understands the same fields as
        ``hostai.ssh.resolve_ssh_endpoint`` so callers can discover SSH the same
        way for every backend.
        """
        from hostai.ssh import resolve_ssh_endpoint

        return resolve_ssh_endpoint(instance)
