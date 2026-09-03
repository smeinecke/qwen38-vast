"""Provider abstraction and registry for hostai backends.

Supported backends:
  * ``vast``  - real Vast.ai SDK (production/default)
  * ``local`` - local Docker containers acting as synthetic Vast hosts
  * ``fake``  - deterministic in-memory provider for fast unit tests
"""

from __future__ import annotations

from typing import Dict, Type

from hostai.config import Config

from .base import Provider, ProviderError
from .fake import FakeProvider
from .local import LocalProvider
from .vast import VastProvider


class UnknownBackendError(ProviderError):
    pass


_REGISTRY: Dict[str, Type[Provider]] = {
    "vast": VastProvider,
    "local": LocalProvider,
    "fake": FakeProvider,
}


def get_provider(config: Config) -> Provider:
    """Return the configured provider instance."""
    backend = (config.provider.backend or "vast").lower().strip()
    if backend not in _REGISTRY:
        raise UnknownBackendError(f"unknown provider backend: {backend}")
    return _REGISTRY[backend](config)


def available_backends() -> list[str]:
    return list(_REGISTRY.keys())


__all__ = [
    "Provider",
    "ProviderError",
    "UnknownBackendError",
    "VastProvider",
    "LocalProvider",
    "FakeProvider",
    "get_provider",
    "available_backends",
]
