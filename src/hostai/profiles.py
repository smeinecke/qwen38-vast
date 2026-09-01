"""Runtime profile and image definitions from profiles.json."""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Type, TypeVar, cast

T = TypeVar("T")


def _from_dict(cls: Type[T], data: Any) -> T:
    if not dataclasses.is_dataclass(cls) or not isinstance(data, dict):
        return data  # type: ignore
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            value = data[f.name]
            kwargs[f.name] = _convert_value(value, f.type)
    return cls(**kwargs)


def _convert_value(value: Any, field_type: Any) -> Any:
    """Recursively convert a JSON value to the target dataclass type."""
    import typing

    origin = typing.get_origin(field_type)
    args = typing.get_args(field_type)

    if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
        return _from_dict(cast(Type[Any], field_type), value)

    if origin is list and isinstance(value, list):
        item_type = args[0] if args else Any
        return [_convert_value(v, item_type) for v in value]

    if origin is dict and isinstance(value, dict):
        _ = args[0] if args else Any
        val_type = args[1] if len(args) > 1 else Any
        return {k: _convert_value(v, val_type) for k, v in value.items()}

    if origin is type and getattr(field_type, "__name__", None) == "OptionalType":
        # Optional[T] is Union[T, None]
        pass

    return value


@dataclass
class Image:
    name: str
    cuda_arch: str
    image_tag: str
    description: str = ""
    builder_base: Optional[str] = None
    runtime_base: Optional[str] = None


@dataclass
class Profile:
    name: str
    image: str
    ctx_size: int
    gpu_query: str
    aliases: List[str] = dataclasses.field(default_factory=list)
    monitor_group: Optional[str] = None
    monitor_search: bool = True
    cache_ram: Optional[int] = None
    ctx_checkpoints: Optional[int] = None


@dataclass
class HardwareRank:
    gpu: str
    aliases: List[str] = dataclasses.field(default_factory=list)
    rank: int = 0


@dataclass
class MonitorHardware:
    policy: str = "same_or_better"
    gpu_ranks: List[HardwareRank] = dataclasses.field(default_factory=list)


@dataclass
class MarketPolicy:
    require_free_traffic: bool = True
    max_inet_down_cost: float = 0.0
    max_inet_up_cost: float = 0.0
    description: str = ""


@dataclass
class Profiles:
    schema_version: int = 0
    default_profile: str = "a6000"
    images: List[Image] = dataclasses.field(default_factory=list)
    profiles: List[Profile] = dataclasses.field(default_factory=list)
    monitor_hardware: MonitorHardware = dataclasses.field(default_factory=MonitorHardware)
    market_policy: MarketPolicy = dataclasses.field(default_factory=MarketPolicy)

    @classmethod
    def from_file(cls, path: Path) -> "Profiles":
        data = json.loads(path.read_text())
        if data.get("schema_version") != 1:
            raise ValueError(f"unsupported profiles.json schema_version: {data.get('schema_version')}")
        obj = _from_dict(cls, data)
        if not isinstance(obj, cls):
            raise ValueError("failed to parse profiles.json")
        return obj

    def image_by_name(self, name: str) -> Optional[Image]:
        for image in self.images:
            if image.name == name:
                return image
        return None

    def resolve_profile(self, name: str) -> Optional[Profile]:
        for profile in self.profiles:
            if profile.name == name or name in profile.aliases:
                return profile
        return None

    def all_monitor_profiles(self, ctx_size: int) -> List[Profile]:
        """Return profiles with the same context size that are not excluded from search."""
        return [p for p in self.profiles if p.ctx_size == ctx_size and p.monitor_search]

    def hardware_rank(self, gpu_name: str) -> Optional[int]:
        if not gpu_name:
            return None
        normalized = re_normalize_gpu(gpu_name)
        for rank in self.monitor_hardware.gpu_ranks:
            if normalized == re_normalize_gpu(rank.gpu) or any(normalized == re_normalize_gpu(a) for a in rank.aliases):
                return rank.rank
        return None

    def gpu_name_matches(self, gpu_name: str, candidate_name: str) -> bool:
        return re_normalize_gpu(gpu_name) == re_normalize_gpu(candidate_name)


def re_normalize_gpu(name: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]", "", name.lower())
