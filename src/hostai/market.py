"""Central market logic: offer search, filtering, hardware ranking, and scoring.

This module is shared by ``hostai up`` and ``hostai monitor`` so both commands
use the same query construction, storage allocation, transfer-cost handling,
eligibility filtering, and optional cost-efficiency scoring.
"""

from __future__ import annotations

import json
import re
import statistics
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

from hostai.config import Config
from hostai.profiles import Profile, Profiles, re_normalize_gpu
from hostai.providers import get_provider

MAX_INET_DOWN_MBPS = 1.0
MAX_DISK_BW_MBPS = 1.0
MIN_STARTUP_SECONDS = 30.0

# Conservative TPS fallbacks for performance scoring when a GPU has no
# historical benchmark data.  They must be used consistently so the resulting
# score stays in cost-per-token units and unknown GPUs are not silently
# compared against measured GPUs using $/h.
PERF_UNKNOWN_PROMPT_TPS = 10.0
PERF_UNKNOWN_DECODE_TPS = 50.0

# Cache state for startup cost estimation.
CACHE_STATE_COLD = "cold"
CACHE_STATE_CACHED = "cached"
CACHE_STATE_UNKNOWN = "unknown"


@dataclass
class OfferScore:
    """Scoring result attached to an offer dict."""

    score: float
    reason: str
    prompt_tps: Optional[float] = None
    decode_tps: Optional[float] = None
    startup_seconds: Optional[float] = None
    session_seconds: Optional[float] = None
    transfer_cost: Optional[float] = None


def resolved_disk_gb(profile: Optional[Profile], config: Config) -> int:
    """Return the disk size for a profile, falling back to the global market default."""
    if profile is not None and isinstance(profile.disk_gb, (int, float)) and not isinstance(profile.disk_gb, bool):
        return int(profile.disk_gb)
    return int(config.market.disk_gb)


def _effective_dph(offer: Dict[str, Any]) -> float:
    for key in ("dph_total", "dph"):
        if key in offer and offer[key] is not None:
            return float(offer[key])
    return float("inf")


def _normalized_gpu_name(offer: Dict[str, Any]) -> str:
    return re_normalize_gpu(str(offer.get("gpu_name", "")))


def hardware_rank_for_offer(profiles: Profiles, offer: Dict[str, Any]) -> Optional[int]:
    """Return the hardware rank for an offer's GPU, or None if unknown."""
    return profiles.hardware_rank(str(offer.get("gpu_name", "")))


def is_same_or_better_gpu(
    profiles: Profiles,
    current_gpu: Optional[str],
    candidate_gpu: Optional[str],
    policy: str = "same_or_better",
) -> bool:
    """Return True when *candidate_gpu* is at least as good as *current_gpu*.

    Unknown GPUs are rejected unless the current GPU is also unknown.
    """
    if not current_gpu or not candidate_gpu:
        return False
    if policy not in ("same_or_better",):
        raise ValueError(f"unsupported monitor hardware policy: {policy}")
    current_rank = profiles.hardware_rank(current_gpu)
    candidate_rank = profiles.hardware_rank(candidate_gpu)
    if current_rank is None and candidate_rank is None:
        return re_normalize_gpu(current_gpu) == re_normalize_gpu(candidate_gpu)
    if current_rank is None or candidate_rank is None:
        return False
    return candidate_rank >= current_rank


def build_search_query(
    config: Config,
    profiles: Profiles,
    profile: Profile,
    *,
    max_price: Optional[float] = None,
    unverified: bool = False,
    offer: Optional[int] = None,
    bid_price: Optional[float] = None,
) -> Tuple[str, float]:
    """Build a Vast query string from profile and config.

    Traffic-cost constraints are appended explicitly with ``<=`` and use the
    configured values, including ``0``.  The resolved disk allocation is the
    single source of truth for the ``disk_space`` host-eligibility constraint.
    Any ``disk_space`` clause already present in the profile query is replaced.

    For interruptible/bid launches, the search ceiling is ``min(max_dph,
    bid_price)`` so the returned offers can actually be rented at the resolved
    bid.

    This keeps ``hostai up`` and the monitor on the same search semantics.
    """
    if max_price is not None and max_price < 0:
        raise click.ClickException("--max-price must be non-negative")

    max_dph = max_price if max_price is not None else config.market.max_dph
    if bid_price is not None:
        max_dph = min(max_dph, bid_price)
    query = config.hostai.gpu_query_override or profile.gpu_query
    disk_gb = resolved_disk_gb(profile, config)

    if unverified:
        query = re.sub(r"\s*reliability\s*(>=?|<=?|=)\s*[^\s]+", "", query)
        query = re.sub(r"\s+", " ", query).strip()
        query += ' verification in ["verified","unverified","deverified"]'

    if profiles.market_policy.require_free_traffic:
        query += f" inet_down_cost<={config.market.max_inet_down_cost}"
        query += f" inet_up_cost<={config.market.max_inet_up_cost}"

    if offer is None and "dph" not in query:
        query += f" dph_total <= {max_dph}"

    # Replace any explicit disk_space constraint with the resolved disk
    # allocation.  A value lower than the resolved allocation would be a
    # contradiction: we could ask a host with less free disk than we intend
    # to rent.  Warnings preserve traceability without failing a search.
    disk_space_pattern = re.compile(r"\s*disk_space\s*(>=?|<=?|=)\s*([^\s]+)")
    for m in disk_space_pattern.finditer(query):
        try:
            val = float(m.group(2))
        except ValueError:
            continue
        if val < disk_gb:
            warnings.warn(
                f"profile {profile.name}: gpu_query disk_space>={val} is lower "
                f"than resolved disk_gb={disk_gb}; using {disk_gb}",
                stacklevel=2,
            )

    query = disk_space_pattern.sub(" ", query)
    query = re.sub(r"\s+", " ", query).strip()
    query += f" disk_space>={disk_gb}"

    # Exclude the consistently slow/unstable China RTX 5090 host.
    query += " machine_id != 148003"

    return query, max_dph


def offer_download_estimate(
    offer: Dict[str, Any],
    config: Config,
    profile: Optional[Profile] = None,
    cache_state: str = CACHE_STATE_COLD,
) -> Tuple[float, float, float]:
    """Estimate download size (GB), time (s), and transfer cost ($).

    The size combines the image size (layers) and the model files.  Transfer
    cost is a rough ingress estimate based on the offer's ``inet_down_cost``.

    *cache_state* controls how much of the model payload is assumed to already
    be present:
      * ``cold``: full model + image download.
      * ``cached``: image only (model already on a persistent volume).
      * ``unknown``: image + half the model; a conservative middle estimate.
    """
    model_gb = float(config.market.model_download_gb)
    image_gb = float(config.market.image_size_gb)

    if cache_state == CACHE_STATE_CACHED:
        size_gb = image_gb
    elif cache_state == CACHE_STATE_UNKNOWN:
        size_gb = image_gb + (model_gb * 0.5)
    else:
        size_gb = image_gb + model_gb

    inet_down = float(offer.get("inet_down", 0) or 0)
    disk_bw = float(offer.get("disk_bw", 0) or 0)

    # Convert Vast's Mbps into seconds for the payload.
    if inet_down > 0:
        net_seconds = (size_gb * 8 * 1000) / inet_down
    else:
        net_seconds = float("inf")

    if disk_bw > 0:
        # Disk bandwidth is reported in MB/s. Image/model extraction is rarely
        # sequential; use a conservative 2x multiplier.
        disk_seconds = (size_gb * 1024 / disk_bw) * 2
    else:
        disk_seconds = float("inf")

    # Startup is a bottleneck model: the model cannot be served until both the
    # network transfer and the local disk processing have completed.  Use the
    # slower of the two while still respecting a fixed minimum startup floor.
    # If one of the two is unknown, the finite one drives the estimate; if both
    # are unknown, fall back to the minimum startup floor.
    finite_seconds = [s for s in (net_seconds, disk_seconds) if s != float("inf")]
    if finite_seconds:
        download_seconds = max(MIN_STARTUP_SECONDS, *finite_seconds)
    else:
        download_seconds = MIN_STARTUP_SECONDS

    inet_down_cost = float(offer.get("inet_down_cost", 0) or 0)
    transfer_cost = size_gb * inet_down_cost
    return size_gb, download_seconds, transfer_cost


def historical_per_gpu_stats(
    runs_dir: Path,
    min_samples: int = 3,
    max_age_days: int = 30,
) -> Dict[str, Dict[str, Any]]:
    """Aggregate prompt/decode/startup statistics per normalized GPU.

    Scans ``.hostai-runs/*/benchmarks/*/metrics.json`` and run metadata.  Only
    runs newer than *max_age_days* are considered.  Robust median aggregation
    prevents a single anomalous benchmark from dominating the score.
    """
    import time

    stats: Dict[str, Dict[str, List[float]]] = {}
    startup_stats: Dict[str, List[float]] = {}
    cutoff = time.time() - (max_age_days * 86400)

    if not runs_dir.exists():
        return {}

    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "metadata.json"
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                pass
        started = meta.get("started_epoch") or meta.get("run_started_epoch")
        if started and int(started) < cutoff:
            continue

        gpu = re_normalize_gpu(str(meta.get("gpu", "")))
        if not gpu:
            continue

        startup = meta.get("startup_seconds")
        if isinstance(startup, (int, float)) and startup > 0:
            startup_stats.setdefault(gpu, []).append(float(startup))

        bench_root = run_dir / "benchmarks"
        if not bench_root.exists():
            continue
        for bench_dir in bench_root.iterdir():
            metrics_path = bench_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            try:
                data = json.loads(metrics_path.read_text())
            except Exception:
                continue
            perf = data.get("performance") or {}
            session = data.get("session") or {}
            bench_gpu = re_normalize_gpu(str(session.get("gpu", meta.get("gpu", ""))))
            if not bench_gpu:
                continue
            entry = stats.setdefault(bench_gpu, {})
            for key in ("prompt_tps", "decode_tps"):
                value = perf.get(key)
                if isinstance(value, (int, float)) and value > 0:
                    entry.setdefault(key, []).append(float(value))

    out: Dict[str, Dict[str, Any]] = {}
    for gpu, values in stats.items():
        prompt = values.get("prompt_tps", [])
        decode = values.get("decode_tps", [])
        if len(prompt) + len(decode) == 0:
            continue
        out[gpu] = {
            "prompt_tps": statistics.median(prompt) if prompt else None,
            "decode_tps": statistics.median(decode) if decode else None,
            "prompt_samples": len(prompt),
            "decode_samples": len(decode),
        }
        if gpu in startup_stats:
            out[gpu]["startup_seconds"] = statistics.median(startup_stats[gpu])
            out[gpu]["startup_samples"] = len(startup_stats[gpu])
    return out


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def _robust_min(values: List[float]) -> Optional[float]:
    """Return the median of the lower half to avoid one fast outlier dominating."""
    if not values:
        return None
    sorted_values = sorted(values)
    n = len(sorted_values)
    lower = sorted_values[: (n // 2) + 1]
    return float(statistics.median(lower))


def estimate_startup_seconds(
    offer: Dict[str, Any],
    config: Config,
    stats: Optional[Dict[str, Dict[str, Any]]] = None,
    min_samples: int = 3,
    cache_state: str = CACHE_STATE_COLD,
) -> float:
    """Return an estimated startup time for an offer.

    Prefer measured historical startup data when enough samples exist.  Fall
    back to a bandwidth-based estimate derived from the model/image size and
    the offer's ``inet_down``/``disk_bw``.
    """
    stats = stats or {}
    gpu = _normalized_gpu_name(offer)
    gpu_stats = stats.get(gpu, {})
    startup_samples = gpu_stats.get("startup_samples", 0)
    startup_seconds = gpu_stats.get("startup_seconds")
    if startup_samples >= min_samples and isinstance(startup_seconds, (int, float)) and startup_seconds > 0:
        return float(startup_seconds)

    _, download_seconds, _ = offer_download_estimate(offer, config, cache_state=cache_state)
    return download_seconds


def _performance_for_offer(
    offer: Dict[str, Any],
    stats: Dict[str, Dict[str, Any]],
    config: Config,
) -> Tuple[Optional[float], Optional[float], str]:
    """Return (prompt_tps, decode_tps, reason) for an offer."""
    gpu = _normalized_gpu_name(offer)
    gpu_stats = stats.get(gpu)
    if gpu_stats is None:
        return None, None, "no historical data"

    prompt_samples = gpu_stats.get("prompt_samples", 0)
    decode_samples = gpu_stats.get("decode_samples", 0)
    total_samples = prompt_samples + decode_samples
    if total_samples < config.market.min_historical_samples:
        return None, None, f"insufficient samples ({total_samples})"

    prompt_tps = gpu_stats.get("prompt_tps")
    decode_tps = gpu_stats.get("decode_tps")
    if not prompt_tps and not decode_tps:
        return None, None, "no positive tps values"
    return prompt_tps, decode_tps, f"history n={total_samples}"


def _seconds_per_token(
    prompt_tps: Optional[float],
    decode_tps: Optional[float],
    config: Config,
) -> Optional[float]:
    """Return the weighted time to process one token.

    Uses the configured prompt/decode weights as fractions of a mixed
    prompt/decode workload.  Missing or non-positive measurements fall back to
    conservative constants so the result is always in comparable units.
    """
    wp = float(config.market.scoring_prompt_weight)
    wd = float(config.market.scoring_decode_weight)
    total_weight = wp + wd
    if total_weight == 0:
        return None
    wp /= total_weight
    wd /= total_weight

    prompt = float(prompt_tps or 0)
    decode = float(decode_tps or 0)
    if prompt <= 0:
        prompt = PERF_UNKNOWN_PROMPT_TPS
    if decode <= 0:
        decode = PERF_UNKNOWN_DECODE_TPS

    # Time for one weighted token = fraction of prompt time + fraction of
    # decode time.  Both denominators are positive by this point.
    return (wp / prompt) + (wd / decode)


def _resolve_cache_state(config: Config) -> str:
    """Return the startup cache state assumed for a fresh search.

    A persistent model volume is assumed to keep the model weights; otherwise
    we default to a full cold start.  ``unknown`` is used conservatively only
    when the caller explicitly chooses it; normal launches do not benefit from
    an optimistic partially-cached estimate.
    """
    if config.vast.volume_id and config.vast.volume_mount_path:
        return CACHE_STATE_CACHED
    return CACHE_STATE_COLD


def score_offer(
    offer: Dict[str, Any],
    config: Config,
    stats: Dict[str, Dict[str, Any]],
    session_seconds: Optional[int] = None,
    cache_state: str = CACHE_STATE_COLD,
) -> OfferScore:
    """Score a single offer.

    * ``dph`` mode simply uses the hourly price (units: $/h).
    * ``perf`` mode uses a cost-per-token score (units: $/token) that never
      falls back to $/h.
    * ``session`` mode returns the estimated total session cost (units: $).

    The returned score is a cost figure: lower is better.
    """
    dph = _effective_dph(offer)
    mode = config.market.scoring_mode

    if mode == "dph":
        return OfferScore(score=dph, reason="dph")

    prompt_tps, decode_tps, reason = _performance_for_offer(offer, stats, config)
    seconds_per_token = _seconds_per_token(prompt_tps, decode_tps, config)

    if mode == "perf":
        if seconds_per_token is None:
            return OfferScore(
                score=dph,
                reason="perf fallback: scoring weights are zero",
                prompt_tps=prompt_tps,
                decode_tps=decode_tps,
            )
        # cost/token = hourly cost per second * seconds per token
        score = (dph / 3600.0) * seconds_per_token
        return OfferScore(
            score=score,
            reason=f"perf: {reason}",
            prompt_tps=prompt_tps,
            decode_tps=decode_tps,
        )

    if mode == "session":
        startup = estimate_startup_seconds(
            offer, config, stats, config.market.min_historical_samples, cache_state=cache_state
        )
        active = session_seconds if session_seconds else 0
        total_seconds = startup + active
        total_cost = dph * total_seconds / 3600.0
        _, _, transfer_cost = offer_download_estimate(offer, config, cache_state=cache_state)
        total_cost += transfer_cost

        return OfferScore(
            score=total_cost,
            reason=f"session: {reason}, startup={startup:.0f}s",
            prompt_tps=prompt_tps,
            decode_tps=decode_tps,
            startup_seconds=startup,
            session_seconds=active,
            transfer_cost=transfer_cost,
        )

    return OfferScore(score=dph, reason=f"unknown scoring mode {mode}")


def score_offers(
    offers: List[Dict[str, Any]],
    config: Config,
    stats: Optional[Dict[str, Dict[str, Any]]] = None,
    session_seconds: Optional[int] = None,
    cache_state: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Attach a ``_hostai_score`` to each offer and sort by score (lower=better)."""
    stats = stats or {}
    if cache_state is None:
        cache_state = _resolve_cache_state(config)
    for offer in offers:
        scoring = score_offer(
            offer, config, stats, session_seconds=session_seconds, cache_state=cache_state
        )
        offer["_hostai_score"] = scoring
    offers.sort(key=lambda o: o["_hostai_score"].score)
    return offers


def filter_eligible_offers(
    offers: List[Dict[str, Any]],
    *,
    max_dph: float,
    offer: Optional[int] = None,
    current_gpu: Optional[str] = None,
    profiles: Optional[Profiles] = None,
    ctx_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Filter search results by price, specific id, hardware rank, and context."""
    matches: List[Dict[str, Any]] = []
    for o in offers:
        if offer is not None:
            if str(o.get("id")) != str(offer) and str(o.get("ask_contract_id")) != str(offer):
                continue
        else:
            if _effective_dph(o) > max_dph:
                continue

        if current_gpu is not None and profiles is not None:
            if not is_same_or_better_gpu(profiles, current_gpu, str(o.get("gpu_name", ""))):
                continue

        if ctx_size is not None and int(o.get("ctx_size", 0)) != 0 and int(o.get("ctx_size", 0)) != ctx_size:
            continue

        matches.append(o)
    return matches


def search_offers(
    config: Config,
    query: str,
    *,
    storage: float,
    max_dph: float,
    offer: Optional[int] = None,
    offer_type: str = "on-demand",
    unverified: bool = False,
    order: str = "dph_total",
    limit: int = 25,
) -> List[Dict[str, Any]]:
    """Search Vast and return raw offers.

    Uses a higher limit when an explicit *offer* id is requested so the
    requested contract is likely to appear in the result set.
    """
    search_limit = 100 if offer is not None else limit
    try:
        provider = get_provider(config)
        return provider.search_offers(
            query,
            limit=search_limit,
            order=order,
            storage=storage,
            offer_type=offer_type,
        )
    except Exception as exc:
        raise click.ClickException(f"search failed: {exc}")


def select_offer(
    config: Config,
    profiles: Profiles,
    query: str,
    *,
    profile: Optional[Profile] = None,
    max_dph: float,
    unverified: bool,
    offer: Optional[int],
    storage: float,
    offer_type: str = "on-demand",
    current_gpu: Optional[str] = None,
    ctx_size: Optional[int] = None,
    session_seconds: Optional[int] = None,
    cache_state: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Search, filter, and select the best offer for ``up`` or ``monitor``.

    Uses the configured scoring mode.  ``_hostai_score`` is attached to the
    returned offer for display.  When no scoring data is available, the legacy
    ``dph_total`` ordering is preserved.
    """
    offers = search_offers(
        config,
        query,
        storage=storage,
        max_dph=max_dph,
        offer=offer,
        offer_type=offer_type,
        unverified=unverified,
    )

    stats: Dict[str, Dict[str, Any]] = {}
    if config.market.scoring_mode in ("perf", "session"):
        from hostai.state import runs_dir

        stats = historical_per_gpu_stats(
            runs_dir(config.root_dir),
            min_samples=config.market.min_historical_samples,
            max_age_days=config.market.max_history_age_days,
        )

    matches = filter_eligible_offers(
        offers,
        max_dph=max_dph,
        offer=offer,
        current_gpu=current_gpu,
        profiles=profiles,
        ctx_size=ctx_size,
    )

    if not matches:
        if offer is not None:
            raise click.ClickException(f"no matching offer for id {offer}")
        raise click.ClickException(f"no matching offer at or below ${max_dph:.2f}/h")

    if cache_state is None:
        cache_state = _resolve_cache_state(config)

    if config.market.scoring_mode in ("perf", "session"):
        matches = score_offers(
            matches, config, stats, session_seconds=session_seconds, cache_state=cache_state
        )
    else:
        matches.sort(key=_effective_dph)

    best = matches[0]
    scoring = best.get("_hostai_score")
    if verbose and scoring:
        click.echo(
            f"[market] selected {best.get('gpu_name')} with score {scoring.score:.6f} "
            f"({scoring.reason})"
        )

    return best


def _format_transfer_cost(cost: float) -> str:
    if cost == 0:
        return "free"
    return f"${cost:.4f}/GB"


def offer_summary(offer: Dict[str, Any]) -> str:
    """Human-readable one-line summary of an offer."""
    dph = _effective_dph(offer)
    gpu = offer.get("gpu_name", "unknown")
    down = offer.get("inet_down", "?")
    up = offer.get("inet_up", "?")
    down_cost = offer.get("inet_down_cost", 0)
    up_cost = offer.get("inet_up_cost", 0)
    offer_id = offer.get("id") or offer.get("ask_contract_id")
    scoring = offer.get("_hostai_score")
    extras = []
    if scoring:
        extras.append(f"score={scoring.score:.6f}")
    if offer.get("is_bid") or offer.get("type") == "bid":
        extras.append("bid")
    summary = (
        f"{gpu} | ${dph:.4f}/h | "
        f"down={down} ({_format_transfer_cost(down_cost)}) | "
        f"up={up} ({_format_transfer_cost(up_cost)}) | offer={offer_id}"
    )
    if extras:
        summary += f" | {' '.join(extras)}"
    return summary
