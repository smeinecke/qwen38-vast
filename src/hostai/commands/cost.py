"""Cost/break-even tooling for persistent model volumes and sessions."""

from __future__ import annotations

import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import click

from hostai import market
from hostai.config import Config
from hostai.state import State, runs_dir, state_dir


def _count_starts_per_month(config: Config, months: int = 1) -> Dict[str, int]:
    """Count run starts per month from ``.hostai-runs/*/metadata.json``."""
    runs = runs_dir(config.root_dir)
    if not runs.exists():
        return {}
    cutoff = time.time() - (months * 30 * 86400)
    counts: Counter = Counter()
    for run_dir in runs.iterdir():
        meta = run_dir / "metadata.json"
        if not meta.exists():
            continue
        try:
            data = market.json.loads(meta.read_text())
        except Exception:
            continue
        started = data.get("started_epoch") or data.get("run_started_epoch")
        if started and int(started) >= cutoff:
            month = datetime.fromtimestamp(int(started), tz=timezone.utc).strftime("%Y-%m")
            counts[month] += 1
    return dict(sorted(counts.items()))


def _get_current_offer_params(config: Config) -> tuple[float, float, float]:
    """Try to read dph, inet_down, and disk_bw from the current state."""
    state_file = state_dir(config.root_dir) / "state.json"
    if state_file.exists():
        state = State.load(state_file)
        if state.instance_id:
            dph = float(state.dph or config.market.max_dph)
            inet_down = float(state.inet_down or 500)
            disk_bw = float(state.disk_bw or 200)
            return dph, inet_down, disk_bw
    return config.market.max_dph, 500.0, 200.0


def _model_fetch_estimate(
    model_gb: float,
    inet_down: float,
    disk_bw: float,
) -> Tuple[float, float]:
    """Return (seconds, transfer_cost) for fetching *model_gb* over an offer.

    A persistent model volume avoids re-downloading the model weights and any
    small draft/MTP weights.  Docker image pulls are not avoided, so the image
    size is excluded from this estimate.

    The model download is bottlenecked by the slower of network transfer and
    local disk processing; missing bandwidth data falls back to the minimum
    startup floor.
    """
    if inet_down > 0:
        net_seconds = (model_gb * 8 * 1000) / inet_down
    else:
        net_seconds = float("inf")

    if disk_bw > 0:
        # Disk bandwidth is reported in MB/s; extraction is rarely sequential,
        # so use a conservative 2x multiplier.
        disk_seconds = (model_gb * 1024 / disk_bw) * 2
    else:
        disk_seconds = float("inf")

    finite = [s for s in (net_seconds, disk_seconds) if s != float("inf")]
    seconds = market.MIN_STARTUP_SECONDS if not finite else max(market.MIN_STARTUP_SECONDS, *finite)
    return seconds, model_gb


@click.group("cost", help="Cost and break-even calculations.")
@click.pass_obj
def cmd_cost(config: Config):
    pass


@cmd_cost.command("volume-break-even", help="Compute break-even starts for a persistent model volume.")
@click.option("--volume-gb", type=float, required=True, help="Persistent volume size in GiB.")
@click.option("--volume-cost-month", type=float, required=True, help="Volume cost in $/month.")
@click.option("--dph", type=float, default=None, help="GPU all-in $/h (default from state/market).")
@click.option("--inet-down", type=float, default=None, help="Host download bandwidth in Mbps (default 500).")
@click.option("--disk-bw", type=float, default=None, help="Host disk bandwidth in MB/s (default 200).")
@click.option("--starts-per-month", type=int, default=None, help="Override starts/month estimate.")
@click.pass_obj
def cmd_volume_break_even(
    config: Config,
    volume_gb: float,
    volume_cost_month: float,
    dph: Optional[float],
    inet_down: Optional[float],
    disk_bw: Optional[float],
    starts_per_month: Optional[int],
):
    """Estimate whether a persistent model volume is cheaper than re-downloading.

    A persistent ``/models`` volume only avoids re-downloading the model
    weights (main + any draft/MTP files) and the associated transfer cost.
    Docker image pulls, container startup, telemetry, and unrelated init are
    not avoided.

    This command uses the current instance's dph/bandwidth when available;
    otherwise it uses the configured market max_dph.
    """
    if volume_gb <= 0 or volume_cost_month < 0:
        raise click.ClickException("--volume-gb must be positive and --volume-cost-month non-negative")

    default_dph, default_inet, default_disk = _get_current_offer_params(config)
    dph = dph if dph is not None else default_dph
    inet_down = inet_down if inet_down is not None else default_inet
    disk_bw = disk_bw if disk_bw is not None else default_disk

    model_gb = config.market.model_download_gb
    # The volume can only save what it can hold.  If it is smaller than the
    # model payload, cap the avoided download and warn that it may not be a
    # useful persistent model volume.
    if volume_gb < model_gb:
        click.echo(
            f"[cost] WARNING: volume ({volume_gb:.1f} GB) is smaller than the model payload ({model_gb:.1f} GB); "
            f"capping avoided download at {volume_gb:.1f} GB",
            err=True,
        )
        effective_model_gb = volume_gb
    else:
        effective_model_gb = model_gb

    # Simulate a representative offer with the given or current bandwidth.
    offer: Dict[str, Any] = {
        "inet_down": inet_down,
        "disk_bw": disk_bw,
        "inet_down_cost": 0.0,
    }
    _, full_startup_seconds, _ = market.offer_download_estimate(offer, config)

    model_fetch_seconds, avoided_download_gb = _model_fetch_estimate(
        effective_model_gb, inet_down, disk_bw
    )
    model_fetch_hours = model_fetch_seconds / 3600.0

    # GPU rental cost during the model fetch: this is the time the GPU is
    # rented while the model is being downloaded/extracted.
    gpu_rental_savings = dph * model_fetch_hours
    # Transfer cost on the model bytes that are no longer downloaded.
    transfer_savings = avoided_download_gb * float(offer.get("inet_down_cost", 0) or 0)
    saving_per_start = gpu_rental_savings + transfer_savings

    if starts_per_month is None:
        counts = _count_starts_per_month(config, months=3)
        total = sum(counts.values())
        months = max(1, len(counts)) if counts else 1
        starts_per_month = total // months if total else 0

    if starts_per_month <= 0:
        starts_per_month = 10
        click.echo(f"[cost] no recent starts; using placeholder {starts_per_month}/month")

    monthly_savings = starts_per_month * saving_per_start
    break_even_starts = volume_cost_month / saving_per_start if saving_per_start > 0 else float("inf")
    break_even_months = break_even_starts / starts_per_month if starts_per_month > 0 else float("inf")

    per_gb_month = volume_cost_month / volume_gb
    per_day_cost = volume_cost_month / 30.44

    click.echo(f"[cost] dph=${dph:.4f}/h")
    click.echo(
        f"[cost] avoided model download: {avoided_download_gb:.2f} GB "
        f"(main + draft/MTP; image pull is not saved)"
    )
    click.echo(
        f"[cost] avoided model fetch: {model_fetch_seconds:.0f}s "
        f"({model_fetch_hours:.3f}h); full cold-start would be ~{full_startup_seconds:.0f}s"
    )
    click.echo(f"[cost] GPU rental savings per start: ${gpu_rental_savings:.4f}")
    if transfer_savings:
        click.echo(f"[cost] transfer savings per start: ${transfer_savings:.4f}")
    else:
        click.echo("[cost] transfer savings per start: $0.00 (free traffic)")
    click.echo(f"[cost] total saving per start: ${saving_per_start:.4f}")
    click.echo(f"[cost] starts/month: {starts_per_month} | monthly savings: ${monthly_savings:.2f}")
    click.echo(f"[cost] volume: {volume_gb:.1f} GiB @ ${volume_cost_month:.2f}/month")
    click.echo(f"[cost] per-GiB cost: ${per_gb_month:.4f}/month | per-day cost: ${per_day_cost:.4f}")
    click.echo(f"[cost] break-even: {break_even_starts:.1f} starts ({break_even_months:.1f} months)")

    if monthly_savings > volume_cost_month:
        net = monthly_savings - volume_cost_month
        click.echo(f"[cost] persistent volume is cheaper by ${net:.2f}/month")
    else:
        net = volume_cost_month - monthly_savings
        click.echo(f"[cost] persistent volume is more expensive by ${net:.2f}/month")
