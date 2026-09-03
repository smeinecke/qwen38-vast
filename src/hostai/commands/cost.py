"""Cost/break-even tooling for persistent model volumes and sessions."""

from __future__ import annotations

import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Optional

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

    A persistent volume avoids re-downloading the model (and pulling the image)
    on every start.  The savings per start is the GPU cost during the download
    and extraction time.  This command uses the current instance's dph/bandwidth
    when available; otherwise it uses the configured market max_dph.
    """
    if volume_gb <= 0 or volume_cost_month < 0:
        raise click.ClickException("--volume-gb must be positive and --volume-cost-month non-negative")

    default_dph, default_inet, default_disk = _get_current_offer_params(config)
    dph = dph if dph is not None else default_dph
    inet_down = inet_down if inet_down is not None else default_inet
    disk_bw = disk_bw if disk_bw is not None else default_disk

    # Simulate a representative offer with the given or current bandwidth.
    offer: Dict[str, Any] = {
        "inet_down": inet_down,
        "disk_bw": disk_bw,
        "inet_down_cost": 0.0,
    }
    download_gb, startup_seconds, transfer_cost = market.offer_download_estimate(offer, config)
    startup_hours = startup_seconds / 3600.0
    saving_per_start = dph * startup_hours

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

    click.echo(f"[cost] dph=${dph:.4f} startup={startup_seconds:.0f}s ({startup_hours:.3f}h) download={download_gb:.1f}GB")
    click.echo(f"[cost] saving per start: ${saving_per_start:.4f}")
    click.echo(f"[cost] starts/month: {starts_per_month} | monthly savings: ${monthly_savings:.2f}")
    click.echo(f"[cost] volume cost: ${volume_cost_month:.2f}/month")
    click.echo(f"[cost] break-even starts/month: {break_even_starts:.1f} ({break_even_months:.1f} months)")

    if monthly_savings > volume_cost_month:
        net = monthly_savings - volume_cost_month
        click.echo(f"[cost] persistent volume is cheaper by ${net:.2f}/month")
    else:
        net = volume_cost_month - monthly_savings
        click.echo(f"[cost] persistent volume is more expensive by ${net:.2f}/month")
