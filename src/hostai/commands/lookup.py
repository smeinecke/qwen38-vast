import csv
import io
import json
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.table import Table

from hostai import market
from hostai.config import Config
from hostai.profiles import Profiles
from hostai.vast import search_instance_offers


def _resolve_query(
    config: Config,
    profiles: Profiles,
    profile: Any,
    max_price: Optional[float],
    unverified: bool,
) -> tuple[str, float, int]:
    """Build the Vast search query, deriving disk_space from resolved disk_gb."""
    if max_price is not None and max_price < 0:
        raise click.ClickException("--max-price must be non-negative")

    query, max_dph = market.build_search_query(
        config,
        profiles,
        profile,
        max_price=max_price,
        unverified=unverified,
    )
    ctx_size = config.hostai.ctx_size_override or profile.ctx_size or 0
    return query, max_dph, ctx_size


def _filter_offers(
    offers: List[Dict[str, Any]], max_dph: float, require_free: bool, max_down: float, max_up: float
) -> List[Dict[str, Any]]:
    out = []
    for o in offers:
        o["_effective_dph"] = o.get("dph_total", 999999)
        if o["_effective_dph"] > max_dph:
            continue
        if require_free:
            down = o.get("inet_down_cost")
            up = o.get("inet_up_cost")
            if (down is None or down > max_down) or (up is None or up > max_up):
                continue
        out.append(o)
    return sorted(out, key=lambda x: x["_effective_dph"])


def _fmt_num(value, fmt=".4f") -> str:
    if isinstance(value, (int, float)):
        return f"{value:{fmt}}"
    return str(value) if value is not None else "?"


def _render_table(offers: List[Dict[str, Any]], max_results: int) -> None:
    console = Console()
    table = Table(
        title=f"Vast offers ({min(max_results, len(offers))} of {len(offers)})",
        show_header=True,
        header_style="bold",
        show_lines=True,
    )
    table.add_column("id", justify="right", no_wrap=True)
    table.add_column("gpu", overflow="fold", no_wrap=False)
    table.add_column("n", justify="right", no_wrap=True)
    table.add_column("dph", justify="right", no_wrap=True)
    table.add_column("disc", justify="right", no_wrap=True)
    table.add_column("rel", justify="right", no_wrap=True)
    table.add_column("loc", overflow="fold", no_wrap=False)
    table.add_column("down", justify="right", no_wrap=True)
    table.add_column("up", justify="right", no_wrap=True)
    for o in offers[:max_results]:
        table.add_row(
            str(o.get("id") or o.get("ask_contract_id") or "?"),
            str(o.get("gpu_name") or "?"),
            _fmt_num(o.get("num_gpus"), ".0f"),
            _fmt_num(o.get("dph_total")),
            _fmt_num(o.get("discounted_dph_total")),
            _fmt_num(o.get("reliability2") or o.get("reliability"), ".2f"),
            str(o.get("geolocation") or "?"),
            _fmt_num(o.get("inet_down_cost"), ".6f"),
            _fmt_num(o.get("inet_up_cost"), ".6f"),
        )
    console.print(table)


def _render_csv(offers: List[Dict[str, Any]]) -> str:
    if not offers:
        return ""
    keys = list(offers[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys)
    writer.writeheader()
    for o in offers:
        writer.writerow({k: (o.get(k) or "") for k in keys})
    return buf.getvalue()


def _resolve_profile(config: Config, profiles: Profiles, name: Optional[str]):
    target = name or config.hostai.default_profile
    return profiles.resolve_profile(target)


@click.command("lookup", help="Search Vast offers for a profile without renting.")
@click.argument("profile", required=False)
@click.option("-p", "--profile", "profile_opt", help="Profile to look up (alias for the positional argument).")
@click.option("--max-price", type=float, default=None, help="Maximum all-in $/h.")
@click.option("--unverified", is_flag=True, default=None, help="Also consider unverified/unknown hosts.")
@click.option("--max-results", type=int, default=10, show_default=True, help="Number of results to show.")
@click.option("--json", "output_format", flag_value="json", help="Output raw JSON array.")
@click.option("--csv", "output_format", flag_value="csv", help="Output CSV.")
@click.pass_obj
def cmd_lookup(config: Config, profile, profile_opt, max_price, unverified, max_results, output_format):
    if max_results <= 0:
        raise click.ClickException("--max-results must be a positive integer")

    profile_name = profile or profile_opt
    profiles = Profiles.from_file(config.root_dir / config.hostai.profiles_file)

    selected = _resolve_profile(config, profiles, profile_name)
    if not selected:
        raise click.ClickException(f"unknown profile '{profile_name or config.hostai.default_profile}'")

    image = profiles.image_by_name(selected.image)
    if not image:
        raise click.ClickException(f"profile '{selected.name}' references unknown image '{selected.image}'")

    unverified = unverified if unverified is not None else config.market.allow_unverified

    query, max_dph, ctx_size = _resolve_query(config, profiles, selected, max_price, unverified)

    click.echo(f"[profile] {selected.name} | sm_{image.cuda_arch} | ctx={ctx_size} | image={selected.image}")
    click.echo(f"[search]  {query}")

    disk_gb = market.resolved_disk_gb(selected, config)

    try:
        offers = search_instance_offers(
            config,
            query,
            limit=50,
            order="dph_total",
            storage=disk_gb,
        )
    except Exception as e:
        raise click.ClickException(f"search failed: {e}")

    matches = _filter_offers(
        offers,
        max_dph,
        profiles.market_policy.require_free_traffic,
        config.market.max_inet_down_cost,
        config.market.max_inet_up_cost,
    )

    if not matches:
        click.echo(f"No matching offers below ${max_dph:.2f}/h.")
        return

    if output_format == "json":
        click.echo(json.dumps(matches[:max_results], indent=2, default=str))
    elif output_format == "csv":
        click.echo(_render_csv(matches[:max_results]))
    else:
        _render_table(matches, max_results)
