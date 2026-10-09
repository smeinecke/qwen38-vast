import csv
import fnmatch
import io
import json
import re
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.table import Table

from hostai import market
from hostai.config import Config
from hostai.profiles import Profiles
from hostai.providers import get_provider


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
    offers: List[Dict[str, Any]],
    max_dph: float,
    require_free: bool,
    max_down: float,
    max_up: float,
    exclusions: Optional[market.OfferExclusions] = None,
) -> List[Dict[str, Any]]:
    out = []
    for o in offers:
        if exclusions and exclusions.is_excluded(o):
            continue
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


def _render_table(offers: List[Dict[str, Any]], max_results: int, show_profiles: bool = False) -> None:
    console = Console()
    table = Table(
        title=f"Vast offers ({min(max_results, len(offers))} of {len(offers)})",
        show_header=True,
        header_style="bold",
        show_lines=True,
    )
    table.add_column("id", justify="right", no_wrap=True)
    table.add_column("machine", justify="right", no_wrap=True)
    table.add_column("gpu", overflow="fold", no_wrap=False)
    table.add_column("n", justify="right", no_wrap=True)
    table.add_column("dph", justify="right", no_wrap=True)
    table.add_column("disc", justify="right", no_wrap=True)
    table.add_column("rel", justify="right", no_wrap=True)
    table.add_column("loc", overflow="fold", no_wrap=False)
    table.add_column("down", justify="right", no_wrap=True)
    table.add_column("up", justify="right", no_wrap=True)
    if show_profiles:
        table.add_column("profiles", overflow="fold", no_wrap=False)
    for o in offers[:max_results]:
        row = [
            str(o.get("id") or o.get("ask_contract_id") or "?"),
            _fmt_num(o.get("machine_id"), ".0f"),
            str(o.get("gpu_name") or "?"),
            _fmt_num(o.get("num_gpus"), ".0f"),
            _fmt_num(o.get("dph_total")),
            _fmt_num(o.get("discounted_dph_total")),
            _fmt_num(o.get("reliability2") or o.get("reliability"), ".2f"),
            market._format_country(o.get("geolocation")),
            _fmt_num(o.get("inet_down_cost"), ".6f"),
            _fmt_num(o.get("inet_up_cost"), ".6f"),
        ]
        if show_profiles:
            row.append(",".join(o.get("_profiles") or []))
        table.add_row(*row)
    console.print(table)


def _render_csv(offers: List[Dict[str, Any]]) -> str:
    if not offers:
        return ""
    keys = list(offers[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys)
    writer.writeheader()
    for o in offers:
        row = {}
        for k in keys:
            v = o.get(k)
            row[k] = ";".join(str(x) for x in v) if isinstance(v, list) else (v if v is not None else "")
        writer.writerow(row)
    return buf.getvalue()


def _resolve_profile(config: Config, profiles: Profiles, name: Optional[str]):
    target = name or config.hostai.default_profile
    return profiles.resolve_profile(target)


def _search_profile(
    config: Config,
    profiles: Profiles,
    provider: Any,
    profile: Any,
    max_price: Optional[float],
    unverified: bool,
    exclusions: Optional[market.OfferExclusions] = None,
) -> List[Dict[str, Any]]:
    """Search and filter offers for one profile, tagging matches with its name."""
    image = profiles.image_by_name(profile.image)
    if not image:
        raise click.ClickException(f"profile '{profile.name}' references unknown image '{profile.image}'")

    query, max_dph, ctx_size = _resolve_query(config, profiles, profile, max_price, unverified)
    click.echo(f"[profile] {profile.name} | sm_{image.cuda_arch} | ctx={ctx_size} | image={profile.image}")
    click.echo(f"[search]  {query}")

    offers = provider.search_offers(
        query,
        limit=50,
        order="dph_total",
        storage=market.resolved_disk_gb(profile, config),
    )
    matches = _filter_offers(
        offers,
        max_dph,
        profiles.market_policy.require_free_traffic,
        config.market.max_inet_down_cost,
        config.market.max_inet_up_cost,
        exclusions=exclusions,
    )
    for o in matches:
        o["_profiles"] = [profile.name]
    return matches


@click.command(
    "lookup",
    help="Search Vast offers for a profile without renting. Supports '*', glob patterns like '*-256k', and comma/semicolon-separated lists.",
)
@click.argument("profile", required=False)
@click.option(
    "-p",
    "--profile",
    "profile_opt",
    help="Profile(s): name, glob pattern ('*-256k'), comma/semicolon-separated list, or '*' for all.",
)
@click.option("--max-price", type=float, default=None, help="Maximum all-in $/h.")
@click.option("--unverified", is_flag=True, default=None, help="Also consider unverified/unknown hosts.")
@click.option("--max-results", type=int, default=10, show_default=True, help="Number of results to show.")
@click.option(
    "--skip-machine",
    "skip_machines",
    type=int,
    multiple=True,
    help="Exclude offers hosted on this Vast machine ID (repeatable).",
)
@click.option(
    "--skip-offer",
    "skip_offers",
    type=int,
    multiple=True,
    help="Exclude this Vast offer ID (repeatable).",
)
@click.option(
    "--skip-country",
    "skip_countries",
    multiple=True,
    help="Exclude offers in this country (alpha-2 code or name, repeatable).",
)
@click.option("--json", "output_format", flag_value="json", help="Output raw JSON array.")
@click.option("--csv", "output_format", flag_value="csv", help="Output CSV.")
@click.pass_obj
def cmd_lookup(
    config: Config,
    profile,
    profile_opt,
    max_price,
    unverified,
    max_results,
    skip_machines,
    skip_offers,
    skip_countries,
    output_format,
):
    if max_results <= 0:
        raise click.ClickException("--max-results must be a positive integer")
    if max_price is not None and max_price < 0:
        raise click.ClickException("--max-price must be non-negative")

    profile_name = profile or profile_opt
    profiles = Profiles.from_file(config.root_dir / config.hostai.profiles_file)

    star = profile_name == "*"
    if star:
        selected_profiles = list(profiles.profiles)
        if not selected_profiles:
            raise click.ClickException("profiles.json defines no profiles")
    else:
        names = [n.strip() for n in re.split(r"[,;]", profile_name or "") if n.strip()] or [None]
        selected_profiles = []
        for n in names:
            if n is not None and "*" in n:
                found = [
                    p
                    for p in profiles.profiles
                    if fnmatch.fnmatchcase(p.name, n) or any(fnmatch.fnmatchcase(a, n) for a in p.aliases or [])
                ]
                if not found:
                    raise click.ClickException(f"profile pattern '{n}' matched no profiles")
            else:
                selected = _resolve_profile(config, profiles, n)
                if not selected:
                    raise click.ClickException(f"unknown profile '{n or config.hostai.default_profile}'")
                found = [selected]
            for p in found:
                if all(q.name != p.name for q in selected_profiles):
                    selected_profiles.append(p)

    multi = len(selected_profiles) > 1

    unverified = unverified if unverified is not None else config.market.allow_unverified
    # max_dph is identical for every profile; track it for the summary line.
    display_max_dph = max_price if max_price is not None else config.market.max_dph

    try:
        provider = get_provider(config)
    except Exception as e:
        raise click.ClickException(f"search failed: {e}")

    exclusions = market.config_exclusions(config).merged(
        market.OfferExclusions(machines=skip_machines, offers=skip_offers, countries=skip_countries)
    )
    if exclusions:
        click.echo(f"[exclude] {exclusions.describe()}")

    merged: Dict[Any, Dict[str, Any]] = {}
    for p in selected_profiles:
        try:
            matches = _search_profile(config, profiles, provider, p, max_price, unverified, exclusions)
        except Exception as e:
            if not (star or multi):
                raise click.ClickException(f"search failed: {e}")
            click.echo(f"[warn] {p.name}: {e}")
            continue
        for o in matches:
            key = o.get("id") or o.get("ask_contract_id")
            if key is None:
                key = ("no-id", len(merged))
            if key in merged:
                merged[key]["_profiles"] = sorted(set(merged[key]["_profiles"]) | set(o["_profiles"]))
            else:
                merged[key] = o

    matches = sorted(merged.values(), key=lambda x: x["_effective_dph"])

    if not matches:
        scope = " across the selected profiles" if (star or multi) else ""
        click.echo(f"No matching offers below ${display_max_dph:.2f}/h{scope}.")
        return

    if output_format == "json":
        click.echo(json.dumps(matches[:max_results], indent=2, default=str))
    elif output_format == "csv":
        click.echo(_render_csv(matches[:max_results]))
    else:
        _render_table(matches, max_results, show_profiles=star or multi)
