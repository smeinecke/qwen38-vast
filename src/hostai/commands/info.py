"""Show provider account balance and every instance on the account.

Unlike ``hostai status`` (which reports on locally tracked deployments),
``hostai info`` asks the provider for ground truth: the account's remaining
credit and all instances it currently holds, including machines that have no
local hostai state and can only be removed with ``hostai down --id``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.table import Table

from hostai import utils
from hostai.commands import _common
from hostai.config import Config
from hostai.providers import get_provider


def _money(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_dph(value: Any) -> str:
    dph = _money(value)
    return f"{dph:.4f}" if dph is not None else "-"


def _status_style(status: str) -> str:
    if status == "running":
        return "green"
    if status in _common.TERMINAL_INSTANCE_STATUSES:
        return "red"
    return "yellow"


def _instance_sort_key(instance: Dict[str, Any]) -> tuple:
    try:
        return (0, int(instance.get("id") or instance.get("instance_id") or 0))
    except (TypeError, ValueError):
        return (1, 0)


def _print_account(provider_name: str, account: Optional[Dict[str, Any]], error: Optional[Exception]) -> None:
    table = Table(title=f"hostai account ({provider_name})", show_header=True, header_style="bold")
    table.add_column("Key")
    table.add_column("Value")
    if account is None:
        reason = f"lookup failed: {error}" if error else "no billing information for this provider"
        table.add_row("Billing", reason)
    else:
        user = str(account.get("username") or account.get("user") or "-")
        email = account.get("email")
        table.add_row("User", f"{user} ({email})" if email else user)
        balance = _money(account.get("balance"))
        table.add_row("Balance", f"${balance:.2f}" if balance is not None else str(account.get("balance") or "-"))
        credit = _money(account.get("credit"))
        if credit:
            table.add_row("Credit", f"${credit:.2f}")
    Console().print(table)


def _print_instances(rows: List[Dict[str, Any]], tracked: Dict[int, str], machines: Dict[int, Any]) -> int:
    table = Table(title=f"provider instances ({len(rows)})", show_header=True, header_style="bold")
    table.add_column("ID", no_wrap=True)
    table.add_column("Machine", no_wrap=True)
    table.add_column("Name", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("GPU", no_wrap=True)
    table.add_column("$/h", no_wrap=True)
    table.add_column("Age", no_wrap=True)
    table.add_column("Label")

    untracked = 0
    burn = 0.0
    for inst in rows:
        raw_id = inst.get("id") or inst.get("instance_id")
        iid = str(raw_id) if raw_id is not None else "-"
        name = "-"
        try:
            name = tracked.get(int(raw_id), "-") if raw_id is not None else "-"
        except (TypeError, ValueError):
            pass
        if name == "-":
            untracked += 1
        machine_id = inst.get("machine_id")
        if machine_id is None and raw_id is not None:
            try:
                machine_id = machines.get(int(raw_id))
            except (TypeError, ValueError):
                machine_id = None
        status = str(inst.get("actual_status") or inst.get("status") or "unknown")
        num = inst.get("num_gpus") or 1
        gpu = str(inst.get("gpu_name") or "-")
        if num != 1:
            gpu = f"{num}x {gpu}"
        if status == "running":
            burn += _money(inst.get("dph_total")) or 0.0
        duration = inst.get("duration")
        age = utils.format_duration(duration) if isinstance(duration, (int, float)) else "-"
        label = str(inst.get("label") or inst.get("container_name") or inst.get("image") or "")
        table.add_row(
            iid,
            str(machine_id) if machine_id is not None else "-",
            name,
            f"[{_status_style(status)}]{status}[/]",
            gpu,
            _fmt_dph(inst.get("dph_total")),
            age,
            label,
        )
    Console().print(table)
    if rows:
        click.echo(f"running burn: ${burn:.4f}/h")
    if untracked:
        click.echo(f"{untracked} instance(s) not tracked by local state; use 'hostai down --id <id>' to destroy them.")
    return untracked


@click.command("info", help="Show provider account balance and all instances on the account.")
@click.pass_obj
def cmd_info(config: Config) -> None:
    try:
        provider = get_provider(config)
    except Exception as exc:
        raise click.ClickException(str(exc))

    account = None
    account_err: Optional[Exception] = None
    try:
        account = provider.get_account_info()
    except Exception as exc:
        account_err = exc
    _print_account(provider.name, account, account_err)

    try:
        rows = provider.list_instances()
    except Exception as exc:
        raise click.ClickException(f"instance lookup failed: {exc}")

    if not rows:
        click.echo("No instances on this account.")
        return
    rows.sort(key=_instance_sort_key)
    _print_instances(
        rows,
        _common.tracked_instance_ids(config.root_dir),
        _common.tracked_instance_machines(config.root_dir),
    )
