"""Run the local OpenAI-compatible, tokenizing proxy."""

from __future__ import annotations

import asyncio
import sys

import click

from hostai.commands import _common
from hostai.proxy import ProxyError, run_proxy


@click.command("proxy", help="Run a local OpenAI-compatible proxy that tokenizes prompts client-side.")
@_common.instance_option
@click.pass_obj
def cmd_proxy(config, instance_name):
    _name, state = _common.resolve_state(config, instance_name)
    if not state.exists:
        raise click.ClickException("no active state; run 'hostai up' first")

    try:
        asyncio.run(run_proxy(config, state))
    except ProxyError as exc:
        raise click.ClickException(str(exc)) from exc
    except KeyboardInterrupt:
        sys.exit(130)
