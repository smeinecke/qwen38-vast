"""Run the local OpenAI-compatible, tokenizing proxy."""

from __future__ import annotations

import asyncio
import sys

import click

from hostai.proxy import ProxyError, run_proxy
from hostai.state import State, state_dir


@click.command("proxy", help="Run a local OpenAI-compatible proxy that tokenizes prompts client-side.")
@click.pass_obj
def cmd_proxy(config):
    state = State(state_dir(config.root_dir) / "state.json")
    if not state.exists:
        raise click.ClickException("no active state; run 'hostai up' first")

    try:
        asyncio.run(run_proxy(config, state))
    except ProxyError as exc:
        raise click.ClickException(str(exc)) from exc
    except KeyboardInterrupt:
        sys.exit(130)
