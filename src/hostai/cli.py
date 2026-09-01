import click

from hostai import __version__
from hostai.commands.bench import cmd_bench
from hostai.commands.cache_cmd import cmd_cache_copy, cmd_cache_setup
from hostai.commands.down import cmd_down
from hostai.commands.lookup import cmd_lookup
from hostai.commands.monitor import (
    cmd_monitor,
    cmd_monitor_logs,
    cmd_monitor_once,
    cmd_monitor_start,
    cmd_monitor_status,
    cmd_monitor_stop,
    cmd_monitor_watch,
)
from hostai.commands.results import cmd_results
from hostai.commands.ssh_cmd import cmd_ssh_prepare
from hostai.commands.status import cmd_status
from hostai.commands.up import cmd_up
from hostai.commands.validate_repo import cmd_validate
from hostai.config import load_config


@click.group(
    name="hostai",
    help="Disposable Vast.ai GPU deployment manager.",
    invoke_without_command=False,
)
@click.version_option(version=__version__, prog_name="hostai")
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()
    ctx.obj = load_config()


cli.add_command(cmd_validate, name="validate")
cli.add_command(cmd_lookup, name="lookup")
cli.add_command(cmd_results, name="results")
cli.add_command(cmd_up, name="up")
cli.add_command(cmd_down, name="down")
cli.add_command(cmd_status, name="status")
cli.add_command(cmd_bench, name="bench")


@cli.group("cache", help="Manage the persistent llama.cpp slot/KV cache.")
@click.pass_context
def cache(ctx):
    pass


cache.add_command(cmd_cache_setup, name="setup")
cache.add_command(cmd_cache_copy, name="copy")
cli.add_command(cache)


@cli.group("ssh", help="SSH key and tunnel helpers.")
@click.pass_context
def ssh(ctx):
    pass


ssh.add_command(cmd_ssh_prepare, name="prepare")
cli.add_command(ssh)


cli.add_command(cmd_monitor, name="monitor")
cmd_monitor.add_command(cmd_monitor_once, name="once")
cmd_monitor.add_command(cmd_monitor_watch, name="watch")
cmd_monitor.add_command(cmd_monitor_start, name="start")
cmd_monitor.add_command(cmd_monitor_stop, name="stop")
cmd_monitor.add_command(cmd_monitor_status, name="status")
cmd_monitor.add_command(cmd_monitor_logs, name="logs")


def main() -> None:
    cli()
