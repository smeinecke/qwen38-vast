import click

from hostai.authorized_keys import prepare_authorized_keys
from hostai.config import Config


@click.command("prepare", help="Prepare authorized_keys for the runtime image.")
@click.option("--strict/--no-strict", default=False, help="Fail on missing keys.")
@click.pass_obj
def cmd_ssh_prepare(config: Config, strict: bool):
    out = prepare_authorized_keys(
        root_dir=config.root_dir,
        env_file=config.env_file,
        strict=strict,
    )
    click.echo(f"[ssh] prepared {out}")
