import click

from hostai.config import Config
from hostai.validate import validate_repo


@click.command("validate")
@click.pass_obj
def cmd_validate(config: Config):
    """Validate the repository layout and configuration."""
    errors = validate_repo(config.root_dir)
    if errors:
        for e in errors:
            click.echo(f"ERROR: {e}", err=True)
        raise click.ClickException("validation failed")
    click.echo("OK")
