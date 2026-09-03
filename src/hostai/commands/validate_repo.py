import os
import re
import shutil
import subprocess
import sys
import time

import click

from hostai.config import Config
from hostai.validate import (
    compare_validations,
    load_last_validation,
    record_validation,
    validate_repo,
    validation_digest,
)


@click.command("validate")
@click.option("--production", is_flag=True, help="Run full local checks before a real Vast rental.")
@click.option("--compare", is_flag=True, help="Compare current state with the last successful validation.")
@click.option("--image", default="hostai-test:latest", help="Local integration image to require.")
@click.pass_obj
def cmd_validate(config: Config, production: bool, compare: bool, image: str):
    """Validate the repository layout and configuration.

    With --production, also verify the local Docker integration image and run
    the integration acceptance tests.  A record of the validation is written to
    .hostai-vast/validation.json; on success, .hostai-vast/validation-last-success.json
    is also written.  With --compare, the current state is compared against the
    last *successful* record so you can detect unvalidated changes before
    spending money on Vast.
    """
    start = time.monotonic()
    errors = validate_repo(config.root_dir, config)
    checks_run = ["repo"]
    level = "repo"

    if production:
        level = "production"
        checks_run.append("docker")
        if not shutil.which("docker"):
            errors.append("docker not found; cannot run production checks")
        else:
            try:
                subprocess.run(
                    ["docker", "image", "inspect", image],
                    capture_output=True,
                    check=True,
                    timeout=30,
                )
                checks_run.append("image-exists")
            except subprocess.CalledProcessError:
                errors.append(
                    f"integration image {image} not built; run "
                    f"'docker build -f tests/integration/Dockerfile.test -t {image} .'"
                )

        if not errors:
            click.echo("[validate] running integration acceptance tests...")
            checks_run.append("integration-tests")
            try:
                env = dict(os.environ)
                env["HOSTAI_LOCAL_IMAGE"] = image
                env["HOSTAI_PROVIDER"] = "local"
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "-m", "not slow", "tests/test_local_integration.py"],
                    cwd=config.root_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if proc.returncode != 0:
                    errors.append("integration acceptance tests failed")
                    if proc.stdout:
                        click.echo(proc.stdout, err=True)
                    if proc.stderr:
                        click.echo(proc.stderr, err=True)
                else:
                    passed = re.search(r"(\d+) passed", proc.stdout or "")
                    skipped = re.search(r"(\d+) skipped", proc.stdout or "")
                    if not passed and skipped:
                        # The suite skipped all lifecycle tests, usually because
                        # the requested image is not what the tests expect.
                        errors.append(
                            "integration acceptance tests were skipped; "
                            f"verify {image} is available and HOSTAI_LOCAL_IMAGE matches"
                        )
            except subprocess.TimeoutExpired:
                errors.append("integration acceptance tests timed out")
            except Exception as exc:
                errors.append(f"could not run integration tests: {exc}")

    duration = time.monotonic() - start
    result = "ok" if not errors else "failed"
    previous = load_last_validation(config.root_dir, success=True) if compare else None
    record = record_validation(
        config.root_dir,
        result,
        duration,
        errors,
        image=image,
        level=level,
        checks_run=checks_run,
    )

    if compare:
        if previous is None:
            click.echo("WARNING: no previous successful validation record found", err=True)
        else:
            diffs = compare_validations(record, previous)
            if diffs:
                for d in diffs:
                    click.echo(f"DRIFT: {d}", err=True)
                if result == "ok":
                    click.echo("WARNING: state drifted since last successful validation", err=True)

    if errors:
        for e in errors:
            click.echo(f"ERROR: {e}", err=True)
        raise click.ClickException("validation failed")

    digest = validation_digest(record)
    if record.image_id:
        display = f"{record.image}@{record.image_id.split(':', 1)[-1][:12]}"
    else:
        display = record.image or record.image_digest
    click.echo(f"OK level={record.level} image={display} (digest {digest})")
