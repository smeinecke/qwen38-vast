import os
import re
import shutil
import subprocess
import sys
import time
from typing import List, Optional, Tuple

import click

from hostai import utils
from hostai.config import Config
from hostai.validate import (
    compare_validations,
    load_last_validation,
    record_validation,
    validate_repo,
    validation_digest,
)


def _run_docker_build(config: Config, image: str) -> Optional[str]:
    """Build the integration image and return an error message, or ``None`` on success."""
    build_env = dict(os.environ)
    build_env["DOCKER_BUILDKIT"] = "1"
    try:
        proc = subprocess.run(
            [
                "docker",
                "build",
                "-f",
                "tests/integration/Dockerfile.test",
                "-t",
                image,
                ".",
            ],
            cwd=config.root_dir,
            env=build_env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            if proc.stdout:
                click.echo(proc.stdout, err=True)
            if proc.stderr:
                click.echo(proc.stderr, err=True)
            return f"failed to build integration image {image}"
    except subprocess.TimeoutExpired:
        return "integration image build timed out"
    except Exception as exc:
        return f"could not build integration image: {exc}"
    return None


def _run_image_inspect(image: str) -> Optional[str]:
    """Return ``None`` when the image exists, otherwise an error string."""
    try:
        subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            check=True,
            timeout=30,
        )
    except subprocess.CalledProcessError:
        return (
            f"integration image {image} not built; run 'docker build -f tests/integration/Dockerfile.test -t {image} .'"
        )
    return None


def _run_integration_tests(config: Config, image: str) -> Optional[str]:
    """Run local integration tests and return an error string, or ``None``."""
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
            if proc.stdout:
                click.echo(proc.stdout, err=True)
            if proc.stderr:
                click.echo(proc.stderr, err=True)
            return "integration acceptance tests failed"
        passed = re.search(r"(\d+) passed", proc.stdout or "")
        skipped = re.search(r"(\d+) skipped", proc.stdout or "")
        if not passed and skipped:
            # The suite skipped all lifecycle tests, usually because
            # the requested image is not what the tests expect.
            return (
                f"integration acceptance tests were skipped; verify {image} is available and HOSTAI_LOCAL_IMAGE matches"
            )
    except subprocess.TimeoutExpired:
        return "integration acceptance tests timed out"
    except Exception as exc:
        return f"could not run integration tests: {exc}"
    return None


def _run_production_checks(config: Config, image: str, no_build: bool) -> Tuple[List[str], List[str]]:
    """Return ``(errors, checks_run)`` for the production validation phase."""
    errors: List[str] = []
    checks_run: List[str] = ["clean-tree", "docker"]

    if utils.is_dirty_tree(config.root_dir):
        errors.append(
            "working tree is dirty; commit or stash before production validation, "
            "or use 'hostai up ... --allow-unvalidated' to bypass the gate"
        )

    if not shutil.which("docker"):
        errors.append("docker not found; cannot run production checks")

    if errors:
        return errors, checks_run

    if not no_build:
        click.echo(f"[validate] building integration image {image}...")
        checks_run.append("image-build")
        err = _run_docker_build(config, image)
        if err:
            errors.append(err)

    if not errors:
        err = _run_image_inspect(image)
        if err:
            errors.append(err)
        else:
            checks_run.append("image-exists")

    if not errors:
        click.echo("[validate] running integration acceptance tests...")
        checks_run.append("integration-tests")
        err = _run_integration_tests(config, image)
        if err:
            errors.append(err)

    return errors, checks_run


@click.command("validate")
@click.option("--production", is_flag=True, help="Run full local checks before a real Vast rental.")
@click.option("--compare", is_flag=True, help="Compare current state with the last successful validation.")
@click.option("--image", default="hostai-test:latest", help="Local integration image to require/build.")
@click.option("--no-build", is_flag=True, help="Do not rebuild the integration image before production tests.")
@click.pass_obj
def cmd_validate(config: Config, production: bool, compare: bool, image: str, no_build: bool):
    """Validate the repository layout and configuration.

    With --production, also build and verify the local Docker integration image
    and run the integration acceptance tests.  A record of the validation is
    written to .hostai-vast/validation.json; on success,
    .hostai-vast/validation-last-success.json is also written.
    With --compare, the current state is compared against the last *successful*
    record so you can detect unvalidated changes before spending money on Vast.
    """
    start = time.monotonic()
    errors = validate_repo(config.root_dir, config)
    checks_run = ["repo"]
    level = "repo"

    if production:
        level = "production"
        prod_errors, prod_checks = _run_production_checks(config, image, no_build)
        errors.extend(prod_errors)
        checks_run.extend(prod_checks)

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
