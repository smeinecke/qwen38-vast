"""Repository validation for hostai."""

from __future__ import annotations

from pathlib import Path
from typing import List

from hostai.profiles import Profiles

REQUIRED_FILES = (
    "pyproject.toml",
    "hostai.toml.example",
    "profiles.json",
    "Dockerfile",
    "start.sh",
    "entrypoint.sh",
    "hostai-init-ssh.sh",
)

DOCKERFILE_MARKERS = (
    "libgomp1",
    "llama-server",
    "CCACHE_SEED",
)

GITIGNORE_PATTERNS = (
    ".hostai-vast/",
    ".hostai-runs/",
    ".hostai-cache/",
    "hostai.toml",
)

PACKAGE_MODULES = (
    "src/hostai/__init__.py",
    "src/hostai/cli.py",
    "src/hostai/config.py",
    "src/hostai/state.py",
    "src/hostai/profiles.py",
    "src/hostai/vast.py",
    "src/hostai/utils.py",
    "src/hostai/ssh.py",
    "src/hostai/tls.py",
    "src/hostai/api.py",
    "src/hostai/cache.py",
    "src/hostai/notify.py",
    "src/hostai/authorized_keys.py",
    "src/hostai/validate.py",
)


def validate_repo(root_dir: Path) -> List[str]:
    """Validate the repository layout and return a list of error messages."""
    errors: List[str] = []

    for name in REQUIRED_FILES:
        if not (root_dir / name).exists():
            errors.append(f"missing required file: {name}")

    pyproject = root_dir / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text()
        if 'name = "hostai"' not in text:
            errors.append("pyproject.toml missing project name 'hostai'")
        if 'hostai = "hostai.cli:main"' not in text:
            errors.append("pyproject.toml missing hostai console script")

    hostai_toml_example = root_dir / "hostai.toml.example"
    if hostai_toml_example.exists():
        try:
            import tomllib

            with hostai_toml_example.open("rb") as f:
                tomllib.load(f)
        except Exception as e:
            errors.append(f"hostai.toml.example is not valid TOML: {e}")

    profiles_path = root_dir / "profiles.json"
    if profiles_path.exists():
        try:
            profiles = Profiles.from_file(profiles_path)
        except Exception as e:
            errors.append(f"profiles.json is invalid: {e}")
        else:
            if not profiles.images:
                errors.append("profiles.json has no images")
            if not profiles.profiles:
                errors.append("profiles.json has no profiles")
            if not profiles.resolve_profile(profiles.default_profile):
                errors.append(f"default profile '{profiles.default_profile}' not found in profiles.json")

    dockerfile = root_dir / "Dockerfile"
    if dockerfile.exists():
        text = dockerfile.read_text()
        for marker in DOCKERFILE_MARKERS:
            if marker not in text:
                errors.append(f"Dockerfile missing marker: {marker}")

    gitignore = root_dir / ".gitignore"
    if gitignore.exists():
        text = gitignore.read_text()
        for pat in GITIGNORE_PATTERNS:
            if pat not in text:
                errors.append(f".gitignore missing pattern: {pat}")

    for mod in PACKAGE_MODULES:
        if not (root_dir / mod).exists():
            errors.append(f"missing package module: {mod}")

    ssh_dir = root_dir / "ssh"
    if not (ssh_dir / "authorized_keys").exists() and not (ssh_dir / "authorized_keys.generated").exists():
        # Only warn; the user may generate it before building an image.
        pass

    return errors


def is_valid(root_dir: Path) -> bool:
    return not validate_repo(root_dir)
