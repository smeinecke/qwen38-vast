"""Repository validation for hostai.

This module also records structured validation metadata in
``.hostai-vast/validation.json`` so a later `hostai up` or `hostai validate
--compare` can detect drift between the last known-good local run and the
current working tree.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hostai import market
from hostai.config import Config
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


def validate_repo(root_dir: Path, config: Optional[Config] = None) -> List[str]:
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

            # Any explicit disk_space constraint in a profile query will be
            # replaced by the resolved disk allocation.  Flag values that are
            # lower than the resolved disk as contradictions.
            if config is not None:
                disk_space_pat = re.compile(r"\s*disk_space\s*(>=?|<=?|=)\s*([^\s]+)")
                for p in profiles.profiles:
                    disk_gb = market.resolved_disk_gb(p, config)
                    for m in disk_space_pat.finditer(p.gpu_query):
                        try:
                            val = float(m.group(2))
                        except ValueError:
                            continue
                        if val < disk_gb:
                            errors.append(
                                f"profile {p.name}: gpu_query disk_space>={val} is lower "
                                f"than resolved disk_gb={disk_gb}"
                            )

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


@dataclasses.dataclass
class ValidationRecord:
    """Structured result of a validation run."""

    timestamp: str
    result: str  # "ok" or "failed"
    duration_seconds: float
    git_commit: str
    dirty: bool
    image_digest: str
    profile_hash: str
    errors: List[str]
    extras: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationRecord":
        return cls(**data)


def _validation_path(root_dir: Path) -> Path:
    path = root_dir / ".hostai-vast" / "validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _git_info(root_dir: Path) -> tuple:
    """Return (commit, dirty) for the repository at *root_dir* or ("", True)."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(root_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        commit = ""
    try:
        status = subprocess.run(
            ["git", "-C", str(root_dir), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        status = "?"
    return commit, bool(status)


def _image_digest(image: str = "hostai-test:latest") -> str:
    """Return the image digest for *image*, or an empty string."""
    try:
        raw = subprocess.run(
            ["docker", "image", "inspect", image, "--format={{index .RepoDigests 0}}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except Exception:
        raw = ""
    return raw or f"{image}:built-locally"


def _profile_hash(root_dir: Path) -> str:
    """Return a hash of the profiles used for this validation."""
    profiles = root_dir / "profiles.json"
    if not profiles.exists():
        return ""
    return hashlib.sha256(profiles.read_bytes()).hexdigest()[:16]


def record_validation(
    root_dir: Path,
    result: str,
    duration: float,
    errors: List[str],
    image: str = "hostai-test:latest",
) -> ValidationRecord:
    """Record a validation run and return the record."""
    commit, dirty = _git_info(root_dir)
    record = ValidationRecord(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        result=result,
        duration_seconds=duration,
        git_commit=commit,
        dirty=dirty,
        image_digest=_image_digest(image),
        profile_hash=_profile_hash(root_dir),
        errors=errors,
    )
    _validation_path(root_dir).write_text(json.dumps(record.to_dict(), indent=2))
    return record


def load_last_validation(root_dir: Path) -> Optional[ValidationRecord]:
    path = _validation_path(root_dir)
    if not path.exists():
        return None
    try:
        return ValidationRecord.from_dict(json.loads(path.read_text()))
    except Exception:
        return None


def validation_digest(record: ValidationRecord) -> str:
    """Return a stable digest that can be compared between validation runs."""
    hasher = hashlib.sha256()
    hasher.update(record.git_commit.encode())
    hasher.update(str(record.dirty).encode())
    hasher.update(record.image_digest.encode())
    hasher.update(record.profile_hash.encode())
    return hasher.hexdigest()[:16]


def compare_validations(current: ValidationRecord, previous: ValidationRecord) -> List[str]:
    """Return a list of human-readable drift messages between two records."""
    diffs: List[str] = []
    if current.git_commit != previous.git_commit:
        diffs.append(
            f"git commit changed: {previous.git_commit[:12]} -> {current.git_commit[:12]}"
        )
    if current.dirty != previous.dirty:
        diffs.append(f"working tree dirty state changed: {previous.dirty} -> {current.dirty}")
    if current.image_digest != previous.image_digest:
        diffs.append(f"integration image changed: {previous.image_digest} -> {current.image_digest}")
    if current.profile_hash != previous.profile_hash:
        diffs.append(f"profiles.json changed: {previous.profile_hash} -> {current.profile_hash}")
    return diffs


def is_valid(root_dir: Path) -> bool:
    return not validate_repo(root_dir)
