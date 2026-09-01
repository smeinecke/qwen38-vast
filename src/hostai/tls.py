"""Local TLS certificate generation and remote tmpfs delivery."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from hostai import ssh, utils


def ensure_local_tls_dir(root_dir: Path) -> Path:
    """Create and return the local TLS directory with 0700 permissions."""
    tls_dir = utils.mkdir_private(root_dir / ".hostai-cache" / "tls")
    return tls_dir


def generate_cert(tls_dir: Path, common_name: str = "localhost") -> Tuple[Path, Path]:
    """Generate a 2048-bit RSA self-signed cert/key pair.

    Returns (server.crt, server.key).  Also writes ca.crt as a copy of
    server.crt and chmods the private key to 0600.
    """
    tls_dir = Path(tls_dir)
    tls_dir.mkdir(parents=True, exist_ok=True)
    tls_dir.chmod(0o700)

    key_path = tls_dir / "server.key"
    cert_path = tls_dir / "server.crt"
    ca_path = tls_dir / "ca.crt"

    cmd = [
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-keyout",
        str(key_path),
        "-out",
        str(cert_path),
        "-days",
        "1",
        "-nodes",
        "-subj",
        f"/CN={common_name}",
        "-addext",
        "subjectAltName=DNS:localhost,IP:127.0.0.1",
    ]
    utils.run(cmd, check=True, capture=True)

    key_path.chmod(0o600)
    cert_path.chmod(0o644)
    ca_path.write_bytes(cert_path.read_bytes())
    ca_path.chmod(0o644)

    return cert_path, key_path


def deliver_cert(
    ssh_url: Optional[str],
    tls_dir: Path,
    *,
    known_hosts: Optional[Path] = None,
    identity: Optional[Path] = None,
) -> bool:
    """Deliver server.crt and server.key to /dev/shm/qwen38/certs over SSH.

    If known_hosts is not provided, the user's default SSH known_hosts file
    (~/.ssh/known_hosts) is used.
    """
    if not ssh_url:
        return False
    tls_dir = Path(tls_dir)
    known_hosts = Path(known_hosts) if known_hosts else Path.home() / ".ssh" / "known_hosts"

    cert_path = tls_dir / "server.crt"
    key_path = tls_dir / "server.key"
    if not cert_path.exists() or not key_path.exists():
        return False

    cert_text = cert_path.read_text()
    key_text = key_path.read_text()

    # Ensure the remote certs directory exists.
    result = ssh.run_remote(
        ssh_url,
        "install -d -m 700 /dev/shm/qwen38/certs",
        known_hosts=known_hosts,
        identity=identity,
        capture=True,
    )
    if result.returncode != 0:
        return False

    # Stream the certificate and key to tmpfs.
    for local_text, remote_name in [(cert_text, "server.crt"), (key_text, "server.key")]:
        result = ssh.run_remote(
            ssh_url,
            f"cat > /dev/shm/qwen38/certs/{remote_name}",
            known_hosts=known_hosts,
            identity=identity,
            input_data=local_text,
            capture=True,
        )
        if result.returncode != 0:
            return False

    # Restrict the private key.
    result = ssh.run_remote(
        ssh_url,
        "chmod 600 /dev/shm/qwen38/certs/server.key",
        known_hosts=known_hosts,
        identity=identity,
        capture=True,
    )
    return result.returncode == 0


def load_cert_pair(tls_dir: Path) -> Tuple[str, str]:
    """Read and return (cert_text, key_text) from a local TLS directory."""
    tls_dir = Path(tls_dir)
    cert_text = (tls_dir / "server.crt").read_text()
    key_text = (tls_dir / "server.key").read_text()
    return cert_text, key_text
