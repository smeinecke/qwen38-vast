"""Desktop / terminal notifications."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys


def notify(title: str, message: str) -> bool:
    """Send a notification.  Never raise."""
    system = platform.system()
    try:
        if system == "Linux" and shutil.which("notify-send"):
            subprocess.run(["notify-send", title, message], check=False)
            return True
        if system == "Darwin" and shutil.which("osascript"):
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    f"display notification {safe_apple(message)} with title {safe_apple(title)}",
                ],
                check=False,
            )
            return True
        if shutil.which("wall") and os.ttyname(0):
            subprocess.run(["wall", f"{title}: {message}"], check=False)
            return True
    except Exception:  # nosec B110 intentionally defensive
        pass
    print(f"[notify] {title}: {message}", file=sys.stderr)
    return False


def safe_apple(s: str) -> str:
    return '"' + s.replace('"', '\\"') + '"'
