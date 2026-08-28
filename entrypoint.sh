#!/usr/bin/env bash
set -Eeuo pipefail

/usr/local/bin/qwen-init-ssh.sh

# We self-manage SSH instead of Vast's SSH launch mode. This avoids Vast
# rebuilding a child /ssh image and makes authorized_keys permissions entirely
# deterministic.
mkdir -p /var/log/qwen38
/usr/sbin/sshd -E /var/log/qwen38/sshd.log

echo "[ssh] sshd ready on 0.0.0.0:22"

# Keep the remote logfile expected by qwen-logs/qwen-down while still emitting
# the same output to the container log visible in the Vast UI.
exec /usr/local/bin/start.sh > >(tee -a /var/log/qwen38/server.log) 2>&1
