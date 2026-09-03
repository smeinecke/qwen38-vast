#!/usr/bin/env bash
# Fake start.sh for the LocalProvider integration image.
set -Eeuo pipefail

HOSTAI_UNSECURE="${HOSTAI_UNSECURE:-0}"
HOSTAI_TMPFS_BASE="${HOSTAI_TMPFS_BASE:-/dev/shm/qwen38}"
HOSTAI_LOG_DIR="${HOSTAI_LOG_DIR:-$HOSTAI_TMPFS_BASE/log}"
HOSTAI_CERTS_DIR="${HOSTAI_CERTS_DIR:-$HOSTAI_TMPFS_BASE/certs}"

mkdir -p "$HOSTAI_TMPFS_BASE" "$HOSTAI_TMPFS_BASE/log" "$HOSTAI_TMPFS_BASE/run" \
         "$HOSTAI_TMPFS_BASE/ssh" "$HOSTAI_TMPFS_BASE/certs" "$HOSTAI_TMPFS_BASE/tmp"
chmod 700 "$HOSTAI_TMPFS_BASE"

rm -rf /var/log/qwen38 /run/qwen38
ln -sfn "$HOSTAI_TMPFS_BASE/log" /var/log/qwen38
ln -sfn "$HOSTAI_TMPFS_BASE/run" /run/qwen38

# Write a fake disk-telemetry log so hostai up can collect it.
mkdir -p "$HOSTAI_LOG_DIR"
cat > "$HOSTAI_LOG_DIR/disk-usage.log" <<'EOF'
time	after-preflight	free_bytes	total_bytes
0	after-preflight	10737418240	16106127360
0	after-main-model	10737418240	16106127360
0	after-draft-model	10737418240	16106127360
0	before-serve	10737418240	16106127360
EOF

# Wait for /dev/shm/qwen38/certs in secure mode; unsecure mode starts immediately.
if [[ "$HOSTAI_UNSECURE" != "1" ]]; then
  for _ in $(seq 1 120); do
    if [[ -f "$HOSTAI_CERTS_DIR/server.crt" && -f "$HOSTAI_CERTS_DIR/server.key" ]]; then
      break
    fi
    echo "[start] waiting for TLS certs..."
    sleep 1
  done
fi

echo "[start] launching fake llama-server (unsecure=$HOSTAI_UNSECURE)"
exec /usr/local/bin/llama-server
