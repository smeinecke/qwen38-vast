#!/usr/bin/env bash
set -Eeuo pipefail

umask 077
install -d -m 0700 -o root -g root /root/.ssh

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

# Preserve any keys already present (for local testing or an operator override).
if [[ -f /root/.ssh/authorized_keys ]]; then
  cat /root/.ssh/authorized_keys >> "$TMP" || true
fi

# Merge all public keys baked by the build workflow.
for f in /etc/qwen38/ssh/authorized_keys*; do
  [[ -f "$f" ]] || continue
  cat "$f" >> "$TMP"
done

# Optional runtime fallback/override. This is deliberately base64 encoded by
# qwen-up so Docker's -e parsing is not confused by spaces in an OpenSSH key.
if [[ -n "${QWEN_SSH_PUBLIC_KEY_B64:-}" ]]; then
  printf '%s' "$QWEN_SSH_PUBLIC_KEY_B64" | base64 -d >> "$TMP"
  printf '\n' >> "$TMP"
fi

grep -E '^[[:space:]]*(ssh-|ecdsa-|sk-)[^[:space:]]+[[:space:]]+[^[:space:]]+' "$TMP" \
  | sed -E 's/^[[:space:]]+//' \
  | awk '!seen[$0]++' > /root/.ssh/authorized_keys.new || true

if [[ ! -s /root/.ssh/authorized_keys.new ]]; then
  echo >&2 "ERROR: no SSH public keys available in the image/runtime environment."
  exit 2
fi

mv /root/.ssh/authorized_keys.new /root/.ssh/authorized_keys
chown root:root /root/.ssh /root/.ssh/authorized_keys
chmod 0700 /root/.ssh
chmod 0600 /root/.ssh/authorized_keys

# Host keys are absent from the baked image and generated on first container
# start. ssh-keygen -A leaves existing keys intact on a later container restart.
ssh-keygen -A >/dev/null
mkdir -p /run/sshd
chmod 0755 /run/sshd

printf '[ssh] authorized keys: %s | mode=%s owner=%s\n' \
  "$(wc -l < /root/.ssh/authorized_keys | tr -d ' ')" \
  "$(stat -c '%a' /root/.ssh/authorized_keys)" \
  "$(stat -c '%U:%G' /root/.ssh/authorized_keys)"
