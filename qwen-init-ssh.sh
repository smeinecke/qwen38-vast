#!/usr/bin/env bash
set -Eeuo pipefail

umask 077
install -d -m 0700 -o root -g root /root/.ssh

QWEN_UNSECURE="${QWEN_UNSECURE:-0}"
QWEN_TMPFS_BASE="${QWEN_TMPFS_BASE:-/dev/shm/qwen38}"

if [[ "$QWEN_UNSECURE" != "1" ]]; then
  # In secure mode host private keys, TLS certs, and the llama-server socket
  # live in tmpfs, but the authorized_keys file is kept in /root/.ssh because
  # sshd's auth_secure_path refuses paths under world-writable directories
  # (/dev/shm is 1777) even when the sticky bit is set. The public keys here
  # are not a secret.
  install -d -m 0700 -o root -g root "$QWEN_TMPFS_BASE" "$QWEN_TMPFS_BASE/log" "$QWEN_TMPFS_BASE/run" "$QWEN_TMPFS_BASE/ssh" "$QWEN_TMPFS_BASE/certs" "$QWEN_TMPFS_BASE/tmp"

  rm -rf /var/log/qwen38 /run/qwen38
  ln -sfn "$QWEN_TMPFS_BASE/log" /var/log/qwen38
  ln -sfn "$QWEN_TMPFS_BASE/run" /run/qwen38
else
  # Legacy/unsecure mode: keep host keys on disk and use standard log/run paths.
  mkdir -p /var/log/qwen38 /run/qwen38
fi

# authorized_keys is always at the standard path; it is the only part of the
# SSH setup that sshd requires to be under a non-world-writable directory.
TMP_AUTHORIZED_KEYS="/root/.ssh/authorized_keys"

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

# Defensive: users may have quoted the key in .env, which the local parser now
# strips. Existing running images still benefit from cleaning up any quotes that
# accidentally made it through.
sed -i -e 's/^[[:space:]]*"//; s/"[[:space:]]*$//' -e "s/^[[:space:]]*'//; s/'[[:space:]]*$//" "$TMP"

grep -E '^[[:space:]]*(ssh-|ecdsa-|sk-)[^[:space:]]+[[:space:]]+[^[:space:]]+' "$TMP" \
  | sed -E 's/^[[:space:]]+//' \
  | awk '!seen[$0]++' > "$TMP_AUTHORIZED_KEYS.new" || true

if [[ ! -s "$TMP_AUTHORIZED_KEYS.new" ]]; then
  echo >&2 "ERROR: no SSH public keys available in the image/runtime environment."
  exit 2
fi

mv "$TMP_AUTHORIZED_KEYS.new" "$TMP_AUTHORIZED_KEYS"
chown root:root /root/.ssh
chmod 0700 /root/.ssh
chown root:root "$TMP_AUTHORIZED_KEYS"
chmod 0600 "$TMP_AUTHORIZED_KEYS"

if [[ "$QWEN_UNSECURE" != "1" ]]; then
  # Some base images do not include sshd_config.d by default, or the Include
  # line is missing. Make sure the drop-in directory is loaded.
  if ! grep -Eq '^[[:space:]]*Include[[:space:]]+/etc/ssh/sshd_config.d/\*\.conf' /etc/ssh/sshd_config; then
    echo 'Include /etc/ssh/sshd_config.d/*.conf' >> /etc/ssh/sshd_config
  fi

  # Force key-only root access and keep host private keys on tmpfs.
  # AuthorizedKeysFile is /root/.ssh/authorized_keys, not tmpfs, because sshd
  # refuses to read authorized_keys under /dev/shm (world-writable 1777).
  # The public keys are not a secret; only the host private keys need tmpfs.
  cat > /etc/ssh/sshd_config.d/99-qwen-secure.conf <<'EOF'
PermitRootLogin prohibit-password
PubkeyAuthentication yes
PasswordAuthentication no
PermitEmptyPasswords no
AuthenticationMethods publickey
UsePAM no
AuthorizedKeysFile /root/.ssh/authorized_keys
HostKey /dev/shm/qwen38/ssh/ssh_host_ed25519_key
HostKey /dev/shm/qwen38/ssh/ssh_host_rsa_key
EOF
  chmod 0600 /etc/ssh/sshd_config.d/99-qwen-secure.conf

  # Generate host keys on tmpfs. Failures are non-fatal; sshd will pick the ones
  # it can use.
  for t in ed25519 rsa; do
    if [[ ! -f "$QWEN_TMPFS_BASE/ssh/ssh_host_${t}_key" ]]; then
      ssh-keygen -t "$t" -f "$QWEN_TMPFS_BASE/ssh/ssh_host_${t}_key" -N "" -C "qwen38" >/dev/null 2>&1 || true
    fi
  done
else
  # Legacy mode: remove any secure drop-in so the default /etc/ssh host keys are
  # used, and generate them on disk if absent. Still force key-only root and
  # make sure sshd_config.d is loaded if the base image omits it.
  rm -f /etc/ssh/sshd_config.d/99-qwen-secure.conf
  if ! grep -Eq '^[[:space:]]*Include[[:space:]]+/etc/ssh/sshd_config.d/\*\.conf' /etc/ssh/sshd_config; then
    echo 'Include /etc/ssh/sshd_config.d/*.conf' >> /etc/ssh/sshd_config
  fi
  cat > /etc/ssh/sshd_config.d/99-qwen.conf <<'EOF'
PermitRootLogin prohibit-password
PubkeyAuthentication yes
PasswordAuthentication no
PermitEmptyPasswords no
AuthenticationMethods publickey
UsePAM no
EOF
  chmod 0600 /etc/ssh/sshd_config.d/99-qwen.conf
  ssh-keygen -A >/dev/null
fi

mkdir -p /run/sshd
chmod 0755 /run/sshd

printf '[ssh] authorized keys: %s | mode=%s owner=%s\n' \
  "$(wc -l < "$TMP_AUTHORIZED_KEYS" | tr -d ' ')" \
  "$(stat -c '%a' "$TMP_AUTHORIZED_KEYS")" \
  "$(stat -c '%U:%G' "$TMP_AUTHORIZED_KEYS")"

# Validate the resulting sshd configuration before entrypoint starts sshd. A
# config error here is far easier to diagnose than sshd exiting in a restart loop.
if ! /usr/sbin/sshd -t 2>&1 | tee /var/log/qwen38/sshd-config-test.log; then
  echo >&2 "ERROR: sshd configuration test failed; see /var/log/qwen38/sshd-config-test.log"
  exit 5
fi
