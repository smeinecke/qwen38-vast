#!/usr/bin/env bash
# PID-1 supervisor for the local integration test image.
set -Eeuo pipefail

/usr/local/bin/hostai-init-ssh.sh

mkdir -p /var/log/qwen38 /run/qwen38
rm -f /run/qwen38/start.exitcode

sshd_pid=""
server_pid=""
shutting_down=0

shutdown() {
  local rc="${1:-143}"
  (( shutting_down == 0 )) || return 0
  shutting_down=1
  trap - TERM INT
  [[ -n "$server_pid" ]] && kill -TERM "$server_pid" 2>/dev/null || true
  [[ -n "$sshd_pid" ]] && kill -TERM "$sshd_pid" 2>/dev/null || true
  wait 2>/dev/null || true
  exit "$rc"
}
trap 'shutdown 143' TERM
trap 'shutdown 130' INT

start_sshd() {
  while (( shutting_down == 0 )); do
    /usr/sbin/sshd -D -e >>/var/log/qwen38/sshd.log 2>&1 &
    sshd_pid=$!
    sleep 0.25
    if kill -0 "$sshd_pid" 2>/dev/null; then
      echo "[ssh] sshd ready (pid=$sshd_pid)"
      return 0
    fi
    wait "$sshd_pid" 2>/dev/null || true
    sshd_pid=""
    echo >&2 "[ssh] sshd failed to start; retrying..."
    sleep 2
  done
}

start_sshd

/usr/local/bin/start.sh > >(tee -a /var/log/qwen38/server.log) 2>&1 &
server_pid=$!

while (( shutting_down == 0 )); do
  wait -n -p ended_pid "$sshd_pid" "$server_pid" 2>/dev/null || true
  if [[ "${ended_pid:-}" == "$server_pid" ]]; then
    echo >&2 "[serve] start.sh exited; keeping container alive for diagnostics"
    server_pid=""
    # Keep the container running so hostai can debug via ssh.
    sleep 1
  elif [[ "${ended_pid:-}" == "$sshd_pid" ]]; then
    echo >&2 "[ssh] sshd exited; restarting"
    sshd_pid=""
    start_sshd
  fi
  # Defensive checks in case wait -p is not available.
  if [[ -n "$sshd_pid" ]] && ! kill -0 "$sshd_pid" 2>/dev/null; then
    sshd_pid=""
    start_sshd
  fi
  if [[ -n "$server_pid" ]] && ! kill -0 "$server_pid" 2>/dev/null; then
    echo >&2 "[serve] start.sh died; keeping container alive"
    server_pid=""
  fi
done
