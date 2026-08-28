#!/usr/bin/env bash
set -Eeuo pipefail

/usr/local/bin/qwen-init-ssh.sh

# Keep SSH infrastructure independent from the inference process. Both sshd and
# start.sh are children of this PID 1 supervisor. A model/server failure must
# never take SSH down, and an unexpected sshd exit is restarted in-place rather
# than relying on Vast to restart the whole container.
mkdir -p /var/log/qwen38 /run/qwen38
rm -f /run/qwen38/start.exitcode

esshd_pid=""
server_pid=""
shutting_down=0
SSH_RESTART_DELAY_SECONDS="${SSH_RESTART_DELAY_SECONDS:-2}"

if ! [[ "$SSH_RESTART_DELAY_SECONDS" =~ ^[0-9]+$ ]]; then
  echo >&2 "ERROR: SSH_RESTART_DELAY_SECONDS must be a non-negative integer"
  exit 2
fi

stop_child() {
  local pid="${1:-}"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill -TERM "$pid" >/dev/null 2>&1 || true
  fi
  wait "$pid" >/dev/null 2>&1 || true
}

shutdown() {
  local rc="${1:-143}"
  (( shutting_down == 0 )) || return 0
  shutting_down=1
  trap - TERM INT
  echo "[entrypoint] shutting down..."
  stop_child "$server_pid"
  server_pid=""
  stop_child "$sshd_pid"
  sshd_pid=""
  exit "$rc"
}
trap 'shutdown 143' TERM
trap 'shutdown 130' INT

start_sshd() {
  local rc=0
  while (( shutting_down == 0 )); do
    # -D keeps sshd in the foreground from sshd's perspective so PID 1 can
    # supervise the exact daemon process. -e sends diagnostics to stderr; they
    # are appended to a persistent per-container log for post-mortem access.
    /usr/sbin/sshd -D -e >>/var/log/qwen38/sshd.log 2>&1 &
    sshd_pid=$!

    # Catch immediate configuration/key/bind failures before advertising SSH as
    # ready. A normally running daemon survives this short check.
    sleep 0.25
    if kill -0 "$sshd_pid" >/dev/null 2>&1; then
      echo "[ssh] sshd ready on 0.0.0.0:22 (pid=$sshd_pid)"
      return 0
    fi

    set +e
    wait "$sshd_pid"
    rc=$?
    set -e
    sshd_pid=""
    echo >&2 "[ssh] ERROR: sshd exited during startup with rc=$rc; retrying in ${SSH_RESTART_DELAY_SECONDS}s"
    tail -n 30 /var/log/qwen38/sshd.log >&2 2>/dev/null || true
    sleep "$SSH_RESTART_DELAY_SECONDS"
  done
  return 1
}

start_sshd

/usr/local/bin/start.sh > >(tee -a /var/log/qwen38/server.log) 2>&1 &
server_pid=$!

while (( shutting_down == 0 )); do
  wait_targets=("$sshd_pid")
  if [[ -n "$server_pid" ]]; then
    wait_targets+=("$server_pid")
  fi

  ended_pid=""
  set +e
  wait -n -p ended_pid "${wait_targets[@]}"
  child_rc=$?
  set -e

  # A trapped TERM/INT can interrupt wait. The trap normally exits directly,
  # but keep this guard so we never attempt a restart during shutdown.
  (( shutting_down == 0 )) || break

  if [[ -n "$server_pid" && "$ended_pid" == "$server_pid" ]]; then
    printf '%s\n' "$child_rc" > /run/qwen38/start.exitcode
    chmod 0644 /run/qwen38/start.exitcode
    server_pid=""
    echo >&2 "[serve] ERROR: start.sh exited with rc=${child_rc}; keeping container alive for SSH diagnostics"
    echo >&2 "[serve] inspect /var/log/qwen38/server.log and /run/qwen38/start.exitcode"
    continue
  fi

  if [[ "$ended_pid" == "$sshd_pid" ]]; then
    echo >&2 "[ssh] WARNING: sshd exited unexpectedly with rc=${child_rc}; restarting sshd in ${SSH_RESTART_DELAY_SECONDS}s"
    sshd_pid=""
    sleep "$SSH_RESTART_DELAY_SECONDS"
    start_sshd
    continue
  fi

  # Defensive fallback for shells where wait -p cannot identify the child.
  # Reconcile both known processes rather than allowing PID 1 to exit.
  if [[ -n "$sshd_pid" ]] && ! kill -0 "$sshd_pid" >/dev/null 2>&1; then
    wait "$sshd_pid" >/dev/null 2>&1 || true
    sshd_pid=""
    echo >&2 "[ssh] WARNING: sshd disappeared; restarting sshd in ${SSH_RESTART_DELAY_SECONDS}s"
    sleep "$SSH_RESTART_DELAY_SECONDS"
    start_sshd
  fi

  if [[ -n "$server_pid" ]] && ! kill -0 "$server_pid" >/dev/null 2>&1; then
    set +e
    wait "$server_pid"
    child_rc=$?
    set -e
    printf '%s\n' "$child_rc" > /run/qwen38/start.exitcode
    chmod 0644 /run/qwen38/start.exitcode
    server_pid=""
    echo >&2 "[serve] ERROR: start.sh exited with rc=${child_rc}; keeping container alive for SSH diagnostics"
  fi
done
