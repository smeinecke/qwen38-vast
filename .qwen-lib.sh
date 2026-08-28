#!/usr/bin/env bash
# Shared Vast/SSH helpers. Source this file after ROOT_DIR/STATE_FILE/KNOWN_HOSTS
# have been defined. Functions deliberately use only standard tools already
# required by the qwen-* scripts.

qwen_parse_ssh_text() {
  python3 -c '
import json, re, sys
from urllib.parse import urlparse
text = sys.stdin.read().strip()

# Some CLI versions return JSON strings/objects, others print a command or URL.
def strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)

candidates = [text]
try:
    candidates = list(strings(json.loads(text))) + candidates
except Exception:
    pass

for s in candidates:
    # ssh://root@host:1234
    m = re.search(r"ssh://([^@\s]+)@([^:/\s]+):(\d+)", s)
    if m:
        user, host, port = m.group(1), m.group(2), int(m.group(3))
        print(f"{user}\t{host}\t{port}\tssh://{user}@{host}:{port}")
        raise SystemExit(0)

    # ssh -p 1234 root@host  (options may occur before/after)
    port_m = re.search(r"(?:^|\s)-p\s+(\d+)(?:\s|$)", s)
    target_m = re.search(r"(?:^|\s)([A-Za-z0-9._-]+)@([A-Za-z0-9._:-]+)(?:\s|$)", s)
    if port_m and target_m:
        user, host, port = target_m.group(1), target_m.group(2), int(port_m.group(1))
        host = host.strip("[]")
        print(f"{user}\t{host}\t{port}\tssh://{user}@{host}:{port}")
        raise SystemExit(0)

raise SystemExit(1)
'
}

qwen_endpoint_from_instance_json() {
  local instance_json="$1"
  local host port user

  # Compatibility with old Vast SSH-mode instances created by v6 and earlier.
  host="$(jq -r '.ssh_host // empty' <<<"$instance_json" 2>/dev/null || true)"
  port="$(jq -r '.ssh_port // empty' <<<"$instance_json" 2>/dev/null || true)"
  user="$(jq -r '.ssh_user // "root"' <<<"$instance_json" 2>/dev/null || echo root)"
  if [[ -n "$host" && "$host" != "null" && "$port" =~ ^[0-9]+$ && "$port" -gt 0 ]]; then
    printf '%s\t%s\t%s\tssh://%s@%s:%s\n' "$user" "$host" "$port" "$user" "$host" "$port"
    return 0
  fi

  # v7 self-manages sshd in normal entrypoint/args mode. Vast therefore does not
  # publish ssh_host/ssh_port; instead port 22 is a normal Docker port mapping.
  host="$(jq -r '.public_ipaddr // .public_ip // empty' <<<"$instance_json" 2>/dev/null || true)"
  port="$(jq -r '.ports["22/tcp"][0].HostPort // empty' <<<"$instance_json" 2>/dev/null || true)"
  user="root"
  if [[ -n "$host" && "$host" != "null" && "$port" =~ ^[0-9]+$ && "$port" -gt 0 ]]; then
    printf '%s\t%s\t%s\tssh://%s@%s:%s\n' "$user" "$host" "$port" "$user" "$host" "$port"
    return 0
  fi
  return 1
}

qwen_resolve_ssh_endpoint() {
  local instance_id="$1"
  local instance_json="${2:-}"
  local out raw

  if [[ -z "$instance_json" ]]; then
    instance_json="$(vastai show instance "$instance_id" --raw 2>/dev/null || true)"
  fi
  if [[ -n "$instance_json" ]] && out="$(qwen_endpoint_from_instance_json "$instance_json" 2>/dev/null)" && [[ -n "$out" ]]; then
    printf '%s\n' "$out"
    return 0
  fi

  raw="$(vastai ssh-url "$instance_id" 2>/dev/null || true)"
  if [[ -n "$raw" ]] && out="$(printf '%s\n' "$raw" | qwen_parse_ssh_text 2>/dev/null)" && [[ -n "$out" ]]; then
    printf '%s\n' "$out"
    return 0
  fi
  return 1
}

qwen_refresh_ssh_state() {
  local instance_id instance_json endpoint ssh_user ssh_host ssh_port ssh_url
  [[ -f "$STATE_FILE" ]] || return 1
  instance_id="$(jq -r '.instance_id // empty' "$STATE_FILE")"
  [[ -n "$instance_id" ]] || return 1
  instance_json="$(vastai show instance "$instance_id" --raw 2>/dev/null || true)"
  [[ -n "$instance_json" ]] || return 1
  endpoint="$(qwen_resolve_ssh_endpoint "$instance_id" "$instance_json" 2>/dev/null || true)"
  [[ -n "$endpoint" ]] || return 1
  IFS=$'\t' read -r ssh_user ssh_host ssh_port ssh_url <<<"$endpoint"
  [[ -n "$ssh_host" && "$ssh_port" =~ ^[0-9]+$ ]] || return 1

  jq --arg ssh_url "$ssh_url" --arg ssh_host "$ssh_host" --arg ssh_user "$ssh_user" --argjson ssh_port "$ssh_port" \
    '. + {ssh_url:$ssh_url,ssh_host:$ssh_host,ssh_port:$ssh_port,ssh_user:$ssh_user}' \
    "$STATE_FILE" > "$STATE_FILE.tmp" && mv "$STATE_FILE.tmp" "$STATE_FILE"
  chmod 600 "$STATE_FILE" 2>/dev/null || true
  printf '%s\n' "$endpoint"
}

qwen_ssh_opts() {
  local port="$1"
  printf '%s\0' \
    -o BatchMode=yes \
    -o ConnectTimeout=8 \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3 \
    -o StrictHostKeyChecking=accept-new \
    -o "UserKnownHostsFile=$KNOWN_HOSTS" \
    -p "$port"
}

qwen_api_healthy() {
  local local_port api_key
  [[ -f "$STATE_FILE" ]] || return 1
  local_port="$(jq -r '.local_port // empty' "$STATE_FILE")"
  local_port="${local_port:-${LOCAL_PORT:-18080}}"
  api_key="$(jq -r '.api_key // empty' "$STATE_FILE")"
  [[ -n "$api_key" ]] || return 1
  curl -fsS --connect-timeout 2 --max-time 4 \
    -H "Authorization: Bearer $api_key" \
    "http://127.0.0.1:${local_port}/health" >/dev/null 2>&1
}

qwen_port_is_free() {
  local port="$1"
  python3 - "$port" <<'PYPORT'
import socket, sys
port = int(sys.argv[1])
s = socket.socket()
try:
    s.bind(("127.0.0.1", port))
except OSError:
    raise SystemExit(1)
finally:
    s.close()
PYPORT
}

qwen_find_free_port() {
  local start_port="${1:-18080}"
  local count="${2:-100}"
  python3 - "$start_port" "$count" <<'PYPORT'
import socket, sys
start = int(sys.argv[1]); count = int(sys.argv[2])
for port in range(start, min(65536, start + count)):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
    except OSError:
        s.close(); continue
    s.close()
    print(port)
    raise SystemExit(0)
raise SystemExit(1)
PYPORT
}

qwen_tunnel_lock_acquire() {
  local lock_dir="${STATE_DIR:-$(dirname "$STATE_FILE")}/tunnel.lock.d"
  local deadline=$(( $(date +%s) + 15 )) pid
  mkdir -p "${STATE_DIR:-$(dirname "$STATE_FILE")}" 2>/dev/null || true
  while ! mkdir "$lock_dir" 2>/dev/null; do
    pid="$(cat "$lock_dir/pid" 2>/dev/null || true)"
    if [[ "$pid" =~ ^[0-9]+$ ]] && ! kill -0 "$pid" >/dev/null 2>&1; then
      rm -rf "$lock_dir" 2>/dev/null || true
      continue
    fi
    if (( $(date +%s) >= deadline )); then
      return 1
    fi
    sleep 0.2
  done
  printf '%s\n' "$$" > "$lock_dir/pid"
  QWEN_TUNNEL_LOCK_DIR="$lock_dir"
  return 0
}

qwen_tunnel_lock_release() {
  if [[ -n "${QWEN_TUNNEL_LOCK_DIR:-}" ]]; then
    rm -rf "$QWEN_TUNNEL_LOCK_DIR" 2>/dev/null || true
    QWEN_TUNNEL_LOCK_DIR=""
  fi
}

_qwen_ensure_tunnel_locked() {
  local quiet="${1:-0}"
  local endpoint ssh_user ssh_host ssh_port ssh_url local_port tunnel_pid api_key auto_port new_port
  local -a opts
  [[ -f "$STATE_FILE" ]] || return 1

  local_port="$(jq -r '.local_port // empty' "$STATE_FILE")"
  local_port="${local_port:-${LOCAL_PORT:-18080}}"
  tunnel_pid="$(jq -r '.tunnel_pid // empty' "$STATE_FILE")"
  api_key="$(jq -r '.api_key // empty' "$STATE_FILE")"
  auto_port="${LOCAL_PORT_AUTO:-1}"

  # Another qwen command (most commonly qwen-status) may have created the
  # tunnel while qwen-up was provisioning. Reuse that managed tunnel instead of
  # racing it for the same local port.
  if [[ -n "$tunnel_pid" ]] && kill -0 "$tunnel_pid" >/dev/null 2>&1; then
    return 0
  fi

  # A manually-created tunnel may already own the port. If it reaches the API,
  # use it rather than trying to steal the port.
  if qwen_api_healthy; then
    jq --argjson local_port "$local_port" 'del(.tunnel_pid) + {local_port:$local_port}' \
      "$STATE_FILE" > "$STATE_FILE.tmp" && mv "$STATE_FILE.tmp" "$STATE_FILE"
    chmod 600 "$STATE_FILE" 2>/dev/null || true
    (( quiet == 1 )) || echo "[tunnel] API already reachable on localhost:$local_port (external/manual tunnel)."
    return 0
  fi

  endpoint="$(qwen_refresh_ssh_state 2>/dev/null || true)"
  [[ -n "$endpoint" ]] || { (( quiet == 1 )) || echo >&2 "[ssh] endpoint not available from Vast yet"; return 1; }
  IFS=$'\t' read -r ssh_user ssh_host ssh_port ssh_url <<<"$endpoint"

  # The port can become occupied after qwen-up's early preflight, e.g. when a
  # concurrently-run qwen-status self-heals the tunnel. Never blindly launch a
  # second ssh -L on the same port. If it is not a usable qwen API endpoint,
  # move to the next free local port by default and persist that decision.
  if ! qwen_port_is_free "$local_port"; then
    if [[ "$auto_port" == "1" ]]; then
      new_port="$(qwen_find_free_port "$(( local_port + 1 ))" 100 2>/dev/null || true)"
      [[ -n "$new_port" ]] || { (( quiet == 1 )) || echo >&2 "[tunnel] no free local port found after $local_port"; return 1; }
      (( quiet == 1 )) || echo >&2 "[tunnel] localhost:$local_port is occupied; switching to localhost:$new_port"
      local_port="$new_port"
      jq --argjson local_port "$local_port" 'del(.tunnel_pid) + {local_port:$local_port}' \
        "$STATE_FILE" > "$STATE_FILE.tmp" && mv "$STATE_FILE.tmp" "$STATE_FILE"
      chmod 600 "$STATE_FILE" 2>/dev/null || true
    else
      (( quiet == 1 )) || echo >&2 "[tunnel] localhost:$local_port is occupied and LOCAL_PORT_AUTO=0"
      return 1
    fi
  fi

  mapfile -d '' -t opts < <(qwen_ssh_opts "$ssh_port")
  mkdir -p "$(dirname "$KNOWN_HOSTS")"
  local tunnel_log="${TUNNEL_LOG:-$(dirname "$STATE_FILE")/tunnel.log}"
  : > "$tunnel_log"
  nohup ssh "${opts[@]}" \
    -o ExitOnForwardFailure=yes \
    -N -L "${local_port}:127.0.0.1:8080" \
    "${ssh_user}@${ssh_host}" >"$tunnel_log" 2>&1 &
  tunnel_pid=$!
  sleep 2
  if ! kill -0 "$tunnel_pid" >/dev/null 2>&1; then
    (( quiet == 1 )) || { echo >&2 "[tunnel] SSH tunnel failed:"; tail -n 20 "$tunnel_log" >&2 2>/dev/null || true; }
    return 1
  fi

  jq --arg ssh_url "$ssh_url" --arg ssh_host "$ssh_host" --arg ssh_user "$ssh_user" \
     --argjson ssh_port "$ssh_port" --argjson local_port "$local_port" --argjson tunnel_pid "$tunnel_pid" \
    '. + {ssh_url:$ssh_url,ssh_host:$ssh_host,ssh_port:$ssh_port,ssh_user:$ssh_user,local_port:$local_port,tunnel_pid:$tunnel_pid}' \
    "$STATE_FILE" > "$STATE_FILE.tmp" && mv "$STATE_FILE.tmp" "$STATE_FILE"
  chmod 600 "$STATE_FILE" 2>/dev/null || true
  (( quiet == 1 )) || echo "[tunnel] connected ${ssh_user}@${ssh_host}:${ssh_port} -> localhost:${local_port} (pid $tunnel_pid)"
  return 0
}

# Destroy a Vast instance non-interactively and with a hard deadline.
#
# Vast's CLI asks for irreversible-action confirmation unless -y is supplied.
# qwen-down captures CLI output, so without -y that prompt is invisible and the
# process appears to hang forever. Keep this helper as the only destroy path so
# shutdown and failure cleanup cannot regress independently.
#
# Results are returned via globals:
#   QWEN_DESTROY_OUTCOME = destroyed | already_absent | timeout | failed
#   QWEN_DESTROY_OUTPUT  = captured CLI stdout/stderr
#   QWEN_DESTROY_RC      = CLI/timeout exit code
qwen_destroy_instance() {
  local instance_id="$1"
  local deadline="${2:-${QWEN_DESTROY_TIMEOUT_SECONDS:-45}}"

  QWEN_DESTROY_OUTCOME="failed"
  QWEN_DESTROY_OUTPUT=""
  QWEN_DESTROY_RC=0

  if [[ ! "$deadline" =~ ^[0-9]+$ ]] || (( deadline < 5 )); then
    deadline=45
  fi

  # Use Python for the deadline instead of coreutils timeout. start_new_session
  # lets us terminate the whole CLI process group, so even a wedged child cannot
  # keep the captured stdout pipe open and make qwen-down appear hung.
  if QWEN_DESTROY_OUTPUT="$(python3 - "$deadline" "$instance_id" <<'PYDESTROY'
import os
import signal
import subprocess
import sys

deadline = int(sys.argv[1])
instance_id = sys.argv[2]
cmd = ["vastai", "destroy", "instance", instance_id, "-y", "--raw"]

try:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
except Exception as exc:
    print(f"failed to start vastai destroy: {exc}")
    raise SystemExit(127)

try:
    output, _ = proc.communicate(timeout=deadline)
    if output:
        sys.stdout.write(output)
    raise SystemExit(proc.returncode)
except subprocess.TimeoutExpired as exc:
    partial = exc.output or ""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        rest, _ = proc.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        rest, _ = proc.communicate()
    if isinstance(partial, bytes):
        partial = partial.decode(errors="replace")
    if isinstance(rest, bytes):
        rest = rest.decode(errors="replace")
    if partial:
        sys.stdout.write(partial)
    if rest and rest != partial:
        sys.stdout.write(rest)
    print(f"vastai destroy timed out after {deadline}s")
    raise SystemExit(124)
PYDESTROY
)"; then
    QWEN_DESTROY_RC=0
  else
    QWEN_DESTROY_RC=$?
  fi

  if grep -Eq '"status_code"[[:space:]]*:[[:space:]]*404|Instance [0-9]+ not found|instance[^[:alnum:]]+not found' <<<"$QWEN_DESTROY_OUTPUT"; then
    QWEN_DESTROY_OUTCOME="already_absent"
    return 0
  fi

  if (( QWEN_DESTROY_RC == 124 )); then
    QWEN_DESTROY_OUTCOME="timeout"
    return 124
  fi

  if (( QWEN_DESTROY_RC != 0 )) || grep -Eq '"error"[[:space:]]*:[[:space:]]*true' <<<"$QWEN_DESTROY_OUTPUT"; then
    QWEN_DESTROY_OUTCOME="failed"
    return "${QWEN_DESTROY_RC:-1}"
  fi

  QWEN_DESTROY_OUTCOME="destroyed"
  return 0
}


qwen_ensure_tunnel() {
  local quiet="${1:-0}" rc=0
  if ! qwen_tunnel_lock_acquire; then
    (( quiet == 1 )) || echo >&2 "[tunnel] could not acquire tunnel lock"
    return 1
  fi
  _qwen_ensure_tunnel_locked "$quiet" || rc=$?
  qwen_tunnel_lock_release
  return "$rc"
}