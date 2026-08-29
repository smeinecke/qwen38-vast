#!/usr/bin/env bash
# Shared Vast/SSH helpers. Source this file after ROOT_DIR/STATE_FILE/KNOWN_HOSTS
# have been defined. Functions deliberately use only standard tools already
# required by the qwen-* scripts.

# Source a .env file without overwriting variables that are already set in the
# current environment. This lets users override .env values with command-line
# env vars such as MAX_DPH=0.22 ./qwen-up ....
qwen_source_env() {
  local env_file="${1:-$ROOT_DIR/.env}"
  [[ -f "$env_file" ]] || return 0
  local line key val
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue
    [[ "$line" == *"="* ]] || continue
    key="${line%%=*}"
    val="${line#*=}"
    # trim leading/trailing whitespace from key
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    # optional `export ` prefix
    if [[ "$key" == export* ]]; then
      key="${key#export}"
      key="${key#"${key%%[![:space:]]*}"}"
      key="${key%"${key##*[![:space:]]}"}"
    fi
    [[ -z "$key" ]] && continue
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    # preserve existing environment (allows CLI override)
    [[ -n "${!key+set}" ]] && continue
    export "$key=$val"
  done < "$env_file"
}

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

# Disposable Vast containers can change host keys (e.g. restart/migration). Remove
# any stale entries for this host:port so StrictHostKeyChecking=accept-new can
# write the current key instead of failing on a mismatch. This only affects the
# per-project known_hosts file, not the user's global one.
qwen_ssh_sanitize_known_hosts() {
  local host="$1" port="${2:-22}"
  [[ -f "$KNOWN_HOSTS" ]] || return 0
  ssh-keygen -R "[${host}]:${port}" -f "$KNOWN_HOSTS" >/dev/null 2>&1 || true
  ssh-keygen -R "${host}" -f "$KNOWN_HOSTS" >/dev/null 2>&1 || true
}

# Stop the remote llama.cpp/llama-server process on a Vast host.  Best-effort:
# the destroy step will hard-kill the container afterwards, but this ensures
# the slot state cannot be reused if Vast takes a while to drop the instance.
qwen_stop_remote_model() {
  local ssh_url="${1:-}"
  local ssh_user ssh_host ssh_port
  [[ -n "$ssh_url" ]] || return 0

  IFS=' ' read -r ssh_user ssh_host ssh_port < <(python3 - "$ssh_url" <<'PY'
from urllib.parse import urlparse
import sys
u = urlparse(sys.argv[1])
print(u.username or "root", u.hostname or "", u.port or 22)
PY
  )
  [[ -n "$ssh_host" ]] || return 0

  local -a ssh_opts
  mapfile -d '' -t ssh_opts < <(qwen_ssh_opts "$ssh_port")
  ssh -n "${ssh_opts[@]}" "${ssh_user}@${ssh_host}" \
    'pkill -TERM llama-server 2>/dev/null || true; sleep 2; pkill -KILL llama-server 2>/dev/null || true' \
    >/dev/null 2>&1 || true
}

# Fetch remote startup status and any new server-log bytes since last_bytes.
# Outputs:
#   __QWEN_START_EXIT__:<rc or empty>
#   __QWEN_LOG_BYTES__:<current byte count>
#   <base64 of new log bytes (no trailing newline)>
# This is used by qwen-up to watch the model download / llama-server load and
# to detect a failing start.sh without issuing a separate SSH call each loop.
# Using bytes instead of lines makes CR-based progress bars (tqdm, hf download)
# stream correctly. Base64 prevents bash command substitution from stripping
# trailing CR/newline bytes.
qwen_remote_startup_status() {
  local ssh_url="${1:-}"
  local last_bytes="${2:-0}"
  local count_only="${3:-}"
  local ssh_user ssh_host ssh_port
  [[ -n "$ssh_url" ]] || return 0

  IFS=' ' read -r ssh_user ssh_host ssh_port < <(python3 - "$ssh_url" <<'PY'
from urllib.parse import urlparse
import sys
u = urlparse(sys.argv[1])
print(u.username or "root", u.hostname or "", u.port or 22)
PY
  )
  [[ -n "$ssh_host" ]] || return 0

  local -a ssh_opts
  local output
  mapfile -d '' -t ssh_opts < <(qwen_ssh_opts "$ssh_port")
  output="$(
    ssh "${ssh_opts[@]}" "${ssh_user}@${ssh_host}" bash -s -- "$last_bytes" "$count_only" <<'REMOTE' 2>/dev/null
set +e
last_bytes="$1"
count_only="$2"
start_exit_rc=$(cat /run/qwen38/start.exitcode 2>/dev/null || true)
log_file="/var/log/qwen38/server.log"
if [[ -s "$log_file" ]]; then
  total_bytes=$(stat -c %s "$log_file" 2>/dev/null || echo 0)
else
  total_bytes=0
fi
printf '%s\n' "__QWEN_START_EXIT__:${start_exit_rc}"
printf '%s\n' "__QWEN_LOG_BYTES__:${total_bytes}"
if [[ -z "$count_only" && "$total_bytes" =~ ^[0-9]+$ && "$last_bytes" =~ ^[0-9]+$ ]]; then
  if (( total_bytes >= last_bytes )); then
    new_bytes=$((total_bytes - last_bytes))
    if (( new_bytes > 0 )); then
      # Output base64 without newlines and do not emit a trailing newline so
      # command substitution does not mangle the bytes (it only strips \n).
      tail -c "$new_bytes" "$log_file" | base64 -w0
    fi
  else
    # Log was truncated/rotated; send the whole file.
    tail -c +1 "$log_file" | base64 -w0
  fi
fi
printf ''
REMOTE
  )" || true
  printf '%s\n' "$output"
}

qwen_api_scheme_and_ca() {
  local tls_ca
  [[ -f "$STATE_FILE" ]] || return 1
  tls_ca="$(jq -r '.tls_ca // empty' "$STATE_FILE")"
  if [[ -n "$tls_ca" && -f "$tls_ca" ]]; then
    printf '%s\t%s\n' "https" "$tls_ca"
  else
    printf '%s\t%s\n' "http" ""
  fi
}

qwen_api_healthy() {
  local local_port api_key scheme ca
  [[ -f "$STATE_FILE" ]] || return 1
  local_port="$(jq -r '.local_port // empty' "$STATE_FILE")"
  local_port="${local_port:-${LOCAL_PORT:-18080}}"
  api_key="$(jq -r '.api_key // empty' "$STATE_FILE")"
  [[ -n "$api_key" ]] || return 1
  IFS=$'\t' read -r scheme ca < <(qwen_api_scheme_and_ca)
  local ca_opt=()
  [[ -n "$ca" ]] && ca_opt=("--cacert" "$ca")
  curl -fsS --connect-timeout 2 --max-time 4 "${ca_opt[@]}" \
    -H "Authorization: Bearer $api_key" \
    "${scheme}://127.0.0.1:${local_port}/health" >/dev/null 2>&1
}

# Build the base (scheme + host:port) and the /v1 endpoint URLs from the
# persisted state. Scripts should call this instead of hard-coding http://.
qwen_api_urls() {
  local local_port scheme ca
  [[ -f "$STATE_FILE" ]] || return 1
  local_port="$(jq -r '.local_port // empty' "$STATE_FILE")"
  local_port="${local_port:-${LOCAL_PORT:-18080}}"
  IFS=$'\t' read -r scheme ca < <(qwen_api_scheme_and_ca)
  printf '%s\t%s\t%s\n' "${scheme}://127.0.0.1:${local_port}" "${scheme}://127.0.0.1:${local_port}/v1" "${ca}"
}

# Return extra curl arguments for the current scheme/CA.
qwen_api_curl_ca() {
  local ca
  IFS=$'\t' read -r _ _ ca < <(qwen_api_urls)
  if [[ -n "$ca" && -f "$ca" ]]; then
    printf '%s\n' "--cacert" "$ca"
  fi
}

qwen_tls_setup() {
  local tls_dir="${1:-}"
  [[ -n "$tls_dir" ]] || { echo >&2 "ERROR: qwen_tls_setup requires a tls_dir"; return 1; }
  if [[ -f "$tls_dir/ca.crt" && -f "$tls_dir/server.crt" && -f "$tls_dir/server.key" ]]; then
    return 0
  fi
  install -d -m 700 "$tls_dir"
  if ! openssl req -x509 -newkey rsa:2048 -keyout "$tls_dir/server.key" -out "$tls_dir/server.crt" \
       -days 1 -nodes -subj "/CN=qwen38" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" 2>/dev/null; then
    echo >&2 "ERROR: failed to generate local TLS certificate. Is OpenSSL >= 1.1.1 installed?"
    return 1
  fi
  chmod 600 "$tls_dir/server.key"
  cp "$tls_dir/server.crt" "$tls_dir/ca.crt"
}

# Deliver the generated TLS certificate and private key to the remote container's
# tmpfs over SSH. The key never lives in Docker environment variables or in the
# container's persistent disk.
qwen_tls_deliver() {
  local ssh_url="${1:-}" tls_dir="${2:-}"
  local ssh_user ssh_host ssh_port
  local -a ssh_opts
  [[ -n "$ssh_url" && -n "$tls_dir" ]] || { echo >&2 "ERROR: qwen_tls_deliver requires ssh_url and tls_dir"; return 1; }

  IFS=' ' read -r ssh_user ssh_host ssh_port < <(python3 - "$ssh_url" <<'PY'
from urllib.parse import urlparse
import sys
u = urlparse(sys.argv[1])
print(u.username or "root", u.hostname or "", u.port or 22)
PY
  )
  [[ -n "$ssh_host" ]] || return 1
  mapfile -d '' -t ssh_opts < <(qwen_ssh_opts "$ssh_port")

  if [[ ! -f "$tls_dir/server.crt" || ! -f "$tls_dir/server.key" ]]; then
    echo >&2 "ERROR: qwen_tls_deliver: missing $tls_dir/server.crt or $tls_dir/server.key"
    return 1
  fi

  echo "[secure] delivering TLS certificate to remote tmpfs..."
  ssh "${ssh_opts[@]}" "${ssh_user}@${ssh_host}" \
    'install -d -m 700 /dev/shm/qwen38/certs && cat > /dev/shm/qwen38/certs/server.crt' < "$tls_dir/server.crt"
  ssh "${ssh_opts[@]}" "${ssh_user}@${ssh_host}" \
    'cat > /dev/shm/qwen38/certs/server.key' < "$tls_dir/server.key"
  ssh "${ssh_opts[@]}" "${ssh_user}@${ssh_host}" \
    'chmod 600 /dev/shm/qwen38/certs/server.key'
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
  qwen_ssh_sanitize_known_hosts "$ssh_host" "$ssh_port"
  mkdir -p "$(dirname "$KNOWN_HOSTS")"
  local tunnel_log="${TUNNEL_LOG:-$(dirname "$STATE_FILE")/tunnel.log}"
  : > "$tunnel_log"

  local remote_dest
  if [[ "$(jq -r '.unsecure // "0"' "$STATE_FILE")" == "1" ]]; then
    remote_dest="127.0.0.1:8080"
  else
    remote_dest="/dev/shm/qwen38/llama.sock"
  fi

  nohup ssh "${opts[@]}" \
    -o ExitOnForwardFailure=yes \
    -N -L "${local_port}:${remote_dest}" \
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

# Try to show a desktop notification using whatever tool is available on the
# current OS. Designed for background monitors that may not have a TTY visible.
qwen_desktop_notify() {
  local title="${1:-qwen38}"
  local body="${2:-}"
  [[ -z "$body" ]] && return 0

  # Linux/libnotify via D-Bus. Nohup'd daemons often lose DBUS_SESSION_BUS_ADDRESS,
  # so fall back to the common systemd user bus path.
  if command -v notify-send >/dev/null 2>&1; then
    if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ]]; then
      local dbus_path="/run/user/$(id - u 2>/dev/null)/bus"
      [[ -S "$dbus_path" ]] && export DBUS_SESSION_BUS_ADDRESS="unix:path=$dbus_path"
    fi
    notify-send "$title" "$body" >/dev/null 2>&1 && return 0
  fi

  # KDE/Plasma.
  if command -v kdialog >/dev/null 2>&1; then
    kdialog --passivepopup "$body" 10 --title "$title" >/dev/null 2>&1 && return 0
  fi

  # macOS.
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display notification \"$body\" with title \"$title\"" >/dev/null 2>&1 && return 0
  fi

  # Windows (WSL).
  if command -v powershell.exe >/dev/null 2>&1; then
    powershell.exe -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show('$body', '$title')" >/dev/null 2>&1 && return 0
  fi

  # Last-ditch: terminal bell and a message to every local login.
  printf '\a'
  if command -v wall >/dev/null 2>&1; then
    printf '%s\n' "$title: $body" | wall -n >/dev/null 2>&1 || true
  fi
  return 1
}

# Run an arbitrary vastai command with a wall-clock timeout. Kills the whole
# process group on timeout and returns the captured stdout.
_qwen_vast_action_timeout() {
  local deadline="$1"
  shift
  if [[ ! "$deadline" =~ ^[0-9]+$ ]] || (( deadline < 5 )); then
    deadline=45
  fi
  python3 - "$deadline" "$@" <<'PYACTION'
import os, signal, subprocess, sys
deadline = int(sys.argv[1])
cmd = sys.argv[2:]
try:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
except Exception as exc:
    print(f"failed to start {' '.join(cmd)}: {exc}")
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
    print(f"vastai {' '.join(cmd[1:])} timed out after {deadline}s")
    raise SystemExit(124)
PYACTION
}

# Pause (stop) a Vast instance while preserving its disk. Returns via globals:
#   QWEN_PAUSE_OUTCOME = paused | not_found | timeout | failed
#   QWEN_PAUSE_OUTPUT  = captured CLI stdout/stderr
#   QWEN_PAUSE_RC      = CLI/timeout exit code
qwen_pause_instance() {
  local instance_id="$1"
  local deadline="${2:-${QWEN_PAUSE_TIMEOUT_SECONDS:-60}}"

  QWEN_PAUSE_OUTCOME="failed"
  QWEN_PAUSE_OUTPUT=""
  QWEN_PAUSE_RC=0

  if QWEN_PAUSE_OUTPUT="$(_qwen_vast_action_timeout "$deadline" vastai stop instance "$instance_id" --raw)"; then
    QWEN_PAUSE_RC=0
  else
    QWEN_PAUSE_RC=$?
  fi

  if grep -Eiq 'Instance [0-9]+ not found|instance[^[:alnum:]]+not found|not found' <<<"$QWEN_PAUSE_OUTPUT"; then
    QWEN_PAUSE_OUTCOME="not_found"
    return 0
  fi

  if (( QWEN_PAUSE_RC == 124 )); then
    QWEN_PAUSE_OUTCOME="timeout"
    return 124
  fi

  if (( QWEN_PAUSE_RC != 0 )) || grep -Eq '"error"[[:space:]]*:[[:space:]]*true' <<<"$QWEN_PAUSE_OUTPUT"; then
    QWEN_PAUSE_OUTCOME="failed"
    return "${QWEN_PAUSE_RC:-1}"
  fi

  QWEN_PAUSE_OUTCOME="paused"
  return 0
}

# Start (resume) a stopped Vast instance. Returns via globals:
#   QWEN_START_OUTCOME = started | not_found | timeout | failed
#   QWEN_START_OUTPUT  = captured CLI stdout/stderr
#   QWEN_START_RC      = CLI/timeout exit code
qwen_start_instance() {
  local instance_id="$1"
  local deadline="${2:-${QWEN_START_TIMEOUT_SECONDS:-120}}"

  QWEN_START_OUTCOME="failed"
  QWEN_START_OUTPUT=""
  QWEN_START_RC=0

  if QWEN_START_OUTPUT="$(_qwen_vast_action_timeout "$deadline" vastai start instance "$instance_id" --raw)"; then
    QWEN_START_RC=0
  else
    QWEN_START_RC=$?
  fi

  if grep -Eiq 'Instance [0-9]+ not found|instance[^[:alnum:]]+not found|not found' <<<"$QWEN_START_OUTPUT"; then
    QWEN_START_OUTCOME="not_found"
    return 0
  fi

  if (( QWEN_START_RC == 124 )); then
    QWEN_START_OUTCOME="timeout"
    return 124
  fi

  if (( QWEN_START_RC != 0 )) || grep -Eq '"error"[[:space:]]*:[[:space:]]*true' <<<"$QWEN_START_OUTPUT"; then
    QWEN_START_OUTCOME="failed"
    return "${QWEN_START_RC:-1}"
  fi

  QWEN_START_OUTCOME="started"
  return 0
}

# Destroy a Vast instance non-interactively and with a hard deadline.
#
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