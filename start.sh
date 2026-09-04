#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

HOSTAI_UNSECURE="${HOSTAI_UNSECURE:-0}"
HOSTAI_TOKENIZED_ONLY="${HOSTAI_TOKENIZED_ONLY:-0}"
HOSTAI_TMPFS_BASE="${HOSTAI_TMPFS_BASE:-/dev/shm/qwen38}"
HOSTAI_TMP_DIR="${HOSTAI_TMP_DIR:-$HOSTAI_TMPFS_BASE/tmp}"
HOSTAI_LOG_DIR="${HOSTAI_LOG_DIR:-$HOSTAI_TMPFS_BASE/log}"
HOSTAI_CERTS_DIR="${HOSTAI_CERTS_DIR:-$HOSTAI_TMPFS_BASE/certs}"
HOSTAI_TLS_WAIT_TIMEOUT="${HOSTAI_TLS_WAIT_TIMEOUT:-600}"

HF_REPO="${HF_REPO:-HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF}"
HF_REVISION="${HF_REVISION:-993a5971fda8f30dd1b7eb2654792ba4415c7460}"
MODEL="${MODEL:-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf}"
DRAFT="${DRAFT:-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf}"
MODEL_DIR="${MODEL_DIR:-/models}"
CTX_SIZE="${CTX_SIZE:-65536}"
DEPTH="${DEPTH:-3}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"
USE_FASTMTP="${USE_FASTMTP:-1}"
HOSTAI_PROFILE="${HOSTAI_PROFILE:-custom}"

# Slot cache location. The client may suggest a path via HOSTAI_SLOT_CACHE_LOCAL_DIR,
# but if that points to /dev/shm and the actual free space is below the minimum,
# fall back to disk so llama-server can save large slot states reliably.
SLOT_SAVE_PATH_DEFAULT="${SLOT_SAVE_PATH:-/var/lib/qwen38/slots}"
SLOT_SAVE_PATH="${HOSTAI_SLOT_CACHE_LOCAL_DIR:-$SLOT_SAVE_PATH_DEFAULT}"
if [[ "$SLOT_SAVE_PATH" == /dev/shm/* ]]; then
  slot_min_gb="${HOSTAI_SLOT_CACHE_MIN_GB:-${HOSTAI_SHM_MIN_GB:-32}}"
  if [[ "$slot_min_gb" =~ ^[0-9]+$ ]]; then
    shm_avail=$(df -P -B1 /dev/shm | awk 'NR==2{print $4}')
    min_bytes=$((slot_min_gb * 1024 * 1024 * 1024))
    if [[ "$shm_avail" -lt "$min_bytes" ]]; then
      echo "[cache] /dev/shm free=${shm_avail}B is below min=${min_bytes}B; using disk slot cache"
      SLOT_SAVE_PATH="/var/lib/qwen38/slots"
    fi
  fi
fi
CACHE_TYPE_K="${CACHE_TYPE_K:-}"
CACHE_TYPE_V="${CACHE_TYPE_V:-}"

if [[ -z "${LLAMA_API_KEY:-}" ]]; then
  echo >&2 "ERROR: LLAMA_API_KEY is required."
  exit 2
fi

if [[ "$HOSTAI_UNSECURE" == "1" && "$HOSTAI_TOKENIZED_ONLY" == "1" ]]; then
  echo >&2 "ERROR: HOSTAI_TOKENIZED_ONLY cannot be combined with HOSTAI_UNSECURE."
  exit 2
fi

if [[ "${HOSTAI_SLOT_CACHE_RCLONE:-0}" == "1" ]] && ! command -v rclone >/dev/null 2>&1; then
  echo "[rclone] installing rclone for slot-cache backend..."
  apt-get update -qq
  apt-get install -y --no-install-recommends rclone
fi

mkdir -p "$MODEL_DIR" "$SLOT_SAVE_PATH" "$HOSTAI_TMP_DIR" "$HOSTAI_LOG_DIR" "$HOSTAI_CERTS_DIR"
chmod 700 "$SLOT_SAVE_PATH"

if [[ "$HOSTAI_UNSECURE" != "1" ]]; then
  # Ensure the log/run paths used by other tools point to tmpfs as well.
  mkdir -p "$HOSTAI_TMPFS_BASE" "$HOSTAI_TMPFS_BASE/log" "$HOSTAI_TMPFS_BASE/run" \
           "$HOSTAI_TMPFS_BASE/ssh" "$HOSTAI_TMPFS_BASE/certs" "$HOSTAI_TMPFS_BASE/tmp"
  rm -rf /var/log/qwen38 /run/qwen38
  ln -sfn "$HOSTAI_TMPFS_BASE/log" /var/log/qwen38
  ln -sfn "$HOSTAI_TMPFS_BASE/run" /run/qwen38
  # Guard the socket and certs on tmpfs so only the container owner can reach
  # them, even though the parent /dev/shm is world-writable.
  chmod 700 "$HOSTAI_TMPFS_BASE"
fi

# Validate the executable before downloading tens of GB. A missing runtime
# library otherwise causes a crash only after model transfer and looks like an
# intermittent SSH/tunnel problem from the client side.
echo "[runtime] validating llama-server dependencies..."
LLAMA_VERSION_OUT="$(mktemp -p "$HOSTAI_TMP_DIR")"
LLAMA_VERSION_ERR="$(mktemp -p "$HOSTAI_TMP_DIR")"
trap 'rm -f "$LLAMA_VERSION_OUT" "$LLAMA_VERSION_ERR"' EXIT

if ! /usr/local/bin/llama-server --version >"$LLAMA_VERSION_OUT" 2>"$LLAMA_VERSION_ERR"; then
  cat "$LLAMA_VERSION_ERR" >&2 || true
  echo >&2 "[runtime] linked libraries:"
  ldd /usr/local/bin/llama-server >&2 || true
  echo >&2 "ERROR: llama-server runtime preflight failed; refusing model download."
  exit 70
fi
cat "$LLAMA_VERSION_OUT"

download_file() {
  local filename="$1"
  echo "[download] ${HF_REPO}/${filename}"
  hf download "$HF_REPO" "$filename" --revision "$HF_REVISION" --local-dir "$MODEL_DIR"
}

record_disk_usage() {
  local stage="$1"
  HOSTAI_DISK_STAGE="$stage" python3 - <<'PY'
import json
import os
import subprocess
import sys

log_dir = os.environ.get("HOSTAI_LOG_DIR", "/dev/shm/qwen38/log")
try:
    df = subprocess.check_output(["df", "-B1", "/"], text=True).strip().splitlines()[1].split()
    total_bytes = int(df[1])
    used_bytes = int(df[2])
    free_bytes = int(df[3])

    def du(path):
        if not os.path.exists(path):
            return 0
        try:
            return int(subprocess.check_output(["du", "-sb", path], text=True).split()[0])
        except Exception:
            return 0

    record = {
        "stage": os.environ.get("HOSTAI_DISK_STAGE", "unknown"),
        "total_bytes": total_bytes,
        "used_bytes": used_bytes,
        "free_bytes": free_bytes,
        "models_bytes": du("/models"),
        "slots_disk_bytes": du("/var/lib/qwen38/slots"),
        "slots_shm_bytes": du("/dev/shm/qwen38/slots"),
        "log_bytes": du(log_dir),
        "run_bytes": du("/run/qwen38"),
        "tmp_bytes": du("/dev/shm/qwen38/tmp"),
    }
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "disk-usage.log"), "a") as f:
        f.write(json.dumps(record) + "\n")
except Exception as e:
    print(f"[disk-usage] could not record {os.environ.get('HOSTAI_DISK_STAGE', 'unknown')}: {e}", file=sys.stderr)
PY
}

# The model repository is publicly downloadable at the time this image was
# prepared. HF_TOKEN is optional; huggingface_hub will automatically use it if
# supplied (for rate limits or if upstream access rules change later).

record_disk_usage "after-preflight"
download_file "$MODEL"
record_disk_usage "after-main-model"

server_args=(
  --model "$MODEL_DIR/$MODEL"
  --ctx-size "$CTX_SIZE"
  --parallel 1
  --batch-size "$BATCH_SIZE"
  --ubatch-size "$UBATCH_SIZE"
  --n-gpu-layers all
  --split-mode none
  --flash-attn on
  --no-mmap
  --temp 1.0
  --top-k 20
  --top-p 0.95
  --min-p 0
  --presence-penalty 0
  --repeat-penalty 1.0
  --jinja
  --reasoning on
  --reasoning-effort "$REASONING_EFFORT"
  --reasoning-preserve
  --reasoning-format deepseek
  --api-key "$LLAMA_API_KEY"
  --metrics
  --slots
  --slot-save-path "$SLOT_SAVE_PATH"
)

if [[ -n "$CACHE_TYPE_K" ]]; then
  server_args+=(--cache-type-k "$CACHE_TYPE_K")
fi
if [[ -n "$CACHE_TYPE_V" ]]; then
  server_args+=(--cache-type-v "$CACHE_TYPE_V")
fi

# llama-server's host prompt cache defaults to 8192 MiB. For 128k/256k context
# the prompt state can be >8 GB, so allow explicit tuning or disabling.
# CACHE_RAM=0 disables the host prompt cache; -1 means no limit.
if [[ -n "${CACHE_RAM:-}" ]]; then
  server_args+=(--cache-ram "$CACHE_RAM")
fi
if [[ -n "${CTX_CHECKPOINTS:-}" ]]; then
  server_args+=(--ctx-checkpoints "$CTX_CHECKPOINTS")
fi

if [[ "$USE_FASTMTP" == "1" ]]; then
  download_file "$DRAFT"
  record_disk_usage "after-draft-model"
  server_args+=(
    --spec-draft-model "$MODEL_DIR/$DRAFT"
    --spec-draft-ngl all
    --spec-type draft-mtp
    --spec-draft-n-max "$DEPTH"
    --spec-draft-p-min 0
  )
else
  # Stock/embedded MTP path; useful as a fallback when debugging FastMTP.
  server_args+=(--spec-type draft-mtp)
fi

# Do not forward the Hugging Face credential into llama-server's environment.
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN || true

if [[ "$HOSTAI_UNSECURE" == "1" ]]; then
  # Legacy path: TCP on loopback. Cert handling is skipped.
  BIND_HOST="${BIND_HOST:-127.0.0.1}"
  PORT="${PORT:-8080}"
  server_args+=(
    --host "$BIND_HOST"
    --port "$PORT"
  )
  llama_bind="$BIND_HOST:$PORT"
else
  # Secure path: HTTPS over a Unix domain socket on tmpfs.
  llama_socket="$HOSTAI_TMPFS_BASE/llama.sock"
  cert_file="$HOSTAI_CERTS_DIR/server.crt"
  key_file="$HOSTAI_CERTS_DIR/server.key"

  echo "[secure] waiting for TLS certificate delivery to tmpfs (timeout ${HOSTAI_TLS_WAIT_TIMEOUT}s)..."
  waited=0
  while [[ ! -f "$cert_file" || ! -f "$key_file" ]]; do
    if (( waited >= HOSTAI_TLS_WAIT_TIMEOUT )); then
      echo >&2 "ERROR: TLS certificate/key did not arrive at $HOSTAI_CERTS_DIR within ${HOSTAI_TLS_WAIT_TIMEOUT}s."
      echo >&2 "       Run hostai up without --unsecure, or check the SSH delivery to the Vast host."
      exit 70
    fi
    sleep 1
    waited=$((waited + 1))
  done

  # Drop certificate material from the environment so it is not visible in
  # /proc/<pid>/environ after we have written it to tmpfs.
  unset HOSTAI_TLS_CERT_B64 HOSTAI_TLS_KEY_B64 || true

  if [[ "$HOSTAI_TOKENIZED_ONLY" == "1" ]]; then
    # In tokenized-only mode llama-server listens on an internal socket with
    # plain HTTP. The hostai-guard binds the public socket, does TLS, and
    # refuses any prompt that is not an array of token IDs.
    llama_internal_socket="$HOSTAI_TMPFS_BASE/llama-internal.sock"
    llama_bind="unix://$llama_internal_socket"
    server_args+=(--host "$llama_internal_socket")
  else
    # The llama-server treats any --host value ending in .sock as a Unix socket.
    server_args+=(
      --host "$llama_socket"
      --ssl-cert-file "$cert_file"
      --ssl-key-file "$key_file"
    )
    llama_bind="unix://$llama_socket"
  fi
fi

record_disk_usage "before-serve"

echo "[serve] profile=$HOSTAI_PROFILE model=$MODEL revision=$HF_REVISION ctx=$CTX_SIZE fastmtp=$USE_FASTMTP bind=$llama_bind slot_save_path=$SLOT_SAVE_PATH"
echo "[runtime] GPU snapshot:"
nvidia-smi --query-gpu=timestamp,index,name,driver_version,memory.total,power.limit --format=csv,noheader 2>&1 || nvidia-smi 2>&1 || true

if [[ "$HOSTAI_UNSECURE" == "1" ]]; then
  # Legacy: just run the server; no key to remove.
  exec /usr/local/bin/llama-server "${server_args[@]}"
fi

# In secure mode the TLS private key is only needed while the TLS endpoint
# builds its SSL context. Start the server, wait for its socket to appear, then
# remove the key file from tmpfs so it is no longer reachable from the
# filesystem even if /dev/shm itself is readable.
/usr/local/bin/llama-server "${server_args[@]}" &
llama_pid=$!

HOSTAI_TLS_SOCKET_TIMEOUT="${HOSTAI_TLS_SOCKET_TIMEOUT:-1800}"
socket_wait=0
target_socket="$llama_socket"
[[ "$HOSTAI_TOKENIZED_ONLY" == "1" ]] && target_socket="$llama_internal_socket"
while [[ ! -S "$target_socket" ]]; do
  if ! kill -0 "$llama_pid" >/dev/null 2>&1; then
    echo >&2 "ERROR: llama-server exited before the Unix socket appeared."
    wait "$llama_pid" || true
    exit 70
  fi
  if (( socket_wait >= HOSTAI_TLS_SOCKET_TIMEOUT )); then
    echo >&2 "ERROR: llama-server did not create $target_socket within ${HOSTAI_TLS_SOCKET_TIMEOUT}s."
    kill -TERM "$llama_pid" >/dev/null 2>&1 || true
    wait "$llama_pid" || true
    exit 70
  fi
  sleep 1
  socket_wait=$((socket_wait + 1))
done

if [[ "$HOSTAI_TOKENIZED_ONLY" == "1" ]]; then
  # Start the tokenizing guard in front of the plain HTTP internal socket.
  # It owns the public TLS socket ($llama_socket) and only accepts token IDs.
  HOSTAI_GUARD_LISTEN="$llama_socket" \
  HOSTAI_GUARD_BACKEND="$llama_internal_socket" \
  HOSTAI_GUARD_CERT="$cert_file" \
  HOSTAI_GUARD_KEY="$key_file" \
    /venv/main/bin/python3 /usr/local/bin/hostai-guard &
  guard_pid=$!

  guard_wait=0
  while [[ ! -S "$llama_socket" ]]; do
    if ! kill -0 "$guard_pid" >/dev/null 2>&1; then
      echo >&2 "ERROR: hostai-guard exited before the public Unix socket appeared."
      wait "$guard_pid" || true
      kill -TERM "$llama_pid" >/dev/null 2>&1 || true
      wait "$llama_pid" || true
      exit 70
    fi
    if (( guard_wait >= HOSTAI_TLS_SOCKET_TIMEOUT )); then
      echo >&2 "ERROR: hostai-guard did not create $llama_socket within ${HOSTAI_TLS_SOCKET_TIMEOUT}s."
      kill -TERM "$guard_pid" >/dev/null 2>&1 || true
      wait "$guard_pid" || true
      kill -TERM "$llama_pid" >/dev/null 2>&1 || true
      wait "$llama_pid" || true
      exit 70
    fi
    sleep 1
    guard_wait=$((guard_wait + 1))
  done
fi

if [[ -f "$key_file" ]]; then
  rm -f "$key_file"
  echo "[secure] TLS private key removed from tmpfs after socket bind"
fi

if [[ "$HOSTAI_TOKENIZED_ONLY" == "1" ]]; then
  # Wait for either llama-server or guard and forward the exit code.
  while kill -0 "$llama_pid" >/dev/null 2>&1 && kill -0 "$guard_pid" >/dev/null 2>&1; do
    sleep 1
  done
  if ! kill -0 "$llama_pid" >/dev/null 2>&1; then
    wait "$llama_pid" || true
    rc=$?
    kill -TERM "$guard_pid" >/dev/null 2>&1 || true
    wait "$guard_pid" || true
    exit "$rc"
  else
    wait "$guard_pid" || true
    rc=$?
    kill -TERM "$llama_pid" >/dev/null 2>&1 || true
    wait "$llama_pid" || true
    exit "$rc"
  fi
fi

# Forward the real exit code to the entrypoint supervisor.
wait "$llama_pid"
