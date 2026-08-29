#!/usr/bin/env bash
# Persistent llama.cpp slot/KV cache helpers.
#
# The cache server only needs SSH + rsync. A dedicated private key lives on the
# local workstation under .qwen-cache/ and is copied to each disposable Vast
# instance after its own SSH endpoint is ready. The key is never baked into the
# Docker image or committed to git.

qwen_cache_load_config() {
  QWEN_SLOT_CACHE_ENABLED="${QWEN_SLOT_CACHE_ENABLED:-1}"
  QWEN_SLOT_CACHE_HOST="${QWEN_SLOT_CACHE_HOST:-}"
  QWEN_SLOT_CACHE_PORT="${QWEN_SLOT_CACHE_PORT:-22}"
  QWEN_SLOT_CACHE_USER="${QWEN_SLOT_CACHE_USER:-qwen-cache}"
  QWEN_SLOT_CACHE_ROOT="${QWEN_SLOT_CACHE_ROOT:-qwen-slot-cache}"
  QWEN_SLOT_CACHE_SESSION="${QWEN_SLOT_CACHE_SESSION:-default}"
  QWEN_SLOT_CACHE_KEY="${QWEN_SLOT_CACHE_KEY:-$ROOT_DIR/.qwen-cache/id_ed25519}"
  QWEN_SLOT_CACHE_MAX_GB="${QWEN_SLOT_CACHE_MAX_GB:-80}"
  QWEN_SLOT_CACHE_REQUIRE_SAVE="${QWEN_SLOT_CACHE_REQUIRE_SAVE:-0}"
  QWEN_SLOT_CACHE_SLOT_ID="${QWEN_SLOT_CACHE_SLOT_ID:-0}"
  QWEN_SLOT_CACHE_LOCAL_DIR="${QWEN_SLOT_CACHE_LOCAL_DIR:-/var/lib/qwen38/slots}"
}

qwen_cache_enabled() {
  [[ "${QWEN_SLOT_CACHE_ENABLED:-0}" == "1" ]]
}

qwen_cache_validate_config() {
  qwen_cache_enabled || return 0

  [[ "$QWEN_SLOT_CACHE_PORT" =~ ^[0-9]+$ ]] && (( QWEN_SLOT_CACHE_PORT > 0 && QWEN_SLOT_CACHE_PORT < 65536 )) \
    || { echo >&2 "ERROR: invalid QWEN_SLOT_CACHE_PORT=$QWEN_SLOT_CACHE_PORT"; return 2; }
  [[ "$QWEN_SLOT_CACHE_MAX_GB" =~ ^[0-9]+$ ]] && (( QWEN_SLOT_CACHE_MAX_GB > 0 )) \
    || { echo >&2 "ERROR: invalid QWEN_SLOT_CACHE_MAX_GB=$QWEN_SLOT_CACHE_MAX_GB"; return 2; }
  [[ "$QWEN_SLOT_CACHE_SLOT_ID" =~ ^[0-9]+$ ]] \
    || { echo >&2 "ERROR: invalid QWEN_SLOT_CACHE_SLOT_ID=$QWEN_SLOT_CACHE_SLOT_ID"; return 2; }
  [[ -n "$QWEN_SLOT_CACHE_HOST" ]] \
    || { echo >&2 "ERROR: QWEN_SLOT_CACHE_HOST is not set; set it in .env to enable the slot cache"; return 2; }
  [[ "$QWEN_SLOT_CACHE_HOST" =~ ^[A-Za-z0-9._:-]+$ ]] \
    || { echo >&2 "ERROR: unsafe QWEN_SLOT_CACHE_HOST=$QWEN_SLOT_CACHE_HOST"; return 2; }
  [[ "$QWEN_SLOT_CACHE_USER" =~ ^[A-Za-z0-9._-]+$ ]] \
    || { echo >&2 "ERROR: unsafe QWEN_SLOT_CACHE_USER"; return 2; }
  [[ "$QWEN_SLOT_CACHE_SESSION" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] \
    || { echo >&2 "ERROR: QWEN_SLOT_CACHE_SESSION must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}"; return 2; }
  [[ "$QWEN_SLOT_CACHE_ROOT" =~ ^[A-Za-z0-9_./-]+$ ]] && [[ "$QWEN_SLOT_CACHE_ROOT" != *".."* ]] \
    || { echo >&2 "ERROR: unsafe QWEN_SLOT_CACHE_ROOT"; return 2; }
  [[ "$QWEN_SLOT_CACHE_LOCAL_DIR" =~ ^/[A-Za-z0-9_./-]+$ ]] && [[ "$QWEN_SLOT_CACHE_LOCAL_DIR" != *".."* ]] \
    || { echo >&2 "ERROR: unsafe QWEN_SLOT_CACHE_LOCAL_DIR"; return 2; }
  return 0
}

qwen_cache_require_key() {
  qwen_cache_enabled || return 0
  if [[ ! -f "$QWEN_SLOT_CACHE_KEY" ]]; then
    echo >&2 "ERROR: slot cache is enabled but dedicated key is missing: $QWEN_SLOT_CACHE_KEY"
    echo >&2 "Run ./qwen-cache-setup once before renting a GPU."
    return 2
  fi
  chmod 600 "$QWEN_SLOT_CACHE_KEY" 2>/dev/null || true
}

qwen_cache_local_preflight() {
  qwen_cache_enabled || return 0
  qwen_cache_require_key || return $?
  ssh \
    -i "$QWEN_SLOT_CACHE_KEY" \
    -p "$QWEN_SLOT_CACHE_PORT" \
    -o BatchMode=yes \
    -o ConnectTimeout=6 \
    -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile="$ROOT_DIR/.qwen-cache/known_hosts" \
    "${QWEN_SLOT_CACHE_USER}@${QWEN_SLOT_CACHE_HOST}" \
    "command -v rsync >/dev/null 2>&1 && mkdir -p '$QWEN_SLOT_CACHE_ROOT' && chmod 700 '$QWEN_SLOT_CACHE_ROOT' && test -w '$QWEN_SLOT_CACHE_ROOT'" \
    >/dev/null 2>&1
}

qwen_cache_signature() {
  local llama_commit="$1"
  local model="$2"
  local revision="$3"
  local ctx_size="$4"
  local use_fastmtp="$5"
  local cache_type_k="${6:-default}"
  local cache_type_v="${7:-default}"

  python3 - "$llama_commit" "$model" "$revision" "$ctx_size" "$use_fastmtp" "$cache_type_k" "$cache_type_v" <<'PY'
import hashlib, json, sys
llama_commit, model, revision, ctx, fastmtp, cache_k, cache_v = sys.argv[1:]
obj = {
    "llama_cpp_commit": llama_commit,
    "model": model,
    "hf_revision": revision,
    "ctx_size": int(ctx),
    "use_fastmtp": int(fastmtp),
    "cache_type_k": cache_k,
    "cache_type_v": cache_v,
}
raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(raw).hexdigest()[:20])
PY
}

qwen_cache_remote_dir() {
  local signature="$1"
  printf '%s/%s/%s\n' "$QWEN_SLOT_CACHE_ROOT" "$QWEN_SLOT_CACHE_SESSION" "$signature"
}

qwen_cache_copy_key_to_vast() {
  local vast_user="$1" vast_host="$2" vast_port="$3"
  local remote_key="/root/.ssh/qwen-slot-cache"
  local -a common_opts scp_opts ssh_opts

  qwen_cache_enabled || return 0
  qwen_cache_require_key || return $?

  common_opts=(
    -o BatchMode=yes
    -o ConnectTimeout=8
    -o ServerAliveInterval=15
    -o ServerAliveCountMax=3
    -o StrictHostKeyChecking=accept-new
    -o "UserKnownHostsFile=$KNOWN_HOSTS"
  )
  scp_opts=("${common_opts[@]}" -P "$vast_port")
  ssh_opts=("${common_opts[@]}" -p "$vast_port")

  qwen_ssh_sanitize_known_hosts "$vast_host" "$vast_port"

  scp -q "${scp_opts[@]}" "$QWEN_SLOT_CACHE_KEY" "${vast_user}@${vast_host}:${remote_key}.tmp" || {
    echo >&2 "[slot-cache] could not copy cache key to ${vast_host}:${vast_port}"
    return 1
  }
  ssh "${ssh_opts[@]}" "${vast_user}@${vast_host}" \
    "install -d -m 700 /root/.ssh && install -m 600 '${remote_key}.tmp' '$remote_key' && rm -f '${remote_key}.tmp'" || {
    echo >&2 "[slot-cache] could not install cache key on ${vast_host}:${vast_port}"
    return 1
  }
}

# Download current.bin from the persistent cache server to the Vast slot-save
# directory. Return codes: 0=cache hit/downloaded, 10=cache miss, other=error.
qwen_cache_prefetch_to_vast() {
  local vast_user="$1" vast_host="$2" vast_port="$3" signature="$4"
  local remote_dir
  local -a ssh_opts
  remote_dir="$(qwen_cache_remote_dir "$signature")"
  mapfile -d '' -t ssh_opts < <(qwen_ssh_opts "$vast_port")

  ssh "${ssh_opts[@]}" "${vast_user}@${vast_host}" bash -s -- \
    "$QWEN_SLOT_CACHE_HOST" "$QWEN_SLOT_CACHE_PORT" "$QWEN_SLOT_CACHE_USER" \
    "$remote_dir" "$QWEN_SLOT_CACHE_LOCAL_DIR" <<'REMOTE'
set -Eeuo pipefail
umask 077
cache_host="$1"; cache_port="$2"; cache_user="$3"; remote_dir="$4"; slot_dir="$5"
key=/root/.ssh/qwen-slot-cache
known=/root/.ssh/qwen-slot-cache-known_hosts
# -n and < /dev/null stop the nested ssh/rsync from slurping this heredoc.
ssh_base=(ssh -n -i "$key" -p "$cache_port" -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$known")
mkdir -p "$slot_dir"
chmod 700 "$slot_dir"

if ! "${ssh_base[@]}" "${cache_user}@${cache_host}" "test -s '$remote_dir/current.bin'"; then
  echo "[slot-cache] miss: ${cache_user}@${cache_host}:${remote_dir}/current.bin"
  exit 10
fi

start=$(date +%s)
remote_bytes=$("${ssh_base[@]}" "${cache_user}@${cache_host}" "stat -c %s '$remote_dir/current.bin'" 2>/dev/null || echo 0)
if [[ "$remote_bytes" =~ ^[0-9]+$ ]] && (( remote_bytes > 0 )); then
  echo "[slot-cache] snapshot size: ${remote_bytes} bytes"
  printf '%s\n' "$remote_bytes" > /run/qwen38/cache-prefetch.total
fi
rm -f "$slot_dir/current.bin.part"
for attempt in 1 2 3; do
  if rsync -a --partial --info=progress2,stats2 \
    -e "ssh -i $key -p $cache_port -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$known" \
    "${cache_user}@${cache_host}:${remote_dir}/current.bin" \
    "$slot_dir/current.bin.part" < /dev/null; then
    break
  fi
  (( attempt == 3 )) && { echo >&2 "[slot-cache] download failed after 3 attempts"; exit 13; }
  echo >&2 "[slot-cache] download attempt $attempt failed; retrying in 3s..."
  sleep 3
done
mv "$slot_dir/current.bin.part" "$slot_dir/current.bin"
chmod 600 "$slot_dir/current.bin"

# Metadata is optional and not needed by llama.cpp; copy it when available.
if "${ssh_base[@]}" "${cache_user}@${cache_host}" "test -s '$remote_dir/current.json'"; then
  rsync -a --partial \
    -e "ssh -i $key -p $cache_port -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$known" \
    "${cache_user}@${cache_host}:${remote_dir}/current.json" \
    "$slot_dir/remote-current.json" < /dev/null || true
fi
bytes=$(stat -c %s "$slot_dir/current.bin")
end=$(date +%s)
echo "[slot-cache] downloaded ${bytes} bytes in $((end-start))s"
REMOTE
}

# Upload current.bin/current.json from the Vast slot directory to the persistent
# cache server. The remote current.* files are atomically replaced after rsync.
qwen_cache_upload_from_vast() {
  local vast_user="$1" vast_host="$2" vast_port="$3" signature="$4"
  local remote_dir
  local -a ssh_opts
  remote_dir="$(qwen_cache_remote_dir "$signature")"
  mapfile -d '' -t ssh_opts < <(qwen_ssh_opts "$vast_port")

  ssh "${ssh_opts[@]}" "${vast_user}@${vast_host}" bash -s -- \
    "$QWEN_SLOT_CACHE_HOST" "$QWEN_SLOT_CACHE_PORT" "$QWEN_SLOT_CACHE_USER" \
    "$QWEN_SLOT_CACHE_ROOT" "$remote_dir" "$QWEN_SLOT_CACHE_LOCAL_DIR" "$QWEN_SLOT_CACHE_MAX_GB" <<'REMOTE'
set -Eeuo pipefail
umask 077
cache_host="$1"; cache_port="$2"; cache_user="$3"; cache_root="$4"; remote_dir="$5"; slot_dir="$6"; max_gb="$7"
key=/root/.ssh/qwen-slot-cache
known=/root/.ssh/qwen-slot-cache-known_hosts
ssh_base=(ssh -i "$key" -p "$cache_port" -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$known")
# -n and < /dev/null stop the nested ssh/rsync from slurping this heredoc.
ssh_base_n=(ssh -n -i "$key" -p "$cache_port" -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$known")
[[ -s "$slot_dir/current.bin" ]] || { echo >&2 "[slot-cache] local current.bin missing/empty"; exit 11; }
[[ -s "$slot_dir/current.json" ]] || { echo >&2 "[slot-cache] local current.json missing/empty"; exit 12; }

"${ssh_base_n[@]}" "${cache_user}@${cache_host}" "mkdir -p '$remote_dir' && chmod 700 '$cache_root' '$cache_root/'* 2>/dev/null || true; mkdir -p '$remote_dir'"
# Keep fixed .part names so a failed upload can resume on the next retry/run,
# while the previous current.bin remains valid until the final atomic rename.
remote_bin="${remote_dir}/.current.bin.part"
remote_json="${remote_dir}/.current.json.part"
local_bytes=$(stat -c %s "$slot_dir/current.bin")
remote_free=$("${ssh_base_n[@]}" "${cache_user}@${cache_host}" "df -PB1 '$remote_dir' | awk 'NR==2 {print \$4}'" 2>/dev/null || echo 0)
if [[ "$remote_free" =~ ^[0-9]+$ ]] && (( remote_free > 0 && remote_free < local_bytes )); then
  echo >&2 "[slot-cache] cache server has only ${remote_free} free bytes; snapshot needs ${local_bytes} bytes before atomic replacement"
  exit 14
fi
start=$(date +%s)
for attempt in 1 2 3; do
  if rsync -a --partial --info=progress2,stats2 \
    -e "ssh -i $key -p $cache_port -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$known" \
    "$slot_dir/current.bin" "${cache_user}@${cache_host}:${remote_bin}" < /dev/null; then
    break
  fi
  (( attempt == 3 )) && { echo >&2 "[slot-cache] upload failed after 3 attempts"; exit 15; }
  echo >&2 "[slot-cache] upload attempt $attempt failed; retrying in 3s..."
  sleep 3
done
for attempt in 1 2 3; do
  if rsync -a --partial \
    -e "ssh -i $key -p $cache_port -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$known" \
    "$slot_dir/current.json" "${cache_user}@${cache_host}:${remote_json}" < /dev/null; then
    break
  fi
  (( attempt == 3 )) && { echo >&2 "[slot-cache] metadata upload failed after 3 attempts"; exit 16; }
  echo >&2 "[slot-cache] metadata upload attempt $attempt failed; retrying in 3s..."
  sleep 3
done

"${ssh_base_n[@]}" "${cache_user}@${cache_host}" \
  "chmod 600 '$remote_bin' '$remote_json' && mv -f '$remote_bin' '$remote_dir/current.bin' && mv -f '$remote_json' '$remote_dir/current.json'"
bytes=$(stat -c %s "$slot_dir/current.bin")
end=$(date +%s)
echo "[slot-cache] uploaded ${bytes} bytes in $((end-start))s"

# Size-based LRU-ish retention across independent session/signature snapshots.
# Never delete the snapshot just uploaded. If that single file exceeds the
# configured budget, retain it and report the overage rather than deleting it.
"${ssh_base[@]}" "${cache_user}@${cache_host}" bash -s -- "$cache_root" "$remote_dir/current.bin" "$max_gb" <<'PRUNE'
set -Eeuo pipefail
umask 077
root="$1"; protect="$2"; max_gb="$3"
max_bytes=$((max_gb * 1024 * 1024 * 1024))
[[ -d "$root" ]] || exit 0
while :; do
  total=$(du -sb "$root" 2>/dev/null | awk '{print $1}')
  total=${total:-0}
  (( total <= max_bytes )) && break
  oldest=$(find "$root" -type f -name current.bin ! -path "$protect" -printf '%T@ %p\n' 2>/dev/null | sort -n | head -n1 | cut -d' ' -f2-)
  if [[ -z "$oldest" ]]; then
    echo "[slot-cache] retention budget exceeded (${total} bytes > ${max_bytes}) but only protected snapshot remains"
    break
  fi
  meta="${oldest%.bin}.json"
  echo "[slot-cache] pruning old snapshot: $oldest"
  rm -f -- "$oldest" "$meta"
  rmdir --ignore-fail-on-non-empty "$(dirname "$oldest")" 2>/dev/null || true
done
PRUNE
REMOTE
}
