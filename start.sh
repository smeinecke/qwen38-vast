#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

HF_REPO="${HF_REPO:-HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF}"
HF_REVISION="${HF_REVISION:-993a5971fda8f30dd1b7eb2654792ba4415c7460}"
MODEL="${MODEL:-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf}"
DRAFT="${DRAFT:-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf}"
MODEL_DIR="${MODEL_DIR:-/models}"
CTX_SIZE="${CTX_SIZE:-65536}"
DEPTH="${DEPTH:-3}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"
USE_FASTMTP="${USE_FASTMTP:-1}"
QWEN_PROFILE="${QWEN_PROFILE:-custom}"
SLOT_SAVE_PATH="${SLOT_SAVE_PATH:-/var/lib/qwen38/slots}"
CACHE_TYPE_K="${CACHE_TYPE_K:-}"
CACHE_TYPE_V="${CACHE_TYPE_V:-}"

if [[ -z "${LLAMA_API_KEY:-}" ]]; then
  echo >&2 "ERROR: LLAMA_API_KEY is required."
  exit 2
fi

mkdir -p "$MODEL_DIR" "$SLOT_SAVE_PATH"
chmod 700 "$SLOT_SAVE_PATH"

# Validate the executable before downloading tens of GB. A missing runtime
# library otherwise causes a crash only after model transfer and looks like an
# intermittent SSH/tunnel problem from the client side.
echo "[runtime] validating llama-server dependencies..."
if ! /usr/local/bin/llama-server --version >/tmp/llama-version.txt 2>/tmp/llama-version.err; then
  cat /tmp/llama-version.err >&2 || true
  echo >&2 "[runtime] linked libraries:"
  ldd /usr/local/bin/llama-server >&2 || true
  echo >&2 "ERROR: llama-server runtime preflight failed; refusing model download."
  exit 70
fi
cat /tmp/llama-version.txt

download_file() {
  local filename="$1"
  echo "[download] ${HF_REPO}/${filename}"
  hf download "$HF_REPO" "$filename" --revision "$HF_REVISION" --local-dir "$MODEL_DIR"
}

# The model repository is publicly downloadable at the time this image was
# prepared. HF_TOKEN is optional; huggingface_hub will automatically use it if
# supplied (for rate limits or if upstream access rules change later).

download_file "$MODEL"

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
  --host "$BIND_HOST"
  --port "$PORT"
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

if [[ "$USE_FASTMTP" == "1" ]]; then
  download_file "$DRAFT"
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

echo "[serve] profile=$QWEN_PROFILE model=$MODEL revision=$HF_REVISION ctx=$CTX_SIZE fastmtp=$USE_FASTMTP bind=$BIND_HOST:$PORT slot_save_path=$SLOT_SAVE_PATH"
echo "[runtime] GPU snapshot:"
nvidia-smi --query-gpu=timestamp,index,name,driver_version,memory.total,power.limit --format=csv,noheader 2>&1 || nvidia-smi 2>&1 || true
exec /usr/local/bin/llama-server "${server_args[@]}"
