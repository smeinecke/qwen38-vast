#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

HF_REPO="${HF_REPO:-HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF}"
HF_REVISION="${HF_REVISION:-993a5971fda8f30dd1b7eb2654792ba4415c7460}"
MODEL="${MODEL:-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf}"
DRAFT="${DRAFT:-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf}"
MODEL_DIR="${MODEL_DIR:-/models}"
CTX_SIZE="${CTX_SIZE:-245760}"
DEPTH="${DEPTH:-3}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"
USE_FASTMTP="${USE_FASTMTP:-1}"

if [[ -z "${LLAMA_API_KEY:-}" ]]; then
  echo >&2 "ERROR: LLAMA_API_KEY is required."
  exit 2
fi

mkdir -p "$MODEL_DIR"

download_file() {
  local filename="$1"
  echo "[download] ${HF_REPO}/${filename}"
  hf download "$HF_REPO" "$filename" --revision "$HF_REVISION" --local-dir "$MODEL_DIR"
}

# The repository is currently gated. HF_TOKEN is automatically consumed by
# huggingface_hub. Anonymous access is still attempted so this keeps working
# if the gate is removed later.
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo >&2 "WARNING: HF_TOKEN is not set; gated Hugging Face downloads may fail."
fi

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
)

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

echo "[serve] model=$MODEL revision=$HF_REVISION ctx=$CTX_SIZE fastmtp=$USE_FASTMTP bind=$BIND_HOST:$PORT"
exec /usr/local/bin/llama-server "${server_args[@]}"
