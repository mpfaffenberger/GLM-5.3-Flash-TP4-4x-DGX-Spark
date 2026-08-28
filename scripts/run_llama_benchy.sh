#!/usr/bin/env bash
# Run the Spark Arena v2 llama-benchy profile against the local GLM endpoint.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE_URL=${GLM53_BASE_URL:-http://127.0.0.1:8000/v1}
MODEL=${GLM53_MODEL:-unsloth/GLM-5.3-Flash-FP8}
SERVED_MODEL=${GLM53_SERVED_NAME:-glm-5.3-flash-fp8}
REVISION=${GLM53_REVISION:-a160e2291674d9e3e92e98fd82faa2544a2867a3}
MODEL_CACHE_DIR=${GLM53_MODEL_CACHE_DIR:-models--unsloth--GLM-5.3-Flash-FP8}
TOKENIZER=${GLM53_TOKENIZER:-$HOME/.cache/huggingface/hub/$MODEL_CACHE_DIR/snapshots/$REVISION}
STAMP=$(date -u +%Y%m%d-%H%M%S)
RESULT_DIR=${1:-$ROOT/results/llama-benchy-spark-arena-v2-$STAMP}

mkdir -p "$RESULT_DIR"
COMMAND_FILE=$RESULT_DIR/command.txt
LOG_FILE=$RESULT_DIR/llama-benchy.log
RESULT_FILE=$RESULT_DIR/results.csv
PROGRESS_FILE=$RESULT_DIR/progress.jsonl

if [[ ! -f "$TOKENIZER/tokenizer_config.json" ]]; then
  echo "tokenizer snapshot is incomplete: $TOKENIZER" >&2
  exit 1
fi

# Force one HTTP connection per streamed request. The tested vLLM Rust
# frontend otherwise leaks shared-memory broadcast blocks across keepalive
# requests and wedges after the third request.
for patch in \
  patch_llama_benchy_keepalive.py \
  patch_llama_benchy_stream_done.py
do
  uvx --from llama-benchy==0.4.0 python "$ROOT/patches/$patch"
done

command=(
  uvx --from llama-benchy==0.4.0 llama-benchy
  --base-url "$BASE_URL"
  --model "$MODEL"
  --served-model-name "$SERVED_MODEL"
  --tokenizer "$TOKENIZER"
  --depth 0 4096 8192 16384 32768 65535 100000
  --pp 2048
  --tg 128
  --runs 3
  # GLM defaults to reasoning; the harness's tiny "Paris" gate can consume
  # its whole budget in reasoning_content despite a healthy engine.
  --skip-coherence
  # vLLM's Rust frontend can wedge its shared-memory output broadcaster when
  # speculative decoding streams the optional token_ids extension. Usage
  # completion_tokens remains the authoritative aggregate token count.
  --extra-body return_token_ids=false
  --enable-prefix-caching
  --concurrency 1 2 5 10
  --save-result "$RESULT_FILE"
  --format csv
  --emit-progress "$PROGRESS_FILE"
)

printf '%q ' "${command[@]}" >"$COMMAND_FILE"
printf '\n' >>"$COMMAND_FILE"
printf '%s\n' "$RESULT_DIR" >"$HOME/llama-benchy-glm53.current"

set +e
"${command[@]}" 2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" >"$RESULT_DIR/exit-code"
exit "$status"
