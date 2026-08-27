#!/usr/bin/env bash
# Launch GLM-5.3 FP8 on the Ray head. Set GLM53_MTP_TOKENS=0 to isolate MTP.
set -euo pipefail

NAME=${GLM53_CONTAINER_NAME:-glm53-tp4}
MODEL=${GLM53_MODEL:-unsloth/GLM-5.3-Flash-FP8}
REVISION=${GLM53_REVISION:-a160e2291674d9e3e92e98fd82faa2544a2867a3}
SERVED_NAME=${GLM53_SERVED_NAME:-glm-5.3-flash-fp8}
PORT=${GLM53_PORT:-8000}
MAX_LEN=${GLM53_MAX_MODEL_LEN:-262144}
MAX_SEQS=${GLM53_MAX_NUM_SEQS:-4}
BATCHED_TOKENS=${GLM53_MAX_BATCHED_TOKENS:-8192}
GMU=${GLM53_GPU_MEMORY_UTILIZATION:-0.80}
KV_CACHE_DTYPE=${GLM53_KV_CACHE_DTYPE:-fp8}
MTP_TOKENS=${GLM53_MTP_TOKENS:-5}
FLASHINFER_AUTOTUNE=${GLM53_FLASHINFER_AUTOTUNE:-0}
LOG=${GLM53_LOG:-$HOME/glm53-serve.log}
SESSION=${GLM53_TMUX_SESSION:-glm53serve}

if ! docker exec "$NAME" ray status 2>/dev/null | grep -Eq '0\.0/4\.0 GPU|4\.0 GPU'; then
  echo "ERROR: Ray does not report four GPUs; refusing partial-rank weight load" >&2
  docker exec "$NAME" ray status || true
  exit 1
fi

AUTOTUNE_ARG=--no-enable-flashinfer-autotune
if [[ "$FLASHINFER_AUTOTUNE" == 1 ]]; then
  AUTOTUNE_ARG=--enable-flashinfer-autotune
fi

SPEC_ARGS=""
if [[ "$MTP_TOKENS" -gt 0 ]]; then
  SPEC_ARGS="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_TOKENS}'"
fi

CMD="vllm serve $MODEL \\
  --revision $REVISION \\
  --served-model-name $SERVED_NAME \\
  --host 0.0.0.0 --port $PORT \\
  --tensor-parallel-size 4 \\
  --distributed-executor-backend ray \\
  --gpu-memory-utilization $GMU \\
  --max-model-len $MAX_LEN \\
  --max-num-batched-tokens $BATCHED_TOKENS \\
  --max-num-seqs $MAX_SEQS \\
  --kv-cache-dtype $KV_CACHE_DTYPE \
  --enable-prefix-caching \\
  --reasoning-parser glm45 \\
  --tool-call-parser glm47 \\
  --enable-auto-tool-choice \\
  $AUTOTUNE_ARG \\
  $SPEC_ARGS"

mkdir -p "$(dirname "$LOG")"
: > "$LOG"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "docker exec $NAME bash -lc $(printf %q "$CMD") > $(printf %q "$LOG") 2>&1; echo SERVE_EXIT=\$? >> $(printf %q "$LOG")"

sleep 3
echo "serve launched: session=$SESSION log=$LOG model=$SERVED_NAME kv_cache=$KV_CACHE_DTYPE mtp_tokens=$MTP_TOKENS flashinfer_autotune=$FLASHINFER_AUTOTUNE"
tmux ls
