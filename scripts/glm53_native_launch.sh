#!/usr/bin/env bash
# Native four-node vLLM TP launcher: PyTorch multiprocessing, no Ray.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HEAD=10.0.0.46
WORKERS=(10.0.0.13 10.0.0.150 10.0.0.246)
NODES=("$HEAD" "${WORKERS[@]}")
NAME=${GLM53_CONTAINER_NAME:-glm53-tp4}
IMAGE=${GLM53_IMAGE:-glm53-vllm-gb10:nope-sm121-topk-compact-v2-ray-2.58}
MODEL=${GLM53_MODEL:-unsloth/GLM-5.3-Flash-FP8}
REVISION=${GLM53_REVISION:-a160e2291674d9e3e92e98fd82faa2544a2867a3}
SERVED_NAME=${GLM53_SERVED_NAME:-glm-5.3-flash-fp8}
PORT=${GLM53_PORT:-8000}
MASTER_PORT=${GLM53_MASTER_PORT:-29500}
GMU=${GLM53_GPU_MEMORY_UTILIZATION:-0.82}
MAX_LEN=${GLM53_MAX_MODEL_LEN:-262144}
MAX_SEQS=${GLM53_MAX_NUM_SEQS:-4}
BATCHED_TOKENS=${GLM53_MAX_BATCHED_TOKENS:-8192}
KV_CACHE_DTYPE=${GLM53_KV_CACHE_DTYPE:-fp8}
MTP_TOKENS=${GLM53_MTP_TOKENS:-3}
FLASHINFER_AUTOTUNE=${GLM53_FLASHINFER_AUTOTUNE:-1}
# GLM53_ENFORCE_EAGER=1 drops CUDA graph capture/replay on every rank. It is
# both the control experiment for the cross-rank MoE divergence and a candidate
# production profile, so pin the env var rather than trusting vLLM's
# auto-enable heuristic for breakable graphs.
ENFORCE_EAGER=${GLM53_ENFORCE_EAGER:-0}
DOCKER_ENV=${GLM53_DOCKER_ENV:-}
if [[ "$ENFORCE_EAGER" == 1 ]]; then
  DOCKER_ENV="$DOCKER_ENV VLLM_USE_BREAKABLE_CUDAGRAPH=0"
fi
ACTION=${1:-start}

stop_cluster() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  for ip in "${WORKERS[@]}"; do
    ssh -o BatchMode=yes "$ip" "docker rm -f '$NAME' >/dev/null 2>&1 || true" &
  done
  wait
}

if [[ "$ACTION" == stop ]]; then
  stop_cluster
  echo "native GLM cluster stopped"
  exit 0
fi
[[ "$ACTION" == start ]] || { echo "usage: $0 [start|stop]" >&2; exit 2; }

# Container setup retains the validated per-node RoCE GID discovery but skips Ray.
stop_cluster
GLM53_START_RAY=0 GLM53_IMAGE="$IMAGE" \
  GLM53_DOCKER_ENV="$DOCKER_ENV" \
  "$ROOT/scripts/glm53_node_up.sh" head "$HEAD" "$HEAD" auto
for i in "${!WORKERS[@]}"; do
  ip=${WORKERS[$i]}
  scp -q "$ROOT/scripts/glm53_node_up.sh" "$ip:~/glm53_node_up.sh"
  ssh -o BatchMode=yes "$ip" \
    "GLM53_START_RAY=0 GLM53_IMAGE='$IMAGE' GLM53_DOCKER_ENV='$DOCKER_ENV' ~/glm53_node_up.sh worker '$ip' '$HEAD' auto" &
done
wait

AUTOTUNE_ARG=--no-enable-flashinfer-autotune
[[ "$FLASHINFER_AUTOTUNE" == 1 ]] && AUTOTUNE_ARG=--enable-flashinfer-autotune
SPEC_ARGS=()
if (( MTP_TOKENS > 0 )); then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_TOKENS}")
fi
EAGER_ARGS=()
[[ "$ENFORCE_EAGER" == 1 ]] && EAGER_ARGS=(--enforce-eager)

base_cmd=(
  vllm serve "$MODEL"
  --revision "$REVISION"
  --served-model-name "$SERVED_NAME"
  --host 0.0.0.0 --port "$PORT"
  --tensor-parallel-size 4
  --distributed-executor-backend mp
  --nnodes 4 --master-addr "$HEAD" --master-port "$MASTER_PORT"
  --gpu-memory-utilization "$GMU"
  --max-model-len "$MAX_LEN"
  --max-num-batched-tokens "$BATCHED_TOKENS"
  --max-num-seqs "$MAX_SEQS"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --enable-prefix-caching
  --reasoning-parser glm45
  --tool-call-parser glm47
  --enable-auto-tool-choice
  "$AUTOTUNE_ARG"
  "${SPEC_ARGS[@]}"
  "${EAGER_ARGS[@]}"
)

quote_cmd() {
  local out='' arg
  for arg in "$@"; do printf -v out '%s %q' "$out" "$arg"; done
  printf '%s' "${out# }"
}

# Workers must enter the torch distributed rendezvous before rank zero.
for rank in 1 2 3; do
  ip=${NODES[$rank]}
  cmd=("${base_cmd[@]}" --node-rank "$rank" --headless)
  quoted=$(quote_cmd "${cmd[@]}")
  echo "launching native worker rank=$rank ip=$ip"
  ssh -o BatchMode=yes "$ip" \
    "docker exec -d '$NAME' bash -lc $(printf %q "exec $quoted >> /proc/1/fd/1 2>&1")"
done
cmd=("${base_cmd[@]}" --node-rank 0)
quoted=$(quote_cmd "${cmd[@]}")
echo "launching native head rank=0 ip=$HEAD"
docker exec -d "$NAME" bash -lc "exec $quoted >> /proc/1/fd/1 2>&1"

echo "native TP=4 launch dispatched: API http://$HEAD:$PORT/v1"
echo "head logs: docker logs -f $NAME"
