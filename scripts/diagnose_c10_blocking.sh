#!/usr/bin/env bash
# Reproduce the 16K-context/c10 hang synchronously, capture stacks, then recover.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=glm53-vllm-gb10:nope-sm121-topk-compact-v2-ray-2.58
MODEL_CACHE=models--unsloth--GLM-5.3-Flash-FP8
TOKENIZER=$HOME/.cache/huggingface/hub/$MODEL_CACHE/snapshots/a160e2291674d9e3e92e98fd82faa2544a2867a3
PYSPY=$HOME/.cache/uv/archive-v0/3eVEIjOvVkqR7Wxj/bin/py-spy
WORKERS='10.0.0.150 10.0.0.13 10.0.0.246'
LOG=$HOME/glm53-c10-blocking-diagnosis.log
DUMPS=$HOME/glm53-hang-dumps/arena-16k-c10-blocking
mkdir -p "$DUMPS"
mark() { echo "$(date -u +%FT%TZ) $1" | tee -a "$LOG"; }
cleanup() {
  tmux kill-session -t glm53serve 2>/dev/null || true
  docker rm -f glm53-tp4 >/dev/null 2>&1 || true
  sudo -n systemctl stop nvidia-persistenced.service 2>/dev/null || true
  sudo -n modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia
  sudo -n modprobe nvidia
  sudo -n modprobe nvidia_uvm
  sudo -n modprobe nvidia_modeset
  sudo -n modprobe nvidia_drm
  sudo -n systemctl start nvidia-persistenced.service
  sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  for ip in $WORKERS; do
    ssh -o BatchMode=yes mpfaffenberger@$ip \
      'docker rm -f glm53-tp4 >/dev/null 2>&1 || true; sudo -n systemctl stop nvidia-persistenced.service 2>/dev/null || true; sudo -n modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia; sudo -n modprobe nvidia; sudo -n modprobe nvidia_uvm; sudo -n modprobe nvidia_modeset; sudo -n modprobe nvidia_drm; sudo -n systemctl start nvidia-persistenced.service; sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"' &
  done
  wait
}
cluster_up() {
  local extra=${1:-}
  cd "$ROOT"
  GLM53_IMAGE=$IMAGE GLM53_MODEL_CACHE_DIR=$MODEL_CACHE GLM53_DOCKER_ENV="$extra" \
    ./scripts/glm53_node_up.sh head 10.0.0.46 10.0.0.46 auto >/dev/null
  for ip in $WORKERS; do
    ssh -o BatchMode=yes mpfaffenberger@$ip \
      "GLM53_IMAGE=$IMAGE GLM53_MODEL_CACHE_DIR=$MODEL_CACHE GLM53_DOCKER_ENV='$extra' ~/glm53_node_up.sh worker $ip 10.0.0.46 auto" >/dev/null &
  done
  wait
  for _ in $(seq 1 60); do
    docker exec glm53-tp4 ray status 2>/dev/null | grep -q '0.0/4.0 GPU' && return 0
    sleep 2
  done
  return 1
}
serve_ready() {
  local label=$1
  cd "$ROOT"
  GLM53_MTP_TOKENS=3 GLM53_FLASHINFER_AUTOTUNE=1 GLM53_LOG="$HOME/glm53-$label-serve.log" ./scripts/glm53_serve.sh >/dev/null
  for _ in $(seq 1 300); do
    curl -fsS --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && return 0
    grep -q 'SERVE_EXIT=' "$HOME/glm53-$label-serve.log" 2>/dev/null && return 1
    sleep 10
  done
  return 1
}
dump_hosts() {
  local pid
  pid=$(ps -ef | awk '/ray::RayWorkerProc/ && !/awk/ {print $2; exit}')
  [[ -n "$pid" ]] && sudo -n "$PYSPY" dump --pid "$pid" > "$DUMPS/head.txt" 2>&1 || true
  for ip in $WORKERS; do
    scp -q "$PYSPY" mpfaffenberger@$ip:/tmp/py-spy
    ssh -o BatchMode=yes mpfaffenberger@$ip \
      'pid=$(ps -ef | awk "/ray::RayWorkerProc/ && !/awk/ {print \$2; exit}"); [[ -n "$pid" ]] && sudo -n /tmp/py-spy dump --pid "$pid"' \
      > "$DUMPS/$ip.txt" 2>&1 || true
  done
}
mark BLOCKING_CLEANUP
cleanup
mark BLOCKING_CLUSTER_UP
cluster_up CUDA_LAUNCH_BLOCKING=1 || { mark BLOCKING_CLUSTER_FAILED; exit 1; }
mark BLOCKING_SERVE_START
serve_ready c10-blocking || { mark BLOCKING_SERVE_FAILED; exit 1; }
mark BLOCKING_API_READY
timeout 900 uvx --from llama-benchy==0.4.0 llama-benchy \
  --base-url http://127.0.0.1:8000/v1 --model unsloth/GLM-5.3-Flash-FP8 \
  --served-model-name glm-5.3-flash-fp8 --tokenizer "$TOKENIZER" \
  --depth 16384 --pp 2048 --tg 128 --runs 1 --skip-coherence \
  --extra-body return_token_ids=false --enable-prefix-caching --concurrency 10 \
  --save-result /tmp/glm53-c10-blocking.csv --format csv \
  --emit-progress /tmp/glm53-c10-blocking.jsonl > "$DUMPS/llama-benchy.log" 2>&1 &
bench_pid=$!
mark "BLOCKING_C10_FIRED pid=$bench_pid"
sleep 180
mark BLOCKING_STACK_CAPTURE
dump_hosts
mark BLOCKING_STACKS_CAPTURED
wait "$bench_pid" 2>/dev/null
mark "BLOCKING_BENCH_EXIT=$?"
mark PRODUCTION_RECOVERY
cleanup
cluster_up '' || { mark RECOVERY_CLUSTER_FAILED; exit 1; }
mark PRODUCTION_SERVE_START
serve_ready c10-final-production || { mark RECOVERY_SERVE_FAILED; exit 1; }
mark PRODUCTION_API_READY
cd "$ROOT"
if GLM53_BENCH_RUNS=3 ./scripts/bench_c1.sh > "$HOME/glm53-c10-final-c1.log" 2>&1; then
  mark "PRODUCTION_C1_OK $(grep C1_RESULT "$HOME/glm53-c10-final-c1.log")"
else
  mark PRODUCTION_C1_FAILED
fi
mark DIAGNOSIS_PIPELINE_DONE
