#!/usr/bin/env bash
# Relaunch an MTP-depth trial with sparse-MLA tuning and benchmark C=1.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${GLM53_IMAGE:-glm53-vllm-gb10:nope-sm121-topk-compact-ray-2.58}
MTP_TOKENS=${GLM53_TRIAL_MTP_TOKENS:-3}
FLASHINFER_AUTOTUNE=${GLM53_TRIAL_FLASHINFER_AUTOTUNE:-1}
DROP_CACHES=${GLM53_DROP_CACHES:-1}
MODEL_CACHE_DIR=${GLM53_MODEL_CACHE_DIR:-models--unsloth--GLM-5.3-Flash-FP8}
TRIAL_LABEL=${GLM53_TRIAL_LABEL:-mtp${MTP_TOKENS}-sparse-tuned}
LOG=$HOME/glm53-${TRIAL_LABEL}-serve.log
RESULT=$HOME/glm53-${TRIAL_LABEL}-c1.log

cleanup_local() {
  tmux kill-session -t glm53serve 2>/dev/null || true
  docker rm -f glm53-tp4 >/dev/null 2>&1 || true
  sudo -n modprobe -r nvidia_uvm
  sudo -n modprobe nvidia_uvm
  if [[ "$DROP_CACHES" == 1 ]]; then
    sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  fi
}

cleanup_remote() {
  local host=$1
  ssh -o BatchMode=yes "mpfaffenberger@$host" \
    "docker rm -f glm53-tp4 >/dev/null 2>&1 || true; \
     sudo -n modprobe -r nvidia_uvm && sudo -n modprobe nvidia_uvm; \
     if [[ '$DROP_CACHES' == 1 ]]; then sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; fi"
}

cleanup_local &
pids=($!)
for host in 10.0.0.150 10.0.0.13 10.0.0.246; do
  cleanup_remote "$host" &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

GLM53_IMAGE=$IMAGE "$ROOT/scripts/glm53_node_up.sh" \
  head 10.0.0.46 10.0.0.46 auto >/tmp/glm53-sparse-head-up.log

ssh -o BatchMode=yes mpfaffenberger@10.0.0.150 \
  "GLM53_IMAGE=$IMAGE GLM53_MODEL_CACHE_DIR=$MODEL_CACHE_DIR ~/glm53_node_up.sh worker 10.0.0.150 10.0.0.46 auto" &
pids=($!)
ssh -o BatchMode=yes mpfaffenberger@10.0.0.13 \
  "GLM53_IMAGE=$IMAGE GLM53_MODEL_CACHE_DIR=$MODEL_CACHE_DIR ~/glm53_node_up.sh worker 10.0.0.13 10.0.0.46 auto" &
pids+=($!)
ssh -o BatchMode=yes mpfaffenberger@10.0.0.246 \
  "GLM53_IMAGE=$IMAGE GLM53_MODEL_CACHE_DIR=$MODEL_CACHE_DIR ~/glm53_node_up.sh worker 10.0.0.246 10.0.0.46 auto" &
pids+=($!)
for pid in "${pids[@]}"; do wait "$pid"; done

for attempt in $(seq 1 60); do
  if docker exec glm53-tp4 ray status 2>/dev/null | grep -q '0.0/4.0 GPU'; then
    break
  fi
  if [[ "$attempt" == 60 ]]; then
    echo 'Ray did not register all four GPUs' >&2
    docker exec glm53-tp4 ray status >&2 || true
    exit 1
  fi
  sleep 2
done

GLM53_MTP_TOKENS=$MTP_TOKENS \
GLM53_FLASHINFER_AUTOTUNE=$FLASHINFER_AUTOTUNE \
GLM53_LOG=$LOG \
  "$ROOT/scripts/glm53_serve.sh"

for attempt in $(seq 1 300); do
  if curl -fsS --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    if [[ "$FLASHINFER_AUTOTUNE" == 1 ]]; then
      grep -q 'Autotuning FlashInfer SM120 sparse MLA' "$LOG"
      grep -q 'dedicated sparse-MLA tuning remains enabled' "$LOG"
    fi
    GLM53_BENCH_RUNS=5 "$ROOT/scripts/bench_c1.sh" | tee "$RESULT"
    echo "SPARSE_TUNED_MTP${MTP_TOKENS}_TRIAL_OK"
    exit 0
  fi
  if grep -q 'SERVE_EXIT=' "$LOG"; then
    tail -100 "$LOG" >&2
    exit 1
  fi
  sleep 10
done

echo "timed out waiting for sparse-tuned MTP${MTP_TOKENS} API" >&2
exit 2
