#!/usr/bin/env bash
# Complete staging, form the four-node Ray cluster, launch GLM, and smoke-test.
# Run on the head after fanout_cluster.sh and the head/.246 nodes are started.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FANOUT_PID_FILE=${GLM53_FANOUT_PID_FILE:-$HOME/glm53-fanout.pid}
LOG=${GLM53_FINALIZE_LOG:-$HOME/glm53-finalize.log}
IMAGE=${GLM53_IMAGE:-glm53-vllm-gb10:nope-sm121-topk-compact-ray-2.58}
SSH="ssh -o BatchMode=yes"

exec > >(tee -a "$LOG") 2>&1

timestamp() { date -u +%FT%TZ; }
remote() { local host=$1; shift; $SSH "mpfaffenberger@$host" "$@"; }

wait_for_pid() {
  local pid=$1 timeout=$2 elapsed=0
  while kill -0 "$pid" 2>/dev/null; do
    if ((elapsed >= timeout)); then
      echo "ERROR: timed out waiting for pid $pid" >&2
      return 1
    fi
    echo "$(timestamp) waiting for fanout pid=$pid elapsed=${elapsed}s"
    sleep 30
    elapsed=$((elapsed + 30))
  done
}

if [[ -f "$FANOUT_PID_FILE" ]]; then
  wait_for_pid "$(cat "$FANOUT_PID_FILE")" 3600
fi
if ! grep -q 'FANOUT_DONE' "$HOME/glm53-fanout.log"; then
  echo "ERROR: fanout did not report FANOUT_DONE" >&2
  tail -50 "$HOME/glm53-fanout.log"
  exit 1
fi

HF_HOME=$HOME/.cache/huggingface python3 "$ROOT/scripts/verify_checkpoint.py"
for host in 10.0.0.150 10.0.0.13; do
  scp -q "$ROOT/scripts/verify_checkpoint.py" "mpfaffenberger@$host:~/verify_glm53.py"
  remote "$host" 'python3 ~/verify_glm53.py'
done

# The patched image must be built once, tested on SM121, and distributed to
# every worker before finalization. Do not silently substitute the stock image.
for host in 10.0.0.150 10.0.0.13; do
  remote "$host" "docker image inspect $IMAGE >/dev/null"
  scp -q "$ROOT/scripts/glm53_node_up.sh" "mpfaffenberger@$host:~/glm53_node_up.sh"
done

# The DeepSeek head has already exited. Stop its idle workers only when each GLM
# replacement is fully staged and ready to join Ray.
remote 10.0.0.150 \
  'chmod +x ~/glm53_node_up.sh; docker stop deepseek-v4-tp4-vllm-dspark-1 >/dev/null 2>&1 || true; ~/glm53_node_up.sh worker 10.0.0.150 10.0.0.46 auto'
remote 10.0.0.13 \
  'chmod +x ~/glm53_node_up.sh; docker stop deepseek-v4-tp4-vllm-dspark-1 >/dev/null 2>&1 || true; ~/glm53_node_up.sh worker 10.0.0.13 10.0.0.46 auto'

for attempt in $(seq 1 30); do
  status=$(docker exec glm53-tp4 ray status 2>&1 || true)
  if grep -Eq '0\.0/4\.0 GPU|4\.0 GPU' <<<"$status"; then
    echo "$(timestamp) Ray reports four GPUs"
    break
  fi
  if ((attempt == 30)); then
    echo "ERROR: Ray did not reach four GPUs" >&2
    echo "$status"
    exit 1
  fi
  sleep 10
done

"$ROOT/scripts/glm53_serve.sh"

for attempt in $(seq 1 180); do
  if curl -fsS --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "$(timestamp) API ready"
    python3 "$ROOT/scripts/smoke_bench.py" http://127.0.0.1:8000
    echo "$(timestamp) GLM53_LAUNCH_OK"
    exit 0
  fi
  if grep -q 'SERVE_EXIT=' "$HOME/glm53-serve.log"; then
    echo "ERROR: server exited before readiness" >&2
    tail -200 "$HOME/glm53-serve.log"
    exit 1
  fi
  if ((attempt % 6 == 0)); then
    echo "$(timestamp) waiting for API; attempt=$attempt"
    tail -5 "$HOME/glm53-serve.log" 2>/dev/null || true
  fi
  sleep 10
done

echo "ERROR: API readiness timed out" >&2
tail -200 "$HOME/glm53-serve.log"
exit 1
