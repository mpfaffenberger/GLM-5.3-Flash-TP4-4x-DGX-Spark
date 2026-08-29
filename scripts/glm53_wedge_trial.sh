#!/usr/bin/env bash
# One-shot: recover wedged GPUs, relaunch native TP=4, gate C=1, run the MoE
# wedge reproducer. Nothing here auto-declares success: a reproducer exit of 42
# is the *desired* signal while the defect is still open (it means the fast
# repro works), and any C=1 gate failure stops the chain.
#
# Env: GLM53_TRIAL_DECODING, GLM53_TRIAL_STAGGER, GLM53_TRIAL_ITERS,
#      GLM53_TRIAL_RAMP, GLM53_TRIAL_TARGET
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HEAD=${GLM53_HEAD:-10.0.0.46}
WORKERS=${GLM53_WORKERS:-"10.0.0.13 10.0.0.150 10.0.0.246"}
DECODING=${GLM53_TRIAL_DECODING:-4}
STAGGER=${GLM53_TRIAL_STAGGER:-1}
ITERS=${GLM53_TRIAL_ITERS:-4}
RAMP=${GLM53_TRIAL_RAMP:-}
TARGET=${GLM53_TRIAL_TARGET:-32768}
LOG=$HOME/glm53-wedge-trial.log

mark() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
: > "$LOG"
cd "$ROOT"

mark "TRIAL_START target=$TARGET decoding=$DECODING stagger=$STAGGER iters=$ITERS ramp=${RAMP:-none}"
./scripts/glm53_gpu_reset.sh "$HEAD" "$WORKERS" 2>&1 | tee -a "$LOG"
mark TRIAL_LAUNCH
./scripts/glm53_native_launch.sh start 2>&1 | tee -a "$LOG"

mark TRIAL_WAIT_API
for _ in $(seq 1 240); do
  curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
  sleep 10
done
if ! curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  mark TRIAL_API_TIMEOUT
  exit 1
fi
mark TRIAL_API_READY

if ! GLM53_BENCH_RUNS=3 ./scripts/bench_c1.sh > "$HOME/glm53-wedge-trial-c1.log" 2>&1; then
  mark "TRIAL_PRE_C1_FAILED $(tail -1 "$HOME/glm53-wedge-trial-c1.log")"
  exit 1
fi
mark "TRIAL_PRE_C1_OK $(grep C1_RESULT "$HOME/glm53-wedge-trial-c1.log")"

mark TRIAL_REPRO_START
python3 scripts/repro_moe_wedge.py \
  --target-tokens "$TARGET" --decoding "$DECODING" --stagger "$STAGGER" \
  --iterations "$ITERS" ${RAMP:+--ramp-depths "$RAMP"}
rc=$?
case "$rc" in
  42) mark TRIAL_WEDGE_REPRODUCED ;;
  0)  mark TRIAL_NO_REPRODUCTION ;;
  *)  mark "TRIAL_INFRA_ERROR rc=$rc" ;;
esac
exit "$rc"
