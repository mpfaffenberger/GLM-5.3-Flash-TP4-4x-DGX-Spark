#!/usr/bin/env bash
# Stage the pinned GLM-5.3 cache from .246 to the head, verify it, then fan out
# from the head to .150 and .13. Run on the head node.
set -euo pipefail

MODEL_DIR=models--unsloth--GLM-5.3-Flash-FP8
SOURCE=${GLM53_STAGE_SOURCE:-mpfaffenberger@10.0.0.246}
DESTINATIONS=${GLM53_STAGE_DESTINATIONS:-"mpfaffenberger@10.0.0.150 mpfaffenberger@10.0.0.13"}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
HUB=$HF_HOME/hub
LOCAL=$HUB/$MODEL_DIR
BWLIMIT_KB=${GLM53_RSYNC_BWLIMIT_KB:-300000}
SSH_OPTIONS=${GLM53_SSH_OPTIONS:-"-o BatchMode=yes -o StrictHostKeyChecking=accept-new"}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

mkdir -p "$LOCAL"
echo "=== pull $SOURCE -> $(hostname) ==="
rsync -a --partial --info=progress2 --bwlimit="$BWLIMIT_KB" \
  -e "ssh $SSH_OPTIONS" \
  "$SOURCE:.cache/huggingface/hub/$MODEL_DIR/" "$LOCAL/"

HF_HOME=$HF_HOME python3 "$ROOT/scripts/verify_checkpoint.py"

echo "=== fan out from $(hostname) ==="
pids=()
for destination in $DESTINATIONS; do
  (
    ssh $SSH_OPTIONS "$destination" "mkdir -p .cache/huggingface/hub/$MODEL_DIR"
    rsync -a --partial --info=progress2 --bwlimit="$BWLIMIT_KB" \
      -e "ssh $SSH_OPTIONS" "$LOCAL/" \
      "$destination:.cache/huggingface/hub/$MODEL_DIR/"
    echo "STAGED $destination"
  ) >"$HOME/glm53-stage-${destination##*@}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if ((failed)); then
  echo "ERROR: one or more worker fan-outs failed; inspect ~/glm53-stage-*.log" >&2
  exit 1
fi

echo "FANOUT_DONE $(date -u +%FT%TZ)"
