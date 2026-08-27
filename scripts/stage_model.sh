#!/usr/bin/env bash
# Fan out the pinned GLM-5.3 FP8 Hugging Face cache directory over RoCE.
# Example: DESTINATIONS='user@10.0.0.150 user@10.0.0.13 user@10.0.0.246' ./scripts/stage_model.sh
set -euo pipefail

MODEL_DIR=models--unsloth--GLM-5.3-Flash-FP8
SRC=${HF_HOME:-$HOME/.cache/huggingface}/hub/$MODEL_DIR
DESTINATIONS=${DESTINATIONS:-}
BWLIMIT_KB=${BWLIMIT_KB:-0}
SSH_OPTIONS=${SSH_OPTIONS:--o StrictHostKeyChecking=accept-new -o BatchMode=yes}

if [[ ! -d "$SRC" ]]; then
  echo "ERROR: source model cache not found: $SRC" >&2
  exit 1
fi
if [[ -z "$DESTINATIONS" ]]; then
  echo "ERROR: set space-separated DESTINATIONS, e.g. user@10.0.0.150" >&2
  exit 2
fi

bw_args=()
if [[ "$BWLIMIT_KB" -gt 0 ]]; then
  bw_args+=(--bwlimit="$BWLIMIT_KB")
fi

for dst in $DESTINATIONS; do
  echo "=== staging to $dst ==="
  ssh $SSH_OPTIONS "$dst" "mkdir -p .cache/huggingface/hub/$MODEL_DIR"
  rsync -a --partial --info=progress2 "${bw_args[@]}" \
    -e "ssh $SSH_OPTIONS" "$SRC/" "$dst:.cache/huggingface/hub/$MODEL_DIR/"
done

echo "STAGE_DONE $(date -u +%FT%TZ)"
