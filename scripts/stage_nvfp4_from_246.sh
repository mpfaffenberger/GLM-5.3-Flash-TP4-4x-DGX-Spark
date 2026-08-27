#!/usr/bin/env bash
# Wait for the pinned NVFP4 download on .246, verify it, then fan it out.
set -euo pipefail

SOURCE_HOST=${NVFP4_SOURCE_HOST:-10.0.0.246}
REMOTE_USER=${NVFP4_REMOTE_USER:-mpfaffenberger}
MODEL_DIR=models--LibertAIDAI--GLM-5.3-Flash-NVFP4
REVISION=11d73216cd636238e82e1d77fe1042ffab36e7fa
EXPECTED_SHARDS=120
EXPECTED_WEIGHT_BYTES=194660206040
REMOTE_ROOT=.cache/huggingface/hub/$MODEL_DIR
LOCAL_ROOT=$HOME/.cache/huggingface/hub/$MODEL_DIR
SSH=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

verify_snapshot() {
  local root=$1
  python3 - "$root/snapshots/$REVISION" "$EXPECTED_SHARDS" \
    "$EXPECTED_WEIGHT_BYTES" <<'PY'
import json
import sys
from pathlib import Path

snapshot = Path(sys.argv[1])
expected_shards = int(sys.argv[2])
expected_bytes = int(sys.argv[3])
index = json.loads((snapshot / "model.safetensors.index.json").read_text())
referenced = {snapshot / name for name in index["weight_map"].values()}
shards = sorted(snapshot.glob("model-*-of-*.safetensors"))
missing = [path for path in referenced if not path.is_file()]
total = sum(path.stat().st_size for path in shards)
if len(shards) != expected_shards or len(referenced) != expected_shards:
    raise SystemExit(
        f"wrong shard count: files={len(shards)} referenced={len(referenced)}"
    )
if missing:
    raise SystemExit(f"missing {len(missing)} referenced shards")
if total != expected_bytes:
    raise SystemExit(f"wrong weight bytes: {total} != {expected_bytes}")
print(f"NVFP4_CHECKPOINT_OK shards={len(shards)} bytes={total}")
PY
}

while ssh "${SSH[@]}" "$REMOTE_USER@$SOURCE_HOST" \
  'p=$(cat "$HOME/logs/glm53-nvfp4-download.pid"); kill -0 "$p" 2>/dev/null'; do
  echo "$(date -u +%FT%TZ) download still running"
  sleep 30
done

ssh "${SSH[@]}" "$REMOTE_USER@$SOURCE_HOST" \
  "test -d '$REMOTE_ROOT/snapshots/$REVISION'"
mkdir -p "$LOCAL_ROOT"
rsync -a --partial --info=progress2 -e "ssh ${SSH[*]}" \
  "$REMOTE_USER@$SOURCE_HOST:$REMOTE_ROOT/" "$LOCAL_ROOT/"
verify_snapshot "$LOCAL_ROOT"

pids=()
for host in 10.0.0.13 10.0.0.150; do
  ssh "${SSH[@]}" "$REMOTE_USER@$host" "mkdir -p '$REMOTE_ROOT'"
  rsync -a --partial --info=progress2 -e "ssh ${SSH[*]}" \
    "$LOCAL_ROOT/" "$REMOTE_USER@$host:$REMOTE_ROOT/" \
    >"$HOME/glm53-nvfp4-rsync-${host}.log" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

for host in 10.0.0.13 10.0.0.150; do
  ssh "${SSH[@]}" "$REMOTE_USER@$host" \
    "test -f '$REMOTE_ROOT/snapshots/$REVISION/model.safetensors.index.json' && \
     test \"\$(find -L '$REMOTE_ROOT/snapshots/$REVISION' -maxdepth 1 -name 'model-*-of-*.safetensors' | wc -l)\" -eq $EXPECTED_SHARDS"
done

echo "NVFP4_FANOUT_OK $(date -u +%FT%TZ)"
