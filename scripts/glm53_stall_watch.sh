#!/usr/bin/env bash
# Capture four-rank py-spy stacks at the FIRST sustained shared-memory-broadcast
# stall, i.e. while the engine is actually wedged instead of 30 minutes later
# when the sample_tokens RPC timeout kills it. Containers lack SYS_PTRACE, so
# py-spy runs on each host via sudo and the binary is copied to /tmp.
#
# Env: WATCH_LOG (progress/markers file), UNTIL (stop marker), GLM53_WORKERS.
set -uo pipefail

CONTAINER=${GLM53_CONTAINER:-glm53-tp4}
WATCH_LOG=${WATCH_LOG:?set WATCH_LOG to the file whose UNTIL marker ends the watch}
UNTIL=${UNTIL:-SUITE_}
WORKERS=${GLM53_WORKERS:-"10.0.0.13 10.0.0.150 10.0.0.246"}
PYSPY=${GLM53_PYSPY:-$HOME/.cache/uv/archive-v0/3eVEIjOvVkqR7Wxj/bin/py-spy}
OUT=$HOME/glm53-hang-dumps/stall-$(date -u +%Y%m%d-%H%M%S)
PATTERN='No available shared memory broadcast block'

dump_rank() { # $1 = host ("local" for this node), $2 = output file
  local host=$1 file=$2 cmd='docker top CONTAINER -eo pid,args | awk "/VLLM::Worker_TP/{print \$1; exit}"'
  cmd=${cmd//CONTAINER/$CONTAINER}
  if [[ $host == local ]]; then
    pid=$(bash -c "$cmd")
    [[ -n $pid ]] && sudo -n "$PYSPY" dump --pid "$pid" > "$file" 2>&1
  else
    pid=$(ssh -o BatchMode=yes "$host" "$cmd")
    [[ -n $pid ]] && ssh -o BatchMode=yes "$host" \
      "sudo -n /tmp/py-spy dump --pid '$pid'" > "$file" 2>&1
  fi
}

capture() {
  mkdir -p "$OUT"
  echo "$(date -u +%FT%TZ) STALL_CAPTURE_START $OUT"
  dump_rank local "$OUT/rank-head.txt"
  nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader \
    > "$OUT/rank-head-gpu.txt" 2>&1
  for ip in $WORKERS; do
    scp -q "$PYSPY" "$ip:/tmp/py-spy"
    dump_rank "$ip" "$OUT/rank-$ip.txt"
    ssh -o BatchMode=yes "$ip" \
      "nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader" \
      > "$OUT/rank-$ip-gpu.txt" 2>&1
  done
  docker logs --since 20m "$CONTAINER" > "$OUT/head-log-20m.txt" 2>&1
  date -u +%FT%TZ > "$OUT/DONE"
  echo "$(date -u +%FT%TZ) STALL_CAPTURE_DONE $OUT"
}

streak=0
captured=0
while ! grep -q "$UNTIL" "$WATCH_LOG" 2>/dev/null; do
  hits=$(docker logs --since 90s "$CONTAINER" 2>&1 | grep -c "$PATTERN" || true)
  if (( hits > 0 )); then ((streak++)); else streak=0; fi
  if (( streak >= 2 && captured == 0 )); then
    capture
    captured=1
  fi
  sleep 25
done
echo "$(date -u +%FT%TZ) STALL_WATCH_DONE captures=$captured"
