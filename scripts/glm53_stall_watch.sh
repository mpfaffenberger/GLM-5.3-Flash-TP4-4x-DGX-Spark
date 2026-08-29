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
PROGRESS=${PROGRESS:?set PROGRESS to the benchmark progress.jsonl being watched}
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
  local dir=$OUT/$(date -u +%H%M%S)
  mkdir -p "$dir"
  echo "$(date -u +%FT%TZ) STALL_CAPTURE_START $dir"
  dump_rank local "$dir/rank-head.txt"
  nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader \
    > "$dir/rank-head-gpu.txt" 2>&1
  for ip in $WORKERS; do
    scp -q "$PYSPY" "$ip:/tmp/py-spy"
    dump_rank "$ip" "$dir/rank-$ip.txt"
    ssh -o BatchMode=yes "$ip" \
      "nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader" \
      > "$dir/rank-$ip-gpu.txt" 2>&1
  done
  docker logs --since 20m "$CONTAINER" > "$dir/head-log-20m.txt" 2>&1
  date -u +%FT%TZ > "$dir/DONE"
  echo "$(date -u +%FT%TZ) STALL_CAPTURE_DONE $dir"
}

captured=0
prev_ends=-1
stale=0
while ! grep -q "$UNTIL" "$WATCH_LOG" 2>/dev/null; do
  hits=$(docker logs --since 90s "$CONTAINER" 2>&1 | grep -c "$PATTERN" || true)
  # The shm-broadcast notice is INFO-level and byte-identical for a healthy
  # engine doing >60s of legitimate work (weight load, KV profiling, a 100K
  # prefill) and for a real deadlock, so it cannot be the sole trigger.
  # Require benchmark progress to be frozen as well.
  ends=$(grep -c '"type": *"request_end"' "$PROGRESS" 2>/dev/null || echo 0)
  if (( hits > 0 )) && [[ "$ends" == "$prev_ends" ]]; then
    ((stale++))
  else
    stale=0
  fi
  prev_ends=$ends
  if (( stale >= 2 )); then
    ((captured++))
    capture
    # Re-arm after a cooldown so a later wedge in the same run is still caught.
    sleep 300
    stale=0
    prev_ends=-1
  fi
  sleep 25
done
echo "$(date -u +%FT%TZ) STALL_WATCH_DONE captures=$captured"
