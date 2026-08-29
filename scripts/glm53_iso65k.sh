#!/usr/bin/env bash
# Isolate the KDA chunk-index wedge on a FRESH engine.
#
# Evidence being tested: at 65K x c10 all four TP ranks parked on the identical
# frame -- prepare_chunk_indices -> .tolist() device-to-host sync inside the
# @tensor_cache KDA path -- twice, six minutes apart, GPUs at 96% / ~20 W.
# The full ladder had ~540 requests of accumulated scheduler and prefix-cache
# state when it wedged. This script asks whether that accumulation is required,
# which decides whether we get a ~20 minute reproducer loop or a ~2 hour one.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HEAD=${GLM53_HEAD:-10.0.0.46}
WORKERS=${GLM53_WORKERS:-"10.0.0.13 10.0.0.150 10.0.0.246"}
DEPTHS=${DEPTHS:-65535}
CONCURRENCIES=${CONCURRENCIES:-5 10}
RUNS=${RUNS:-3}
API=${GLM53_API:-http://127.0.0.1:8000/v1/models}
STAMP=$(date -u +%Y%m%d-%H%M%S)
RESULT=$ROOT/results/iso-${DEPTHS// /_}-c${CONCURRENCIES// /_}-$STAMP
LOG=$HOME/glm53-iso.log
: >"$LOG"
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

cd "$ROOT"

say "ISO_START depths=[$DEPTHS] concurrency=[$CONCURRENCIES] runs=$RUNS image=${GLM53_IMAGE:-launcher-default} result=$RESULT"

# SKIP_PRELUDE=1 probes an already-running engine instead of recovering and
# relaunching, so a healthy ~20 minute cold start is not thrown away.
if [[ "${SKIP_PRELUDE:-0}" != "1" ]]; then
./scripts/glm53_gpu_reset.sh "$HEAD" "$WORKERS" >>"$LOG" 2>&1 \
  || { say "ISO_RESET_FAILED"; exit 1; }

GLM53_ENFORCE_EAGER=1 ./scripts/glm53_native_launch.sh start >>"$LOG" 2>&1 \
  || { say "ISO_LAUNCH_FAILED"; exit 1; }
fi

# The API wait is unconditional: SKIP_PRELUDE only skips recovery and launch,
# never the readiness gate. Cold start is ~9.3 min per weight pass (two passes) plus KV profiling and
# autotuning, so ~18-22 min is normal. A 15-minute budget aborted a healthy
# engine mid-load and produced a bogus ISO_API_TIMEOUT.
for _ in $(seq 1 "${API_WAIT_LOOPS:-180}"); do
  [[ "$(curl -sS -o /dev/null -w '%{http_code}' --max-time 4 "$API" 2>/dev/null)" == 200 ]] \
    && { say "ISO_API_READY"; break; }
  sleep 10
done
[[ "$(curl -sS -o /dev/null -w '%{http_code}' --max-time 4 "$API" 2>/dev/null)" == 200 ]] \
  || { say "ISO_API_TIMEOUT"; exit 1; }

mkdir -p "$RESULT"
if ! GLM53_BENCH_RUNS=3 ./scripts/bench_c1.sh >"$RESULT/pre-c1.log" 2>&1; then
  say "ISO_PRE_C1_FAILED $(grep -o 'C1_RESULT.*' "$RESULT/pre-c1.log" | tail -1)"
  exit 1
fi
say "ISO_PRE_C1_OK $(grep -o 'C1_RESULT.*' "$RESULT/pre-c1.log" | tail -1)"

# Armed before the ladder so the first sustained stall is captured live.
PROGRESS="$RESULT/progress.jsonl" WATCH_LOG="$LOG" \
  UNTIL='ISO_BENCH_EXIT\|ISO_DONE\|ISO_PRE_C1_FAILED\|ISO_API_TIMEOUT' \
  nohup ./scripts/glm53_stall_watch.sh >"$HOME/glm53-isowatch.log" 2>&1 &
watcher=$!
say "ISO_WATCHER_ARMED pid=$watcher"

DEPTHS="$DEPTHS" CONCURRENCIES="$CONCURRENCIES" RUNS="$RUNS" \
  ./scripts/run_llama_benchy.sh "$RESULT" >>"$RESULT/runner.log" 2>&1
rc=$?
say "ISO_BENCH_EXIT=$rc"

if GLM53_BENCH_RUNS=3 ./scripts/bench_c1.sh >"$RESULT/post-c1.log" 2>&1; then
  say "ISO_POST_C1_OK $(grep -o 'C1_RESULT.*' "$RESULT/post-c1.log" | tail -1)"
else
  say "ISO_POST_C1_FAILED $(grep -o 'C1_RESULT.*' "$RESULT/post-c1.log" | tail -1)"
fi
say "ISO_DONE rc=$rc"
exit "$rc"
