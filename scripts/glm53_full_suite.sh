#!/usr/bin/env bash
# Strict Spark Arena v2 suite with fail-closed validation: pre/post C=1 gates
# and the validated llama-benchy matrix through 65K. Exits non-zero unless
# every stage passes, so an invalid run can never be mistaken for a valid one.
#
# Set GLM53_ENFORCE_EAGER=1 to run the whole suite against an eager engine.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG=$HOME/glm53-full-suite.log
STAMP=$(date -u +%Y%m%d-%H%M%S)
PROFILE=native
[[ "${GLM53_ENFORCE_EAGER:-0}" == 1 ]] && PROFILE=native-eager
RESULT=$ROOT/results/llama-benchy-spark-arena-v2-$PROFILE-$STAMP
mark() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

: > "$LOG"
printf '%s\n' "$RESULT" > "$HOME/glm53-full-suite.current"
mark "SUITE_START profile=$PROFILE result=$RESULT"

for _ in $(seq 1 360); do
  curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
  docker exec glm53-tp4 pgrep -f 'vllm serve' >/dev/null 2>&1 || {
    mark SUITE_HEAD_DIED; exit 1; }
  sleep 10
done
if ! curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  mark SUITE_API_TIMEOUT; exit 1
fi
mark SUITE_API_READY

cd "$ROOT"
if ! GLM53_BENCH_RUNS=3 ./scripts/bench_c1.sh > "$HOME/glm53-suite-pre-c1.log" 2>&1; then
  mark "SUITE_PRE_C1_FAILED $(tail -1 "$HOME/glm53-suite-pre-c1.log")"; exit 1
fi
mark "SUITE_PRE_C1_OK $(grep C1_RESULT "$HOME/glm53-suite-pre-c1.log")"

mark SUITE_BENCH_START
./scripts/run_llama_benchy.sh "$RESULT"
rc=$?
mark "SUITE_BENCH_EXIT=$rc"
(( rc == 0 )) || exit "$rc"

if ! GLM53_BENCH_RUNS=3 ./scripts/bench_c1.sh > "$HOME/glm53-suite-post-c1.log" 2>&1; then
  mark "SUITE_POST_C1_FAILED $(tail -1 "$HOME/glm53-suite-post-c1.log")"; exit 1
fi
mark "SUITE_POST_C1_OK $(grep C1_RESULT "$HOME/glm53-suite-post-c1.log")"
mark SUITE_VALIDATED
