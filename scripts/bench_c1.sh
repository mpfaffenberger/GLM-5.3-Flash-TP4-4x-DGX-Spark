#!/usr/bin/env bash
# Repeatable C=1 decode benchmark matching the MiniMax recipe methodology.
set -euo pipefail

URL=${GLM53_BASE_URL:-http://127.0.0.1:8000}
MODEL=${GLM53_SERVED_NAME:-glm-5.3-flash-fp8}
RUNS=${GLM53_BENCH_RUNS:-5}
TOKENS=${GLM53_BENCH_TOKENS:-256}
PROMPT=${GLM53_BENCH_PROMPT:-Write a detailed guide to reselling sneakers:}

metric() {
  curl -fsS --max-time 10 "$URL/metrics" \
    | awk '/^vllm:generation_tokens_total/ {print $2; exit}'
}

body=$(python3 - "$MODEL" "$PROMPT" "$TOKENS" <<'PY'
import json
import sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": sys.argv[2]}],
    "max_tokens": int(sys.argv[3]),
    "temperature": 0,
    "ignore_eos": True,
}))
PY
)

rates=()
for run in $(seq 1 "$RUNS"); do
  before=$(metric)
  started=$(date +%s.%N)
  response=$(mktemp)
  status=$(curl -sS --max-time 300 -o "$response" -w '%{http_code}' \
    "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
  finished=$(date +%s.%N)
  after=$(metric)
  if [[ "$status" != 200 ]]; then
    cat "$response" >&2
    rm -f "$response"
    echo "run=$run HTTP=$status" >&2
    exit 1
  fi
  reported=$(python3 - "$response" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("usage", {}).get("completion_tokens", 0))
PY
)
  rm -f "$response"
  result=$(python3 - "$before" "$after" "$started" "$finished" <<'PY'
import sys
before, after, started, finished = map(float, sys.argv[1:])
wall = finished - started
tokens = after - before
print(f"{tokens / wall:.4f} {tokens:.0f} {wall:.4f}")
PY
)
  read -r rate measured wall <<<"$result"
  rates+=("$rate")
  printf 'run=%d reported=%s measured=%s wall=%.2fs rate=%.2f tok/s\n' \
    "$run" "$reported" "$measured" "$wall" "$rate"
done

python3 - "${rates[@]}" <<'PY'
import statistics
import sys
rates = list(map(float, sys.argv[1:]))
print(
    f"C1_RESULT runs={len(rates)} mean={statistics.mean(rates):.2f} "
    f"median={statistics.median(rates):.2f} min={min(rates):.2f} "
    f"max={max(rates):.2f} tok/s"
)
PY
