#!/usr/bin/env bash
# Persistently watch one B12x trial and preserve a compact status timeline.
set -uo pipefail

TRIAL_PID_FILE=${B12X_TRIAL_PID_FILE:-$HOME/glm53-nvfp4-b12x-mtp0-watch.pid}
TRIAL_LOG=${B12X_TRIAL_LOG:-$HOME/glm53-nvfp4-b12x-mtp0-watch.log}
SERVE_LOG=${B12X_SERVE_LOG:-$HOME/glm53-nvfp4-b12x-mtp0-serve.log}
RESULT_LOG=${B12X_RESULT_LOG:-$HOME/glm53-nvfp4-b12x-mtp0-c1.log}
WATCH_LOG=${B12X_WATCH_LOG:-$HOME/glm53-nvfp4-b12x-monitor.log}
INTERVAL=${B12X_WATCH_INTERVAL:-30}
TIMEOUT=${B12X_WATCH_TIMEOUT:-5400}
START=$(date +%s)
LAST_EVENT=""

mkdir -p "$(dirname "$WATCH_LOG")"
: >"$WATCH_LOG"

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$WATCH_LOG"
}

latest_event() {
  grep -E \
    "Using .*NvFp4|B12x|b12x|Loading safetensors checkpoint shards: 100%|Model loading took|Available KV cache memory|GPU KV cache size|Graph capturing finished|Application startup complete|C1_RESULT|TRIAL_OK|ERROR|Traceback|RuntimeError|ValueError|MMA|SERVE_EXIT" \
    "$SERVE_LOG" "$TRIAL_LOG" "$RESULT_LOG" 2>/dev/null | tail -1
}

capture_diagnostics() {
  log "DIAGNOSTICS_BEGIN"
  {
    echo "=== trial tail ==="
    tail -80 "$TRIAL_LOG" 2>/dev/null || true
    echo "=== serve tail ==="
    tail -160 "$SERVE_LOG" 2>/dev/null || true
    echo "=== local container ==="
    docker inspect glm53-tp4 \
      --format 'status={{.State.Status}} oom={{.State.OOMKilled}} restarts={{.RestartCount}} image={{.Config.Image}}' \
      2>/dev/null || true
    echo "=== ray ==="
    docker exec glm53-tp4 ray status 2>/dev/null || true
  } >>"$WATCH_LOG" 2>&1
  log "DIAGNOSTICS_END"
}

if [[ ! -s "$TRIAL_PID_FILE" ]]; then
  log "FAIL missing trial PID file: $TRIAL_PID_FILE"
  exit 2
fi
TRIAL_PID=$(cat "$TRIAL_PID_FILE")
log "START trial_pid=$TRIAL_PID interval=${INTERVAL}s timeout=${TIMEOUT}s"

while true; do
  now=$(date +%s)
  elapsed=$((now - START))

  event=$(latest_event || true)
  if [[ -n "$event" && "$event" != "$LAST_EVENT" ]]; then
    log "EVENT $event"
    LAST_EVENT=$event
  fi

  if grep -q 'C1_RESULT' "$RESULT_LOG" 2>/dev/null; then
    result=$(grep 'C1_RESULT' "$RESULT_LOG" | tail -1)
    log "SUCCESS $result"
    exit 0
  fi

  api=down
  if curl -fsS --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    api=ready
  fi

  if ! kill -0 "$TRIAL_PID" 2>/dev/null; then
    log "FAIL trial exited before a C1 result api=$api elapsed=${elapsed}s"
    capture_diagnostics
    exit 1
  fi

  if ((elapsed >= TIMEOUT)); then
    log "FAIL watcher timeout api=$api elapsed=${elapsed}s"
    capture_diagnostics
    exit 124
  fi

  container=$(docker inspect glm53-tp4 \
    --format '{{.State.Status}}/oom={{.State.OOMKilled}}/r={{.RestartCount}}' \
    2>/dev/null || echo missing)
  log "HEARTBEAT elapsed=${elapsed}s api=$api container=$container"
  sleep "$INTERVAL"
done
