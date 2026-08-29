#!/usr/bin/env bash
# Run the isolated 65K control with the host-metadata KDA patch image.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${GLM53_KDA_IMAGE:-glm53-vllm-gb10:nope-sm121-topk-compact-v2-kda-hostmeta}
HEAD=${GLM53_HEAD:-10.0.0.46}
WORKERS=${GLM53_WORKERS:-"10.0.0.13 10.0.0.150 10.0.0.246"}
EXPECTED_LABEL=host-metadata-no-d2h-v1

verify_image() {
  local host=$1 value
  if [[ "$host" == "$HEAD" ]]; then
    value=$(docker image inspect --format \
      '{{index .Config.Labels "ai.code-puppy.glm53-kda-chunk-indices"}}' "$IMAGE")
  else
    value=$(ssh -o BatchMode=yes "$host" docker image inspect --format \
      "'{{index .Config.Labels \"ai.code-puppy.glm53-kda-chunk-indices\"}}'" \
      "'$IMAGE'")
  fi
  [[ "$value" == "$EXPECTED_LABEL" ]] || {
    echo "KDA patch image verification failed on $host: label=$value" >&2
    return 1
  }
  echo "$host image=$IMAGE label=$value"
}

for host in "$HEAD" $WORKERS; do
  verify_image "$host"
done

[[ "${GLM53_KDA_VERIFY_ONLY:-0}" == 1 ]] && exit 0

cd "$ROOT"
GLM53_IMAGE="$IMAGE" ./scripts/glm53_iso65k.sh
