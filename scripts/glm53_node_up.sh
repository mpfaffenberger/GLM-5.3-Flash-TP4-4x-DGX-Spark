#!/usr/bin/env bash
# Start one GLM-5.3 TP=4 container and start/join Ray.
# Usage: glm53_node_up.sh <head|worker> <self_fabric_ip> <head_fabric_ip> <gid_index|auto>
set -euo pipefail

if [[ $# -ne 4 || ! "$1" =~ ^(head|worker)$ ]]; then
  echo "usage: $0 <head|worker> <self_fabric_ip> <head_fabric_ip> <gid_index|auto>" >&2
  exit 2
fi

ROLE=$1
SELF_IP=$2
HEAD_IP=$3
GID=$4
IMAGE=${GLM53_IMAGE:-glm53-vllm-gb10:nope-sm121-topk-compact-v2-ray-2.58}
NAME=${GLM53_CONTAINER_NAME:-glm53-tp4}
HF_CACHE=${HF_HOME:-$HOME/.cache/huggingface}
MODEL_CACHE_DIR=${GLM53_MODEL_CACHE_DIR:-models--unsloth--GLM-5.3-Flash-FP8}
HCA=${GLM53_NCCL_HCA:-rocep1s0f1}
SOCKET_IFACE=${GLM53_SOCKET_IFACE:-enp1s0f1np1}
MTU_IFACE=${GLM53_MTU_IFACE:-enp1s0f1np1}

if [[ "$GID" == auto ]]; then
  GID=$(show_gids | awk -v ip="$SELF_IP" '
    $5 == ip && $6 == "v2" && first == "" { first = $3 }
    END { print first }
  ')
  if [[ -z "$GID" ]]; then
    echo "ERROR: no RoCE-v2 GID found for $SELF_IP" >&2
    exit 1
  fi
fi

if [[ ! -d /dev/infiniband ]]; then
  echo "ERROR: /dev/infiniband is missing on the host" >&2
  exit 1
fi
if [[ ! -d "$HF_CACHE/hub/$MODEL_CACHE_DIR" ]]; then
  echo "ERROR: checkpoint $MODEL_CACHE_DIR is not staged under $HF_CACHE/hub" >&2
  exit 1
fi

MTU=$(ip -o link show "$MTU_IFACE" | sed -n 's/.* mtu \([0-9][0-9]*\).*/\1/p')
if [[ "$MTU" != 9000 ]]; then
  echo "ERROR: $MTU_IFACE MTU is ${MTU:-unknown}; require 9000 before NCCL bring-up" >&2
  exit 1
fi

# Optional extra container env (space-separated KEY=VALUE pairs), e.g.
# GLM53_DOCKER_ENV='CUDA_LAUNCH_BLOCKING=1' for synchronous-launch debugging.
EXTRA_ENV_ARGS=()
for kv in ${GLM53_DOCKER_ENV:-}; do
  EXTRA_ENV_ARGS+=(-e "$kv")
done

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --restart no --network host --ipc host --shm-size 64gb --gpus all \
  "${EXTRA_ENV_ARGS[@]}" \
  --entrypoint /bin/bash \
  --device /dev/infiniband:/dev/infiniband \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
  --cap-add IPC_LOCK \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e VLLM_HOST_IP="$SELF_IP" \
  -e NCCL_IB_HCA="$HCA" \
  -e NCCL_SOCKET_IFNAME="$SOCKET_IFACE" \
  -e GLOO_SOCKET_IFNAME="$SOCKET_IFACE" \
  -e TP_SOCKET_IFNAME="$SOCKET_IFACE" \
  -e NCCL_IB_GID_INDEX="$GID" \
  -e NCCL_CROSS_NIC=1 -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_DEBUG=WARN \
  "$IMAGE" -lc 'sleep infinity' >/dev/null

sleep 3
echo "container=$NAME role=$ROLE fabric_ip=$SELF_IP gid=$GID hca=$HCA socket_iface=$SOCKET_IFACE mtu=$MTU"
docker exec "$NAME" bash -lc '
  test "$(ulimit -n)" -ge 1048576
  test -d /dev/infiniband
  python3 - <<"PY"
import torch, vllm
print("cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())
print("capability", torch.cuda.get_device_capability())
print("vllm", vllm.__version__)
PY'

if [[ "$ROLE" == head ]]; then
  docker exec "$NAME" ray start --head --node-ip-address "$SELF_IP" --port 6379 \
    --num-gpus 1 --disable-usage-stats
else
  docker exec "$NAME" ray start --address "$HEAD_IP:6379" \
    --node-ip-address "$SELF_IP" --num-gpus 1 --disable-usage-stats
fi

echo "=== NODE $SELF_IP ($ROLE) UP ==="
