#!/usr/bin/env bash
# Clear a wedged GB10 GPU (96% util at ~20W, spinning kernel) without a reboot.
#
# A hung CUDA context survives container removal: the compute engine stays
# "busy" and every fresh vLLM process inherits a dead device. The only
# software-only fix found on sm_121a is a full NVIDIA module reload, which
# requires that no process still holds /dev/nvidia*.
#
# Usage: scripts/glm53_gpu_reset.sh [head-ip] [worker-ip ...]
set -uo pipefail

HEAD=${1:-10.0.0.46}
shift 2>/dev/null || true
WORKERS=${*:-"10.0.0.13 10.0.0.150 10.0.0.246"}
CONTAINER=${GLM53_CONTAINER:-glm53-tp4}

log() { echo "$(date -u +%FT%TZ) $*"; }

# Ordering matters: nvidia_drm depends on nvidia_modeset, which depends on
# nvidia_uvm and the core module. Reload in the reverse order of removal.
RESET_SNIPPET='
docker rm -f '"$CONTAINER"' >/dev/null 2>&1 || true
sudo -n systemctl stop nvidia-persistenced.service 2>/dev/null || true
sudo -n modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia
sleep 2
sudo -n modprobe nvidia
sudo -n modprobe nvidia_uvm
sudo -n modprobe nvidia_modeset
sudo -n modprobe nvidia_drm
sudo -n systemctl start nvidia-persistenced.service
sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"
'

log "RESET_HEAD $HEAD"
eval "$RESET_SNIPPET"

for ip in $WORKERS; do
  log "RESET_WORKER $ip"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$ip" "$RESET_SNIPPET" &
done
wait

# Verify the wedge is actually gone: idle utilisation should be ~0% and power
# draw single-digit-to-teens watts. The pipeline is a string so that it can be
# evaluated both locally and over ssh (shell functions do not cross ssh).
PROBE='nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits \
  | awk -F", *" "{ printf \"util=%s%% power=%sW\n\", \$1, \$2 }"'
log "VERIFY"
bad=0
for ip in $HEAD $WORKERS; do
  if [[ "$ip" == "$HEAD" ]]; then out=$(bash -c "$PROBE")
  else out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$ip" "$PROBE")
  fi
  printf '%s %s\n' "$ip" "$out"
  if grep -qE 'util=([1-9][0-9]|100)%' <<<"$out"; then
    log "STILL_WEDGED $ip"
    bad=1
  fi
done
exit "$bad"
