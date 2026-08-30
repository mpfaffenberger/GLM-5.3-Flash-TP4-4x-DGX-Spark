#!/usr/bin/env bash
set -euo pipefail

# Opt-in DFlash2 trial profile. The validated MTP3 defaults remain untouched.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export GLM53_IMAGE=${GLM53_IMAGE:-glm53-vllm-gb10:dflash2-experimental}
export GLM53_MAX_MODEL_LEN=${GLM53_MAX_MODEL_LEN:-8192}
export GLM53_MAX_NUM_SEQS=${GLM53_MAX_NUM_SEQS:-1}
export GLM53_GPU_MEMORY_UTILIZATION=${GLM53_GPU_MEMORY_UTILIZATION:-0.82}
export GLM53_ENFORCE_EAGER=${GLM53_ENFORCE_EAGER:-0}
export GLM53_SPECULATIVE_CONFIG=${GLM53_SPECULATIVE_CONFIG:-'{"model":"incoai/GLM-5.3-Flash-DFlash2","revision":"dc77ff1c99eeb2df044ee3d4f0094eb033fee410","method":"dflash","num_speculative_tokens":7,"draft_tensor_parallel_size":1,"attention_backend":"FLASH_ATTN","kv_cache_dtype":"auto"}'}

exec "$ROOT/scripts/glm53_native_launch.sh" "${1:-start}"
