# TROUBLESHOOTING — GLM-5.3 FP8 on GB10

## Runtime does not recognize `Glm5NextForConditionalGeneration`

The image predates GLM-5.3 integration. DeepSeek V4 support does not imply GLM-5.3 support. Use a vLLM 0.27.0+ integration image or build the integration commit for aarch64/`sm_121a`.

## FlashInfer sparse-MLA import or dispatch failure

GLM-5.3's NoPE sparse MLA requires FlashInfer 0.6.17+ in the official vLLM recipe. Confirm both the Python package and compiled extension match the image's torch/CUDA ABI. An x86 or GB200 wheel is not an aarch64 GB10 wheel wearing a fake moustache.

## FP8 kernel reports unsupported architecture

The checkpoint uses blockwise FP8 (`128×128`). Confirm the selected quantized MoE/GEMM backend has `sm_121a` kernels. Hopper-or-newer support in an upstream recipe does not guarantee GB10 support for every compiled backend. Capture the exact operator and backend before changing launch flags.

## OOM during weight load

The checkpoint is 305.79 GiB, roughly 76.45 GiB of files per TP rank before runtime overhead. Confirm TP=4 is formed before loading. Start with utilization 0.80 and 256K context. Do not try BF16. Check host memory consumers because GB10 uses unified memory.

## NCCL silently stalls at first collective

Check all four fabric NICs for MTU 9000. A reimaged node at MTU 1500 has previously caused a silent RoCE stall with no useful NCCL warning. Confirm HCA, socket interface, per-node RoCE-v2 GID index, and `/dev/infiniband` visibility inside every container.

## `Too many open files`

Set container `nofile` soft/hard to 1048576. Reimaged nodes may inherit Docker's 1024 limit. Verify inside the running container with `ulimit -n`.

## MTP initialization/capture fails

Set `GLM53_MTP_TOKENS=0` and retry the baseline. The checkpoint has one next-token prediction layer and upstream recommends k=5, but baseline model support and MTP support are separate gates on GB10.

## Reasoning or tool calls are malformed

Keep `--reasoning-parser glm45`, `--tool-call-parser glm47`, and `--enable-auto-tool-choice`. The checkpoint defaults to thinking. Check `message.reasoning_content` separately from `message.content`.

## Ray kills a healthy-looking worker

Check disk utilization. Every node stores the complete ~306 GiB Hugging Face snapshot, and Ray's disk monitor may evict processes above 95%. Also check container OOM state and restart count; API silence is a symptom, not a diagnosis.

## `pe_dim must be 64 for fp8_ds_mla`

The stock dedicated image selects `FLASHINFER_MLA_SPARSE_SM120` on GB10, canonicalizes `auto` to packed `fp8_ds_mla`, and rejects GLM-5.3 NoPE (`qk_rope_head_dim=0`). Use the patched image from this repository. Its fixed-ABI adapter stores the 512 FP8 latent bytes and four FP32 scales normally, zero-fills FlashInfer's fixed 128-byte RoPE region, pads queries with 64 zeros only at the backend boundary, and slices GLM's 2176-wide ranked index buffer to FlashInfer's 2048-entry ABI. Run both focused tests before launching; do not paper over this with BF16 KV, which the SM120 backend does not support.

## Distributed FlashInfer autotune stalls after KV allocation

The generic FlashInfer autotuner can hang all four ranks in its synchronized
8192-token dummy run after sparse-MLA autotuning has already completed. Typical
symptoms are four GPUs near 96% utilization, low power, no capture progress,
and repeated `No available shared memory broadcast block` messages.

The recipe disables this optional tuner by default with
`--no-enable-flashinfer-autotune`. Set `GLM53_FLASHINFER_AUTOTUNE=1` only for a
controlled tuning experiment. If a stalled run leaves GB10 unified memory
pinned, remove the containers, reload `nvidia_uvm`, drop filesystem caches, and
verify about 117 GiB is available on every node before restarting.

## 1M context fails after 256K succeeds

That is capacity tuning, not baseline failure. Record live KV-cache tokens, lower max sequences, and raise memory utilization cautiously. Test 512K before 1M after the SM121 kernel blocker is resolved. FP8 KV reduces cache cost but does not repeal arithmetic.
