# TROUBLESHOOTING — GLM-5.3 FP8 on GB10

## Runtime does not recognize `Glm5NextForConditionalGeneration`

The image predates GLM-5.3 integration. DeepSeek V4 support does not imply GLM-5.3 support. Use a vLLM 0.27.0+ integration image or build the integration commit for aarch64/`sm_121a`.

## FlashInfer sparse-MLA import or dispatch failure

GLM-5.3's NoPE sparse MLA requires FlashInfer 0.6.17+ in the official vLLM recipe. Confirm both the Python package and compiled extension match the image's torch/CUDA ABI. An x86 or GB200 wheel is not an aarch64 GB10 wheel wearing a fake moustache.

## FP8 kernel reports unsupported architecture

The checkpoint uses blockwise FP8 (`128×128`). Confirm the selected quantized MoE/GEMM backend has `sm_121a` kernels. Hopper-or-newer support in an upstream recipe does not guarantee GB10 support for every compiled backend. Capture the exact operator and backend before changing launch flags.

## OOM during weight load

The checkpoint is 305.79 GiB, roughly 76.45 GiB of files per TP rank before runtime overhead. Confirm TP=4 is formed before loading. Use the validated 0.82 production utilization and 256K context; drop to 0.80 only while isolating startup headroom issues. Do not try BF16. Check host memory consumers because GB10 uses unified memory.

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

## NVFP4 is fast enough to benchmark but returns repeated tokens

Treat coherence as a hard gate before publishing throughput. The tested ModelOpt
NVFP4 checkpoint with `FLASHINFER_CUTLASS`, MTP k=5, and memory utilization
0.70 reached 17.42 tok/s median C=1, but deterministic and sampled requests both
degenerated into repeated `lockhandle`/`lock` tokens with thinking enabled or
disabled. Those numbers are not a valid serving result. The experimental B12x
backend also spent over 40 minutes in thousands of synchronous per-expert
`_load_w2` copies and never reached inference. Use the validated FP8/DeepGemm
profile until NVFP4 passes an independent coherence gate and B12x's expert
layout can be prepacked or loaded in batches.

GB10 unified memory makes retained filesystem cache harmful during these large
loads. Drop caches before a controlled relaunch; do not preserve tens of GiB of
checkpoint page cache while allocating model weights.

## `llama-benchy` wedges after three streamed requests

The tested vLLM Rust frontend can retain shared-memory broadcast blocks when
`llama-benchy` reuses one HTTP/1.1 connection. The engine then reports no active
request while new completions hang and `shm_broadcast.py` reports no available
block. `scripts/run_llama_benchy.sh` applies the narrow, idempotent
`patch_llama_benchy_keepalive.py` compatibility patch to use one connection per
streamed request. It also disables the optional `return_token_ids` extension;
OpenAI usage metadata remains the authoritative aggregate token count.

If the broadcaster is already wedged, stopping the client is insufficient.
Stop the API, remove all four rank containers, reload `nvidia_uvm`, drop caches,
and relaunch native TP=4 from clean unified memory before benchmarking again.

## Requests beyond 2048 total tokens wedge without top-k compaction

GLM-5.3's sparse attention selects `index_topk = 2048` tokens. vLLM's kpool
indexer short-circuits contexts at or below 2048 (causal `arange`, no real
selection), so small benchmarks can pass while the first request beyond 2048
wedges all four GPUs at ~96% utilization and ~20 W.

Synchronous-launch captures in
`results/sm121-sparse-mla-hang-diagnosis/` proved that one rank never returns
from FlashInfer's SM120 sparse-MLA paged attention while the other ranks wait
in the post-attention `ncclAllReduce`. The remapped kpool table contained
interspersed `-1` slots, but the kernel requires a dense valid prefix with an
explicit length.

The production fix is
`patches/patch_sm121_sparse_topk_compact.py`: call
`triton_convert_req_index_to_global_index(..., return_valid_counts=True)` to
compact valid indices to `[0, count)`, then pass those counts as `seq_lens`
(the kernel's `topk_length`). Use image
`glm53-vllm-gb10:nope-sm121-topk-compact-v2-ray-2.58`. Validation passed 1024,
1900, 2100, 4096, and 8192-token probes, a 3200-token recall gate, and the 2K
llama-benchy canary with no C=1 regression. V2 also mirrors the generic
backend's CUDA-graph padding guard: zero-length rows launch against one dummy
slot with `topk_length=1`, then their ignored output is zeroed. Without that
guard, the 16K-context concurrency-10 warmup can wedge the same kernel.

Legacy Ray runs used `RAY_memory_usage_threshold=0.97` because GB10 GPU
allocations count as host unified memory. The published native profile does not
run Ray; keep this setting only when reproducing historical Ray evidence.

## KDA wedge at long context and concurrency

The host-metadata patch fixes two concrete defects in GLM's model-specific KDA
path:

- an in-forward CUDA-to-host `.tolist()` synchronization;
- stale identity-cached chunk maps from mutable scheduler CPU tensors.

It also brings GLM mixed decode/prefill handling in line with Qwen GDN. Focused
65K × `{c5,c10}` validation completed 90/90 requests and 8/8 result rows. The
old accumulated-state failure at request 544/65K c10 was crossed cleanly.

A deeper KDA kernel defect remains at 100K. The tested scheduler ladder is:

| max sequences | 100K c5 | 100K c10 |
|---:|---|---|
| 10 | wedge | not reached |
| 5 | wedge after 10 requests | not reached |
| 4 | pass, 30/30 | wedge after 5 c10 requests |

The wedge signature is unambiguous:

- `request_end` progress stops;
- `No available shared memory broadcast block found in 60 seconds` repeats;
- all GPUs report about 96% utilization at only 19–24 W;
- live rank stacks diverge inside fused chunk KDA kernels.

The API `/v1/models` route may still return 200 because the frontend process is
alive while the engine worker is spinning. API health alone is therefore not a
progress signal. `scripts/glm53_stall_watch.sh` requires repeated notices plus
stagnant request accounting before taking a live stack capture.

Representative captures:

- max-seqs 10, 100K c5: `~/glm53-hang-dumps/stall-20260829-231037/014943`
- max-seqs 5, 100K c5: `~/glm53-hang-dumps/stall-20260830-025747/030903`
- max-seqs 4, 100K c10: `~/glm53-hang-dumps/stall-20260830-033341/041254`

Do not promote `max-num-seqs` above 4, and do not advertise 100K c10 on this
runtime. Nominal KV capacity is not the blocker: the max-seqs-5 c5 wedge
occurred around half cache usage. A blocking host-to-device metadata experiment
also reproduced the wedge and was reverted, falsifying that race hypothesis.

A wedged compute engine survives container removal. After `docker rm -f`, every
fresh vLLM process can inherit the dead device. Recover all ranks with the full
module-stack reset:

```bash
./scripts/glm53_gpu_reset.sh \
  10.0.0.46 10.0.0.13 10.0.0.150 10.0.0.246
```

The script stops persistence, reloads `nvidia_drm`, `nvidia_modeset`,
`nvidia_uvm`, and `nvidia` in dependency order, then verifies idle utilization
and power before another launch.

## 1M context fails after 256K succeeds

That is capacity tuning, not baseline failure. Record live KV-cache tokens, lower max sequences, and raise memory utilization cautiously. Test 512K before 1M after the SM121 kernel blocker is resolved. FP8 KV reduces cache cost but does not repeal arithmetic.
