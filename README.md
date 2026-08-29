# GLM-5.3-Flash on 4× DGX Spark — TP=4, FP8/NVFP4, FP8 KV, MTP

Serve GLM-5.3-Flash across four NVIDIA DGX Sparks (GB10, `sm_121a`) over dedicated RoCE. The validated baseline uses [`unsloth/GLM-5.3-Flash-FP8`](https://huggingface.co/unsloth/GLM-5.3-Flash-FP8); the recipe also includes staging and launch support for [`LibertAIDAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4).

> **Status:** the patched FP8 TP=4 language path is validated on four GB10 nodes: all 62 shards load, NoPE sparse-MLA cache packing works, CUDA graphs capture, MTP works, smoke tests pass, and a clean restored launch measured **32.07 tok/s median C=1** with MTP k=3 plus sparse-only autotuning. NVFP4 is retained only as experimental research material: CUTLASS reached 17.42 tok/s but failed coherence, while B12x never completed its operational startup gate.
>
> **SM121 long-context fixes:** real sparse selection begins beyond GLM's `index_topk = 2048`. The production V2 image compacts valid entries to a dense prefix, passes explicit per-token `topk_length`, and safely substitutes CUDA-graph padding rows that would otherwise launch with length zero. Validation passed the original boundary probes, 16K/32K/65K at concurrency 10, and the exact 100K×c10 cache-capacity canary at GMU 0.82. Root-cause stack evidence remains in [`results/sm121-sparse-mla-hang-diagnosis/`](results/sm121-sparse-mla-hang-diagnosis/).
>
> **Multi-node topology:** the active launch path is native vLLM `mp` with `--nnodes 4`, rank 0 serving the OpenAI API and ranks 1–3 running headless over PyTorch distributed/NCCL. No Ray process runs in this topology; Ray support remains only as legacy fallback. The current image tag still contains `ray-2.58` because it was cut before the topology change.
>
> **Open defect (not yet green):** neither Ray nor native `mp` completes the full 104-row Spark Arena matrix. Ray wedged at repeated 32K×c10; native `mp` reaches 100K but wedges earlier at 32K×c5. Live four-rank stacks show TP0 still in the MoE router while TP1–TP3 have advanced into shared-expert FP8 input-quant, followed by shared-memory broadcast starvation and `RPC call to sample_tokens timed out`. Standalone 32K×c10 passes, so this is a sequence/mixed-batch-dependent defect rather than a broken kernel path. See [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) and `scripts/repro_moe_wedge.py`. No runtime image has been published until a clean 104/104 run exists.

## Why quantized weights

GLM-5.3-Flash is a 320B-total / 18B-active multimodal MoE. The Unsloth FP8 checkpoint contains 62 safetensors shards totaling **328,337,455,672 bytes (305.79 GiB)**. TP=4 loads about **78.13 GiB per rank** in the validated runtime.

The optional ModelOpt NVFP4-A16 checkpoint contains 120 shards totaling **194,660,206,040 bytes (181.29 GiB)**. It quantizes routed-expert weights while retaining attention, indexer, shared-expert, and MTP tensors in BF16. The BF16 source is roughly 598 GiB and does not leave adequate four-node runtime headroom, so this recipe has no BF16 serving profile.

## Model facts

- 45 text layers: 3 dense, 42 sparse MoE
- 288 routed experts + 1 shared expert; 8 routed experts active per token
- hybrid KDA linear attention, NoPE sparse MLA, and full-attention layers
- one in-model next-token prediction layer for MTP
- native multimodal model; this first recipe launches **language-only**
- native context ceiling: 1,048,576 tokens
- blockwise FP8 weights (`128×128`)
- checkpoint revision: `a160e2291674d9e3e92e98fd82faa2544a2867a3`

## Conservative first-boot profile

| Setting | Initial value |
|---|---:|
| checkpoint | `unsloth/GLM-5.3-Flash-FP8` |
| served model | `glm-5.3-flash-fp8` |
| tensor parallelism | 4 |
| context | 262,144 |
| KV dtype | FP8 (`fp8_ds_mla`, patched fixed NoPE ABI) |
| max sequences | 4 |
| batched tokens | 8,192 |
| GPU memory utilization | 0.82 |
| MTP draft tokens | 3 |

The official vLLM recipe uses TP=4, FP8 KV, and MTP k=5. It requires vLLM 0.27.0+ integration and FlashInfer 0.6.17+ for NoPE sparse MLA. GB10 is Blackwell, but `sm_121a`/aarch64 is not the GB200 configuration used by the upstream example; treat kernel compatibility as a bring-up gate, not a paperwork exercise.

## Repository layout

- [`RECIPE.md`](RECIPE.md) — staged deployment procedure
- [`recipe.yaml`](recipe.yaml) — machine-readable serve recipe
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — expected GB10 failure gates
- `scripts/glm53_node_up.sh` — start a rank container with validated RoCE setup
- `scripts/glm53_native_launch.sh` — **no-Ray** TP=4 launch: rank 0 serves the
  API, ranks 1–3 run `--headless` over PyTorch distributed rendezvous
- `scripts/glm53_serve.sh` — launch vLLM on the head
- `scripts/glm53_gpu_reset.sh` — clear a wedged GB10 compute engine without rebooting
- `scripts/repro_moe_wedge.py` — fast targeted reproducer for the MoE rank-divergence wedge
- `scripts/stage_model.sh` — FP8 checkpoint fan-out
- `scripts/stage_nvfp4_from_246.sh` — pinned NVFP4 verify-and-fan-out pipeline
- `scripts/fanout_cluster.sh` — this cluster's `.246 → head → workers` FP8 staging pipeline
- `scripts/smoke_bench.py` — model, reasoning, tool-free smoke, and basic decode timing
- `scripts/run_llama_benchy.sh` — full Spark Arena v2 depth/concurrency profile with the vLLM keepalive workaround
- `scripts/verify_checkpoint.py` — verify revision, shard count, and index completeness

## Quick start

Build the small GB10 multi-node derivative of upstream's dedicated arm64 image:

```bash
docker pull vllm/vllm-openai:glm53-flash
# Build the NoPE/Ray base and sparse-tuned runtime as described in RECIPE.md,
# then add the production dense-prefix top-k fix:
docker build -f Dockerfile.gb10-topk-compact \
  -t glm53-vllm-gb10:nope-sm121-topk-compact-v2-ray-2.58 .
export GLM53_IMAGE=glm53-vllm-gb10:nope-sm121-topk-compact-v2-ray-2.58
```

The upstream image supplies GLM-5.3, CUDA 13.0, and FlashInfer 0.6.17. The derivative only adds Ray 2.58.0 because upstream's published image targets single-node deployments.

Stage the checkpoint on every node, then start the head and workers using their fabric addresses and live RoCE-v2 GID indices:

```bash
./scripts/glm53_node_up.sh head   10.0.0.46  10.0.0.46  auto
./scripts/glm53_node_up.sh worker 10.0.0.150 10.0.0.46  auto
./scripts/glm53_node_up.sh worker 10.0.0.13  10.0.0.46  auto
./scripts/glm53_node_up.sh worker 10.0.0.246 10.0.0.46  auto
```

GID indices drift after network changes. `auto` resolves the RoCE-v2 index matching each node's fabric IPv4 address at launch time; do not bake observed indexes into automation.

On the head:

```bash
./scripts/glm53_serve.sh
python3 ./scripts/smoke_bench.py http://127.0.0.1:8000
```

See [`RECIPE.md`](RECIPE.md) before running this on production nodes. The SM121 NoPE fixed-ABI cache-kernel compatibility design and focused tests are documented in [`SM121_NOPE_PATCH.md`](SM121_NOPE_PATCH.md). The patch preserves FlashInfer's 656-byte cache entry by zero-filling the fixed RoPE region and padding NoPE queries only at the backend boundary.

## Sources

- Unsloth checkpoint/model card and checked-in config
- Z.ai GLM-5.3-Flash model card
- official vLLM GLM-5.3 recipe (vLLM 0.27.0+, TP=4, FP8 KV, MTP k=5)
- official SGLang GLM-5.3 cookbook (dedicated integration image and GB10-relevant backend caveats)

## License

Deployment scripts in this repository are MIT-licensed. Model weights retain their upstream license; see the checkpoint's `LICENSE` and model card.
