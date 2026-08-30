# GLM-5.3-Flash FP8 on 4× DGX Spark

Native, no-Ray tensor-parallel serving of
[`unsloth/GLM-5.3-Flash-FP8`](https://huggingface.co/unsloth/GLM-5.3-Flash-FP8)
across four NVIDIA DGX Spark/GB10 nodes over dedicated RoCE.

## Validation status

This repository publishes the strongest profile that completed repeatable
testing. It does **not** claim that every advertised model context/concurrency
combination is safe.

| Workload | Result |
|---|---|
| C=1 decode | 36–37 tok/s median after warmup |
| 65,535 tokens, concurrency 5 and 10 | **PASS**, 90/90 requests, 8/8 rows |
| accumulated max-seqs-10 experiment through 65K c10 | **PASS** through the historical request-544 failure point |
| 100,000 tokens, concurrency 5, `max-num-seqs=4` | **PASS**, 30/30 requests |
| 100,000 tokens, concurrency 10 | **UNSUPPORTED**: KDA kernel wedge |
| `max-num-seqs=5` or `10` at 100K c5 | **UNSUPPORTED**: KDA kernel wedge |

The published default is therefore **65K validated**, with
`--max-num-seqs 4`. The model is configured with a 262K ceiling so individual
requests are not artificially rejected, but that ceiling is not a promise of
high-concurrency stability. The full original 104-row matrix, which includes
100K c10, is not green.

The runtime image is published to GHCR. Model weights are not embedded:

```bash
docker pull ghcr.io/mpfaffenberger/glm-5.3-flash-fp8-tp4-4x-dgx-spark@sha256:717d12c5fba5731562511bdca7abe60a13d54fbf310276508fbe8eb6fa5d3341
```

The source patch, executable contracts, and image label remain reviewable here.

## Validated profile

| Setting | Value |
|---|---:|
| checkpoint | `unsloth/GLM-5.3-Flash-FP8` |
| revision | `a160e2291674d9e3e92e98fd82faa2544a2867a3` |
| served name | `glm-5.3-flash-fp8` |
| tensor parallelism | 4 nodes × 1 GB10 |
| executor | native vLLM multiprocessing/NCCL, **no Ray** |
| model context ceiling | 1,048,576 |
| validated concurrent context | 65,535 × c10 |
| KV dtype | FP8 |
| max active sequences | 4 |
| max batched tokens | 8,192 |
| GPU memory utilization | 0.82 |
| speculative decoding | MTP, 3 draft tokens |
| execution mode | eager |

GLM-5.3-Flash is a 320B-total/18B-active multimodal MoE. This recipe validates
the language path only. The pinned FP8 checkpoint contains 62 shards totaling
305.79 GiB; each node stores the complete snapshot and each TP rank loads
roughly one quarter of the weights.

## Why the KDA patch exists

GLM's model-specific KDA path recomputed chunk indices in the forward pass and
called `.tolist()` on a CUDA tensor. That introduced a blocking GPU→CPU sync
inside multi-rank execution. The patch:

1. builds chunk indices and offsets from scheduler CPU metadata;
2. copies them host→device and passes them into fused KDA;
3. splits mixed decode+prefill batches like the Qwen GDN implementation;
4. bypasses FLA's identity cache for mutable scheduler CPU buffers.

The original cache keyed tensors by object identity. vLLM reuses and mutates
those tensors, so a shape transition could return stale chunk maps. The
regression is executable in `patches/test_glm53_kda_chunk_indices.py`.

This fixes the repeatable 65K accumulated-state wedge. A deeper long-prefill
KDA kernel defect remains at 100K concurrency; it is documented rather than
papered over with a cheerful YAML value.

## Build

The KDA image layers over the locally built SM121 sparse-MLA runtime:

```bash
docker build -f Dockerfile.gb10-kda-hostmeta \
  -t glm53-vllm-gb10:nope-sm121-topk-compact-v2-kda-hostmeta .

docker image inspect --format \
  '{{index .Config.Labels "ai.code-puppy.glm53-kda-chunk-indices"}}' \
  glm53-vllm-gb10:nope-sm121-topk-compact-v2-kda-hostmeta
# host-metadata-no-d2h-v1
```

The base image construction and SM121 NoPE sparse-MLA compatibility patch are
documented in [`RECIPE.md`](RECIPE.md) and
[`SM121_NOPE_PATCH.md`](SM121_NOPE_PATCH.md).

Published tags `v0.1.0` and `latest` currently resolve to the immutable digest
shown above. Production automation should use the digest, not `latest`.

## Launch

Stage the pinned model snapshot on every node, then launch all four native TP
ranks from the head:

```bash
export GLM53_IMAGE=ghcr.io/mpfaffenberger/glm-5.3-flash-fp8-tp4-4x-dgx-spark@sha256:717d12c5fba5731562511bdca7abe60a13d54fbf310276508fbe8eb6fa5d3341
export GLM53_MAX_NUM_SEQS=4
export GLM53_MAX_BATCHED_TOKENS=8192
export GLM53_ENFORCE_EAGER=1
./scripts/glm53_native_launch.sh start
```

The launcher starts ranks 1–3 headless before rank 0 enters the distributed
rendezvous. It discovers each node's live RoCE-v2 GID; do not hard-code GID
indices because they drift after network changes.

Wait for readiness, then smoke test:

```bash
curl -fsS http://127.0.0.1:8000/v1/models
python3 scripts/smoke_bench.py http://127.0.0.1:8000
```

## Benchmark the supported envelope

The default llama-benchy matrix stops at the validated 65K boundary:

```bash
GLM53_ENFORCE_EAGER=1 ./scripts/glm53_full_suite.sh
```

Run the unsupported 100K probe explicitly; never merge its partial output with
validated results:

```bash
DEPTHS=100000 CONCURRENCIES='5 10' RUNS=3 \
  ./scripts/run_llama_benchy.sh results/experimental-100k
```

The harness fails closed on process failures, request-count mismatches, missing
CSV rows, request errors, or an unhealthy API. `scripts/glm53_stall_watch.sh`
captures all rank stacks when repeated broadcast starvation coincides with
stagnant request progress.

## Recovery

A wedged CUDA context survives container deletion on GB10. Recover all ranks:

```bash
./scripts/glm53_gpu_reset.sh \
  10.0.0.46 10.0.0.13 10.0.0.150 10.0.0.246
```

See [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for the failure signature,
evidence paths, and why max-seqs 5/10 are not recipe defaults.

## Repository map

- `recipe.yaml` — machine-readable native TP=4 profile
- `scripts/glm53_native_launch.sh` — no-Ray four-rank launcher
- `Dockerfile.gb10-kda-hostmeta` — fail-closed KDA patch image
- `patches/patch_glm53_kda_chunk_indices.py` — idempotent source patcher
- `patches/test_glm53_kda_chunk_indices.py` — static and executable contracts
- `scripts/run_llama_benchy.sh` — fail-closed benchmark harness
- `scripts/glm53_stall_watch.sh` — live wedge evidence capture
- `scripts/glm53_gpu_reset.sh` — module-reload recovery

## License

Deployment code is MIT-licensed. Model weights retain their upstream license.
