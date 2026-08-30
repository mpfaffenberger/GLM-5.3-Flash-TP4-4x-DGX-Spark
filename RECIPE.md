# Deployment recipe

This is the supported language-only deployment path for GLM-5.3-Flash FP8 on
four DGX Spark nodes. It uses native vLLM multi-node multiprocessing; Ray is
not part of the production topology.

## 1. Hard gates

- Four GB10 nodes, one GPU each, connected through switched RoCE.
- Fabric MTU 9000 on every node.
- Docker host networking/IPC, `/dev/infiniband`, unlimited memlock, and
  `nofile >= 1048576`.
- Approximately 306 GiB plus healthy disk margin for the full checkpoint on
  every node.
- Pinned model revision
  `a160e2291674d9e3e92e98fd82faa2544a2867a3`.
- `max-num-seqs=4`. Five active 100K requests reproducibly wedge KDA.

Do not substitute BF16; it does not leave adequate four-node runtime headroom.

## 2. Build the runtime

Pull the published immutable image:

```bash
docker pull ghcr.io/mpfaffenberger/glm-5.3-flash-fp8-tp4-4x-dgx-spark@sha256:717d12c5fba5731562511bdca7abe60a13d54fbf310276508fbe8eb6fa5d3341
```

Or build the same reviewable derivative locally.

Build the repository's SM121 NoPE/top-k base chain first, then the KDA host
metadata derivative:

```bash
docker build -f Dockerfile.gb10-kda-hostmeta \
  --build-arg BASE_IMAGE=glm53-vllm-gb10:nope-sm121-topk-compact-v2-ray-2.58 \
  -t glm53-vllm-gb10:nope-sm121-topk-compact-v2-kda-hostmeta .
```

The historical base tag contains `ray-2.58`; the native launcher explicitly
sets `GLM53_START_RAY=0`, and no Ray process participates in serving. The build
executes the KDA patch contract before producing an image.

Verify the image label on all four nodes:

```bash
GLM53_KDA_VERIFY_ONLY=1 ./scripts/glm53_kda_patch_trial.sh
```

## 3. Stage and verify the checkpoint

```bash
export HF_HOME="$HOME/.cache/huggingface"
hf download unsloth/GLM-5.3-Flash-FP8 \
  --revision a160e2291674d9e3e92e98fd82faa2544a2867a3
python3 scripts/verify_checkpoint.py
./scripts/stage_model.sh
```

Require all 62 shards and the index on every node. Use resumable transfer; a
half-copied 306 GiB model is not a creative quantization format.

## 4. Audit the fabric

```bash
ip -d link show enp1s0f1np1
show_gids
ulimit -n
```

Select the RoCE-v2 GID matching each node's fabric IPv4 address. The launch
scripts discover this with `auto`. Last-known GIDs are not configuration
because reboots and network changes can renumber them.

## 5. Launch native TP=4

From rank 0/head (`10.0.0.46`):

```bash
export GLM53_IMAGE=ghcr.io/mpfaffenberger/glm-5.3-flash-fp8-tp4-4x-dgx-spark@sha256:717d12c5fba5731562511bdca7abe60a13d54fbf310276508fbe8eb6fa5d3341
export GLM53_MAX_MODEL_LEN=262144
export GLM53_MAX_NUM_SEQS=4
export GLM53_MAX_BATCHED_TOKENS=8192
export GLM53_GPU_MEMORY_UTILIZATION=0.82
export GLM53_KV_CACHE_DTYPE=fp8
export GLM53_MTP_TOKENS=3
export GLM53_ENFORCE_EAGER=1
./scripts/glm53_native_launch.sh start
```

The launcher removes stale containers, starts containers with Ray disabled,
waits for GB10 unified memory to return, starts worker ranks 1–3 headless, and
then starts API rank 0. Cold startup commonly takes 18–25 minutes.

Do not edit an executing launch or benchmark script. Bash is surprisingly
literal about moving floorboards.

## 6. Readiness and smoke gates

```bash
until curl -fsS http://127.0.0.1:8000/v1/models >/dev/null; do sleep 10; done
python3 scripts/smoke_bench.py http://127.0.0.1:8000
GLM53_BENCH_RUNS=3 ./scripts/bench_c1.sh
```

Require all ranks alive, coherent output, a successful second request, no NCCL
errors, recorded KV capacity, and a passing C=1 gate.

## 7. Validate the supported envelope

The strict suite runs pre/post C=1 and the default matrix through 65K:

```bash
GLM53_ENFORCE_EAGER=1 ./scripts/glm53_full_suite.sh
```

Acceptance is fail-closed: request starts equal request ends, no request errors,
every expected row exists, the API remains healthy, and post-suite C=1 passes.

The focused patch validation that completed cleanly was 65K × `{c5,c10}`:
90/90 requests, 8/8 rows, zero errors, and healthy post-C1.

## 8. Unsupported 100K concurrency

Testing established this boundary:

- max-seqs 10: wedges at 100K c5;
- max-seqs 5: wedges on the third 100K c5 wave;
- max-seqs 4: 100K c5 passes 30/30, but 100K c10 wedges.

The signature is repeated shared-memory broadcast starvation, no request-end
progress, and all GPUs at approximately 96% utilization but only 19–24 W.
Treat 100K c10 as unsupported until the FLA/KDA kernel defect is fixed.

Do not increase max sequences based only on nominal KV capacity. Arithmetic can
say “fits” while the kernel says “absolutely not.”

## 9. Recovery

Container removal does not clear a spinning GB10 CUDA context:

```bash
./scripts/glm53_gpu_reset.sh \
  10.0.0.46 10.0.0.13 10.0.0.150 10.0.0.246
```

The script reloads the complete NVIDIA module stack and verifies idle power and
utilization before another launch.
