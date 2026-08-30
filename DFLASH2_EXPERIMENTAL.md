# DFlash2 experimental profile

This is an **opt-in development profile**, not the production recipe. The
validated and published default remains MTP3. Nothing here changes
`recipe.yaml`, the production image digest, or the documented 65K support
boundary.

## What works

The trial combines:

- target: `unsloth/GLM-5.3-Flash-FP8` at revision
  `a160e2291674d9e3e92e98fd82faa2544a2867a3`;
- draft: `incoai/GLM-5.3-Flash-DFlash2` at revision
  `dc77ff1c99eeb2df044ee3d4f0094eb033fee410`;
- native TP=4 target execution over four nodes, with a TP=1 draft;
- seven speculative tokens and the FlashAttention draft backend;
- GLM mHC auxiliary-state capture at completed target layers
  `(5, 14, 24, 33, 42)`;
- a separate DFlash sliding-window KV-cache group.

The first successful gate used an intentionally small startup profile:

- max model length: 8,192;
- max sequences: 1;
- GPU memory utilization: 0.82;
- eager mode;
- API startup succeeded;
- repeated temperature-zero completions were byte-identical;
- 53 of 98 drafted tokens were accepted after three short requests.

That proves the integration executes. It does **not** prove a throughput win,
multi-request stability, or the production 65K envelope.

The subsequent five-run C=1 gate produced a **24.45 tok/s median** (24.14
minimum, 24.65 maximum). The accumulated speculative counters were 833
accepted tokens out of 3,563 drafted tokens (23.4%). This is substantially
slower than the validated MTP3 C=1 measurements around 36.5--37.2 tok/s, so the
current DFlash2 profile fails the promotion gate even before long-context
stress. Working is nice; winning is the point.

A second five-run gate with breakable CUDA graphs enabled produced a **24.10
tok/s median** (22.00 minimum, 24.16 maximum). CUDA graphs were stable, but did
not recover the performance deficit. The experimental launcher therefore
enables graphs by default; set `GLM53_ENFORCE_EAGER=1` only for control runs.

## Build

```bash
docker build \
  -f Dockerfile.gb10-dflash2-experimental \
  -t glm53-vllm-gb10:dflash2-experimental \
  .
```

Build the same image on every node. The Dockerfile starts from the immutable
published MTP3 image and applies reviewable patches; model weights are not
embedded.

## Launch

Download the pinned draft snapshot on every node, then run:

```bash
./scripts/glm53_gpu_reset.sh \
  10.0.0.46 10.0.0.13 10.0.0.150 10.0.0.246
./scripts/glm53_dflash2_trial.sh start
```

The wrapper deliberately defaults to the small profile above. Override its
environment variables only for explicit experiments; do not infer support
from a successful startup.

## Current blockers to promotion

At max length 65,535, max sequences 4, and GPU utilization 0.82, startup
correctly fails closed because the DFlash-aware cache layout requires about
15.6 GiB while only about 12.1 GiB is available. Raising utilization to 0.86
provided enough cache for a 32K trial, but exhausted practical unified-memory
headroom during/after warmup and the rank-0 worker died. That is not a viable
production configuration.

Before promotion, this profile still needs:

1. cache accounting and memory-footprint review;
2. a deterministic comparison against a non-speculative target server;
3. investigation of the failed C=1 throughput and acceptance-efficiency gate;
4. long-context C=5/C=10 stress tests with post-run C=1 verification;
5. an immutable image digest built and deployed identically on every node.

Until those pass, DFlash2 stays in the lab where experimental puppies belong.
