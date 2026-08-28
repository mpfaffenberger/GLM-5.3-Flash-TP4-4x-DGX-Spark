# RECIPE — GLM-5.3-Flash FP8 TP=4 on 4× DGX Spark

This is the validated FP8 deployment path. It copies the operational shape of the MiniMax M3 TP=4 repository and includes the GLM-specific SM121 NoPE fixed-ABI patches required by FlashInfer sparse MLA.

## 0. Hard gates

- Four GB10 nodes connected through the switched RoCE fabric; all fabric NICs must use MTU 9000.
- Docker must expose `/dev/infiniband`, host networking, host IPC, memlock, and at least `nofile=1048576`.
- Each node needs ~306 GiB for its local full Hugging Face snapshot plus healthy disk margin. Ray may evict workers when disks exceed 95%.
- Use `glm53-vllm-gb10:nope-sm121-topk-compact-ray-2.58`. It layers the validated dense-prefix sparse-index fix over the sparse-tuned SM121 runtime. Without that final patch, the first real sparse request beyond `index_topk=2048` can permanently wedge FlashInfer's kernel.
- Do not substitute BF16. It cannot fit this four-node cluster.

## 1. Validate the runtime image before downloading 306 GiB four times

```bash
docker pull vllm/vllm-openai:glm53-flash
# Build/stage the validated sparse-tuned base first, then add the production
# top-k compaction layer. Override BASE_IMAGE if your local base tag differs.
docker build -f Dockerfile.gb10-topk-compact \
  --build-arg BASE_IMAGE=glm53-vllm-gb10:nope-sm121-sparse-tuned-ray-2.58 \
  -t glm53-vllm-gb10:nope-sm121-topk-compact-ray-2.58 .
export GLM53_IMAGE=glm53-vllm-gb10:nope-sm121-topk-compact-ray-2.58
docker run --rm --gpus all --entrypoint python3 "$GLM53_IMAGE" - <<'PY'
import torch, vllm
import flashinfer
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(), torch.cuda.get_device_capability())
print("vllm", vllm.__version__)
print("flashinfer", flashinfer.__version__)
PY
```

Also confirm that the installed vLLM recognizes `Glm5NextForConditionalGeneration`. A generic vLLM 0.25 image is too old even if its DeepSeek kernels work beautifully; model registration is not inherited through optimism.

## 2. Stage the pinned FP8 checkpoint

On one node with internet:

```bash
export HF_HOME="$HOME/.cache/huggingface"
hf download unsloth/GLM-5.3-Flash-FP8 \
  --revision a160e2291674d9e3e92e98fd82faa2544a2867a3
python3 scripts/verify_checkpoint.py
```

Then edit the destination list in `scripts/stage_model.sh` and fan it out over the fabric:

```bash
./scripts/stage_model.sh
```

Use `rsync --partial`, not tar pipes. Verify all 62 shards and the snapshot index on every node.

## 3. Audit the fabric on every rank

The last validated four-rank topology was:

| Rank | Fabric IP | Last known GID |
|---:|---|---:|
| 0 | `10.0.0.46` | 3 |
| 1 | `10.0.0.150` | 3 |
| 2 | `10.0.0.13` | 6 |
| 3 | `10.0.0.246` | 3 |

Check rather than trust the table:

```bash
ip -d link show enp1s0f1np1
show_gids
ulimit -n
```

All `enp1s0f1np1` fabric links must report MTU 9000. Select each node's RoCE-v2 IPv4 GID. Reimages have previously reset MTU to 1500 and Docker `nofile` to 1024; either defect can stall or kill NCCL.

## 4. Start Ray containers

Run head first, then workers:

```bash
./scripts/glm53_node_up.sh head   10.0.0.46  10.0.0.46  auto
./scripts/glm53_node_up.sh worker 10.0.0.150 10.0.0.46  auto
./scripts/glm53_node_up.sh worker 10.0.0.13  10.0.0.46  auto
./scripts/glm53_node_up.sh worker 10.0.0.246 10.0.0.46  auto
```

On the head:

```bash
docker exec glm53-tp4 ray status
```

Require four GPUs before launching the engine. GID indices drift across reboots; `auto` resolves the RoCE-v2 GID matching each node's fabric IPv4 address and is the production setting.

## 5. First serve

On rank 0:

```bash
./scripts/glm53_serve.sh
follow="${HOME}/glm53-serve.log"
tail -f "$follow"
```

The default launch uses:

- TP=4 through Ray
- 256K context
- FP8 KV (`fp8_ds_mla`) through the tested SM121 NoPE fixed-ABI adapter
- MTP k=5
- reasoning parser `glm45`
- tool parser `glm47`
- GPU memory utilization 0.80

If MTP is the first failing gate, retry with `GLM53_MTP_TOKENS=0`; the script omits speculative config when set to zero. This diagnoses MTP separately rather than deleting it from the recipe.

## 6. Acceptance gates

```bash
python3 scripts/smoke_bench.py http://127.0.0.1:8000
```

Do not call the deployment verified until all of these pass:

1. all 62 weight shards load on all four ranks;
2. startup reaches `Application startup complete`;
3. no rank is OOM-killed and restart counts remain zero;
4. no NCCL warnings/errors appear in the successful launch log;
5. plain chat returns coherent content;
6. reasoning lands in `reasoning_content` rather than leaking markers;
7. a second request succeeds (first-request-only success is a surprisingly popular lie);
8. the live KV-cache token capacity is recorded;
9. at least a C1 decode benchmark and a short concurrency sweep complete.

## 7. Expand only after baseline validation

Change one variable at a time:

1. raise GPU memory utilization in 0.01 steps;
2. test 512K, then 1M context ceilings;
3. compare MTP depths with identical prompts; k=3 measured 32.07 tok/s median C=1 in the clean restored FP8 profile;
4. increase max sequences based on measured KV capacity;
5. add multimodal encoder serving only after the language path is stable.

YAGNI applies to kernel debugging too: do not add expert parallelism, PD disaggregation, custom quant loaders, and multimodal encoder separation to the same first boot. That is not a test matrix; it is a séance.
