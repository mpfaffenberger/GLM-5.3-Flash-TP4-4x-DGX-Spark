# SM121 NoPE sparse-MLA compatibility patch

## Failure

The dedicated `vllm/vllm-openai:glm53-flash` image selects
`FLASHINFER_MLA_SPARSE_SM120` on DGX Spark (`sm_121`). GLM-5.3 uses native
NoPE sparse MLA (`qk_rope_head_dim=0`), but the backend and packed-cache CUDA
kernel expose FlashInfer's fixed DeepSeek-style ABI:

```text
512 bytes  FP8 latent values
 16 bytes  four FP32 tile scales
128 bytes  64 BF16 RoPE values
---------
656 bytes  per token
```

The stock cache-update operation rejects GLM before profiling completes:

```text
concat_and_cache_mla: pe_dim must be 64 for fp8_ds_mla
```

`--kv-cache-dtype auto` is not a workaround. The SM120 backend canonicalizes
`auto` to `fp8_ds_mla` and does not support BF16 KV.

## Compatibility design

The patch preserves the existing 656-byte FlashInfer ABI:

1. Accept `pe_dim` 0 or 64 in `concat_and_cache_mla`.
2. For NoPE, quantize and store the 512 latent values normally.
3. Preserve GLM's arbitrary FP32 scale format.
4. Zero-fill bytes 528–655 instead of reading an empty `k_pe` tensor.
5. Pad the 512-wide query with 64 BF16 zeros at the FlashInfer boundary.
6. Pass an effective `qk_rope_head_dim=64` to the fixed ABI.
7. Slice the indexer's kpool-expanded/padded 2176-entry table to the fixed
   2048-entry SM120 v32/GLM ABI capacity.

This is mathematically exact for a NoPE dot product:

```text
[Q, 0] · [K, 0] = Q · K
```

It does not invent positional values or modify checkpoint configuration.

A compact 528-byte native NoPE cache would save about 19.5% versus 656 bytes,
but requires a coordinated change to FlashInfer's SM120 kernel and allocator
contract. That is deliberately out of scope for the compatibility patch.

## Files

- `patches/vllm-glm53-nope-sm121.patch` — CUDA cache packer
- `patches/patch_flashinfer_sm121_backend.py` — query ABI adapter
- `Dockerfile.gb10-nope` — reproducible patched image
- `scripts/test_nope_cache_kernel.py` — poisoned-cache CUDA test
- `scripts/test_nope_backend_adapter.py` — query padding and fixed-top-k ABI unit test

## Required tests

Before a four-rank launch:

```bash
docker run --rm --gpus all --entrypoint python3 \
  glm53-vllm-gb10:nope-sm121-ray-2.58 \
  /usr/local/bin/test_nope_cache_kernel.py

docker run --rm --gpus all --entrypoint python3 \
  glm53-vllm-gb10:nope-sm121-ray-2.58 \
  /usr/local/bin/test_nope_backend_adapter.py
```

The CUDA test initializes cache memory with `0xA5`, proves that mapped NoPE
entries get a zeroed ABI RoPE region, proves an unmapped slot remains poisoned,
and proves unsupported `pe_dim=32` is rejected.

After those focused tests pass, run the normal TP=4 startup and smoke tests.
