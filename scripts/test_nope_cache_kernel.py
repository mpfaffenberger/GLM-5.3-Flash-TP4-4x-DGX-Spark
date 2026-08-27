#!/usr/bin/env python3
"""Focused SM121 test for GLM NoPE packing into fp8_ds_mla cache entries."""

import torch

import vllm._custom_ops  # noqa: F401 - registers _C_cache_ops

ENTRY_BYTES = 656
LATENT_BYTES = 512
SCALE_BYTES = 16
ROPE_OFFSET = LATENT_BYTES + SCALE_BYTES


def run_nope_case() -> None:
    torch.manual_seed(53)
    kv_c = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
    k_pe = torch.empty(3, 0, device="cuda", dtype=torch.bfloat16)
    cache = torch.full(
        (1, 64, ENTRY_BYTES),
        0xA5,
        device="cuda",
        dtype=torch.uint8,
    )
    slots = torch.tensor([0, 1, -1], device="cuda", dtype=torch.int64)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)

    torch.ops._C_cache_ops.concat_and_cache_mla(
        kv_c, k_pe, cache, slots, "fp8_ds_mla", scale
    )
    torch.cuda.synchronize()

    assert torch.count_nonzero(cache[:1, :2, ROPE_OFFSET:]).item() == 0
    assert torch.count_nonzero(cache[0, 0, :LATENT_BYTES] != 0xA5).item() > 0
    assert torch.all(cache[0, 2] == 0xA5), "unmapped cache slot was modified"


def run_invalid_dimension_case() -> None:
    kv_c = torch.zeros(1, 512, device="cuda", dtype=torch.bfloat16)
    k_pe = torch.zeros(1, 32, device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros(1, 64, ENTRY_BYTES, device="cuda", dtype=torch.uint8)
    slots = torch.zeros(1, device="cuda", dtype=torch.int64)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)

    try:
        torch.ops._C_cache_ops.concat_and_cache_mla(
            kv_c, k_pe, cache, slots, "fp8_ds_mla", scale
        )
    except RuntimeError as exc:
        assert "pe_dim must be 0 or 64" in str(exc)
    else:
        raise AssertionError("pe_dim=32 was unexpectedly accepted")


def main() -> None:
    capability = torch.cuda.get_device_capability()
    assert capability == (12, 1), f"expected GB10 SM121, got {capability}"
    run_nope_case()
    run_invalid_dimension_case()
    print("GLM53_NOPE_CACHE_KERNEL_OK")


if __name__ == "__main__":
    main()
