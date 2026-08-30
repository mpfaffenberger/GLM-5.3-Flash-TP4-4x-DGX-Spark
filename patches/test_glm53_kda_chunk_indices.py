#!/usr/bin/env python3
"""Static contract tests for the GLM5-Next KDA chunk-index patch."""

import inspect

import torch
from vllm.models.glm5next.nvidia.kda import Glm5NextLinearAttention
from vllm.third_party.flash_linear_attention.ops import index, kda
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder


def main() -> None:
    fwd_signature = inspect.signature(kda.chunk_kda_with_fused_gate_fwd)
    wrapper_signature = inspect.signature(kda.chunk_kda_with_fused_gate)
    assert "chunk_indices" in fwd_signature.parameters
    assert "chunk_indices" in wrapper_signature.parameters

    fwd_source = inspect.getsource(kda.chunk_kda_with_fused_gate_fwd)
    assert "if chunk_indices is None and cu_seqlens is not None" in fwd_source
    assert "chunk_indices = prepare_chunk_indices" in fwd_source

    wrapper_source = inspect.getsource(kda.chunk_kda_with_fused_gate)
    assert "chunk_indices=chunk_indices" in wrapper_source

    # Scheduler CPU buffers are mutated and reused, so identity-cached results
    # are stale after a shape change. The metadata builder must use the explicit
    # uncached variants.
    assert callable(index.prepare_chunk_indices_uncached)
    assert callable(index.prepare_chunk_offsets_uncached)
    builder_source = inspect.getsource(GDNAttentionMetadataBuilder.build)
    assert "prepare_chunk_indices_uncached" in builder_source
    assert "prepare_chunk_offsets_uncached" in builder_source
    assert builder_source.count("non_blocking=False") >= 2

    reused = torch.tensor([0, 65], dtype=torch.int32)
    cached_before = index.prepare_chunk_indices(reused, 64)
    reused.copy_(torch.tensor([0, 129], dtype=torch.int32))
    cached_after = index.prepare_chunk_indices(reused, 64)
    fresh_after = index.prepare_chunk_indices_uncached(reused, 64)
    assert cached_after is cached_before
    assert cached_after.shape == (2, 2)
    assert fresh_after.shape == (3, 2)

    glm_source = inspect.getsource(Glm5NextLinearAttention._forward)
    assert "split_non_spec" in glm_source
    assert "q_ns[:num_decode_tokens]" in glm_source
    assert "q_prefill = q_ns[num_decode_tokens:]" in glm_source
    assert "cu_seqlens=prefill_query_start_loc" in glm_source
    assert "prefill_state_indices" in glm_source
    assert "assert attn_metadata_narrowed.chunk_indices is not None" in glm_source
    assert "chunk_indices=attn_metadata_narrowed.chunk_indices" in glm_source

    # Exercise the branch without launching GPU kernels. A supplied metadata
    # tensor must bypass prepare_chunk_indices entirely and flow through both
    # downstream helpers unchanged.
    supplied = object()
    observed = []
    originals = (
        kda.prepare_chunk_indices,
        kda.fused_kda_gate_chunk_cumsum,
        kda._chunk_kda_fwd_with_cumulative_g,
    )

    def forbidden_prepare(*_args, **_kwargs):
        raise AssertionError("precomputed chunk_indices unexpectedly ignored")

    def fake_cumsum(raw_g, **kwargs):
        observed.append(kwargs["chunk_indices"])
        return raw_g

    def fake_forward(**kwargs):
        observed.append(kwargs["chunk_indices"])
        return kwargs["v"], None

    try:
        kda.prepare_chunk_indices = forbidden_prepare
        kda.fused_kda_gate_chunk_cumsum = fake_cumsum
        kda._chunk_kda_fwd_with_cumulative_g = fake_forward
        marker = object()
        result, final_state = kda.chunk_kda_with_fused_gate_fwd(
            q=marker,
            k=marker,
            v=marker,
            raw_g=marker,
            beta=marker,
            A_log=marker,
            g_bias=None,
            scale=1.0,
            initial_state=marker,
            output_final_state=True,
            cu_seqlens=marker,
            chunk_indices=supplied,
        )
        assert result is marker
        assert final_state is None
        assert observed == [supplied, supplied]
    finally:
        (
            kda.prepare_chunk_indices,
            kda.fused_kda_gate_chunk_cumsum,
            kda._chunk_kda_fwd_with_cumulative_g,
        ) = originals

    print("GLM5-Next KDA chunk-index patch contract: PASS")


if __name__ == "__main__":
    main()
