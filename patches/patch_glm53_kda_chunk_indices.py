#!/usr/bin/env python3
"""Use host-precomputed FLA chunk indices in GLM5-Next KDA prefills.

The generic GDN metadata builder already constructs ``chunk_indices`` from
``query_start_loc_cpu`` and asynchronously copies the tiny tensor to the GPU.
GLM5-Next's model-specific KDA fork ignored it and called
``prepare_chunk_indices(cu_seqlens)`` inside every layer instead. That helper
uses ``Tensor.tolist()``, forcing a device-to-host synchronization in the
middle of a tensor-parallel forward pass.

This patch wires the existing metadata through the fused KDA entry point. The
old computation remains as a compatibility fallback for other callers.
"""

import os
from pathlib import Path

VLLM_ROOT = Path(
    os.environ.get(
        "VLLM_ROOT",
        "/usr/local/lib/python3.12/dist-packages/vllm",
    )
)
FLA_KDA = VLLM_ROOT / "third_party/flash_linear_attention/ops/kda.py"
GLM_KDA = VLLM_ROOT / "models/glm5next/nvidia/kda.py"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        print(f"already patched: {path}")
        return
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one patch context in {path}, found {count}")
    path.write_text(text.replace(old, new, 1))
    print(f"patched: {path}")


def patch_fla_entry_points() -> None:
    old_fwd = """def chunk_kda_with_fused_gate_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
):
    chunk_size = FLA_CHUNK_SIZE
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
"""
    new_fwd = """def chunk_kda_with_fused_gate_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
):
    chunk_size = FLA_CHUNK_SIZE
    # GDNAttentionMetadataBuilder already computes this from CPU scheduler
    # metadata and performs an async H2D copy. Keep the legacy path only for
    # callers that have not been migrated; it performs a blocking D2H sync.
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
"""
    replace_once(FLA_KDA, old_fwd, new_fwd)

    old_wrapper = """    cu_seqlens: torch.Tensor | None = None,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
    **kwargs,
):
    \"\"\"Run chunk KDA from raw gate projection using fused gate+cumsum.\"\"\"
"""
    new_wrapper = """    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
    **kwargs,
):
    \"\"\"Run chunk KDA from raw gate projection using fused gate+cumsum.\"\"\"
"""
    replace_once(FLA_KDA, old_wrapper, new_wrapper)

    old_call = """        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        safe_gate=safe_gate,
"""
    new_call = """        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
"""
    replace_once(FLA_KDA, old_call, new_call)


def patch_glm_call_site() -> None:
    old_metadata = """        has_initial_state = attn_metadata_narrowed.has_initial_state
        non_spec_query_start_loc = attn_metadata_narrowed.non_spec_query_start_loc
"""
    new_metadata = """        has_initial_state = attn_metadata_narrowed.has_initial_state
        non_spec_query_start_loc = attn_metadata_narrowed.non_spec_query_start_loc
        prefill_query_start_loc = attn_metadata_narrowed.prefill_query_start_loc
        prefill_state_indices = attn_metadata_narrowed.prefill_state_indices
        prefill_has_initial_state = attn_metadata_narrowed.prefill_has_initial_state
        num_decode_tokens = attn_metadata_narrowed.num_decode_tokens
"""
    replace_once(GLM_KDA, old_metadata, new_metadata)

    old_split = """        use_spec = spec_sequence_masks is not None and num_spec_decodes > 0
        # KDA gate variant: GLM5-Next checkpoints with
"""
    new_split = """        use_spec = spec_sequence_masks is not None and num_spec_decodes > 0
        # Generic GDN metadata peels a plain-decode prefix from mixed batches.
        # Process that prefix recurrently and send only the prefill tail to the
        # chunk kernel, matching qwen_gdn_linear_attn.py.
        split_non_spec = (
            spec_sequence_masks is None
            and attn_metadata_narrowed.num_prefills > 0
            and attn_metadata_narrowed.num_decodes > 0
        )
        # KDA gate variant: GLM5-Next checkpoints with
"""
    replace_once(GLM_KDA, old_split, new_split)

    old_prefill = """        if attn_metadata_narrowed.num_prefills > 0:
            assert q_ns is not None
            assert non_spec_state_indices_tensor is not None
            assert has_initial_state is not None
            initial_state = gather_initial_states(
                recurrent_state, non_spec_state_indices_tensor, has_initial_state
            )
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_kda_with_fused_gate(
                q=_rearr(q_ns),
                k=_rearr(k_ns),
                v=_rearr(v_ns),
                raw_g=g1_ns,
                # Chunk path wants the pre-sigmoided fp32 beta (its kernels
                # don't sigmoid); beta_ns is raw bf16 from forward.
                beta=_cast_sigmoid(beta_ns.squeeze(0)).unsqueeze(0),
                A_log=self.A_log,
                g_bias=self.dt_bias,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
            )
            # Init cache
            scatter_states(
                recurrent_state,
                last_recurrent_state,
                non_spec_state_indices_tensor,
            )
"""
    new_prefill = """        if attn_metadata_narrowed.num_prefills > 0:
            assert q_ns is not None
            assert k_ns is not None
            assert v_ns is not None
            assert g1_ns is not None
            assert beta_ns is not None
            assert prefill_query_start_loc is not None
            assert prefill_state_indices is not None
            assert prefill_has_initial_state is not None
            assert attn_metadata_narrowed.chunk_indices is not None

            if split_non_spec:
                assert non_spec_query_start_loc is not None
                assert non_spec_state_indices_tensor is not None
                core_attn_out_decode, _ = fused_recurrent_kda(
                    q=_rearr(q_ns[:num_decode_tokens]),
                    k=_rearr(k_ns[:num_decode_tokens]),
                    v=_rearr(v_ns[:num_decode_tokens]),
                    g=g1_ns[:, :num_decode_tokens],
                    beta=beta_ns[:, :num_decode_tokens],
                    initial_state=recurrent_state,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=non_spec_query_start_loc[
                        : attn_metadata_narrowed.num_decodes + 1
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    sigmoid_beta=True,
                    a_log=self.A_log,
                    g_bias=self.dt_bias,
                    compute_gate=True,
                    lower_bound=lower_bound,
                )
                q_prefill = q_ns[num_decode_tokens:]
                k_prefill = k_ns[num_decode_tokens:]
                v_prefill = v_ns[num_decode_tokens:]
                g1_prefill = g1_ns[:, num_decode_tokens:]
                beta_prefill = beta_ns[:, num_decode_tokens:]
            else:
                core_attn_out_decode = None
                q_prefill, k_prefill, v_prefill = q_ns, k_ns, v_ns
                g1_prefill, beta_prefill = g1_ns, beta_ns

            initial_state = gather_initial_states(
                recurrent_state, prefill_state_indices, prefill_has_initial_state
            )
            (
                core_attn_out_prefill,
                last_recurrent_state,
            ) = chunk_kda_with_fused_gate(
                q=_rearr(q_prefill),
                k=_rearr(k_prefill),
                v=_rearr(v_prefill),
                raw_g=g1_prefill,
                # Chunk path wants the pre-sigmoided fp32 beta (its kernels
                # don't sigmoid); beta_prefill is raw bf16 from forward.
                beta=_cast_sigmoid(beta_prefill.squeeze(0)).unsqueeze(0),
                A_log=self.A_log,
                g_bias=self.dt_bias,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=prefill_query_start_loc,
                # Built from query_start_loc_cpu by GDNAttentionMetadataBuilder;
                # never calls prepare_chunk_indices(...).tolist() in GLM.
                chunk_indices=attn_metadata_narrowed.chunk_indices,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
            )
            # Init cache for prefill requests. Recurrent KDA updated any peeled
            # decode requests in place above.
            scatter_states(
                recurrent_state,
                last_recurrent_state,
                prefill_state_indices,
            )
            if split_non_spec:
                core_attn_out_non_spec = torch.cat(
                    [core_attn_out_decode, core_attn_out_prefill], dim=1
                )
            else:
                core_attn_out_non_spec = core_attn_out_prefill
"""
    replace_once(GLM_KDA, old_prefill, new_prefill)


def main() -> None:
    patch_fla_entry_points()
    patch_glm_call_site()


if __name__ == "__main__":
    main()
