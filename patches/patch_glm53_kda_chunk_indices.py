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
    old_assert = """            initial_state = gather_initial_states(
                recurrent_state, non_spec_state_indices_tensor, has_initial_state
            )
            (
"""
    new_assert = """            initial_state = gather_initial_states(
                recurrent_state, non_spec_state_indices_tensor, has_initial_state
            )
            # Prefill metadata must be host-derived. Never silently re-enable
            # the unsafe in-forward D2H fallback for GLM5-Next.
            assert attn_metadata_narrowed.chunk_indices is not None
            (
"""
    replace_once(GLM_KDA, old_assert, new_assert)

    old = """                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc,
                safe_gate=safe_gate,
"""
    new = """                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc,
                # Built from query_start_loc_cpu by GDNAttentionMetadataBuilder;
                # avoids prepare_chunk_indices(...).tolist() in every KDA layer.
                chunk_indices=attn_metadata_narrowed.chunk_indices,
                safe_gate=safe_gate,
"""
    replace_once(GLM_KDA, old, new)


def main() -> None:
    patch_fla_entry_points()
    patch_glm_call_site()


if __name__ == "__main__":
    main()
