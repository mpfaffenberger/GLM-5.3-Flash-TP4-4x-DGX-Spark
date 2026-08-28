#!/usr/bin/env python3
"""Feed the SM120/SM121 sparse-MLA kernel a dense-prefix top-k table.

GLM-5.3 contexts beyond ``index_topk`` (2048) produce the first real kpool
top-k tables. Stored in place, those tables carry interspersed ``-1`` slots
and unsorted entries, and FlashInfer's SM120 sparse-MLA page walk never
terminates on GB10 when it consumes them (see
results/sm121-sparse-mla-hang-diagnosis/). The remap kernel already supports
compacting valid entries to a contiguous prefix and counting them in the same
pass; use that mode and hand the exact per-token valid count to the kernel as
``topk_length`` instead of ``seq_lens=None``.
"""

from pathlib import Path

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/"
    "flashinfer_mla_sparse_sm120.py"
)

OLD_CONVERT = """        topk_indices_physical = cast(
            torch.Tensor,
            triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
            ),
        )
"""
NEW_CONVERT = """        # Compact valid entries to a contiguous prefix and count them in the
        # same remap pass. The kpool indexer's real (>index_topk context)
        # tables otherwise interleave -1 slots mid-row, and the SM120 sparse
        # kernel's page walk never terminates on GB10 when fed such rows.
        topk_indices_physical, topk_valid_counts = cast(
            "tuple[torch.Tensor, torch.Tensor]",
            triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            ),
        )
"""

OLD_SEQ = """            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=None,
"""
NEW_SEQ = """            block_tables=topk_indices_physical.unsqueeze(1),
            # Exact per-token valid count -> kernel topk_length; the dense
            # prefix plus explicit length removes any reliance on in-kernel
            # -1 sentinel scanning.
            seq_lens=topk_valid_counts,
"""


def main() -> None:
    text = TARGET.read_text()
    if NEW_CONVERT in text and NEW_SEQ in text:
        print(f"already patched: {TARGET}")
        return
    for old, name in ((OLD_CONVERT, "convert block"), (OLD_SEQ, "seq_lens line")):
        if old not in text:
            raise SystemExit(f"expected {name} not found in {TARGET}")
    text = text.replace(OLD_CONVERT, NEW_CONVERT, 1)
    text = text.replace(OLD_SEQ, NEW_SEQ, 1)
    TARGET.write_text(text)
    print(f"patched: {TARGET}")


if __name__ == "__main__":
    main()
