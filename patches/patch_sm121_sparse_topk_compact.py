#!/usr/bin/env python3
"""Make SM120 sparse-MLA top-k tables safe for real and padded rows.

Real GLM kpool tables contain interspersed invalid slots. Compact valid entries
to a dense prefix and pass their exact count as ``topk_length``. CUDA graph
padding can also create empty rows; FlashInfer's native NoPE kernel rejects
length zero, so launch those rows against one dummy slot and zero their ignored
output, matching vLLM's generic sparse backend.
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
NEW_CONVERT = """        # Compact valid slots to the prefix expected by FlashInfer and count
        # them in the same remap pass.
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

        # CUDA graph padding can leave empty query rows. Native NoPE sparse MLA
        # rejects topk_length=0; use one dummy slot and discard its output.
        empty_rows = topk_valid_counts == 0
        topk_indices_physical[:, 0] = topk_indices_physical[:, 0].masked_fill(
            empty_rows, 0
        )
        kernel_topk_lengths = topk_valid_counts.clamp(min=1)
"""
OLD_SEQ = """            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=None,
"""
NEW_SEQ = """            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=kernel_topk_lengths,
"""
OLD_RETURN = """        )
        return out.squeeze(1), None
"""
NEW_RETURN = """        )
        out = out.squeeze(1)
        out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
        return out, None
"""


def main() -> None:
    text = TARGET.read_text()
    if NEW_CONVERT in text and NEW_SEQ in text and NEW_RETURN in text:
        print(f"already patched: {TARGET}")
        return
    for old, name in (
        (OLD_CONVERT, "convert block"),
        (OLD_SEQ, "seq_lens line"),
        (OLD_RETURN, "return block"),
    ):
        if old not in text:
            raise SystemExit(f"expected {name} not found in {TARGET}")
    text = text.replace(OLD_CONVERT, NEW_CONVERT, 1)
    text = text.replace(OLD_SEQ, NEW_SEQ, 1)
    text = text.replace(OLD_RETURN, NEW_RETURN, 1)
    TARGET.write_text(text)
    print(f"patched: {TARGET}")


if __name__ == "__main__":
    main()
