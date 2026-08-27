#!/usr/bin/env python3
"""Adapt GLM NoPE queries to FlashInfer's fixed SM120/SM121 sparse-MLA ABI."""

from pathlib import Path

BACKEND = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/"
    "flashinfer_mla_sparse_sm120.py"
)

TOPK_OLD = '''        topk_indices = self.topk_indices_buffer[:num_actual_toks]
'''

TOPK_NEW = '''        # GLM's indexer buffer is kpool-expanded and padded to 2176, while
        # FlashInfer's SM120 v32/GLM ABI has a fixed 2048-entry table.
        flashinfer_topk = 2048
        topk_indices = self.topk_indices_buffer[
            :num_actual_toks, :flashinfer_topk
        ]
'''

OLD = '''        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
'''

NEW = '''        # FlashInfer's SM120/SM121 sparse-MLA ABI has a fixed
        # 512-latent + 64-RoPE query shape. GLM-5.3 is native NoPE, so pad
        # both sides of that dot product with zeros. The matching cache
        # region is zero-filled by the patched concat_and_cache_mla kernel.
        flashinfer_rope_dim = self.qk_rope_head_dim
        if self.qk_rope_head_dim == 0:
            if self.kv_lora_rank != 512 or q.shape[-1] != 512:
                raise ValueError(
                    "SM121 NoPE adaptation requires kv_lora_rank=512 "
                    "and a 512-dimensional query"
                )
            q = torch.nn.functional.pad(q, (0, 64))
            flashinfer_rope_dim = 64

        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=flashinfer_rope_dim,
'''


def main() -> None:
    text = BACKEND.read_text()
    if NEW in text and TOPK_NEW in text:
        print(f"already patched: {BACKEND}")
        return
    if OLD not in text or TOPK_OLD not in text:
        raise SystemExit(f"expected backend source block not found: {BACKEND}")
    text = text.replace(TOPK_OLD, TOPK_NEW, 1)
    text = text.replace(OLD, NEW, 1)
    text = text.replace(
        "max_seq_len=attn_metadata.topk_tokens,", "max_seq_len=flashinfer_topk,", 1
    )
    text = text.replace(
        "sparse_mla_top_k=attn_metadata.topk_tokens,",
        "sparse_mla_top_k=flashinfer_topk,",
        1,
    )
    BACKEND.write_text(text)
    print(f"patched: {BACKEND}")


if __name__ == "__main__":
    main()
