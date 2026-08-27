#!/usr/bin/env python3
"""Unit-test the SM121 NoPE-to-fixed-FlashInfer query adapter."""

from types import SimpleNamespace

import torch

from vllm.v1.attention.backends.mla import flashinfer_mla_sparse_sm120 as backend


def main() -> None:
    captured: dict[str, object] = {}

    def fake_index_conversion(*args, **kwargs):
        del args
        captured["convert_topk"] = kwargs["NUM_TOPK_TOKENS"]
        return torch.zeros(
            2, kwargs["NUM_TOPK_TOKENS"], device="cuda", dtype=torch.int32
        )

    def fake_flashinfer(**kwargs):
        captured.update(kwargs)
        return kwargs["out"]

    backend.triton_convert_req_index_to_global_index = fake_index_conversion

    import vllm.utils.flashinfer

    vllm.utils.flashinfer.flashinfer_trtllm_batch_decode_with_kv_cache_mla = (
        fake_flashinfer
    )

    impl = object.__new__(backend.FlashInferMLASparseSM120Impl)
    impl.num_heads = 8
    impl.kv_lora_rank = 512
    impl.qk_nope_head_dim = 256
    impl.qk_rope_head_dim = 0
    impl.topk_indices_buffer = torch.zeros(
        2, 2176, device="cuda", dtype=torch.int32
    )
    impl._workspace_buffer = torch.empty(1, device="cuda", dtype=torch.uint8)
    impl.scale = 1.0
    impl.kv_scale_format = "arbitrary_fp32"

    q = torch.randn(2, 8, 512, device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros(1, 64, 656, device="cuda", dtype=torch.uint8)
    metadata = SimpleNamespace(
        req_id_per_token=torch.zeros(2, device="cuda", dtype=torch.int32),
        block_table=torch.zeros(2, 1, device="cuda", dtype=torch.int32),
        block_size=64,
        topk_tokens=2176,
    )

    output, aux = impl.forward_mqa(q, cache, metadata, None)
    adapted = captured["query"]
    assert isinstance(adapted, torch.Tensor)
    assert adapted.shape == (2, 1, 8, 576)
    torch.testing.assert_close(adapted[:, 0, :, :512], q)
    assert torch.count_nonzero(adapted[..., 512:]).item() == 0
    assert captured["qk_rope_head_dim"] == 64
    assert captured["convert_topk"] == 2048
    assert captured["block_tables"].shape == (2, 1, 2048)
    assert captured["max_seq_len"] == 2048
    assert captured["sparse_mla_top_k"] == 2048
    assert captured["kv_scale_format"] == "arbitrary_fp32"
    assert output.shape == (2, 8, 512)
    assert aux is None
    print("GLM53_NOPE_BACKEND_ADAPTER_OK")


if __name__ == "__main__":
    main()
