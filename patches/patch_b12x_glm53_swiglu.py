#!/usr/bin/env python3
"""Enable GLM's clamped SwiGLU on FlashInfer's fused SM12x B12x MoE path."""

from pathlib import Path

EXPERTS = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "fused_moe/experts/flashinfer_b12x_moe.py"
)
ORACLE = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "fused_moe/oracle/nvfp4.py"
)

EXPERTS_OLD_STATE = """        self.max_num_tokens = moe_config.max_num_tokens
        self.local_expert_offset = self.ep_rank * self.num_local_experts
"""
EXPERTS_NEW_STATE = """        self.max_num_tokens = moe_config.max_num_tokens
        self.local_expert_offset = self.ep_rank * self.num_local_experts
        # FlashInfer's SM120/SM121 B12x API already implements the same
        # clamped SwiGLU semantics used by vLLM. Preserve the model value.
        self.swiglu_limit = moe_config.swiglu_limit
"""

EXPERTS_OLD_WRAPPER = """            activation=self._activation_str,
        )
"""
EXPERTS_NEW_WRAPPER = """            activation=self._activation_str,
            swiglu_limit=self.swiglu_limit,
        )
"""

ORACLE_OLD = """        NvFp4MoeBackend.FLASHINFER_CUTEDSL,
        NvFp4MoeBackend.VLLM_CUTLASS,
"""
ORACLE_NEW = """        NvFp4MoeBackend.FLASHINFER_CUTEDSL,
        # The bundled SM12x B12x kernel accepts and applies swiglu_limit.
        NvFp4MoeBackend.FLASHINFER_B12X,
        NvFp4MoeBackend.VLLM_CUTLASS,
"""


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        print(f"already patched: {path}")
        return
    if old not in text:
        raise SystemExit(f"expected patch context not found: {path}")
    path.write_text(text.replace(old, new, 1))
    print(f"patched: {path}")


def main() -> None:
    replace_once(EXPERTS, EXPERTS_OLD_STATE, EXPERTS_NEW_STATE)
    replace_once(EXPERTS, EXPERTS_OLD_WRAPPER, EXPERTS_NEW_WRAPPER)
    replace_once(ORACLE, ORACLE_OLD, ORACLE_NEW)


if __name__ == "__main__":
    main()
