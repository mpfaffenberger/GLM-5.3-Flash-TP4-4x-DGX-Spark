#!/usr/bin/env python3
"""Use the DeepGemm-supported FP8 indexer quant block on SM120/SM121."""

from pathlib import Path

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/attention.py"
)
OLD = """        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128  # TODO: get from config
"""
NEW = """        self.scale_fmt = "ue8m0"
        # DeepGemm's paged MQA kernel supports FP8 block_kv=64 on SM120/SM121;
        # the generic 128-wide default asserts during CUDA-graph profiling.
        self.quant_block_size = (
            64
            if current_platform.is_device_capability_family(120)
            else 128
        )
"""


def main() -> None:
    text = TARGET.read_text()
    if NEW in text:
        print(f"already patched: {TARGET}")
        return
    if OLD not in text:
        raise SystemExit(f"expected quant-block context not found: {TARGET}")
    TARGET.write_text(text.replace(OLD, NEW, 1))
    print(f"patched: {TARGET}")


if __name__ == "__main__":
    main()
