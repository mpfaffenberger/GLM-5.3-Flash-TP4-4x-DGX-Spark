#!/usr/bin/env python3
"""Keep sparse-MLA tuning but skip FlashInfer's hanging generic TP tuner."""

from pathlib import Path

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/warmup/"
    "kernel_warmup.py"
)

OLD = '''    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)
'''

NEW = '''    elif has_flashinfer() and current_platform.has_device_capability(90):
        # GLM-5.3 on distributed GB10 already runs the dedicated SM120 sparse-
        # MLA tuner above. The generic 8192-token TP autotune dummy run can
        # deadlock all ranks, so deliberately retain heuristic configs for
        # unrelated FlashInfer operations.
        logger.info_once(
            "Skipping generic distributed FlashInfer autotune; "
            "dedicated sparse-MLA tuning remains enabled."
        )
'''


def main() -> None:
    text = TARGET.read_text()
    if NEW in text:
        print(f"already patched: {TARGET}")
        return
    if OLD not in text:
        raise SystemExit(f"expected generic autotune call not found: {TARGET}")
    TARGET.write_text(text.replace(OLD, NEW, 1))
    print(f"patched: {TARGET}")


if __name__ == "__main__":
    main()
