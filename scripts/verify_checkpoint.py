#!/usr/bin/env python3
"""Verify the pinned GLM-5.3 FP8 Hugging Face snapshot without loading weights."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

MODEL_CACHE = "models--unsloth--GLM-5.3-Flash-FP8"
EXPECTED_REVISION = "a160e2291674d9e3e92e98fd82faa2544a2867a3"
EXPECTED_SHARDS = 62
EXPECTED_WEIGHT_BYTES = 328_337_455_672


def cache_root() -> Path:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return hf_home / "hub" / MODEL_CACHE


def resolve_snapshot(root: Path) -> Path:
    pinned = root / "snapshots" / EXPECTED_REVISION
    if pinned.is_dir():
        return pinned

    ref = root / "refs" / "main"
    if ref.is_file():
        revision = ref.read_text(encoding="utf-8").strip()
        candidate = root / "snapshots" / revision
        if revision != EXPECTED_REVISION:
            raise RuntimeError(f"main points to {revision}, expected {EXPECTED_REVISION}")
        if candidate.is_dir():
            return candidate

    raise RuntimeError(f"pinned snapshot not found below {root}")


def main() -> int:
    try:
        snapshot = resolve_snapshot(cache_root())
        index_path = snapshot / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        referenced = {snapshot / name for name in weight_map.values()}
        shards = sorted(snapshot.glob("model-*-of-*.safetensors"))
        missing = sorted(str(path.name) for path in referenced if not path.is_file())

        if len(shards) != EXPECTED_SHARDS:
            raise RuntimeError(f"found {len(shards)} shards, expected {EXPECTED_SHARDS}")
        if len(referenced) != EXPECTED_SHARDS:
            raise RuntimeError(f"index references {len(referenced)} shards, expected {EXPECTED_SHARDS}")
        if missing:
            raise RuntimeError(f"missing index shards: {', '.join(missing[:5])}")

        total = sum(path.stat().st_size for path in shards)
        if total != EXPECTED_WEIGHT_BYTES:
            raise RuntimeError(
                f"weight bytes {total:,} != expected {EXPECTED_WEIGHT_BYTES:,}"
            )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"CHECKPOINT_INVALID: {exc}", file=sys.stderr)
        return 1

    print(f"CHECKPOINT_OK revision={EXPECTED_REVISION}")
    print(f"snapshot={snapshot}")
    print(f"shards={len(shards)} tensors={len(weight_map)} bytes={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
