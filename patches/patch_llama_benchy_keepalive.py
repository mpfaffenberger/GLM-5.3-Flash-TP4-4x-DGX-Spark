#!/usr/bin/env python3
"""Disable llama-benchy HTTP keepalive for the vLLM Rust frontend.

The tested frontend can retain shared-memory broadcast blocks across streamed
requests on one connection and wedge after the third request. Fresh connections
complete normally. This patch is intentionally narrow and idempotent.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("llama_benchy.runner")
if spec is None or spec.origin is None:
    raise SystemExit("llama_benchy.runner not found")

path = Path(spec.origin)
text = path.read_text(encoding="utf-8")
old = "force_close=False, keepalive_timeout=600"
new = "force_close=True"

if new in text:
    print(f"already patched: {path}")
elif old in text:
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched: {path}")
else:
    raise SystemExit(f"expected TCPConnector configuration not found in {path}")
