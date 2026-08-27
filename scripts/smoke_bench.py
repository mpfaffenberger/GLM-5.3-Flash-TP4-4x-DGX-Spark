#!/usr/bin/env python3
"""Run model-list, chat/reasoning, repeat-request, and simple decode checks."""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
MODEL = "glm-5.3-flash-fp8"


def request(path: str, payload: dict | None = None, timeout: int = 900) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def chat(prompt: str, max_tokens: int = 256) -> tuple[dict, float]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
    }
    started = time.monotonic()
    result = request("/v1/chat/completions", payload)
    return result, time.monotonic() - started


def message(result: dict) -> dict:
    return result["choices"][0]["message"]


def main() -> int:
    print("=== /v1/models ===")
    models = request("/v1/models", timeout=30)
    ids = [entry["id"] for entry in models.get("data", [])]
    print(ids)
    if MODEL not in ids:
        raise RuntimeError(f"served model {MODEL!r} not advertised")

    print("\n=== coherent chat + reasoning parser ===")
    first, elapsed = chat("In exactly two sentences, explain tensor parallel inference.", 1024)
    msg = message(first)
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    print("reasoning:", reasoning[:300] or "(none)")
    print("content:", content[:600])
    print(f"elapsed={elapsed:.2f}s")
    if not content.strip():
        raise RuntimeError("first chat returned empty content")
    leaked = ("<think>" in content) or ("</think>" in content)
    if leaked:
        raise RuntimeError("reasoning markers leaked into content")

    print("\n=== repeat request ===")
    second, elapsed = chat("Reply with exactly: GLM53_SECOND_OK", 512)
    second_content = message(second).get("content") or ""
    print(second_content)
    if "GLM53_SECOND_OK" not in second_content:
        raise RuntimeError("second request did not return the expected marker")
    print(f"elapsed={elapsed:.2f}s")

    print("\n=== basic decode timing ===")
    bench, elapsed = chat("Write a detailed 250-word history of GPU computing.", 768)
    completion_tokens = bench.get("usage", {}).get("completion_tokens", 0)
    rate = completion_tokens / elapsed if elapsed else 0.0
    print(f"completion_tokens={completion_tokens} elapsed={elapsed:.2f}s rate={rate:.2f} tok/s")
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, RuntimeError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"SMOKE_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
