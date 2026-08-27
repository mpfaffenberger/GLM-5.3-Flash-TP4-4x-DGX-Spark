#!/usr/bin/env python3
"""Small streaming benchmark for the OpenAI-compatible GLM vLLM endpoint."""

from __future__ import annotations

import concurrent.futures
import json
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
MODEL = "glm-5.3-flash-fp8"
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 256
PROMPT = (
    "Write a technically precise explanation of distributed GPU inference, "
    "continuing until the output token limit. Do not use markdown headings."
)


@dataclass(frozen=True)
class Result:
    ttft: float
    elapsed: float
    tokens: int

    @property
    def decode_rate(self) -> float:
        decode_time = max(self.elapsed - self.ttft, 1e-9)
        return max(self.tokens - 1, 0) / decode_time


def stream_once() -> Result:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    first_token_at: float | None = None
    completion_tokens = 0

    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or {}
            completion_tokens = max(
                completion_tokens, int(usage.get("completion_tokens") or 0)
            )
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if any(delta.get(key) for key in ("content", "reasoning_content")):
                    first_token_at = first_token_at or time.monotonic()

    finished = time.monotonic()
    if first_token_at is None:
        raise RuntimeError("stream completed without a content or reasoning token")
    if completion_tokens <= 0:
        raise RuntimeError("stream did not report completion token usage")
    return Result(first_token_at - started, finished - started, completion_tokens)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def run_batch(concurrency: int) -> None:
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda _: stream_once(), range(concurrency)))
    wall = time.monotonic() - started
    total_tokens = sum(result.tokens for result in results)
    ttfts = [result.ttft for result in results]
    rates = [result.decode_rate for result in results]
    print(
        f"concurrency={concurrency} requests={len(results)} "
        f"tokens={total_tokens} wall={wall:.2f}s "
        f"aggregate={total_tokens / wall:.2f} tok/s "
        f"ttft_p50={statistics.median(ttfts):.3f}s "
        f"ttft_p95={percentile(ttfts, 0.95):.3f}s "
        f"decode_p50={statistics.median(rates):.2f} tok/s"
    )


def main() -> None:
    print(f"endpoint={BASE} model={MODEL} max_tokens={MAX_TOKENS}")
    print("warmup=1")
    warmup = stream_once()
    print(
        f"warmup_tokens={warmup.tokens} ttft={warmup.ttft:.3f}s "
        f"decode={warmup.decode_rate:.2f} tok/s"
    )
    run_batch(1)
    run_batch(4)
    print("BENCH_OK")


if __name__ == "__main__":
    main()
