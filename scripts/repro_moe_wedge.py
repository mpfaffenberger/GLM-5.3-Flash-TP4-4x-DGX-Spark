#!/usr/bin/env python3
"""Fast single-shot reproducer for the GLM-5.3-Flash-FP8 MoE rank wedge.

Root-cause evidence (see results/ + ~/glm53-hang-dumps) shows the full Spark
Arena suite wedges when a *long prefill is admitted while other requests are
already decoding*. At that moment the four TP ranks diverge inside a single MoE
layer: rank 0 sits in the router (``GateLinear.forward``) while ranks 1-3 are
already inside ``_maybe_apply_shared_experts`` -> FP8 input-quant. The engine
then reports "No available shared memory broadcast block found in 60 seconds"
and dies with "RPC call to sample_tokens timed out" while the GPUs spin at
~20 W.

llama-benchy needs ~50 minutes of unrelated shapes to reach that state. This
script builds the state directly: fire N long requests, wait until every one of
them is decoding, then admit one more long prompt into the running batch.

Exit codes: 0 = no wedge, 42 = wedge reproduced, 2 = API/infra problem.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request

FILLER = "The quick brown fox jumps over the lazy dog while analysts review. "
DEFAULT_WORKERS = "10.0.0.13 10.0.0.150 10.0.0.246"
DEFAULT_PYSPY = os.path.expanduser(
    "~/.cache/uv/archive-v0/3eVEIjOvVkqR7Wxj/bin/py-spy"
)

_lock = threading.Lock()
_state = {"last_event": time.monotonic(), "active": 0, "stalled": False}


def post(url: str, payload: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def note_event() -> None:
    with _lock:
        _state["last_event"] = time.monotonic()


def calibrate(base: str, model: str, target: int) -> str:
    """Return a prompt text whose token length is within 2% of ``target``."""
    probe = FILLER * 64
    count = post(f"{base}/tokenize", {"model": model, "prompt": probe})["count"]
    per_unit = count / 64
    repeats = max(1, math.ceil(target / per_unit))
    prompt = FILLER * repeats
    for _ in range(6):
        actual = post(f"{base}/tokenize", {"model": model, "prompt": prompt})["count"]
        ratio = target / actual
        if abs(ratio - 1.0) <= 0.02:
            return prompt
        repeats = max(1, int(repeats * ratio))
        prompt = FILLER * repeats
    print(f"WARN prompt calibrated to {actual} tokens for target {target}")
    return prompt


def stream_chat(base: str, model: str, prompt: str, max_new: int, tag: str,
                reader: threading.Event) -> bool:
    """Stream one completion. Returns True when it finished cleanly."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with _lock:
        _state["active"] += 1
        _state["last_event"] = time.monotonic()
    tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                delta = (json.loads(payload).get("choices") or [{}])[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    tokens += 1
                    note_event()
                    reader.set()
        print(f"  {tag} done tokens={tokens}", flush=True)
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"  {tag} FAILED after {tokens} tokens: {exc}", flush=True)
        return False
    finally:
        with _lock:
            _state["active"] -= 1


DUMP_SNIPPET = """
pid=$(docker top {container} -eo pid,args | awk '/VLLM::Worker_TP/{{print $1; exit}}')
if [ -n "$pid" ]; then sudo -n {pyspy} dump --pid "$pid"; else echo "no worker process"; fi
"""


def _dump_one(target: str | None, container: str, pyspy: str, out_dir: str) -> None:
    """Dump the TP worker stack from `target` (None = this host) into out_dir."""
    snippet = DUMP_SNIPPET.format(container=container, pyspy=pyspy)
    cmd = ["bash", "-c", snippet] if target is None else \
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", target, "bash -s"]
    suffix = "head" if target is None else target
    with open(f"{out_dir}/rank-{suffix}.txt", "w") as stack_file:
        if target is None:
            subprocess.run(cmd, stdout=stack_file, stderr=subprocess.STDOUT, check=False)
        else:
            subprocess.run(cmd, input=snippet.encode(), stdout=stack_file,
                           stderr=subprocess.STDOUT, check=False)
    gpu_cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,memory.used",
               "--format=csv,noheader"]
    if target is not None:
        gpu_cmd = ["ssh", "-o", "BatchMode=yes", target, *gpu_cmd]
    with open(f"{out_dir}/rank-{suffix}-gpu.txt", "w") as gpu_file:
        subprocess.run(gpu_cmd, stdout=gpu_file, stderr=subprocess.STDOUT, check=False)


def capture(dirpath: str, workers: str, pyspy: str) -> None:
    os.makedirs(dirpath, exist_ok=True)
    print(f"  capturing rank stacks -> {dirpath}", flush=True)
    container = os.environ.get("GLM53_CONTAINER", "glm53-tp4")
    _dump_one(None, container, pyspy, dirpath)
    for ip in workers.split():
        # The container has no SYS_PTRACE, so py-spy must run on the host.
        subprocess.run(["scp", "-q", pyspy, f"{ip}:/tmp/py-spy"], check=False)
        _dump_one(ip, container, "/tmp/py-spy", dirpath)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="glm-5.3-flash-fp8")
    ap.add_argument("--target-tokens", type=int, default=32768)
    ap.add_argument("--decoding", type=int, default=4,
                    help="requests already decoding before the extra prefill")
    ap.add_argument("--stagger", type=int, default=1,
                    help="long prefills admitted while the above decode")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--ramp-depths", default="",
                    help="optional comma list, e.g. 8192,16384, run first")
    ap.add_argument("--stall-seconds", type=float, default=150.0)
    ap.add_argument("--capture-dir", default=os.path.expanduser(
        "~/glm53-hang-dumps/repro-moe-wedge"))
    ap.add_argument("--workers", default=os.environ.get("GLM53_WORKERS", DEFAULT_WORKERS))
    ap.add_argument("--pyspy", default=os.environ.get("GLM53_PYSPY", DEFAULT_PYSPY))
    args = ap.parse_args()

    try:
        post(f"{args.base_url}/models", {})
    except Exception as exc:  # noqa: BLE001 - report any handshake failure
        print(f"API unreachable at {args.base_url}: {exc}")
        return 2

    print(f"calibrating prompt at {args.target_tokens} tokens ...", flush=True)
    prompt = calibrate(args.base_url, args.model, args.target_tokens)

    def watchdog() -> None:
        while not _state["stalled"]:
            time.sleep(5)
            with _lock:
                idle = time.monotonic() - _state["last_event"]
                busy = _state["active"] > 0
            if busy and idle > args.stall_seconds:
                _state["stalled"] = True
                stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
                print(f"WEDGE: no token for {idle:.0f}s with {_state['active']} active", flush=True)
                capture(f"{args.capture_dir}/{stamp}", args.workers, args.pyspy)
                return

    threading.Thread(target=watchdog, daemon=True).start()

    def round_at(depth_tokens: int, label: str, stagger: int) -> bool:
        text = prompt if depth_tokens == args.target_tokens else calibrate(
            args.base_url, args.model, depth_tokens)
        readers = [threading.Event() for _ in range(args.decoding)]
        threads = [
            threading.Thread(
                target=stream_chat,
                args=(args.base_url, args.model, text, args.max_new,
                      f"{label}#{i}", readers[i]),
            )
            for i in range(args.decoding)
        ]
        for t in threads:
            t.start()
        # The wedge needs a prefill landing on top of live decode, so wait for
        # every request to emit its first token before admitting more work.
        deadline = time.monotonic() + 300
        for i, ev in enumerate(readers):
            remaining = deadline - time.monotonic()
            if not ev.wait(timeout=max(1.0, remaining)):
                print(f"  {label} reader {i} never emitted a token", flush=True)
        late = [
            threading.Thread(
                target=stream_chat,
                args=(args.base_url, args.model, text, args.max_new,
                      f"{label}+stagger{j}", threading.Event()),
            )
            for j in range(stagger)
        ]
        for t in late:
            t.start()
            time.sleep(0.4)
        for t in threads + late:
            t.join(timeout=900)
        return not _state["stalled"]

    for depth in [int(d) for d in args.ramp_depths.split(",") if d.strip()]:
        stamp = time.strftime("%H:%M:%S", time.gmtime())
        print(f"[{stamp}] ramp depth={depth} c={args.decoding + args.stagger}", flush=True)
        if not round_at(depth, f"ramp{depth}", stagger=0):
            return 42

    for it in range(1, args.iterations + 1):
        stamp = time.strftime("%H:%M:%S", time.gmtime())
        print(f"[{stamp}] iter {it}/{args.iterations} depth={args.target_tokens} "
              f"decoding={args.decoding} +{args.stagger} admitted mid-decode", flush=True)
        if not round_at(args.target_tokens, f"i{it}", stagger=args.stagger):
            print("REPRODUCED (exit 42)", flush=True)
            return 42
    print("NO REPRODUCTION: every request completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
