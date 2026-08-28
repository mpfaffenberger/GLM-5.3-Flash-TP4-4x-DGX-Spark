#!/usr/bin/env python3
"""Make llama-benchy release streamed responses as soon as SSE sends DONE."""
from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("llama_benchy.client")
if spec is None or spec.origin is None:
    raise SystemExit("llama_benchy.client not found")

path = Path(spec.origin)
text = path.read_text(encoding="utf-8")
marker = "stream_done = False"

if marker in text:
    print(f"already patched: {path}")
else:
    old_loop = """                async for chunk_bytes in response.content.iter_any():
                    chunk_time = time.perf_counter()
"""
    new_loop = """                stream_done = False
                async for chunk_bytes in response.content.iter_any():
                    chunk_time = time.perf_counter()
"""
    old_done = """                        if line == 'data: [DONE]' or line == 'data:[DONE]':
                            continue
"""
    new_done = """                        if line == 'data: [DONE]' or line == 'data:[DONE]':
                            stream_done = True
                            break
"""
    old_finalize = """                            except json.JSONDecodeError:
                                continue

                self._finalize_stream_tokens(result, content_chunks, usage_completion_tokens, tokenizer)
"""
    new_finalize = """                            except json.JSONDecodeError:
                                continue

                    if stream_done:
                        break

                self._finalize_stream_tokens(result, content_chunks, usage_completion_tokens, tokenizer)
"""
    for old in (old_loop, old_done, old_finalize):
        if old not in text:
            raise SystemExit(f"expected llama-benchy stream block not found in {path}")
    text = text.replace(old_loop, new_loop, 1)
    text = text.replace(old_done, new_done, 1)
    text = text.replace(old_finalize, new_finalize, 1)
    path.write_text(text, encoding="utf-8")
    print(f"patched: {path}")
