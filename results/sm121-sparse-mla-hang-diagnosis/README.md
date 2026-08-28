# SM121 sparse-MLA >2048-context hang — diagnosis evidence

## Symptom

Any request whose total context exceeds GLM-5.3's `index_topk = 2048` wedges
the TP=4 engine permanently: all four GB10 GPUs sit at ~96% utilization and
~20 W, the Ray shm command ring stops advancing
(`No available shared memory broadcast block found`), and no request ever
completes again. Every request at or below 2048 total tokens works, because
vLLM's kpool indexer port fills the top-k buffer with a causal `arange`
shortcut and never runs the real sparse-selection path
(`sparse_attn_indexer_kpool.py`, "Real sparsity only kicks in for contexts
> topk_tokens").

## Method

1. Reproduced the hang on the production image with a 2100-token probe and
   captured host-side `py-spy dump` stacks of all four ranks while wedged
   (`pyspy-host-*.txt`). All four CPUs were blocked at CUDA kernel *launches*
   inside the DeepGemm MoE path — streams jammed behind an earlier
   never-terminating kernel.
2. Rebuilt the cluster with `CUDA_LAUNCH_BLOCKING=1`
   (`GLM53_DOCKER_ENV` support in `glm53_node_up.sh`) so each launch blocks
   until its kernel completes, and repeated the probe
   (`blocking-rank-*.txt`).

## Finding

With synchronous launches the stacks are unambiguous:

- One rank is inside FlashInfer's SM120 sparse-MLA paged attention and never
  returns:
  `flashinfer/mla/_sparse_mla_sm120.py:392 _paged_attention` via
  `trtllm_batch_decode_with_kv_cache_mla` (DSv3.2/v32 path), called from
  vLLM's `flashinfer_mla_sparse_sm120.py:162 forward_mqa`.
- The other three ranks are parked in `ncclAllReduce` from the o_proj
  row-parallel projection immediately after attention (`mla.py:245`),
  waiting on the stuck rank.

Root cause: the FlashInfer SM120 sparse-MLA kernel does not terminate on
GB10 (`sm_121a`) when consuming a fully populated top-k index table from
GLM's kpool indexer. Contexts ≤ 2048 never build a real table, which is why
the failure cleanly bisects at the boundary.

Note: GLM's indexer buffer is 2176 entries (2048 pool-expanded + kpool
tail); the fixed-ABI NoPE patch slices the first 2048 for FlashInfer's
table. A table containing out-of-range, duplicate, or `-1` mid-table
entries that the kernel's page-walk does not guard is the leading suspect
and the first thing to check upstream.

## Ruled out

- `llama-benchy` HTTP keepalive and SSE `[DONE]` handling (patched;
  small-context suites now pass).
- The optional `return_token_ids` streaming extension.
- The indexer FP8 quant block: blanket 64 breaks the required Hadamard-128
  query quant (`attention.py:382` asserts `quant_block_size == 128`); the
  hang reproduces with stock 128 and is downstream of the indexer.
- MoE/DeepGemm kernels: they appear in async stacks only because their
  launches queue behind the stuck attention kernel.
