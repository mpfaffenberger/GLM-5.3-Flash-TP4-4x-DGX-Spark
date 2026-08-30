# Benchmarks

## 65K focused concurrency validation

[`llama-benchy-65k-c5-c10.csv`](llama-benchy-65k-c5-c10.csv) is the validated
focused stress result for the published host-metadata KDA image:

- context: 65,535 tokens;
- concurrency: 5 and 10;
- runs: 3;
- request starts/ends: 90/90;
- request errors: 0;
- result rows: 8/8;
- benchmark exit code: 0;
- pre-C1 median: 37.15 tok/s;
- post-C1 median: 36.51 tok/s.

Source run:
`results/iso-65535-c5_10-20260829-215902` (runtime artifacts are intentionally
ignored by Git).

This is **not** a full 0→65K matrix. It validates the long-context c5/c10
boundary used by the published 65K support claim. Failed or partial 100K runs
are not included.
