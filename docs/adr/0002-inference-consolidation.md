# 0002 — Consolidate inference reconstruction to a single module + script

Status: Accepted (2026-06-04)

Builds on [0001](0001-inference-sr-speedup.md) — the SR speedup decision in 0001
stands; this record only changes the file/module organization around it.

## Context

After the SR speedup (0001), the inference path had accumulated redundant,
partly-broken variants:

- **Two inference scripts:** `scripts/inference.py` (separate encoder/decoder
  compile, but wired to a slow triple-loop NN and a v1 trilinear that crashed at
  `3072x1536x48`) and `scripts/inference_linear_fast.py` (vectorized v2 NN/linear
  + per-phase timing, whole-model compile).
- **Three reconstruct modules:** `reconstruct.py` (the good
  `reconstruct_volume_sr_fast` + dead v1 NN/linear), `reconstruct_fast_v2.py`
  (the good v2 NN/linear), and `reconstruct_orig.py` (a pristine pre-optimization
  backup, for A/B).
- Dev harnesses (`_verify_recon.py`, `_bench_recon.py`) and a full
  `_speedup_backup/` tree.

Measured on UF181 at `3072x1536x48`, **solo**, the whole-model compile reaches the
same SR time as the separate-compile benchmark (104 s vs 105 s) — the 140 s seen
when two reconstructions ran at once was host-side contention, not the compile
strategy. So 0001's separate-compile complexity buys nothing measurable here.

## Decision

Consolidate to one module and one script:

- `geominr/models/utils/reconstruct.py` holds everything: `extract_slices`,
  `save_volume`, `reconstruct_volume_sr_fast` (unchanged), and the vectorized
  `reconstruct_volume_nn` / `reconstruct_volume_linear_fast` (the former `_v2`
  functions, promoted in and renamed without the suffix). Shared geometry helpers
  `_prepare_slices` / `_prepare_geometry` moved in alongside.
- `scripts/inference.py` is the single entry point: v2 baselines + whole-model
  `torch.compile` + per-phase timing, importing only from that module.
- Deleted: `scripts/inference_linear_fast.py`,
  `geominr/models/utils/reconstruct_fast_v2.py`,
  `geominr/models/utils/reconstruct_orig.py`, `scripts/_verify_recon.py`,
  `scripts/_bench_recon.py`, `_speedup_backup/`.

## Consequences

- Verified equivalence: the consolidated path's NN and linear volumes are
  byte-identical to the prior output (the v2 code was moved, not changed); SR is
  within bf16 compile-boundary noise (maxdiff 1/255). `sr_fast` was proven
  byte-identical by diff (only its sibling functions changed).
- Table 2 timing reference at `3072x1536x48` (UF181, solo): NN ~0.8 s,
  trilinear ~0.7 s, model SR ~104 s.
- The pre-optimization A/B reference is gone; git history is the record if a
  future A/B is needed.
