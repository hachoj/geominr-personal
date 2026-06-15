# Architecture Decision Records

- [0001](0001-inference-sr-speedup.md) — Speeding up model SR reconstruction in inference.py (~5.4× via masking-loop factorization; reject kernel-level changes as compute-bound under torch.compile)
- [0002](0002-inference-consolidation.md) — Consolidate inference to a single reconstruct module + inference.py (promote v2 NN/linear, delete dead variants/backups; drop separate-compile as solo-equivalent)
