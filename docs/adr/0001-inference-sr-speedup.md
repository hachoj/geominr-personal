# 0001 — Speeding up model SR reconstruction in inference.py

Status: Accepted (2026-06-04)

## Context

`scripts/inference.py` reconstructs a full axial volume by querying the HAT INR
over every covered voxel (`reconstruct_volume_sr_fast` in
`geominr/models/utils/reconstruct.py`). At the manuscript resolution
(`ap_voxel_count=1536`, `volume_depth=48` → a `3072×1536×48 ≈ 226 M`-voxel grid),
a single patient (UF181, 247 slices) took **~572 s** on one B200.

We profiled the hot path empirically (canonical checkpoint, reduced and full
configs) before changing anything. Two findings drove every decision:

1. **The per-(wedge × row-tile × col-tile) masking loop dominated at full
   resolution.** For each of ~110 700 tiles it built full-volume boolean masks
   and ran `torch.nonzero` over the entire `[LR,AP,SI]` grid — `O(wedges·tiles·V)`.
   At the reduced profiling config (`192/8`) this was only ~16 s (≈14 %), which
   *under-represented* it; at full resolution it was **~467 s (≈82 %)**.
2. **The model forward is compute-bound, not launch-bound, under
   `torch.compile`.** It is ~entirely the HAT encoder, linear in the tile-batch
   size and independent of the query count.

## Decision

Keep the math/weights identical; optimize only the host-side work and the
compile setup. Concretely:

- **Factorize the masking loop (the big win).** `theta`/`j_idx` depend only on
  the `(LR,AP)` plane and `i_idx` only on `SI`, so each tile's voxel set is the
  Cartesian product of a plane-voxel set and an SI set. Replace the full-volume
  boolean-AND + `torch.nonzero` per tile with cheap 2D/1D masks, emitting
  contributions in the same `(lr,ap,si)` order. `O(wedges·tiles·V)` → `O(contributions)`.
- **Parallelize `extract_slices`** with a `ThreadPoolExecutor` (SimpleITK
  releases the GIL; reads are independent). ~4.1 s → ~0.9 s.
- **Stabilize compilation:** compile `model.encoder` and `model.decoder`
  separately (the expensive encoder graph then depends only on the tile-batch
  size, never the query count) and pad the per-batch query count to a fixed
  power-of-two bucket. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Rejected (measured, did not help)

Under `torch.compile`, inductor already optimizes the GPU path, so manual
kernel/layout changes gave no steady-state gain and hurt compile time:

- **Twin-tower encoder batching** (run both slice streams as one `2B` batch):
  no steady-state change, ~3× slower compile, doubled peak memory (OOM without
  SDPA). Reverted.
- **Flash SDPA** in SW-MSA/SW-MCA: numerically fine but no speedup at the small
  `L=64` windows; only useful to make twin-tower fit memory, which we dropped.
  Reverted.
- **channels_last (NHWC):** steady 31.6 s → 37.8 s and compile 53.6 s → 103.7 s,
  bit-identical output (inductor ignored the layout hint, paid for conversions).
- **CUDA graphs (reduce-overhead):** not pursued — halving launches via
  twin-tower showed the forward is compute-bound, so launch-overhead removal has
  no headroom; high risk (buffer aliasing) for no expected gain.

fp8 / fewer tiles were out of scope: they change the output.

## Consequences

- **~5.4× faster** at manuscript resolution (UF181, 1 B200): **572 s → 105 s**.
- Output equivalent within bf16 noise: vs the original compiled output,
  `max_abs ≤ 3`, ~all differing voxels by ±1, mean ≈ 0, nonzero count matches.
  (The shipped path already used bf16 autocast + `torch.compile`, which itself
  differs from eager by up to 42 levels, so this is well inside existing noise.)
- The masking rewrite is bit-exact in eager (verified at the reduced config).
- Touches only `reconstruct_volume_sr_fast` and the compile setup; the model
  modules (`hat.py`, attention) are unchanged, so training/eval are unaffected.
