# Stitching artifact correction — investigation notes

Feature branch: `fix/stitching-artifact-correction` (off `james/spool_zarr`).
Evaluation sample: 3 adjacent strips of `sub-MF283/sample-slice036`, full Y=1600,
Z[64:128], X[12000:16096] (a tissue region — see below), extracted by
`scripts/sample/extract_sample.py` into scratch (never committed; restricted data).

## Baseline (current `james/spool_zarr` behavior, `--blend`, no stripe maps)

Measured on the representative sample (mid-Z plane), via `scripts/eval/evaluate_stitch.py`:

| metric | value | meaning |
|--------|-------|---------|
| seam discontinuity | 4.03 | seams present |
| stripe prominence (along Y) | 22.1 | strong periodic stripes along Y |
| in-strip mean / std | 125 / 64 | real tissue contrast |

Per-strip body means: **127.8 / 204.2 / 265.3** — strips get progressively brighter.

## Findings

1. **Region matters.** A naive crop at X[0:4096] is *background* (mean ~102, std ~0.9) and
   shows almost no artifact. Tissue with contrast is around X≈12000–16000, Z≈96. The
   extraction script now targets that window (`--x-start 12000 --z-start 64`).

2. **Stripes are periodic along Y, not X.** The branch's correction is a per-(z,y)-line
   model (constant along X). The evaluation stripe metric was corrected to measure spectral
   prominence along Y (axis=0); it then reports 22.1 on the sample (vs 1.07 measuring X).

3. **The seam is a whole-strip brightness offset, not a local edge.** The raised-cosine
   blend already smooths the *local* transition over the 192-px overlap (local row jumps are
   ~few units). The visible seam is the large strip-to-strip body difference (128→204→265).

4. **A naive in-blend gain equalization is the wrong fix.** Matching adjacent strips' overlap
   means and multiplying (sequential) overcorrects via compounding (strip3 → 101, below
   strip1's 128) and can flatten genuine signal. It *raised* the seam metric (4.03 → 5.06).
   Reverted. Per-strip illumination normalization belongs in the stripe/white-matter
   normalization path (`stripes` → `white_matter_intensity / corr`), applied to whole strips
   against a common target — which requires the `mip → stripes` chain working first.

5. **`mip → stripes` filename mismatch (R5) blocks the stripe path.** `mip.py` writes
   `{name}.tiff`; `stripes.py` reads `{name}_proc-mip.tiff` (with a `slice0`→`slice` rewrite).
   Must be reconciled before a stripe baseline/after can be produced.

6. **Inverted background mask (R3) fixed.** `data*(data < threshold)` kept background and
   zeroed foreground; changed to `data*(data >= threshold)`, gated by `background_threshold`
   (default unchanged). Latent data-erasing bug (constitution: never fail silently).

## Next steps (remaining tasks)

- Decide artifact-vs-signal for the inter-strip brightness ramp (domain input); likely
  normalize per-strip illumination via the stripe/white-matter path, not the blend.
- Fix R5 naming so `mip → stripes → stitch` chains; produce a stripe "before".
- Harden stripe maps (empty-line fallback + logging, R4) and re-measure stripe prominence.
- Make the seam metric target the inter-strip body step; re-validate against SC-002/003.
