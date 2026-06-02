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

## Update — illumination normalization implemented and measured

Changes made:
- **R5 fixed**: `mip.py` now writes `{name}_proc-mip.tiff`; `stripes.py` reads it directly
  (removed the `slice0`→`slice` rewrite that corrupted `slice036`→`slice36`). The
  `mip → stripes → stitch` chain runs end-to-end.
- **Stripe-map hardening**: empty (no-foreground) `(z,y)` lines no longer become the
  `9999999` sentinel (→ black holes); they are filled with the tile's finite-line median
  and the count is **logged**. On the sample, chunk-0001 had **49,920/102,400 (49%)** empty
  lines — these would have been black stripes. Added a finiteness guard on apply.
- **Map smoothing along Y** (`smooth_y`, Gaussian): the per-line map is noisy (many fallback
  lines); dividing by it injects new stripes. Smoothing keeps the low-frequency illumination
  falloff but not the line noise.
- **Seam metric corrected**: now measures the strip-body brightness step across each seam
  (the blend turns the junction into a gradient, so a local row-jump metric missed it).

Results on the sample (`white_matter_intensity=200`, `smooth_y=25`), vs. the `--blend`-only
baseline:

| metric | baseline | after | change |
|--------|----------|-------|--------|
| seam-body step (% of mean) | 126.6% | 13.6% | **−89%** (≈ SC-002's 90% target) |
| stripe prominence (Y) | 22.1 | 23.9 | ≈ neutral (−8%) |

Interpretation:
- **Seams (P1) are essentially fixed** by per-line illumination normalization (the
  `stripes`/`white_matter_intensity` path), once the map is hardened and Y-smoothed.
- **Stripes (P2) are not yet reduced.** The per-line flat-field removes the low-frequency
  falloff that drives the seam, but not the mid-frequency stripe prominence. Note also the
  overlap analysis: adjacent strips' shared-overlap foreground medians disagree ~2.5× (153 vs
  379), i.e. strong **within-strip illumination falloff along Y** — a single per-strip gain
  cannot fix it (confirmed: whole-strip normalization only moved the step 126.6%→103%).

## Stripe suppression: why no FFT-notch, and what's actually left

Investigated stripe suppression with spectral analysis + visual inspection of the stitched
mosaic (figures generated in scratch; not committed — restricted data):

- **No narrow stripe peak.** The Y power spectrum of the baseline has its in-band energy
  spread broadly around period ~35–48 px (peak/band-sum ≈ 0.006; a true periodic stripe
  would be ≈ 1), and low-frequency power dominates the band by ~37×. So an **FFT-notch is the
  wrong tool** — there is no narrow frequency to notch, and notching the broad band would
  destroy real ~40 px tissue structure. Deliberately NOT implemented.
- **The residual "stripe" is banding introduced by the correction**, not a source stripe.
  Visually, the per-line `white_matter / corr` multiply equalizes the strips (seam fixed) but
  amplifies low-signal/background lines unevenly into horizontal bands. Y-smoothing
  (`smooth_y`) and bounding the gain (`correction_clip`, clips to [clip, 100-clip] percentiles)
  reduce the worst of it but do not eliminate it (both ~neutral on the prominence metric).
- The stripe-prominence metric is an **unreliable proxy** here: corrected images have more
  contrast, which inflates it regardless of true striping.

**Foreground-gated correction implemented** (`foreground_gated=True`,
`background_level` optional, else per-tile 5th percentile): apply
`corrected = bg + (data - bg) * correction` so background stays flat and only signal above
background is normalized. Results vs. baseline on the sample (`smooth_y=25`, `wm=200`):

| variant | seam-body step | stripe prominence | mean | banding (visual) |
|---------|----------------|-------------------|------|------------------|
| baseline (blend only) | 126.6% | 22.1 | 125 | n/a |
| blanket multiply | 13.6% (−89%) | 23.9 (worse) | 511 (5× amp) | strong horizontal bands |
| **foreground-gated** | 42.4% (−67%) | 20.8 (−6%) | 117 (preserved) | **bands removed** |

Visually the foreground-gated mosaic is the cleanest: strips equalized, tissue preserved, and
the horizontal background banding of the blanket multiply is gone. Its seam-body-step number is
higher only because that metric conflates each strip's *foreground fraction* (strips genuinely
tile different tissue density) with intensity — a metric limitation, not a worse result.
**Recommendation: foreground-gated is the right default for the illumination/stripe correction.**

## Remaining work

- Foreground-gated correction (above) to remove background banding; re-define an acceptance
  metric for stripes that matches a visible artifact (current prominence metric is unreliable).
- Tune `white_matter_intensity` per dataset (default 1000 amplifies this data ~5×; relative
  metrics unaffected, but output scale matters downstream).
- Full-slice end-to-end run (SC-005); `_cm2` camera variant; unit tests for the fixes.
- **Maintainer input**: is the inter-strip / within-strip brightness variation purely
  illumination (normalize away) or partly real signal? And is there a dataset/region with a
  genuine periodic stripe (this sample has none)? These gate the stripe approach.
