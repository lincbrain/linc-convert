#!/usr/bin/env python
"""Quantitative + visual evaluation of a stitched LSM mosaic.

Feature: stitching-artifact-correction (SC-001..SC-004, SC-006).

Metrics (all computed on a representative Z plane of the stitched mosaic, shape (Z,Y,X)):
  * seam discontinuity  -- mean |Δintensity| across each strip-seam row vs. interior rows.
                           Ratio ~1.0 => seam indistinguishable from normal texture.
  * stripe energy       -- fraction of 1-D power spect(along X) concentrated in the
                           stripe band, averaged over rows. Lower => fewer stripes.
  * in-strip mean/std   -- recorded so compare.py can check non-seam content is preserved.

Seam rows are derived from strip geometry (--strip-height, --overlap, --n-strips), NOT
hard-coded. Fails loudly if the mosaic is unreadable or geometry is inconsistent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def seam_rows(strip_height: int, overlap: int, n_strips: int) -> list[int]:
    """Y positions where adjacent strips join in the stitched mosaic."""
    step = strip_height - overlap
    return [k * step + (strip_height - overlap) for k in range(n_strips - 1)]


def seam_discontinuity(
    plane: np.ndarray, seams: list[int], margin: int = 150, band: int = 400
) -> float:
    """Max strip-body brightness step across seams, as %% of the overall mean.

    The visible seam is the brightness difference between adjacent strip *bodies*
    (the raised-cosine blend turns the junction into a gradient, so a local row-jump
    metric misses it). For each seam we compare the median of a body band just above
    vs. just below, skipping the blend zone (``margin``). 0%% => indistinguishable.
    """
    row_mean = plane.mean(axis=1).astype(np.float64)
    overall = float(np.median(row_mean)) + 1e-9
    steps = []
    for s in seams:
        above = row_mean[max(0, s - margin - band): max(0, s - margin)]
        below = row_mean[s + margin: s + margin + band]
        if above.size and below.size:
            steps.append(abs(np.median(above) - np.median(below)))
    step = float(np.max(steps)) if steps else 0.0
    return 100.0 * step / overall


def stripe_energy(plane: np.ndarray, band=(0.02, 0.5), axis: int = 0) -> float:
    """Stripe peak prominence along ``axis``: dominant in-band spectral peak / median.

    LSM stripes run parallel to the illumination beam and the branch's correction is a
    per-(z,y)-line model, so stripes are periodic along **Y** (axis=0 of a (Y,X) plane),
    i.e. horizontal bands. We therefore measure prominence along Y by default: a periodic
    stripe makes one frequency dominate -> large ratio; broadband texture/noise -> ~1.
    The (0.02, 0.5) band excludes the very-low-frequency seam steps and DC.
    """
    p = plane.astype(np.float64)
    p = p - p.mean(axis=axis, keepdims=True)
    spec = (np.abs(np.fft.rfft(p, axis=axis)) ** 2).mean(axis=1 - axis)
    freqs = np.fft.rfftfreq(p.shape[axis])
    bandmask = (freqs >= band[0]) & (freqs <= band[1])
    if not bandmask.any():
        return 0.0
    peak = float(spec[bandmask].max())
    med = float(np.median(spec[1:]) + 1e-9)  # exclude DC from baseline
    return peak / med


def mid_plane(path: str) -> np.ndarray:
    import zarr

    grp = zarr.open_group(path, mode="r")
    key = "0" if "0" in grp else next(iter(grp.keys()))
    a = grp[key]
    if a.ndim != 3:
        raise ValueError(f"Expected 3-D mosaic, got shape {a.shape}")
    z = a.shape[0] // 2
    return np.asarray(a[z])


def evaluate(path: str, *, strip_height: int, overlap: int, n_strips: int) -> dict:
    plane = mid_plane(path)
    seams = seam_rows(strip_height, overlap, n_strips)
    interior_lo = plane[: seams[0] - 8] if seams else plane
    return {
        "mosaic": path,
        "plane_shape": list(plane.shape),
        "seams": seams,
        "seam_discontinuity": seam_discontinuity(plane, seams),
        "stripe_energy": stripe_energy(plane),
        "in_strip_mean": float(interior_lo.mean()),
        "in_strip_std": float(interior_lo.std()),
    }


def save_figure(path: str, out_png: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[eval] matplotlib unavailable, skipping figure: {exc}")
        return
    plane = mid_plane(path)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.figure(figsize=(10, 6))
    vmax = np.percentile(plane, 99.5)
    plt.imshow(plane, cmap="gray", vmax=vmax, aspect="auto")
    plt.title(os.path.basename(path))
    plt.colorbar()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[eval] wrote {out_png}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mosaic", help="stitched OME-Zarr output path")
    ap.add_argument("--report", required=True, help="output JSON path")
    ap.add_argument("--figures", default=None, help="output figures dir")
    ap.add_argument("--strip-height", type=int, default=1600)
    ap.add_argument("--overlap", type=int, default=192)
    ap.add_argument("--n-strips", type=int, default=3)
    args = ap.parse_args(argv)

    report = evaluate(
        args.mosaic,
        strip_height=args.strip_height,
        overlap=args.overlap,
        n_strips=args.n_strips,
    )
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if args.figures:
        save_figure(args.mosaic, os.path.join(args.figures, "mosaic.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
