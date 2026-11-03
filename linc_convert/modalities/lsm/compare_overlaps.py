"""Compare neighbouring chunks in their overlap, before stitching.

Chunk ``i + 1`` images the last rows of chunk ``i`` again, so after correction the
two copies of the overlap should agree. For every pair this script cuts the
overlap out of both chunks, aligns them with the registration
(``weighted_average.y_offset`` and ``dz``), and compares them in three versions:

    raw          the chunk as stored (no correction)
    noise only   ``(vol - noise).clip(0)``
    corrected    ``(vol - noise).clip(0) * scaler``  (what the stitch uses)

Per pair and version it reports, on tissue blocks present in both copies:

    ratio    median(second copy) / median(first copy)
    gain,
    offset   line through the matched quantiles, ``second = gain * first + offset``
             (a gain far from 1 means a brightness factor between the chunks, an
             offset far from 0 a constant such as a background difference)
    scatter  robust spread (%) of log(second / first) over blocks; this is the
             noise level plus any residual misregistration
    corr     Pearson correlation of the block values

``corrected`` is also split into three bands along y and three along z, to show
whether the mismatch sits at the chunk edges or at one end of the z range. A shift
search (+-3 px of the layer along y and along z) shows whether the registration
is off. Per chunk, the script prints the background that is left in the overlap
rows after the shared noise map is subtracted (the chunk's own trailing-column
median minus the map). Finally it prints how bright every chunk is relative to the
first one, which is what a band pattern across chunks looks like.

    python -m linc_convert.modalities.lsm.compare_overlaps \\
        --input_path .../sample-slice060 --yaml_path .../LSM_metadata.yaml \\
        --camera_id 2 --slice_number 60 --channel 561 \\
        --coords .../slice060_561.yaml --corrections_dir .../60_corrections_noreg_L2 \\
        --layer 2 --min_y 13000 --max_y 17000

The registration file and the corrections must be the ones the stitch uses (the
shared registration of ``weighted_average``). The x window is applied without the
oblique x shear, so it only roughly matches the stitch window; leave it out to use
the whole width.
"""

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

VARIANTS = ("raw", "noise only", "corrected")
LEVELS = np.arange(10, 91, 10)       # percentiles matched between the copies
TISSUE_FRACTION = 0.3                # tissue = block mean above this x the 90th percentile
MAX_TISSUE_SHARE = 0.9               # more tissue blocks than this = no tissue contrast
MIN_BLOCKS = 500                     # fewest tissue blocks for a fit
SHIFTS = range(-3, 4)


def block_mean(a: np.ndarray, block: int) -> np.ndarray:
    """Mean over ``block`` x ``block`` x ``block`` cells (edges trimmed)."""
    z, y, x = (s // block * block for s in a.shape)
    if min(z, y, x) == 0:
        return np.empty((0, 0, 0), dtype=np.float32)
    a = a[:z, :y, :x]
    return a.reshape(z // block, block, y // block, block, x // block, block
                     ).mean(axis=(1, 3, 5))


def corrected_versions(vol: np.ndarray, noise: np.ndarray, scaler: np.ndarray
                       ) -> Dict[str, np.ndarray]:
    """The three versions of a (z, y, x) piece; maps have shape (z, y)."""
    sub = np.clip(vol - noise[:, :, None], 0, None)
    return {"raw": vol, "noise only": sub, "corrected": sub * scaler[:, :, None]}


def tissue_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Blocks that are tissue in both copies (from the noise-only version)."""
    mean = 0.5 * (a + b)
    pos = mean[mean > 0]
    if pos.size == 0:
        return np.zeros(a.shape, dtype=bool)
    return mean > TISSUE_FRACTION * np.percentile(pos, 90)


def compare(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> Optional[dict]:
    """Agreement of two block arrays on the masked blocks."""
    if int(mask.sum()) < MIN_BLOCKS:
        return None
    va, vb = a[mask].astype(np.float64), b[mask].astype(np.float64)
    ok = (va > 0) & (vb > 0)
    if int(ok.sum()) < MIN_BLOCKS:
        return None
    gain, offset = np.polyfit(np.percentile(va, LEVELS), np.percentile(vb, LEVELS), 1)
    log_ratio = np.log(vb[ok] / va[ok])
    scatter = 1.4826 * np.median(np.abs(log_ratio - np.median(log_ratio)))
    return {"n": int(mask.sum()), "ratio": float(np.median(vb) / np.median(va)),
            "gain": float(gain), "offset": float(offset),
            "scatter": 100 * float(scatter),
            "corr": float(np.corrcoef(va, vb)[0, 1])}


def band_ratios(a: np.ndarray, b: np.ndarray, mask: np.ndarray, axis: int
                ) -> List[float]:
    """Median ratio second / first in three equal bands along ``axis``."""
    out = []
    for part in np.array_split(np.arange(a.shape[axis]), 3):
        sel = np.zeros(a.shape, dtype=bool)
        index = [slice(None)] * 3
        index[axis] = part
        sel[tuple(index)] = True
        sel &= mask & (a > 0) & (b > 0)
        out.append(float(np.median(b[sel]) / np.median(a[sel]))
                   if int(sel.sum()) >= 50 else float("nan"))
    return out


def shift_search(a: np.ndarray, b: np.ndarray, axis: int
                 ) -> Tuple[int, float, float]:
    """Best shift (px) of ``b`` against ``a`` along y (1) or z (0).

    Returns ``(best shift, correlation at 0, correlation at the best shift)``.
    A best shift different from 0 means the registration is off by that much.
    ``b`` at index ``i + s`` is compared with ``a`` at index ``i``.
    """
    a, b = a[::2, :, ::2], b[::2, :, ::2]
    floor = TISSUE_FRACTION * float(np.median(a[a > 0])) if (a > 0).any() else 0
    scores = {}
    for s in SHIFTS:
        step = s
        n = a.shape[axis]
        if abs(step) >= n - 4:
            continue
        lo, hi = max(0, -step), min(n, n - step)
        ia = [slice(None)] * 3
        ib = [slice(None)] * 3
        ia[axis], ib[axis] = slice(lo, hi), slice(lo + step, hi + step)
        pa, pb = a[tuple(ia)], b[tuple(ib)]
        keep = (pa > floor) & (pb > floor)
        if int(keep.sum()) < 1000:
            continue
        scores[s] = float(np.corrcoef(pa[keep], pb[keep])[0, 1])
    if 0 not in scores:
        return 0, float("nan"), float("nan")
    best = max(scores, key=scores.get)
    return best, scores[0], scores[best]


def own_background(vol, background_length: int) -> np.ndarray:
    """Background of every (z, y) row from the trailing columns of a chunk."""
    from linc_convert.modalities.lsm.stitching_utils.corrections import (
        compute_noise_map,
    )
    return compute_noise_map(vol, background_length)


def pair_rows(step: int, sy: int, sz: int, dz: int
              ) -> Optional[Tuple[slice, slice, slice, slice]]:
    """Overlap rows and z ranges: (A rows, B rows, A z, B z).

    A row ``r`` is B row ``r - step``; A z ``z`` is B z ``z - dz``.
    """
    height = sy - step
    zlo, zhi = max(0, dz), min(sz, sz + dz)
    if height <= 0 or zhi - zlo <= 0:
        return None
    return (slice(step, sy), slice(0, height),
            slice(zlo, zhi), slice(zlo - dz, zhi - dz))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input_path", required=True, help="slice folder with the chunks")
    p.add_argument("--yaml_path", required=True)
    p.add_argument("--camera_id", type=int, required=True)
    p.add_argument("--slice_number", type=int, required=True)
    p.add_argument("--channel", required=True)
    p.add_argument("--coords", required=True, help="register_chunks YAML")
    p.add_argument("--corrections_dir", required=True)
    p.add_argument("--layer", type=int, default=2)
    p.add_argument("--background_length", type=int, default=5000,
                   help="full-res columns used for the own background")
    p.add_argument("--block", type=int, default=4, help="block size (layer px)")
    p.add_argument("--min_x", type=float, help="full-res px")
    p.add_argument("--max_x", type=float)
    p.add_argument("--min_y", type=float, help="stitch window, full-res px")
    p.add_argument("--max_y", type=float)
    p.add_argument("--max_pairs", type=int, help="only the first N pairs of the window")
    p.add_argument("--out_csv")
    args = p.parse_args()

    from linc_convert.modalities.lsm.stitching_utils.corrections import (
        load_corrections,
    )
    from linc_convert.modalities.lsm.stitching_utils.io import (
        crop_and_flip,
        discover_tile_paths,
        open_tile_array,
    )
    from linc_convert.modalities.lsm.stitching_utils.scan_metadata import (
        get_crops,
        get_vertical_flip,
    )

    scale = 2 ** args.layer
    with open(args.yaml_path) as f:
        scan = yaml.safe_load(f)
    crop = get_crops(scan, args.camera_id, args.slice_number)[args.channel]
    flip = get_vertical_flip(scan, args.camera_id)
    with open(args.coords) as f:
        reg = yaml.safe_load(f)["weighted_average"]
    y_offset, dz_full = float(reg["y_offset"]), float(reg.get("dz", 0.0))
    step_full = y_offset
    dz = int(round(dz_full / scale))
    noise, scaler = load_corrections(args.corrections_dir, args.channel)
    paths = discover_tile_paths(args.input_path, args.camera_id)
    bg_len = max(1, args.background_length // scale)

    def open_chunk(i: int):
        return crop_and_flip(open_tile_array(paths[i], args.layer), crop, flip,
                             args.layer)

    sz, sy, sx = open_chunk(0).shape
    if noise.shape != (sz, sy):
        raise SystemExit(f"corrections have shape {noise.shape} but chunks at layer "
                         f"{args.layer} are (Z, Y) = ({sz}, {sy})")
    x0 = 0 if args.min_x is None else max(0, int(round(args.min_x / scale)))
    x1 = sx if args.max_x is None else min(sx, int(round(args.max_x / scale)))
    # the stitch puts chunk i at round(i * y_offset / scale); use its step
    step = int(round(y_offset / scale))
    chunk_full_y = sy * scale
    print(f"{len(paths)} chunks of {args.channel}; chunk (Z,Y,X) = ({sz},{sy},{sx}) at "
          f"layer {args.layer}; y_offset {y_offset:.2f} px full-res -> step {step} rows; "
          f"overlap {sy - step} rows; dz {dz_full:.2f} px -> {dz} rows at this layer; "
          f"x {x0}:{x1}; block {args.block}")

    pairs = []
    for i in range(len(paths) - 1):
        lo, hi = (i + 1) * step_full, i * step_full + chunk_full_y  # overlap, full-res y
        if args.min_y is not None and hi < args.min_y:
            continue
        if args.max_y is not None and lo > args.max_y:
            continue
        pairs.append(i)
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    if not pairs:
        raise SystemExit("no chunk pair lies in the y window")
    rows = pair_rows(step, sy, sz, dz)
    if rows is None:
        raise SystemExit("the chunks do not overlap with this registration")
    a_rows, b_rows, a_z, b_z = rows

    results = []        # (pair, {variant: stats}, bands, shifts, backgrounds)
    own_cache: Dict[int, np.ndarray] = {}

    def own(i: int) -> np.ndarray:
        if i not in own_cache:
            for old in [k for k in own_cache if k < i - 1]:
                del own_cache[old]
            own_cache[i] = own_background(open_chunk(i), bg_len)
        return own_cache[i]

    print("\npair   tissue  variant       ratio   gain  offset  scatter%  corr")
    for i in pairs:
        va = np.asarray(open_chunk(i)[a_z, a_rows, x0:x1].compute(), dtype=np.float32)
        vb = np.asarray(open_chunk(i + 1)[b_z, b_rows, x0:x1].compute(),
                        dtype=np.float32)
        ca = corrected_versions(va, noise[a_z, a_rows], scaler[a_z, a_rows])
        cb = corrected_versions(vb, noise[b_z, b_rows], scaler[b_z, b_rows])
        blk = {v: (block_mean(ca[v], args.block), block_mean(cb[v], args.block))
               for v in VARIANTS}
        mask = tissue_mask(*blk["noise only"])
        if mask.mean() > MAX_TISSUE_SHARE:
            # almost every block passes: the pair lies in background, not tissue
            mask = np.zeros_like(mask)
            print(f"{i + 1:>2}-{i + 2:<2}  (mostly background: skipped)")
        stats = {v: compare(blk[v][0], blk[v][1], mask) for v in VARIANTS}
        nan3 = [float("nan")] * 3
        bands = (band_ratios(*blk["corrected"], mask, 1),
                 band_ratios(*blk["corrected"], mask, 0)) if mask.any() else (nan3, nan3)
        shifts = (shift_search(ca["corrected"], cb["corrected"], 1),
                  shift_search(ca["corrected"], cb["corrected"], 0)) if mask.any() else (
            (0, float("nan"), float("nan")),) * 2
        # background left in the overlap rows after the shared map is subtracted
        left_a = float(np.median(own(i)[a_z, a_rows] - noise[a_z, a_rows]))
        left_b = float(np.median(own(i + 1)[b_z, b_rows] - noise[b_z, b_rows]))
        results.append((i + 1, stats, bands, shifts, (left_a, left_b)))
        for v in VARIANTS:
            s = stats[v]
            if not mask.any():
                break
            print(f"{i + 1:>2}-{i + 2:<2} " + (f"{s['n']:>6}  {v:<11} {s['ratio']:7.3f}"
                  f" {s['gain']:6.3f} {s['offset']:7.1f} {s['scatter']:8.1f} "
                  f"{s['corr']:6.3f}" if s else f"{'':>6}  {v:<11}  (too little tissue)"))

    ok = [r for r in results if r[1]["corrected"] is not None]
    print("\nwhere the corrected mismatch sits (median ratio second/first copy)")
    print("pair    y bands (start of overlap -> end)     z bands (low -> high z)"
          "     best shift y/z  corr at 0 -> best")
    for pair, stats, (yb, zb), (sy_, sz_), _ in results:
        if stats["corrected"] is None:
            continue
        print(f"{pair:>2}-{pair + 1:<2}  " + " ".join(f"{v:6.3f}" for v in yb)
              + "       " + " ".join(f"{v:6.3f}" for v in zb)
              + f"      {sy_[0]:+d}/{sz_[0] * 2:+d} px"
              + f"   {sy_[1]:.3f}->{sy_[2]:.3f} / {sz_[1]:.3f}->{sz_[2]:.3f}")
    print("(z shifts are reported in layer px; the search ran on every 2nd z plane)")

    print("\nbackground left after the shared map (own trailing-column median - map),"
          "\nin the overlap rows of each pair, counts:")
    print("pair    first copy  second copy")
    for pair, stats, _, _, (la, lb) in results:
        print(f"{pair:>2}-{pair + 1:<2}  {la:10.1f}  {lb:11.1f}")

    print("\nsummary over the pairs (corrected vs the other versions)")
    for v in VARIANTS:
        vals = [r[1][v] for r in results if r[1][v] is not None]
        if not vals:
            continue
        print(f"{v:<11} median |ratio-1| {100 * np.median([abs(s['ratio'] - 1) for s in vals]):5.2f}%"
              f"   median |gain-1| {100 * np.median([abs(s['gain'] - 1) for s in vals]):5.2f}%"
              f"   median |offset| {np.median([abs(s['offset']) for s in vals]):6.1f}"
              f"   median scatter {np.median([s['scatter'] for s in vals]):5.1f}%")

    if ok:
        ratios = np.array([r[1]["corrected"]["ratio"] for r in ok])
        print(f"\nratio second/first copy, corrected, over the {len(ok)} pairs with tissue:"
              f" median {np.median(ratios):.3f}, min {ratios.min():.3f}, "
              f"max {ratios.max():.3f}")
        print("Perfect corrections give 1.000 for every pair. A median clearly above or "
              "below 1 that\nis the same for every pair is NOT a drift from chunk to chunk "
              "(it cannot compound over\nthe chunks); it means the first rows of a chunk and "
              "its last rows are at different\nlevels after correction, so every chunk has a "
              "step or ramp of that size at the same\nplace: bands one chunk apart. "
              "Scatter of the ratio around 1 with no sign is a random\nchunk-to-chunk gain.")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair", "variant", "n", "ratio", "gain", "offset", "scatter_pct",
                        "corr", "bg_left_first", "bg_left_second"])
            for pair, stats, _, _, (la, lb) in results:
                for v in VARIANTS:
                    s = stats[v]
                    w.writerow([pair, v] + ([s["n"], s["ratio"], s["gain"], s["offset"],
                                             s["scatter"], s["corr"]] if s else [""] * 6)
                               + [la, lb])
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()