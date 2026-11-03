"""Measure the brightness steps at the chunk seams of a stitched channel.

Reads the stitched output (OME-Zarr, array ``0``) and, at every seam between two
chunks, compares the rows just before the blended overlap with the rows just after
it. Tissue values on both sides are matched by quantiles and a line

    after = gain * before + offset

is fitted through the matched quantiles. A *gain* different from 1 means the two
chunks differ in brightness by a factor; an *offset* different from 0 means they
differ by a constant (for example a different background level); both small means
the step is not a chunk effect. The same measurement at positions half a chunk
away from every seam (``control``) shows how much change ordinary anatomy gives, so
a real seam effect is the amount by which the seams exceed the controls.

If the raw chunks are given (``--raw_input`` and ``--corrections_dir``), the script
also prints, per chunk, how far its own background (median of its trailing
columns) is from the background map the corrections subtracted; a chunk-to-chunk
change there is the usual cause of an offset step.

    python -m linc_convert.modalities.lsm.diagnose_seams \\
        --stitched .../corrected/60_noreg_L2/561 --coords .../slice060_561.yaml \\
        --layer 2 --overlap 150 --min_y 13000

Positions: chunk ``i`` starts at row ``round(i * y_offset / 2**layer) - y0`` of
the stitched array, where ``y0 = round(min_y / 2**layer)`` and ``y_offset`` is
the registration file's ``weighted_average.y_offset``; the blended overlap is the
first ``round(overlap / 2**layer)`` rows of each chunk.
"""

import argparse
import csv
import os
from typing import List, Optional, Tuple

import numpy as np
import yaml

LEVELS = np.arange(10, 91, 10)       # percentiles matched across a seam
MAX_VALUES = 3_000_000               # values read per side


def open_stitched(path: str):
    """The full-resolution array of a stitched channel."""
    import zarr
    root = zarr.open(path, mode="r")
    return root["0"] if hasattr(root, "keys") and "0" in list(root.keys()) else root


def read_rows(array, row0: int, row1: int) -> np.ndarray:
    """Rows ``[row0, row1)`` of the array, thinned in z and x to ``MAX_VALUES``."""
    n_z, n_y, n_x = array.shape
    row0, row1 = max(row0, 0), min(row1, n_y)
    if row1 <= row0:
        return np.empty(0, dtype=np.float32)
    thin = max(1, int(np.ceil(np.sqrt(n_z * (row1 - row0) * n_x / MAX_VALUES))))
    block = array[::thin, row0:row1, ::thin]
    return np.asarray(block, dtype=np.float32).ravel()


def match(before: np.ndarray, after: np.ndarray, floor: float
          ) -> Optional[Tuple[float, float, float, int, float]]:
    """Fit ``after = gain * before + offset`` on tissue quantiles.

    Returns ``(gain, offset, level_ratio, n, model_ratio)`` or None if either
    side has fewer than 1000 tissue values. Tissue means above ``floor``;
    ``model_ratio`` is the squared error of the best pure gain (``after = gain *
    before``) over that of the best pure offset (``after = before + offset``):
    below 1 the step looks like a brightness factor, above 1 like a constant.
    """
    b, a = before[before > floor], after[after > floor]
    if b.size < 1000 or a.size < 1000:
        return None
    qb, qa = np.percentile(b, LEVELS), np.percentile(a, LEVELS)
    gain, offset = np.polyfit(qb, qa, 1)
    pure_gain = float((qa * qb).sum() / (qb * qb).sum())
    rss_gain = float(((qa - pure_gain * qb) ** 2).sum())
    rss_offset = float(((qa - qb - (qa - qb).mean()) ** 2).sum())
    return (float(gain), float(offset), float(np.median(a) / np.median(b)),
            int(min(b.size, a.size)), rss_gain / max(rss_offset, 1e-12))


def seam_positions(n_rows: int, y_offset: float, layer: int, min_y: float
                   ) -> List[int]:
    """Start row (in the stitched array) of every chunk after the first."""
    scale = 2 ** layer
    y0 = int(round(min_y / scale))
    out, i = [], 1
    while True:
        row = int(round(i * y_offset / scale)) - y0
        if row >= n_rows:
            return out
        if row > 0:
            out.append(row)
        i += 1


def measure(array, seams: List[int], overlap: int, window: int, step: int
            ) -> Tuple[list, list, float]:
    """Seam and control fits; also the tissue floor used."""
    n_z, n_y, n_x = array.shape
    sample = read_rows(array, 0, n_y)
    nonzero = sample[sample > 0]
    floor = 0.25 * float(np.median(nonzero)) if nonzero.size else 0.0
    seam_rows, control_rows = [], []
    for s in seams:
        before = read_rows(array, s - window, s)
        after = read_rows(array, s + overlap, s + overlap + window)
        seam_rows.append((s, match(before, after, floor)))
        c = s + step // 2
        if c + overlap + window <= n_y and c - window >= 0:
            control_rows.append((c, match(read_rows(array, c - window, c),
                                          read_rows(array, c + overlap,
                                                    c + overlap + window), floor)))
    return seam_rows, control_rows, floor


def summarise(rows: list) -> Tuple[float, float, float, float]:
    """Median |gain - 1| (%), |offset|, |level ratio - 1| (%), share gain-like."""
    ok = [r for _, r in rows if r is not None]
    if not ok:
        return (float("nan"),) * 4
    return (100 * float(np.median([abs(r[0] - 1) for r in ok])),
            float(np.median([abs(r[1]) for r in ok])),
            100 * float(np.median([abs(r[2] - 1) for r in ok])),
            float(np.mean([r[4] < 1 for r in ok])))


def chunk_backgrounds(args) -> List[Tuple[int, float, float]]:
    """(chunk, own background median, shared noise map median) of every chunk."""
    from linc_convert.modalities.lsm.stitching_utils.corrections import (
        compute_noise_map,
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
    with open(args.yaml_path) as f:
        scan = yaml.safe_load(f)
    crop = get_crops(scan, args.camera_id, args.slice_number)[args.channel]
    flip = get_vertical_flip(scan, args.camera_id)
    noise, _ = load_corrections(args.corrections_dir, args.channel)
    shared = float(np.median(noise))
    bg_len = max(1, args.background_length // 2 ** args.layer)
    out = []
    for i, path in enumerate(discover_tile_paths(args.raw_input, args.camera_id), 1):
        tile = open_tile_array(path, args.layer, chunks=(128, 128, 128))
        own = compute_noise_map(crop_and_flip(tile, crop, flip, args.layer), bg_len)
        out.append((i, float(np.median(own)), shared))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stitched", required=True, help="stitched channel folder")
    p.add_argument("--coords", required=True, help="registration YAML (y_offset)")
    p.add_argument("--layer", type=int, default=2)
    p.add_argument("--overlap", type=float, default=150, help="full-res px")
    p.add_argument("--min_y", type=float, default=0, help="stitch window min_y, full-res px")
    p.add_argument("--window", type=int, default=30, help="rows compared on each side")
    p.add_argument("--out_csv")
    p.add_argument("--raw_input", help="slice folder with the raw chunks")
    p.add_argument("--corrections_dir")
    p.add_argument("--yaml_path")
    p.add_argument("--camera_id", type=int, default=2)
    p.add_argument("--slice_number", type=int)
    p.add_argument("--channel", default="561")
    p.add_argument("--background_length", type=int, default=5000)
    args = p.parse_args()

    with open(args.coords) as f:
        y_offset = float(yaml.safe_load(f)["weighted_average"]["y_offset"])
    scale = 2 ** args.layer
    overlap = max(1, int(round(args.overlap / scale)))
    step = max(2, int(round(y_offset / scale)))
    array = open_stitched(args.stitched)
    print(f"stitched array {array.shape}; chunk step {step} rows, blended overlap "
          f"{overlap} rows, comparing {args.window} rows on each side")
    seams = seam_positions(array.shape[1], y_offset, args.layer, args.min_y)
    seam_rows, control_rows, floor = measure(array, seams, overlap, args.window, step)
    print(f"tissue = values above {floor:.1f}\n")
    print("seam  row    gain    offset  level ratio  looks like")
    for s, r in seam_rows:
        print(f"{seams.index(s) + 1:>4} {s:>5}  " + (
            f"{r[0]:6.3f} {r[1]:9.1f} {r[2]:10.3f}  "
            f"{'gain' if r[4] < 1 else 'offset'}" if r else "  (too little tissue)"))
    sg, so, sr, sf = summarise(seam_rows)
    cg, co, cr, cf = summarise(control_rows)
    print(f"\nmedian |gain-1|   seams {sg:5.1f}%   controls {cg:5.1f}%")
    print(f"median |offset|   seams {so:7.1f}    controls {co:7.1f}   (counts)")
    print(f"median |level ratio-1|  seams {sr:5.1f}%  controls {cr:5.1f}%")
    print(f"share of steps that look like a brightness factor (not a constant): "
          f"seams {sf:.0%}  controls {cf:.0%}")
    print("A chunk effect is the amount by which the seams exceed the controls.")

    if args.raw_input and args.corrections_dir and args.yaml_path:
        print("\nchunk  own background  shared map  difference")
        rows = chunk_backgrounds(args)
        for i, own, shared in rows:
            print(f"{i:>5} {own:14.1f} {shared:11.1f} {own - shared:+11.1f}")
        diffs = np.array([o - s for _, o, s in rows])
        print(f"spread of the difference between chunks: {diffs.std():.1f} counts")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kind", "row", "gain", "offset", "level_ratio", "n", "model_ratio"])
            for kind, rows in (("seam", seam_rows), ("control", control_rows)):
                for row, r in rows:
                    w.writerow([kind, row] + (list(r) if r else [""] * 5))
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()