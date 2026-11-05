"""Refine the background offset of a channel with the overlaps between chunks.

``create_corrections`` fits one multiplier per (z, y) row for a given background
offset. The right offset cannot be told from a single chunk, but the overlap can: chunk
``i + 1`` images row ``step + k`` of chunk ``i`` again as its row ``k`` (z shifted by
``dz``), so after a correct offset and multipliers the two copies of the same tissue
have the same percentiles. This script finds that offset and feeds it back:

1. Chunk pairs. Only overlaps with a good amount of tissue are used (see
   ``--min_pair_columns`` and ``--min_pair_rows``).
2. Masking, the procedure of ``create_corrections``: a pixel is masked if its raw value
   is below ``median(background) * zy_threshold_multiplier``; an x column is masked for
   ALL rows of the overlap (both copies) if more than ``--column_mask_fraction`` of
   them are masked there, and kept for ALL rows otherwise. Every row is then measured
   on the same columns.
3. Percentiles. For every (z, row) of both copies the percentiles ``--levels`` of the
   kept columns are taken along x. They are of the RAW values, so they do not depend
   on the correction maps and are measured once.
4. Offset. With the current maps a corrected percentile is
   ``multiplier * (percentile - noise + delta)``. ``delta`` (one number of counts for
   the whole channel) is the value that makes the percentiles of the two copies as
   close as possible. A multiplier is a gain: it moves all levels of a row by the same
   factor (and ``create_corrections`` fits it again for every offset). So each row of
   the two copies, ``pa - noise + delta`` and ``pb - noise + delta`` over the percentile
   levels, is allowed its own gain, and what counts is how far it is from being
   proportional: an offset error adds the same counts to every level, which no gain can
   undo. ``delta`` minimises the squared residual of the best proportional fit over all
   rows and pairs (a one-dimensional search, grid then golden section).
5. The offsets add up (``offset += delta``), ``create_corrections`` is run again for the
   channel with ``fixed_offset=offset`` to get the multipliers for it, and steps 4-5 are
   repeated with the new maps until ``|delta| < --offset_tol`` or ``--max_iterations``
   (5) rounds have been made. A last measurement shows what is left.

The maps of the channel are replaced; the originals are kept once as
``<channel>_alt_zy_scaler_prior.tiff`` / ``..._noise_prior.tiff`` and always used as the
starting point, so running again does not stack. ``--restore`` puts them back. The
history is written to ``<corrections_dir>/refine_offset_<channel>.yaml``.

    python -m linc_convert.modalities.lsm.refine_corrections \\
        --input_path .../sample-slice060 --yaml_path .../LSM_metadata.yaml \\
        --camera_id 2 --slice_number 60 --channel 561 \\
        --coords .../slice060_561.yaml --corrections_dir .../60_corrections_noreg_L2 \\
        --layer 2 --levels 25 37.5 50 62.5 75

The earlier version (smooth multiplier correction from the overlaps) is kept as
``refine_corrections_gain.py``.
"""

import argparse
import math
import os
import shutil
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

CLIP = 0.3          # residuals of the log ratio are limited to +-CLIP (a few bad rows)


def percentiles_of_overlap(va: np.ndarray, vb: np.ndarray, threshold: float,
                           levels: List[float], column_mask_fraction: Optional[float],
                           min_pixels: int = 8
                           ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]:
    """Percentiles along x of every (z, row) of both copies, on shared columns.

    Parameters
    ----------
    va, vb : numpy.ndarray
        RAW values of the two copies of the overlap, shape (z, rows, columns) (columns
        already strided and without the background columns).
    threshold : float
        A pixel is masked if it is below this raw value.
    levels : list of float
        Percentiles (0-100).
    column_mask_fraction : float, optional
        The ``create_corrections`` column rule: a column is kept for all rows only if
        at most this share of the rows of both copies is masked there. None keeps every
        column.
    min_pixels : int
        Kept columns needed.

    Returns
    -------
    (pa, pb, weight, kept, total) or None
        ``pa`` / ``pb``: (levels, z, rows) raw percentiles of the first / second copy,
        NaN where a row's percentile is itself below the threshold; ``weight``
        (z, rows): share of the kept pixels that are tissue in the poorer copy;
        ``kept`` / ``total``: columns kept and columns offered. None if too few
        columns are kept.
    """
    from linc_convert.modalities.lsm.stitching_utils.corrections import keep_columns
    total = va.shape[2]
    if column_mask_fraction is None:
        keep = np.ones(total, dtype=bool)
    else:
        keep = keep_columns(np.concatenate([va, vb], axis=0), threshold,
                            column_mask_fraction)
    kept = int(keep.sum())
    if kept < min_pixels:
        return None
    ka, kb = va[:, :, keep], vb[:, :, keep]
    q = np.asarray(levels, dtype=float) / 100.0
    out = []
    weight = []
    for k in (ka, kb):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            vals = np.quantile(k, q, axis=2).astype(np.float32)
        count = (np.isfinite(k) & (k >= threshold)).sum(axis=2)
        good = (count >= min_pixels) & np.all(vals >= threshold, axis=0)
        out.append(np.where(good[None], vals, np.nan))
        weight.append(np.where(good, count / kept, 0.0))
    return out[0], out[1], np.minimum(weight[0], weight[1]).astype(np.float32), kept, total


def mismatch(delta: float, pa: np.ndarray, pb: np.ndarray, weight: np.ndarray,
             na: np.ndarray, nb: np.ndarray, min_levels: int = 3) -> Tuple[float, int]:
    """How far the two copies are from being the same up to a gain, for an offset.

    ``pa`` / ``pb`` are (pairs, levels, z, rows) raw percentiles of the first and second
    copy, ``weight`` (pairs, z, rows), ``na`` / ``nb`` the (z, rows) noise maps of the
    rows of the two copies. With the offset ``delta`` a row's percentiles are
    ``u = pa - na + delta`` (first copy) and ``v = pb - nb + delta`` (second copy).
    The same tissue seen through a gain ``g`` gives ``v = g * u`` at every level, for
    the right ``delta`` only: an offset error adds the same number of counts to every
    level, which no gain can undo. The multiplier of a row is such a gain (and
    ``create_corrections`` fits it again for every offset), so it is left free: for each
    (pair, z, row) the best ``g`` is found and the squared residual
    ``sum_levels (v - g * u) ** 2`` is what counts. The score is its weighted mean.

    Returns
    -------
    (score, n)
        Weighted mean squared residual (counts squared) and the number of rows used;
        inf if there are none.
    """
    u = pa - na[None, None] + delta
    v = pb - nb[None, None] + delta
    ok = (u > 0) & (v > 0) & np.isfinite(u) & np.isfinite(v)
    ok &= ok.sum(axis=1, keepdims=True) >= min_levels   # a row needs a few levels
    u = np.where(ok, u, 0.0).astype(np.float64)
    v = np.where(ok, v, 0.0).astype(np.float64)
    suu, suv, svv = (u * u).sum(axis=1), (u * v).sum(axis=1), (v * v).sum(axis=1)
    rows = ok.any(axis=1) & (weight > 0)
    if not rows.any():
        return float("inf"), 0
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = np.where(suu > 0, svv - suv * suv / suu, 0.0)
    w = np.where(rows, weight, 0.0)
    return float((w * np.maximum(resid, 0.0)).sum() / w.sum()), int(rows.sum())


def search_offset(evaluate, limit: float, grid_points: int = 21, golden_steps: int = 60,
                  precision: float = 0.005) -> Tuple[float, bool, List[Tuple[float, float]]]:
    """Minimise ``evaluate(delta)`` over ``[-limit, limit]``.

    A grid first, then golden-section steps between the neighbours of the best grid
    point, until the bracket is narrower than ``precision`` counts. Returns
    ``(delta, interior, curve)``: ``interior`` is False if the best
    grid point is at an end of the range, and ``curve`` lists the grid
    ``(delta, value)``.
    """
    grid = np.linspace(-limit, limit, grid_points)
    values = [evaluate(float(g)) for g in grid]
    curve = list(zip(grid.tolist(), values))
    finite = [i for i, v in enumerate(values) if np.isfinite(v)]
    if not finite:
        return 0.0, False, curve
    best = min(finite, key=lambda i: values[i])
    if best in (0, grid_points - 1):
        return float(grid[best]), False, curve
    lo, hi = float(grid[best - 1]), float(grid[best + 1])
    phi = (math.sqrt(5) - 1) / 2
    x1, x2 = hi - phi * (hi - lo), lo + phi * (hi - lo)
    f1, f2 = evaluate(x1), evaluate(x2)
    for _ in range(golden_steps):
        if hi - lo < precision:
            break
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - phi * (hi - lo)
            f1 = evaluate(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + phi * (hi - lo)
            f2 = evaluate(x2)
    return 0.5 * (lo + hi), True, curve


def level_ratios(delta: float, pa, pb, weight, na, nb, sa, sb) -> np.ndarray:
    """Median ratio second / first per percentile level (no pair gain removed)."""
    ua = pa - na[None, None] + delta
    ub = pb - nb[None, None] + delta
    ok = (ua > 0) & (ub > 0) & np.isfinite(ua) & np.isfinite(ub)
    ok &= (weight[:, None] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (sb[None, None] * ub) / (sa[None, None] * ua)
    out = []
    for lv in range(pa.shape[1]):
        sel = ok[:, lv]
        out.append(float(np.median(r[:, lv][sel])) if sel.any() else float("nan"))
    return np.array(out)


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
    p.add_argument("--levels", type=float, nargs="+", default=[25, 37.5, 50, 62.5, 75],
                   help="percentiles (0-100) of the rows' values along x that are made "
                        "equal in the two copies")
    p.add_argument("--column_mask_fraction", type=float, default=0.2,
                   help="an x column is used for every row only if at most this share of "
                        "the rows is masked there (as in create_corrections)")
    p.add_argument("--zy_threshold_multiplier", type=float, default=1.1,
                   help="a pixel is masked below median(background) times this (use the "
                        "value of create_corrections)")
    p.add_argument("--background_length", type=int, default=5000,
                   help="trailing columns (full-res px) the noise map was taken from")
    p.add_argument("--max_columns", type=int, default=512,
                   help="at most this many x columns (evenly strided) per row")
    p.add_argument("--min_x", type=float, help="full-res px, stitching coordinates "
                                               "(optional, the mask makes it unnecessary)")
    p.add_argument("--max_x", type=float)
    p.add_argument("--min_y", type=float,
                   help="only pairs whose overlap lies in this y range (full-res px)")
    p.add_argument("--max_y", type=float)
    p.add_argument("--min_pair_columns", type=float, default=0.3,
                   help="a chunk pair is used only if at least this share of its x "
                        "columns passes the column mask")
    p.add_argument("--min_pair_rows", type=float, default=0.5,
                   help="... and at least this share of its (z, row)s has tissue "
                        "percentiles in both copies")
    p.add_argument("--min_pairs", type=int, default=3, help="pairs needed")
    p.add_argument("--offset_range", type=float, default=0.5,
                   help="search delta within +- this multiple of the typical lowest "
                        "percentile above the background")
    p.add_argument("--offset_limit", type=float,
                   help="search range in counts (replaces --offset_range)")
    p.add_argument("--offset_tol", type=float, default=0.5,
                   help="stop when |delta| of a round is below this many counts (or the "
                        "mismatch no longer improves by 0.1%%)")
    p.add_argument("--max_iterations", type=int, default=5)
    p.add_argument("--n_reference_tiles", type=int, default=16,
                   help="reference tiles create_corrections picks automatically")
    p.add_argument("--reference_tiles", type=int, nargs="+",
                   help="reference tiles of create_corrections (1-based) instead of the "
                        "automatic pick")
    p.add_argument("--create_percentiles", type=float, nargs="+",
                   help="--offset_percentiles of create_corrections (default: --levels)")
    p.add_argument("--dry_run", action="store_true",
                   help="measure and report, change nothing")
    p.add_argument("--restore", action="store_true",
                   help="put the original maps back and stop")
    args = p.parse_args()

    from linc_convert.modalities.lsm.compare_overlaps import pair_rows
    from linc_convert.modalities.lsm.create_corrections import create_corrections
    from linc_convert.modalities.lsm.stitching_utils.corrections import (
        correction_paths,
        load_corrections,
    )
    from linc_convert.modalities.lsm.stitching_utils.io import (
        crop_and_flip,
        discover_tile_paths,
        open_tile_array,
        write_tiff,
        write_yaml,
    )
    from linc_convert.modalities.lsm.stitching_utils.scan_metadata import (
        get_crops,
        get_skew_shear,
        get_vertical_flip,
    )

    paths_ = correction_paths(args.corrections_dir, args.channel)
    scaler_file, noise_file = paths_["scaler"], paths_["noise"]
    scaler_backup = scaler_file.replace("_alt_zy_scaler.tiff", "_alt_zy_scaler_prior.tiff")
    noise_backup = noise_file.replace("_alt_zy_noise.tiff", "_alt_zy_noise_prior.tiff")
    state_file = os.path.join(args.corrections_dir, f"refine_offset_{args.channel}.yaml")
    if args.restore:
        if not os.path.exists(scaler_backup):
            raise SystemExit(f"no original multiplier map at {scaler_backup}")
        shutil.copyfile(scaler_backup, scaler_file)
        shutil.copyfile(noise_backup, noise_file)
        print(f"restored {scaler_file} and {noise_file}")
        return

    # the offset the original maps were made with: remembered from the first run, else
    # taken from the report of create_corrections (0 if there is none)
    start_offset = 0.0
    if os.path.exists(state_file) and os.path.exists(scaler_backup):
        with open(state_file) as f:
            start_offset = float(yaml.safe_load(f)["start_offset"])
    else:
        report = os.path.join(args.corrections_dir,
                              f"offset_fit_camera{args.camera_id}.yaml")
        if os.path.exists(report):
            with open(report) as f:
                rep = yaml.safe_load(f) or {}
            start_offset = float(rep.get("channels", {}).get(args.channel, {})
                                 .get("offset_applied", 0.0))
        if not args.dry_run:
            if not os.path.exists(scaler_backup):
                shutil.copyfile(scaler_file, scaler_backup)
                shutil.copyfile(noise_file, noise_backup)
                print(f"kept the originals at {scaler_backup} and {noise_backup}")
            write_yaml(state_file, {"start_offset": start_offset})
    if os.path.exists(scaler_backup) and not args.dry_run:
        # always start from the originals
        shutil.copyfile(scaler_backup, scaler_file)
        shutil.copyfile(noise_backup, noise_file)
    noise, scaler = load_corrections(args.corrections_dir, args.channel)
    offset = start_offset
    print(f"channel {args.channel}: the original maps were made with offset "
          f"{start_offset:+.2f} counts")

    # ------------------------------------------------------------------ geometry
    scale = 2 ** args.layer
    with open(args.yaml_path) as f:
        scan = yaml.safe_load(f)
    crop = get_crops(scan, args.camera_id, args.slice_number)[args.channel]
    flip = get_vertical_flip(scan, args.camera_id)
    with open(args.coords) as f:
        reg = yaml.safe_load(f)["weighted_average"]
    y_offset, dz_full = float(reg["y_offset"]), float(reg.get("dz", 0.0))
    dz = int(round(dz_full / scale))
    step = int(round(y_offset / scale))
    chunks = discover_tile_paths(args.input_path, args.camera_id)

    def open_chunk(i: int):
        return crop_and_flip(open_tile_array(chunks[i], args.layer), crop, flip,
                             args.layer)

    sz, sy, sx = open_chunk(0).shape
    if noise.shape != (sz, sy) or scaler.shape != (sz, sy):
        raise SystemExit(f"corrections {noise.shape} do not fit chunks (Z, Y) = "
                         f"({sz}, {sy}) at layer {args.layer}")
    rows = pair_rows(step, sy, sz, dz)
    if rows is None:
        raise SystemExit("the chunks do not overlap with this registration")
    a_rows, b_rows, a_z, b_z = rows
    bg_len = max(1, args.background_length // scale)
    shift = int(math.ceil(get_skew_shear(scan) * (sz - 1)))
    x0 = 0 if args.min_x is None else max(0, int(round(args.min_x / scale)))
    x1 = sx - bg_len if args.max_x is None else int(round(args.max_x / scale)) - shift
    x1 = min(sx - bg_len, x1)
    if x1 <= x0:
        raise SystemExit("the x window leaves no columns")
    stride = max(1, int(math.ceil((x1 - x0) / args.max_columns)))
    # the background level the threshold of the column mask refers to: the noise map
    # before any offset was taken off
    threshold = float(np.median(noise + offset)) * args.zy_threshold_multiplier
    print(f"{len(chunks)} chunks, chunk (Z,Y,X)=({sz},{sy},{sx}) at layer {args.layer}, "
          f"step {step}, overlap {sy - step} rows, dz {dz}; x {x0}:{x1} every {stride}; "
          f"tissue threshold {threshold:.1f} (raw)")

    selected = []
    for i in range(len(chunks) - 1):
        lo, hi = (i + 1) * y_offset, i * y_offset + sy * scale
        if args.min_y is not None and hi < args.min_y:
            continue
        if args.max_y is not None and lo > args.max_y:
            continue
        selected.append(i)
    if not selected:
        raise SystemExit("no chunk pair in the y range")

    # ------------------------------------------------------ measure (once)
    pas, pbs, wts, used = [], [], [], []
    for n, i in enumerate(selected, 1):
        va = np.asarray(open_chunk(i)[a_z, a_rows, x0:x1:stride].compute(),
                        dtype=np.float32)
        vb = np.asarray(open_chunk(i + 1)[b_z, b_rows, x0:x1:stride].compute(),
                        dtype=np.float32)
        res = percentiles_of_overlap(va, vb, threshold, args.levels,
                                     args.column_mask_fraction)
        why = ""
        if res is None:
            why = "  -> too few columns pass the mask, skipped"
            kept_share, row_share = 0.0, 0.0
        else:
            pa, pb, wt, kept, total = res
            kept_share, row_share = kept / total, float((wt > 0).mean())
            if kept_share < args.min_pair_columns:
                why = f"  -> below --min_pair_columns {args.min_pair_columns:g}, skipped"
            elif row_share < args.min_pair_rows:
                why = f"  -> below --min_pair_rows {args.min_pair_rows:g}, skipped"
        print(f"  pair {i + 1}-{i + 2} ({n}/{len(selected)}): "
              f"{100 * kept_share:4.1f}% of columns kept, "
              f"{100 * row_share:4.1f}% of rows with tissue{why}", flush=True)
        if not why:
            pas.append(pa)
            pbs.append(pb)
            wts.append(wt)
            used.append(i)
    if len(used) < max(args.min_pairs, 2):
        raise SystemExit(f"only {len(used)} pairs have enough tissue; need at least "
                         f"{max(args.min_pairs, 2)} (see --min_pair_columns, "
                         "--min_pair_rows, --column_mask_fraction)")
    pa, pb, wt = np.stack(pas), np.stack(pbs), np.stack(wts)
    print(f"\n{len(used)} of {len(selected)} pairs used")
    del pas, pbs, wts

    def current_rows():
        return (noise[a_z, a_rows], noise[b_z, b_rows],
                scaler[a_z, a_rows], scaler[b_z, b_rows])

    def measure_round(label: str) -> Tuple[float, bool, float, float, float]:
        na, nb, sa, sb = current_rows()
        if args.offset_limit is not None:
            limit = float(args.offset_limit)
        else:
            low = (pa[:, 0] - na[None])[np.isfinite(pa[:, 0])]
            level = float(np.median(low[low > 0])) if (low > 0).any() else 0.0
            limit = max(args.offset_range * level, 1.0)
        ev = lambda x: mismatch(x, pa, pb, wt, na, nb)[0]  # noqa: E731
        before = ev(0.0)
        delta, interior, curve = search_offset(ev, limit)
        after = ev(delta)
        print(f"\n{label}: search range +-{limit:.1f} counts; mismatch "
              f"{before:.5f} at delta 0 -> {after:.5f} at delta {delta:+.2f}"
              + ("" if interior else "  (at the EDGE of the range)"))
        r0 = level_ratios(0.0, pa, pb, wt, na, nb, sa, sb)
        r1 = level_ratios(delta, pa, pb, wt, na, nb, sa, sb)
        print("  median ratio second/first at percentiles "
              f"{[float(v) for v in args.levels]}")
        print("    delta 0:     " + " ".join(f"{v:.3f}" for v in r0))
        print(f"    delta {delta:+.2f}: " + " ".join(f"{v:.3f}" for v in r1))
        return delta, interior, before, after, limit

    history: List[Dict] = []
    stable = False
    for it in range(1, args.max_iterations + 1):
        delta, interior, before, after, limit = measure_round(
            f"round {it}: offset so far {offset:+.2f}")
        history.append({"round": it, "offset_before": float(offset),
                        "delta": float(delta), "interior": bool(interior),
                        "mismatch_at_zero": float(before),
                        "mismatch_at_delta": float(after), "search_limit": float(limit)})
        if abs(delta) < args.offset_tol or before - after <= 1e-3 * before:
            why = (f"|delta| {abs(delta):.2f} < {args.offset_tol:g}"
                   if abs(delta) < args.offset_tol else
                   "the mismatch does not improve by 0.1% or more")
            print(f"  {why}: the offset is stable")
            stable = True
            break
        if args.dry_run:
            print("  dry run: no new maps")
            break
        offset += delta
        print(f"  offset becomes {offset:+.2f}; running create_corrections again")
        history[-1]["offset_after"] = float(offset)
        create_corrections(
            input_path=args.input_path, yaml_path=args.yaml_path,
            camera_id=args.camera_id, slice_number=args.slice_number,
            reference_tiles=args.reference_tiles,
            auto_reference_tiles=args.reference_tiles is None,
            n_reference_tiles=args.n_reference_tiles,
            output_dir=args.corrections_dir, background_length=args.background_length,
            zy_threshold_multiplier=args.zy_threshold_multiplier,
            only_channel=args.channel, force=True, layer=args.layer, fit_offset=True,
            fixed_offset=float(offset),
            offset_percentiles=args.create_percentiles or args.levels,
            column_mask=True, column_mask_fraction=args.column_mask_fraction)
        noise, scaler = load_corrections(args.corrections_dir, args.channel)
    else:
        if not args.dry_run:
            delta, interior, before, after, limit = measure_round(
                f"after round {args.max_iterations}: offset {offset:+.2f}")
            history.append({"round": "final", "offset_before": float(offset),
                            "delta": float(delta), "interior": bool(interior),
                            "mismatch_at_zero": float(before),
                            "mismatch_at_delta": float(after)})
            stable = abs(delta) < args.offset_tol
            print("  (not applied)")

    print(f"\nfinal offset {offset:+.2f} counts (started at {start_offset:+.2f}); "
          + ("stable" if stable else f"not stable after {args.max_iterations} rounds"))
    if not args.dry_run:
        write_yaml(state_file, {"start_offset": start_offset,
                                "final_offset": float(offset), "stable": bool(stable),
                                "pairs_used": [int(i) + 1 for i in used],
                                "levels": [float(v) for v in args.levels],
                                "column_mask_fraction": args.column_mask_fraction,
                                        "rounds": history})
        print(f"wrote {scaler_file} and {noise_file}; history in {state_file}")


if __name__ == "__main__":
    main()