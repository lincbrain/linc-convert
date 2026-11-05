"""Second refinement: smooth multiplier curves from the overlaps, plus a multiplier-noise map.

Run after ``refine_corrections`` (or ``create_corrections``). It does NOT use the
multiplier map that is there. It uses the full background subtraction of the channel:
the (z, y) background map with the offset already taken out (``background - offset``,
the offset found by ``refine_corrections``), so every row is measured as
``raw - background + offset`` exactly as the stitching will correct it.

1. Overlaps. The chunk pairs, the stitching crop, the registration (``--coords``), the
   column mask and the percentiles are the ones of ``refine_corrections``: the two copies
   of every overlap row give raw percentiles ``--levels`` along x on the same columns.
2. Curves. With ``u`` and ``v`` the offset-corrected percentiles (``raw - noise``) of the
   first and second copy of a row, a multiplier map ``M(z, y)`` makes the copies equal
   when ``M(za, ya) * u = M(zb, yb) * v``. The map is searched as a product of a curve
   along y and a curve along z, ``M = f(y) * g(z)``, so in logs
   ``log f(ya) - log f(yb) + log g(za) - log g(zb) + p = log v - log u`` for every row,
   level-median over ``--levels``. ``p`` is one free constant per chunk pair (chunks
   differ a little in brightness, which no map can or should fix). ``log f`` and
   ``log g`` are cubic B-splines with ``--y_knots`` / ``--z_knots`` intervals and a
   smoothness penalty ``--smooth``; the fit is a robust (Huber) least squares.
   Parts of y that are in no overlap follow from the smoothness only.
3. Multiplier noise. The multiplier map that ``create_corrections`` made at the current
   offset (the existing ``<ch>_alt_zy_scaler.tiff``) is divided by the median of its
   surroundings (``--noise_window`` pixels in z and y). A pixel of 1 means no
   fine structure; the rest is multiplier noise.

With ``--base_scaler`` the existing map ``S`` is kept and only a small smooth residual is
fitted on top of it (``M = S * exp(a(y) + b(z))``, few knots, strong smoothness). The
residual is applied only if it lowers the disagreement of overlap pairs it was not
fitted on (the pairs are split in two halves, each scored by the fit of the other) by
``--min_gain``; otherwise the files reproduce ``S`` exactly. ``multiplier_correction``
is then the median-filtered ``S`` times the residual and ``multiplier_noise`` the rest
of ``S``, so their product is ``S * residual``.

Three files are written to ``<corrections_dir>/<channel>/`` (nothing existing is changed):

* ``<ch>_background_subtraction.tiff``  -- background minus offset (the noise map)
* ``<ch>_multiplier_correction.tiff``   -- ``f(y) * g(z)``, scaled to the median of the
                                           current multiplier map
* ``<ch>_multiplier_noise.tiff``        -- multiplier map / median of its surroundings
                                           (1e-9 where the row has no valid multiplier)

``stitch_chunks --refined_corrections`` applies
``(raw - background_subtraction).clip(0) * multiplier_correction * multiplier_noise``.
The curves and the comparison are written to ``<corrections_dir>/refine_multipliers_<ch>.yaml``.

    python -m linc_convert.modalities.lsm.refine_multipliers \\
        --input_path .../sample-slice060 --yaml_path .../LSM_metadata.yaml \\
        --camera_id 2 --slice_number 60 --channel 561 \\
        --coords .../slice060_561.yaml --corrections_dir .../60_corrections_noreg_L2 --layer 2
"""

import argparse
import math
import os
from typing import List, Optional, Tuple

import numpy as np
import yaml
from scipy import ndimage
from scipy.interpolate import BSpline

from linc_convert.modalities.lsm.refine_corrections import percentiles_of_overlap

MIN_SCALER = 1e-3


def spline_basis(n: int, intervals: int) -> np.ndarray:
    """Cubic B-spline basis on ``n`` points: (n, intervals + 3)."""
    intervals = max(1, min(intervals, max(1, n - 1)))
    inner = np.linspace(0, n - 1, intervals + 1)
    t = np.concatenate([[0.0] * 3, inner, [n - 1.0] * 3])
    return BSpline.design_matrix(np.arange(n, dtype=float), t, 3).toarray()


def second_differences(m: int) -> np.ndarray:
    """(m - 2, m) second-difference matrix."""
    d = np.zeros((max(m - 2, 0), m))
    for i in range(max(m - 2, 0)):
        d[i, i:i + 3] = (1.0, -2.0, 1.0)
    return d


def row_log_ratio(pa: np.ndarray, pb: np.ndarray, na: np.ndarray, nb: np.ndarray,
                  weight: np.ndarray, min_levels: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Level-median of ``log(v / u)`` per (z, row) of one pair and its weights.

    ``pa`` / ``pb`` are (levels, z, rows) raw percentiles; ``na`` / ``nb`` the (z, rows)
    background maps (offset already taken out).
    """
    u = pa - na[None]
    v = pb - nb[None]
    ok = (u > 0) & (v > 0) & np.isfinite(u) & np.isfinite(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.where(ok, np.log(v) - np.log(u), np.nan)
    good = ok.sum(axis=0) >= min_levels
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        d = np.nanmedian(lr, axis=0)
    w = np.where(good & np.isfinite(d) & (weight > 0), weight, 0.0)
    return np.where(w > 0, d, 0.0), w


def fit_curves(data: List[Tuple[np.ndarray, np.ndarray]], step: int, sy: int, sz: int,
               a_z0: int, b_z0: int, y_knots: int, z_knots: int, smooth: float,
               pair_ridge: float = 1e-2, huber_rounds: int = 4
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit ``log f(y)`` and ``log g(z)`` (and a constant per pair) to the row ratios.

    ``data`` holds ``(d, w)`` per pair, both (z, rows), from ``row_log_ratio``. Row ``r``
    of the first copy is row ``r - step`` of the second, z index ``i`` of the first is
    ``i - dz`` of the second (``a_z0`` / ``b_z0`` are the first z index of each).
    Returns ``(log f, log g, pair constants, robust residual scale)``.
    """
    by, bz = spline_basis(sy, y_knots), spline_basis(sz, z_knots)
    ny, nz, npair = by.shape[1], bz.shape[1], len(data)
    n_par = ny + nz + npair
    zo, ro = data[0][0].shape
    ya, yb = np.arange(ro) + step, np.arange(ro)
    za, zb = np.arange(zo) + a_z0, np.arange(zo) + b_z0
    dy = by[ya] - by[yb]            # (rows, ny)
    dz_ = bz[za] - bz[zb]           # (z, nz)
    total_w = float(sum(w.sum() for _, w in data))
    if total_w <= 0:
        raise SystemExit("no usable overlap rows")
    # penalties (scaled with the total weight so --smooth does not depend on the amount of data)
    pen = np.zeros((n_par, n_par))
    d2y, d2z = second_differences(ny), second_differences(nz)
    pen[:ny, :ny] += smooth * total_w * d2y.T @ d2y
    pen[ny:ny + nz, ny:ny + nz] += smooth * total_w * d2z.T @ d2z
    pen[ny + nz:, ny + nz:] += pair_ridge * total_w / npair * np.eye(npair)
    tiny = 1e-9 * total_w            # constants of log f / log g cancel; fixed afterwards
    pen[:ny + nz, :ny + nz] += tiny * np.eye(ny + nz)

    robust = [np.ones_like(w) for _, w in data]
    coef = np.zeros(n_par)
    scale = 1.0
    for round_ in range(huber_rounds):
        ata = pen.copy()
        atb = np.zeros(n_par)
        for k, (d, w) in enumerate(data):
            ww = w * robust[k]                       # (z, rows)
            if ww.sum() <= 0:
                continue
            wr, wz = ww.sum(axis=0), ww.sum(axis=1)  # per row, per z
            wd = ww * d
            k_ = ny + nz + k
            ata[:ny, :ny] += dy.T @ (wr[:, None] * dy)
            ata[ny:ny + nz, ny:ny + nz] += dz_.T @ (wz[:, None] * dz_)
            cross = dy.T @ ww.T @ dz_                # (ny, nz)
            ata[:ny, ny:ny + nz] += cross
            ata[ny:ny + nz, :ny] += cross.T
            py, pz = dy.T @ wr, dz_.T @ wz
            ata[:ny, k_] += py
            ata[k_, :ny] += py
            ata[ny:ny + nz, k_] += pz
            ata[k_, ny:ny + nz] += pz
            ata[k_, k_] += ww.sum()
            atb[:ny] += dy.T @ wd.sum(axis=0)
            atb[ny:ny + nz] += dz_.T @ wd.sum(axis=1)
            atb[k_] += wd.sum()
        coef = np.linalg.solve(ata, atb)
        res_all, w_all = [], []
        for k, (d, w) in enumerate(data):
            fit = (dy @ coef[:ny])[None, :] + (dz_ @ coef[ny:ny + nz])[:, None] \
                + coef[ny + nz + k]
            r = d - fit
            robust[k] = np.ones_like(r)
            res_all.append(r[w > 0])
            w_all.append(w[w > 0])
        r_cat = np.concatenate(res_all)
        scale = 1.4826 * float(np.median(np.abs(r_cat - np.median(r_cat)))) + 1e-6
        cut = 1.345 * scale
        for k, (d, w) in enumerate(data):
            fit = (dy @ coef[:ny])[None, :] + (dz_ @ coef[ny:ny + nz])[:, None] \
                + coef[ny + nz + k]
            r = np.abs(d - fit)
            robust[k] = np.where(r > cut, cut / np.maximum(r, 1e-12), 1.0)
    log_f = by @ coef[:ny]
    log_g = bz @ coef[ny:ny + nz]
    p = coef[ny + nz:].copy()
    log_f -= log_f.mean()
    log_g -= log_g.mean()
    return log_f, log_g, p, scale


def noise_map(scaler: np.ndarray, window: Tuple[int, int]) -> np.ndarray:
    """``scaler`` divided by the median of its surroundings; 1e-9 where invalid."""
    valid = scaler > MIN_SCALER
    if not valid.any():
        raise SystemExit("the multiplier map has no valid rows")
    idx = ndimage.distance_transform_edt(~valid, return_distances=False,
                                         return_indices=True)
    filled = scaler[tuple(idx)]
    med = ndimage.median_filter(filled, size=window, mode="nearest")
    ratio = np.where(valid, scaler / np.maximum(med, MIN_SCALER), 1e-9)
    return ratio.astype(np.float32)


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
    p.add_argument("--levels", type=float, nargs="+",
                   default=[10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95],
                   help="percentiles (0-100) whose median ratio gives each row's value")
    p.add_argument("--column_mask_fraction", type=float, default=0.2)
    p.add_argument("--zy_threshold_multiplier", type=float, default=1.1)
    p.add_argument("--background_length", type=int, default=5000)
    p.add_argument("--max_columns", type=int, default=512)
    p.add_argument("--min_x", type=float)
    p.add_argument("--max_x", type=float)
    p.add_argument("--min_y", type=float)
    p.add_argument("--max_y", type=float)
    p.add_argument("--min_pair_columns", type=float, default=0.3)
    p.add_argument("--min_pair_rows", type=float, default=0.5)
    p.add_argument("--min_pairs", type=int, default=3)
    p.add_argument("--y_knots", type=int, default=12,
                   help="spline intervals of the curve along y")
    p.add_argument("--z_knots", type=int, default=8,
                   help="spline intervals of the curve along z")
    p.add_argument("--smooth", type=float, default=1e-3,
                   help="penalty on the curvature of the curves (relative to the data)")
    p.add_argument("--noise_window", type=int, nargs=2, default=[15, 15],
                   metavar=("Z", "Y"),
                   help="size (layer pixels) of the median filter of the multiplier noise")
    p.add_argument("--base_scaler", action="store_true",
                   help="keep the existing multiplier map S and fit only a small smooth "
                        "residual on top of it, M = S * exp(a(y) + b(z)); the residual is "
                        "used only if it improves pairs it was not fitted on")
    p.add_argument("--base_y_knots", type=int, default=4,
                   help="--base_scaler: spline intervals of the residual along y")
    p.add_argument("--base_z_knots", type=int, default=3,
                   help="--base_scaler: spline intervals of the residual along z")
    p.add_argument("--base_smooth", type=float, default=1e-2,
                   help="--base_scaler: curvature penalty of the residual")
    p.add_argument("--min_gain", type=float, default=0.05,
                   help="--base_scaler: the residual is applied only if the rms "
                        "disagreement of the held-out pairs falls by at least this share in BOTH halves")
    args = p.parse_args()

    from linc_convert.modalities.lsm.compare_overlaps import pair_rows
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

    noise, scaler = load_corrections(args.corrections_dir, args.channel)
    state = os.path.join(args.corrections_dir, f"refine_offset_{args.channel}.yaml")
    offset = None
    if os.path.exists(state):
        with open(state) as f:
            offset = yaml.safe_load(f).get("final_offset")
    print(f"channel {args.channel}: background subtraction map (background - offset, "
          f"median {float(np.median(noise)):.1f})"
          + ("" if offset is None else f"; offset {float(offset):+.2f} counts is in it"))

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
        return crop_and_flip(open_tile_array(chunks[i], args.layer), crop, flip, args.layer)

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
    # threshold of the column mask refers to the background before the offset was taken off
    threshold = float(np.median(noise + (0.0 if offset is None else float(offset)))) \
        * args.zy_threshold_multiplier
    print(f"{len(chunks)} chunks, chunk (Z,Y,X)=({sz},{sy},{sx}) at layer {args.layer}, "
          f"step {step}, overlap {sy - step} rows, dz {dz}; x {x0}:{x1} every {stride}")

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

    na, nb = noise[a_z, a_rows], noise[b_z, b_rows]
    data, used = [], []
    pairs_raw = []
    for n, i in enumerate(selected, 1):
        va = np.asarray(open_chunk(i)[a_z, a_rows, x0:x1:stride].compute(), dtype=np.float32)
        vb = np.asarray(open_chunk(i + 1)[b_z, b_rows, x0:x1:stride].compute(),
                        dtype=np.float32)
        res = percentiles_of_overlap(va, vb, threshold, args.levels, args.column_mask_fraction)
        why = ""
        if res is None:
            why = "  -> too few columns pass the mask, skipped"
            kept_share = row_share = 0.0
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
            d, w = row_log_ratio(pa, pb, na, nb, wt)
            data.append((d, w))
            pairs_raw.append((pa, pb, wt))
            used.append(i)
    if len(used) < max(args.min_pairs, 2):
        raise SystemExit(f"only {len(used)} pairs have enough tissue; need at least "
                         f"{max(args.min_pairs, 2)}")
    print(f"\n{len(used)} of {len(selected)} pairs used")

    valid = scaler > MIN_SCALER
    mult_noise = noise_map(scaler, tuple(args.noise_window))
    held_out = None
    if args.base_scaler:
        # residual on top of the existing map: fit what is left after applying it
        ls = np.log(np.maximum(scaler, 1e-12))
        ok_rows = valid[a_z, a_rows] & valid[b_z, b_rows]
        shift = np.where(ok_rows, ls[a_z, a_rows] - ls[b_z, b_rows], 0.0)
        data_b = [(np.where(ok_rows, d - shift, 0.0), np.where(ok_rows, w, 0.0))
                  for d, w in data]

        def fit_on(idx):
            return fit_curves([data_b[i] for i in idx], step, sy, sz, a_z.start, b_z.start,
                              args.base_y_knots, args.base_z_knots, args.base_smooth)

        def rms_of(idx, lf, lg):
            """Rms disagreement of pairs ``idx`` after the residual curves (free pair constant)."""
            ya, yb = np.arange(a_rows.start, a_rows.stop), np.arange(b_rows.start, b_rows.stop)
            za, zb = np.arange(a_z.start, a_z.stop), np.arange(b_z.start, b_z.stop)
            model = (lf[ya] - lf[yb])[None, :] + (lg[za] - lg[zb])[:, None]
            sq = ww = 0.0
            for i in idx:
                d, w = data_b[i]
                if w.sum() <= 0:
                    continue
                r = d - model
                r = r - float((w * r).sum() / w.sum())
                sq += float((w * r * r).sum())
                ww += float(w.sum())
            return sq / max(ww, 1e-12), ww

        folds = [list(range(0, len(data_b), 2)), list(range(1, len(data_b), 2))]
        sq0 = sq1 = wsum = 0.0
        fold_gain = []
        for k in range(2):
            train, test = folds[1 - k], folds[k]
            lf_k, lg_k, _, _ = fit_on(train)
            m1, w1 = rms_of(test, lf_k, lg_k)
            m0, _ = rms_of(test, np.zeros(sy), np.zeros(sz))
            sq0 += m0 * w1
            sq1 += m1 * w1
            wsum += w1
            fold_gain.append(1.0 - math.sqrt(m1 / max(m0, 1e-30)))
        rms0, rms1 = math.sqrt(sq0 / wsum), math.sqrt(sq1 / wsum)
        apply = all(g >= args.min_gain for g in fold_gain)   # both halves must improve
        held_out = {"rms_existing_map": rms0, "rms_with_residual": rms1,
                    "fold_gain": [float(g) for g in fold_gain],
                    "min_gain": args.min_gain, "applied": bool(apply)}
        print(f"\nheld-out check (fit on every other pair, scored on the others): rms "
              f"{rms0:.4f} with the existing map -> {rms1:.4f} with the residual curves: "
              + f"(per half: {100 * fold_gain[0]:+.1f}%, {100 * fold_gain[1]:+.1f}%) "
              + ("residual APPLIED" if apply else
                 f"both halves need at least {100 * args.min_gain:.0f}% improvement, "
                 "residual NOT applied (existing map kept)"))
        if apply:
            log_f, log_g, pair_const, rscale = fit_on(list(range(len(data_b))))
        else:
            log_f, log_g = np.zeros(sy), np.zeros(sz)
            pair_const, rscale = np.zeros(len(data_b)), float("nan")
        resid = np.exp(log_f[None, :] + log_g[:, None])
        # smooth part of the existing map times the residual; the fine part stays in the
        # multiplier noise, so the product is exactly S * residual
        idx = ndimage.distance_transform_edt(~valid, return_distances=False,
                                             return_indices=True)
        smooth_part = ndimage.median_filter(scaler[tuple(idx)],
                                            size=tuple(args.noise_window), mode="nearest")
        curve = (smooth_part * resid).astype(np.float32)
    else:
        log_f, log_g, pair_const, rscale = fit_curves(
            data, step, sy, sz, a_z.start, b_z.start, args.y_knots, args.z_knots,
            args.smooth)
        curve = np.exp(log_f[None, :] + log_g[:, None])
        curve = (curve * float(np.median(scaler[valid])) / float(np.median(curve[valid]))
                 ).astype(np.float32)
    combined = np.where(valid, curve * mult_noise, 1e-9).astype(np.float32)

    # ---- how well the two copies agree with each map (weighted rms of the row ratios,
    # one free constant per pair)
    def disagreement(mult: Optional[np.ndarray]) -> Tuple[float, float]:
        sq, ww, absd = 0.0, 0.0, []
        for (d, w) in data:
            if mult is None:
                model = np.zeros_like(d)
            else:
                ok = (mult[a_z, a_rows] > MIN_SCALER) & (mult[b_z, b_rows] > MIN_SCALER)
                model = np.where(ok, np.log(np.maximum(mult[a_z, a_rows], 1e-12))
                                 - np.log(np.maximum(mult[b_z, b_rows], 1e-12)), 0.0)
                w = np.where(ok, w, 0.0)
            r = d - model
            if w.sum() <= 0:
                continue
            r = r - float((w * r).sum() / w.sum())
            sq += float((w * r * r).sum())
            ww += float(w.sum())
            absd.append(np.abs(r[w > 0]))
        return math.sqrt(sq / max(ww, 1e-12)), float(np.median(np.concatenate(absd)))

    rows_report = {}
    print("\ndisagreement of the two copies of the overlaps (log ratio, one free constant "
          "per pair; rms / median abs):")
    for name, m in (("no multiplier", None), ("current multiplier map (create_corrections)",
                                              scaler),
                    (("existing map * residual (smooth part)" if args.base_scaler
                      else "new curves f(y)*g(z)"), curve),
                    (("existing map * residual (final)" if args.base_scaler
                      else "new curves * multiplier noise"), combined)):
        rms, med = disagreement(m)
        rows_report[name] = {"rms": rms, "median_abs": med}
        print(f"  {name:46s} {rms:.4f} / {med:.4f}")
    print("  (the curves were fitted on these overlaps; the current map was not"
          + ("; the held-out check above is the fair test of the residual)" if args.base_scaler
             else ")"))

    paths = correction_paths(args.corrections_dir, args.channel)
    folder = os.path.dirname(paths["noise"])
    out = {"background_subtraction": os.path.join(folder, f"{args.channel}_background_subtraction.tiff"),
           "multiplier_correction": os.path.join(folder, f"{args.channel}_multiplier_correction.tiff"),
           "multiplier_noise": os.path.join(folder, f"{args.channel}_multiplier_noise.tiff")}
    write_tiff(out["background_subtraction"], noise)
    write_tiff(out["multiplier_correction"], curve)
    write_tiff(out["multiplier_noise"], mult_noise)
    nvalid = mult_noise[valid]
    print(f"\nmultiplier noise: median {np.median(nvalid):.4f}, 1st-99th percentile "
          f"{np.percentile(nvalid, 1):.3f}-{np.percentile(nvalid, 99):.3f}")
    write_yaml(os.path.join(args.corrections_dir, f"refine_multipliers_{args.channel}.yaml"), {
        "offset": None if offset is None else float(offset),
        "pairs_used": [int(i) + 1 for i in used],
        "levels": [float(v) for v in args.levels],
        "y_knots": args.y_knots, "z_knots": args.z_knots, "smooth": args.smooth,
        "noise_window": list(args.noise_window),
        "log_f_of_y": [float(v) for v in log_f], "log_g_of_z": [float(v) for v in log_g],
        "pair_constants": [float(v) for v in pair_const],
        "base_scaler": bool(args.base_scaler), "held_out": held_out,
        "residual_scale": float(rscale), "disagreement": rows_report, "files": out})
    for path in out.values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()