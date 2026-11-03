"""Refine the correction maps of a channel with the overlaps between neighbouring chunks.

``create_corrections`` finds the multiplier of every (z, y) row from tissue statistics.
That cannot tell whether the first rows of a chunk and its last rows are at the right
level *relative to each other*, because anatomy hides a smooth tilt. The overlap can:
chunk ``i + 1`` images row ``step + k`` of chunk ``i`` again as its row ``k`` (z shifted
by ``dz``), so after a perfect correction the two copies of the same tissue agree.

The refinement keeps the existing maps and multiplies the multiplier map by a *smooth*
factor ``exp(phi(z, y))`` found by one sparse least-squares problem:

    data        the two copies agree in the overlap. For every (z, k) the median over
                tissue of log(second copy / first copy) is taken over every pair of
                chunks (anatomy and random chunk gains average out), weighted by how
                much tissue went into it
    smoothness  second differences of phi along y and along z are small
                (so no stripes can be added; row-to-row detail stays as it was)
    ridge       phi stays small where the overlap says nothing

Default: only ``phi`` is fitted, the noise map is left alone.

With ``--fit_offset`` the channel also gets ONE static offset ``delta`` (a number of
counts, the same for every row and chunk): the noise map becomes ``noise - delta``, so
``corrected' = exp(phi) * multiplier * (raw - noise + delta)``. A gain error and an
offset error look the same at one brightness but differ across brightness, and an
offset error also acts differently on rows with different multipliers. The overlap is
compared at several tissue levels (``--levels``, percentiles of the tissue values of each
row), and ``delta`` is the value for which the best smooth ``phi`` leaves the smallest
mismatch between the two copies, over all rows, z planes and chunk pairs together. It is
found by a one-dimensional search (grid, then golden section). If the best value sits at
the edge of the search range (``--offset_range``), it is not trusted and ``delta`` stays 0.

Only differences between row ``step + k`` and row ``k`` are observed. The smoothness
prior decides how the correction is spread over the chunk, so a smooth tilt or bow is
recovered well, an isolated bump in the middle of a chunk is not (nothing sees it).

    python -m linc_convert.modalities.lsm.refine_corrections \\
        --input_path .../sample-slice060 --yaml_path .../LSM_metadata.yaml \\
        --camera_id 2 --slice_number 60 --channel 561 \\
        --coords .../slice060_561.yaml --corrections_dir .../60_corrections_noreg_L2 \\
        --layer 2 [--fit_offset]

The pair data are split into two halves first: the fit on one half is tested on the
other, and the before/after agreement of the held-out pairs is printed (with
``--fit_offset`` also the offset found by each half on its own, which shows how well it
is determined). Then the fit uses all pairs and the TIFFs of the channel are replaced.
The originals are kept once as ``<channel>_alt_zy_scaler_prior.tiff`` and
``<channel>_alt_zy_noise_prior.tiff`` and always used as the starting point, so running
again does not stack corrections; ``--restore`` puts them back.
"""

import argparse
import math
import os
import shutil
import warnings
from typing import List, Optional, Tuple

import numpy as np
import yaml

MIN_TISSUE_SHARE = 0.01      # fewer tissue blocks than this: the pair lies in background
VALID_SCALER = 1e-6          # rows with a smaller multiplier are invalid
COUNT_SCALE = 300.0          # tissue blocks at which a row has weight 1/2


def prior_path(path: str) -> str:
    """Where the original of a correction TIFF is kept."""
    return (path.replace("_alt_zy_scaler.tiff", "_alt_zy_scaler_prior.tiff")
            .replace("_alt_zy_noise.tiff", "_alt_zy_noise_prior.tiff"))


def tissue_blocks(ca: np.ndarray, cb: np.ndarray, block: int, tissue: np.ndarray
                  ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """x-block means of both copies and the tissue mask; None if there is no tissue.

    ``ca`` and ``cb`` are the corrected overlap of the first and the second chunk,
    shape (z, rows, x); ``tissue`` is the (z, rows, x // block) mask of blocks that
    stand out from the background in the later chunk (``compare_overlaps.block_tissue``).
    It is applied to both copies, so the same blocks are used on both sides.
    """
    nz, h, nx = ca.shape
    nb = nx // block
    if nb == 0:
        return None
    a = ca[:, :, :nb * block].reshape(nz, h, nb, block).mean(axis=3)
    b = cb[:, :, :nb * block].reshape(nz, h, nb, block).mean(axis=3)
    mask = tissue[:, :, :nb] & (a > 0) & (b > 0)
    if mask.mean() < MIN_TISSUE_SHARE:
        return None
    return a, b, mask


def pair_log_ratios(ca: np.ndarray, cb: np.ndarray, block: int, tissue: np.ndarray
                    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Per (z, row) median of log(second / first) over tissue, and the tissue count."""
    blocks = tissue_blocks(ca, cb, block, tissue)
    if blocks is None:
        return None
    a, b, mask = blocks
    with np.errstate(divide="ignore", invalid="ignore"):
        logr = np.where(mask, np.log(np.where(b > 0, b, 1) / np.where(a > 0, a, 1)),
                        np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(logr, axis=2)
    return med.astype(np.float32), mask.sum(axis=2).astype(np.float32)


def pair_quantiles(ca: np.ndarray, cb: np.ndarray, block: int, levels: List[float],
                   tissue: np.ndarray
                   ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per (z, row) percentiles of the tissue values of both copies, and the count.

    Returns ``(first, second, count)`` with shapes (levels, z, rows), (levels, z, rows)
    and (z, rows).
    """
    blocks = tissue_blocks(ca, cb, block, tissue)
    if blocks is None:
        return None
    a, b, mask = blocks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        qa = np.nanpercentile(np.where(mask, a, np.nan), levels, axis=2)
        qb = np.nanpercentile(np.where(mask, b, np.nan), levels, axis=2)
    return (qa.astype(np.float32), qb.astype(np.float32),
            mask.sum(axis=2).astype(np.float32))


def aggregate(logs: List[np.ndarray], counts: List[np.ndarray], min_blocks: int,
              min_pairs: int) -> Tuple[np.ndarray, np.ndarray]:
    """Combine the pairs: median log ratio per (z, row) and a weight in 0..1.

    A (z, row) needs ``min_pairs`` pairs with at least ``min_blocks`` tissue blocks;
    others get weight 0. The weight grows with the total tissue count.
    """
    stack = np.stack(logs)
    n = np.stack(counts)
    stack = np.where(n >= min_blocks, stack, np.nan)
    ok = np.isfinite(stack)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(stack, axis=0)
    total = np.where(ok, n, 0).sum(axis=0)
    weight = total / (total + COUNT_SCALE)
    weight = np.where(ok.sum(axis=0) >= min_pairs, weight, 0.0)
    return np.nan_to_num(med), weight.astype(np.float64)


def offset_mismatch(firsts: List[np.ndarray], seconds: List[np.ndarray],
                    counts: List[np.ndarray], s_first: np.ndarray, s_second: np.ndarray,
                    delta: float, min_blocks: int, min_pairs: int
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Mismatch of the two copies at every level if the noise map were lowered by ``delta``.

    ``firsts`` / ``seconds`` are the per-pair percentiles (levels, z, rows) of the
    corrected overlap with the current maps, ``s_first`` / ``s_second`` the current
    multipliers of the rows of the two copies (z, rows). Lowering the noise by ``delta``
    raises every raw-minus-noise value by ``delta``, so a corrected percentile ``q`` of
    a row with multiplier ``s`` becomes ``q + s * delta`` (percentiles commute with this
    monotone change). Returns ``(log(second / first) median over pairs, weight)``,
    shapes (levels, z, rows) and (z, rows).
    """
    qa, qb = np.stack(firsts), np.stack(seconds)          # (pairs, levels, z, rows)
    n = np.stack(counts)                                  # (pairs, z, rows)
    na = qa + s_first[None, None] * delta
    nb = qb + s_second[None, None] * delta
    good = (n >= min_blocks)[:, None] & (qa > 0) & (qb > 0) & (na > 0) & (nb > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        logd = np.where(good, np.log(np.where(nb > 0, nb, 1) / np.where(na > 0, na, 1)),
                        np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        d = np.nanmedian(logd, axis=0)
    all_levels = good.all(axis=1)                         # (pairs, z, rows)
    total = np.where(all_levels, n, 0).sum(axis=0)
    weight = total / (total + COUNT_SCALE)
    weight = np.where(all_levels.sum(axis=0) >= min_pairs, weight, 0.0)
    return np.nan_to_num(d), weight.astype(np.float64)


def _second_difference(sz: int, sy: int, axis: int, weight: float):
    """Rows ``weight * (x[i-1] - 2 x[i] + x[i+1])`` along y (axis 1) or z (axis 0)."""
    if axis == 1:
        zz, yy = np.meshgrid(np.arange(sz), np.arange(1, sy - 1), indexing="ij")
        zz, yy = zz.ravel(), yy.ravel()
        parts = [(zz, yy - 1), (zz, yy), (zz, yy + 1)]
    else:
        zz, yy = np.meshgrid(np.arange(1, sz - 1), np.arange(sy), indexing="ij")
        zz, yy = zz.ravel(), yy.ravel()
        parts = [(zz - 1, yy), (zz, yy), (zz + 1, yy)]
    r = np.arange(len(zz))
    cols = [z * sy + y for z, y in parts]
    vals = [np.full(len(r), weight), np.full(len(r), -2 * weight), np.full(len(r), weight)]
    return (np.concatenate(vals), np.concatenate([r, r, r]), np.concatenate(cols),
            len(r))


def solve_phi(shape: Tuple[int, int], step: int, dz: int, z_lo: int, d: np.ndarray,
              weight: np.ndarray, smooth_y: float, smooth_z: float, ridge: float
              ) -> Tuple[np.ndarray, float]:
    """Solve for the log gain ``phi`` (z, y) from the overlap data ``d``.

    ``d[l, i, k]`` is the log mismatch at level ``l`` of first-copy z index ``z_lo + i``
    and overlap row ``k``; ``weight`` has the same shape. The equation is
    ``phi(z - dz, k) - phi(z, step + k) = -d``.

    Returns ``phi`` and the weighted mean squared residual of the data equations (the
    mismatch the smooth ``phi`` could not remove).
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    sz, sy = shape
    n = sz * sy
    nlev, nzo, h = d.shape
    iz, ik = np.meshgrid(np.arange(nzo), np.arange(h), indexing="ij")
    z_a = (z_lo + iz).ravel()
    z_b = z_a - dz
    k = ik.ravel()
    inside = (z_b >= 0) & (z_b < sz) & (z_a >= 0) & (z_a < sz)

    blocks, rhs, n_data = [], [], 0
    sq_weight, sq_resid_rows = 0.0, []
    for lev in range(nlev):
        w = np.sqrt(weight[lev].ravel() / nlev)
        keep = (w > 0) & inside
        za, zb, kk, ww = z_a[keep], z_b[keep], k[keep], w[keep]
        rows = np.arange(len(kk))
        blocks.append(sp.coo_matrix(
            (np.concatenate([ww, -ww]),
             (np.concatenate([rows, rows]),
              np.concatenate([zb * sy + kk, za * sy + step + kk]))),
            shape=(len(kk), n)))
        rhs.append((-d[lev].ravel()[keep]) * ww)
        n_data += len(kk)
    data_rows = n_data
    for axis, weight_s, enabled in ((1, smooth_y, smooth_y > 0 and sy > 2),
                                    (0, smooth_z, smooth_z > 0 and sz > 2)):
        if enabled:
            vals, r_idx, c_idx, nrows = _second_difference(sz, sy, axis, weight_s)
            blocks.append(sp.coo_matrix((vals, (r_idx, c_idx)), shape=(nrows, n)))
            rhs.append(np.zeros(nrows))
    if ridge > 0:
        blocks.append(sp.coo_matrix((np.full(n, ridge), (np.arange(n), np.arange(n))),
                                    shape=(n, n)))
        rhs.append(np.zeros(n))
    a = sp.vstack(blocks).tocsr()
    b = np.concatenate(rhs)
    solution = np.asarray(spsolve((a.T @ a).tocsc(), a.T @ b), dtype=np.float64)
    resid = (a @ solution - b)[:data_rows]
    # the data rows are already scaled by sqrt(weight / levels), so the sum of squares is
    # a weighted sum; divide by the total weight (= sum of the squared scales)
    scale_sq = float(sum((blk.data[:len(blk.data) // 2] ** 2).sum() for blk in blocks[:nlev]))
    msr = float((resid ** 2).sum() / scale_sq) if scale_sq > 0 else float("nan")
    return solution.reshape(sz, sy), msr


def implied_change(field: np.ndarray, step: int, dz: int, z_lo: int,
                   shape: Tuple[int, int]) -> np.ndarray:
    """``field(z - dz, k) - field(z, step + k)`` over the overlap (z, rows)."""
    sz, sy = shape
    h = sy - step
    z_hi = min(sz, sz + dz)
    za = np.arange(z_lo, z_hi)
    return field[za - dz][:, :h] - field[za][:, step:step + h]


def robust_sd(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")


def search_offset(evaluate, limit: float, grid_points: int = 11, golden_steps: int = 8
                  ) -> Tuple[float, bool, List[Tuple[float, float]]]:
    """Minimise ``evaluate(delta)`` (the leftover mismatch) over ``[-limit, limit]``.

    A grid first, then golden-section steps between the neighbours of the best grid
    point. Returns ``(delta, interior, curve)``: ``interior`` is False if the best grid
    point is at an end of the range, and ``curve`` lists the grid ``(delta, value)``.
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
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - phi * (hi - lo)
            f1 = evaluate(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + phi * (hi - lo)
            f2 = evaluate(x2)
    return 0.5 * (lo + hi), True, curve


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
    p.add_argument("--block", type=int, default=4, help="x block size (layer px)")
    p.add_argument("--tissue_sigma", type=float, default=5.0,
                   help="a block is tissue if its mean is this many noise sigmas above "
                        "the background map")
    p.add_argument("--background_length", type=int, default=5000,
                   help="trailing columns (full-res px) the noise map was taken from; "
                        "their spread gives the noise level")
    p.add_argument("--min_x", type=float, help="full-res px, stitching coordinates")
    p.add_argument("--max_x", type=float)
    p.add_argument("--min_y", type=float,
                   help="only pairs whose overlap lies in this y range (full-res px); "
                        "default: all pairs of the slice")
    p.add_argument("--max_y", type=float)
    p.add_argument("--smooth_y", type=float, default=30.0,
                   help="weight of the second-difference penalty along y")
    p.add_argument("--smooth_z", type=float, default=30.0, help="same along z")
    p.add_argument("--ridge", type=float, default=0.05, help="pull of phi towards 0")
    p.add_argument("--fit_offset", action="store_true",
                   help="also fit one static offset for the channel (noise map - delta)")
    p.add_argument("--min_pair_tissue", type=float, default=0.3,
                   help="a chunk pair is used only if at least this share of its overlap "
                        "blocks is tissue; pairs at the edge of the sample (mostly "
                        "background) otherwise pull the fit")
    p.add_argument("--levels", type=float, nargs="+", default=[20, 35, 50, 65, 80],
                   help="tissue percentiles compared when the offset is fitted")
    p.add_argument("--offset_range", type=float, default=1.0,
                   help="search delta within +- this share of the raw tissue level "
                        "above background (20th percentile)")
    p.add_argument("--min_blocks", type=int, default=50,
                   help="tissue blocks a (z, row) needs in a pair")
    p.add_argument("--min_pairs", type=int, default=3,
                   help="pairs a (z, row) needs")
    p.add_argument("--max_correction", type=float, default=0.5,
                   help="largest |log change| of the multiplier allowed")
    p.add_argument("--no_validate", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="do not write the TIFFs")
    p.add_argument("--restore", action="store_true",
                   help="put the original maps back and stop")
    args = p.parse_args()
    if args.fit_offset and len(args.levels) < 2:
        raise SystemExit("--fit_offset needs at least two --levels")

    from linc_convert.modalities.lsm.compare_overlaps import (
        background_sigma,
        block_tissue,
        pair_rows,
    )
    from linc_convert.modalities.lsm.stitching_utils.corrections import (
        correction_paths,
        load_corrections,
    )
    from linc_convert.modalities.lsm.stitching_utils.io import (
        crop_and_flip,
        discover_tile_paths,
        open_tile_array,
        write_tiff,
    )
    from linc_convert.modalities.lsm.stitching_utils.scan_metadata import (
        get_crops,
        get_skew_shear,
        get_vertical_flip,
    )

    paths_ = correction_paths(args.corrections_dir, args.channel)
    scaler_file, noise_file = paths_["scaler"], paths_["noise"]
    scaler_backup, noise_backup = prior_path(scaler_file), prior_path(noise_file)
    if args.restore:
        if not os.path.exists(scaler_backup):
            raise SystemExit(f"no original multiplier map at {scaler_backup}")
        shutil.copyfile(scaler_backup, scaler_file)
        print(f"restored {scaler_file}")
        if os.path.exists(noise_backup):
            shutil.copyfile(noise_backup, noise_file)
            print(f"restored {noise_file}")
        return
    import tifffile
    if os.path.exists(scaler_backup):
        print(f"starting from the original maps ({scaler_backup})")
        scaler = tifffile.imread(scaler_backup).astype(np.float32)
        noise = (tifffile.imread(noise_backup) if os.path.exists(noise_backup)
                 else tifffile.imread(noise_file)).astype(np.float32)
    else:
        noise, scaler = load_corrections(args.corrections_dir, args.channel)

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
    # --min_x / --max_x are in stitching coordinates (x of the sheared mosaic, x' = x +
    # shear_x * z). A chunk column x is inside the window at every z only for
    # min_x <= x and x < max_x - shear_x * (Z - 1), so that is the range used.
    shift = int(math.ceil(get_skew_shear(scan) * (sz - 1)))
    x0 = 0 if args.min_x is None else max(0, int(round(args.min_x / scale)))
    x1 = sx if args.max_x is None else int(round(args.max_x / scale)) - shift
    x1 = min(sx, x1)
    if x1 <= x0:
        raise SystemExit(f"the x window leaves no chunk columns after the shear "
                         f"(shift {shift} px at this layer)")
    if args.min_x is not None or args.max_x is not None:
        print(f"x window {args.min_x}:{args.max_x} (stitching coordinates, full-res) -> "
              f"chunk columns {x0}:{x1} at layer {args.layer} (shear shifts the window by "
              f"up to {shift} px over z)")
    h = sy - step
    bg_len = max(1, args.background_length // scale)
    mode = "gain + static offset" if args.fit_offset else "gain only"
    print(f"channel {args.channel} ({mode}): {len(chunks)} chunks, chunk (Z,Y,X)="
          f"({sz},{sy},{sx}) at layer {args.layer}, step {step}, overlap {h} rows, "
          f"dz {dz} rows; x {x0}:{x1}")

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

    valid = scaler > VALID_SCALER
    data, used = [], []        # per pair: (log ratios, counts) or (first, second, counts)
    for n, i in enumerate(selected, 1):
        va = np.asarray(open_chunk(i)[a_z, a_rows, x0:x1].compute(), dtype=np.float32)
        vb = np.asarray(open_chunk(i + 1)[b_z, b_rows, x0:x1].compute(), dtype=np.float32)
        tail_b = np.asarray(open_chunk(i + 1)[b_z, b_rows, sx - bg_len:sx].compute(),
                            dtype=np.float32)
        # one mask for both copies: tissue is judged on the top rows of the later chunk
        # (its own overlap rows, against its own background level) and applied to the
        # same blocks of the earlier chunk's overlap
        tissue = block_tissue(vb, noise[b_z, b_rows],
                              background_sigma(tail_b, noise[b_z, b_rows]),
                              1, 1, args.block, args.tissue_sigma)
        ca = np.clip(va - noise[a_z, a_rows][:, :, None], 0, None) \
            * scaler[a_z, a_rows][:, :, None]
        cb = np.clip(vb - noise[b_z, b_rows][:, :, None], 0, None) \
            * scaler[b_z, b_rows][:, :, None]
        if args.fit_offset:
            res = pair_quantiles(ca, cb, args.block, args.levels, tissue)
        else:
            res = pair_log_ratios(ca, cb, args.block, tissue)
        share = float(tissue.mean())
        why = ""
        if res is None:
            why = "  -> no tissue, skipped"
        elif share < args.min_pair_tissue:
            why = f"  -> below --min_pair_tissue {args.min_pair_tissue:g}, skipped"
            res = None
        print(f"  pair {i + 1}-{i + 2} ({n}/{len(selected)}): tissue "
              f"{100 * share:4.1f}% of blocks{why}", flush=True)
        if res is not None:
            data.append(res)
            used.append(i)
    if len(used) < max(args.min_pairs, 2):
        raise SystemExit(f"only {len(used)} pairs have enough tissue; need at least "
                         f"{max(args.min_pairs, 2)} (see --min_pair_tissue)")
    print(f"\n{len(used)} of {len(selected)} pairs used (pairs {used[0] + 1}-{used[0] + 2} "
          f"... {used[-1] + 1}-{used[-1] + 2})")
    z_lo = a_z.start
    levels = np.asarray(args.levels, dtype=float)
    s_a, s_b = scaler[a_z, a_rows], scaler[b_z, b_rows]
    row_ok = (valid[a_z, a_rows] & valid[b_z, b_rows])
    shape = (sz, sy)

    def fit_gain(sel: List[int]):
        d, w = aggregate([data[j][0] for j in sel], [data[j][1] for j in sel],
                         args.min_blocks, args.min_pairs)
        w = w * row_ok
        phi, msr = solve_phi(shape, step, dz, z_lo, d[None],
                             np.broadcast_to(w, d.shape)[None], args.smooth_y,
                             args.smooth_z, args.ridge)
        return phi, d[None], w, msr

    def fit_at(sel: List[int], delta: float):
        """phi, mismatch data, weight and leftover mismatch for a given offset."""
        d, w = offset_mismatch([data[j][0] for j in sel], [data[j][1] for j in sel],
                               [data[j][2] for j in sel], s_a, s_b, delta,
                               args.min_blocks, args.min_pairs)
        w = w * row_ok
        phi, msr = solve_phi(shape, step, dz, z_lo, d, np.broadcast_to(w, d.shape),
                             args.smooth_y, args.smooth_z, args.ridge)
        return phi, d, w, msr

    def find_offset(sel: List[int]):
        """Static offset of the pairs ``sel``: (delta, interior, curve, limit)."""
        low = np.stack([data[j][0][0] for j in sel]) / np.maximum(s_a, 1e-12)[None]
        ok = np.isfinite(low) & (low > 0) & row_ok[None]
        level = float(np.median(low[ok])) if ok.any() else 0.0
        limit = args.offset_range * level
        if limit <= 0:
            return 0.0, False, [], 0.0
        delta, interior, curve = search_offset(
            lambda x: fit_at(sel, x)[3], limit)
        return delta, interior, curve, limit

    def ratios_after(j: int, phi: np.ndarray, delta: float) -> np.ndarray:
        """Median second/first ratio per level of pair j after phi and delta."""
        qa, qb, cnt = data[j]
        na, nb = qa + s_a[None] * delta, qb + s_b[None] * delta
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (nb / na) * np.exp(implied_change(phi, step, dz, z_lo, shape))[None]
        good = (cnt >= args.min_blocks)[None] & (na > 0) & (nb > 0) & row_ok[None]
        return np.array([np.median(r[lv][good[lv]]) if good[lv].any() else np.nan
                         for lv in range(len(levels))])

    if not args.no_validate and len(used) >= 2 * args.min_pairs:
        train = list(range(0, len(used), 2))
        test = list(range(1, len(used), 2))
        print(f"\nheld-out check: fit on {len(train)} pairs, test on {len(test)} others")
        if args.fit_offset:
            d_tr, ok_tr, _, lim = find_offset(train)
            d_te, ok_te, _, _ = find_offset(test)
            note = lambda ok: "" if ok else " (at the edge of the range: not trusted)"  # noqa: E731
            print(f"  static offset found by the train half {d_tr:+.1f}{note(ok_tr)}, by "
                  f"the test half {d_te:+.1f}{note(ok_te)} counts (search range +-{lim:.1f})")
            delta_tr = d_tr if ok_tr else 0.0
            phi_tr = fit_at(train, delta_tr)[0]
            print("  pair   ratio second/first at the tissue percentiles "
                  f"{[int(v) for v in levels]}: before -> after (train offset and phi)")
            for j in test:
                b4 = ratios_after(j, np.zeros((sz, sy)), 0.0)
                af = ratios_after(j, phi_tr, delta_tr)
                print(f"  {used[j] + 1:>2}-{used[j] + 2:<2}  "
                      + " ".join(f"{v:.3f}" for v in b4) + "  ->  "
                      + " ".join(f"{v:.3f}" for v in af))
            print("  (1.000 is perfect; a gain error moves all percentiles by the same "
                  "factor, an offset moves the low ones more)")
        else:
            phi_tr, _, _, _ = fit_gain(train)
            print("  pair   median ratio   row scatter (%)")
            for j in test:
                before, cnt = data[j]
                after = before + implied_change(phi_tr, step, dz, z_lo, shape)
                good = cnt >= args.min_blocks
                print(f"  {used[j] + 1:>2}-{used[j] + 2:<2}  "
                      f"{np.exp(np.median(before[good])):.3f} -> "
                      f"{np.exp(np.median(after[good])):.3f}   "
                      f"{100 * robust_sd(before[good]):.1f} -> "
                      f"{100 * robust_sd(after[good]):.1f}")
            print("  (ratio = second / first copy of the overlap, 1.000 is perfect; "
                  "row scatter is the spread of the log ratio over the rows)")

    everything = list(range(len(used)))
    delta = 0.0
    if args.fit_offset:
        d_all, interior, curve, limit = find_offset(everything)
        base = fit_at(everything, 0.0)[3]
        print(f"\nstatic offset from all {len(used)} pairs: search range +-{limit:.1f} "
              f"counts")
        print("  offset   leftover mismatch (mean square of log ratio, x1e-4)")
        for g, v in curve:
            print(f"  {g:+8.1f}   {1e4 * v:8.2f}")
        if interior:
            delta = d_all
            best = fit_at(everything, delta)[3]
            print(f"  best {delta:+.2f} counts: leftover {1e4 * base:.2f} -> "
                  f"{1e4 * best:.2f} (x1e-4) against no offset")
        else:
            print(f"  the best value ({d_all:+.1f}) is at the edge of the range: not "
                  "trusted, offset stays 0 (widen --offset_range to look further)")
        phi, d, w, _ = fit_at(everything, delta)
    else:
        phi, d, w, _ = fit_gain(everything)
    if not (w > 0).any():
        raise SystemExit("no (z, row) has enough tissue in enough pairs "
                         f"(--min_blocks {args.min_blocks}, --min_pairs {args.min_pairs})")
    phi = phi - float(np.median(phi[valid]))
    clipped = float(np.mean(np.abs(phi) > args.max_correction))
    phi = np.clip(phi, -args.max_correction, args.max_correction)
    new_scaler = np.where(valid, scaler * np.exp(phi), scaler).astype(np.float32)
    new_noise = (noise - delta).astype(np.float32)

    good = w > 0
    after = d + implied_change(phi, step, dz, z_lo, shape)[None]
    print(f"\nfit on all {len(used)} pairs: rows with data {int(good.sum())} of {good.size}")
    if args.fit_offset:
        lvl = lambda arr: "  ".join(  # noqa: E731
            f"p{int(lv)} {np.exp(np.median(arr[i][good])):.3f}"
            for i, lv in enumerate(levels))
        print("  overlap ratio second/first by tissue percentile, before: " + lvl(d))
        print("                                                    after:  " + lvl(after))
    else:
        print(f"  overlap mismatch (median ratio): {np.exp(np.median(d[0][good])):.3f} -> "
              f"{np.exp(np.median(after[0][good])):.3f}; row scatter "
              f"{100 * robust_sd(d[0][good]):.1f}% -> {100 * robust_sd(after[0][good]):.1f}%")
    print(f"  phi: min {phi[valid].min():+.3f}, max {phi[valid].max():+.3f} "
          f"(log factors; {100 * clipped:.1f}% of rows at the limit)")
    if args.fit_offset:
        print(f"  static offset: noise map lowered by {delta:+.2f} counts")
    profile = np.array([phi[valid[:, y], y].mean() if valid[:, y].any() else np.nan
                        for y in range(sy)])
    print("  mean correction by chunk row (factor on the multiplier):")
    for y0 in range(0, sy, max(1, sy // 12)):
        y1 = min(sy, y0 + max(1, sy // 12))
        tag = "  <- overlap with the previous chunk" if y1 <= h else (
            "  <- overlap with the next chunk" if y0 >= step else "")
        print(f"    rows {y0:>3}-{y1 - 1:<3} {np.exp(np.nanmean(profile[y0:y1])):.3f}{tag}")

    if args.dry_run:
        print("dry run: nothing written")
        return
    if not os.path.exists(scaler_backup):
        shutil.copyfile(scaler_file, scaler_backup)
        shutil.copyfile(noise_file, noise_backup)
        print(f"kept the originals at {scaler_backup} and {noise_backup}")
    write_tiff(scaler_file, new_scaler)
    write_tiff(noise_file, new_noise)
    print(f"wrote {scaler_file} and {noise_file}"
          + ("" if args.fit_offset else " (noise map unchanged)"))


if __name__ == "__main__":
    main()