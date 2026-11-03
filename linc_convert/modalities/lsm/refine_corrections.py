"""Refine the correction maps of a channel with the overlaps between neighbouring chunks.

``create_corrections`` finds the multiplier of every (z, y) row from tissue statistics.
That cannot tell whether the first rows of a chunk and its last rows are at the right
level *relative to each other*, because anatomy hides a smooth tilt. The overlap can:
chunk ``i + 1`` images row ``step + k`` of chunk ``i`` again as its row ``k`` (z shifted
by ``dz``), so after a perfect correction the two copies of the same tissue agree.

The refinement keeps the existing maps and adds *smooth* fields to them, found by one
sparse least-squares problem:

    data        the two copies agree in the overlap. For every (z, k) the median over
                tissue of log(second copy / first copy) is taken over every pair of
                chunks (anatomy and random chunk gains average out), weighted by how
                much tissue went into it
    smoothness  second differences of the fields along y and along z are small
                (so no stripes can be added; row-to-row detail stays as it was)
    ridge       the fields stay small where the overlap says nothing

Default (multiplicative): ``corrected' = exp(phi(z, y)) * corrected``. Only the
multiplier map changes.

With ``--offset_term`` the correction is also allowed a smooth additive part,
``corrected' = exp(phi) * corrected + psi``. A gain error and an offset error look
the same at one brightness; they differ across brightness, so the data are the
mismatch of the two copies at several tissue levels (``--levels``, percentiles of
the tissue values of each row): ``(second - first) / mean`` at level ``j`` is
``-(phi_B - phi_A) - (q_ref / q_j) (psi_B - psi_A) / q_ref`` for a gain and an
offset together, with ``q_j`` the tissue level of that percentile. ``psi`` is
penalised more than ``phi`` (``--ridge_offset``), so an offset is only used where
the levels ask for it. It is written into the noise map:
``noise' = noise - psi / multiplier'``.

Only differences between row ``step + k`` and row ``k`` are observed. The smoothness
prior decides how the correction is spread over the chunk, so a smooth tilt or bow is
recovered well, an isolated bump in the middle of a chunk is not (nothing sees it).

    python -m linc_convert.modalities.lsm.refine_corrections \\
        --input_path .../sample-slice060 --yaml_path .../LSM_metadata.yaml \\
        --camera_id 2 --slice_number 60 --channel 561 \\
        --coords .../slice060_561.yaml --corrections_dir .../60_corrections_noreg_L2 \\
        --layer 2 [--offset_term]

The pair data are split into two halves first: the fit on one half is tested on the
other, and the before/after agreement of the held-out pairs is printed. Then the fit
uses all pairs and the TIFFs of the channel are replaced. The originals are kept once
as ``<channel>_alt_zy_scaler_prior.tiff`` and ``<channel>_alt_zy_noise_prior.tiff``
and always used as the starting point, so running again does not stack corrections;
``--restore`` puts them back.
"""

import argparse
import os
import shutil
import warnings
from typing import List, Optional, Tuple

import numpy as np
import yaml

MIN_TISSUE_SHARE = 0.01      # fewer tissue blocks than this: the pair lies in background
VALID_SCALER = 1e-6          # rows with a smaller multiplier are invalid
COUNT_SCALE = 300.0          # tissue blocks at which a row has weight 1/2
MAX_KAPPA = 20.0             # largest q_ref / q_j leverage of the offset


def prior_path(path: str) -> str:
    """Where the original of a correction TIFF is kept."""
    return (path.replace("_alt_zy_scaler.tiff", "_alt_zy_scaler_prior.tiff")
            .replace("_alt_zy_noise.tiff", "_alt_zy_noise_prior.tiff"))


def tissue_blocks(ca: np.ndarray, cb: np.ndarray, block: int, tissue: np.ndarray
                  ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """x-block means of both copies and the tissue mask; None if there is no tissue.

    ``ca`` and ``cb`` are the corrected overlap of the first and the second chunk,
    shape (z, rows, x); ``tissue`` is the (z, rows, x // block) mask of blocks that
    stand out from the background in both copies (``compare_overlaps.block_tissue``).
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


def aggregate_quantiles(firsts: List[np.ndarray], seconds: List[np.ndarray],
                        counts: List[np.ndarray], min_blocks: int, min_pairs: int
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine the pairs at every level.

    Returns ``(mismatch, level, weight)``: the median over pairs of
    ``(second - first) / mean`` and of the mean of the two, both (levels, z, rows),
    and the (z, rows) weight as in ``aggregate``.
    """
    qa, qb = np.stack(firsts), np.stack(seconds)          # (pairs, levels, z, rows)
    n = np.stack(counts)                                  # (pairs, z, rows)
    mean = 0.5 * (qa + qb)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(mean > 0, (qb - qa) / mean, np.nan)
    good = (n >= min_blocks)[:, None] & np.isfinite(rel) & (mean > 0)
    rel = np.where(good, rel, np.nan)
    mean = np.where(good, mean, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        d = np.nanmedian(rel, axis=0)
        level = np.nanmedian(mean, axis=0)
    all_levels = good.all(axis=1)                         # (pairs, z, rows)
    total = np.where(all_levels, n, 0).sum(axis=0)
    weight = total / (total + COUNT_SCALE)
    weight = np.where(all_levels.sum(axis=0) >= min_pairs, weight, 0.0)
    return np.nan_to_num(d), np.nan_to_num(level), weight.astype(np.float64)


def _second_difference(sz: int, sy: int, axis: int, weight: float, base: int):
    """Rows ``weight * (x[i-1] - 2 x[i] + x[i+1])`` along y (axis 1) or z (axis 0)."""
    import scipy.sparse as sp
    if axis == 1:
        zz, yy = np.meshgrid(np.arange(sz), np.arange(1, sy - 1), indexing="ij")
        zz, yy = zz.ravel(), yy.ravel()
        parts = [(zz, yy - 1), (zz, yy), (zz, yy + 1)]
    else:
        zz, yy = np.meshgrid(np.arange(1, sz - 1), np.arange(sy), indexing="ij")
        zz, yy = zz.ravel(), yy.ravel()
        parts = [(zz - 1, yy), (zz, yy), (zz + 1, yy)]
    r = np.arange(len(zz))
    cols = [base + z * sy + y for z, y in parts]
    vals = [np.full(len(r), weight), np.full(len(r), -2 * weight), np.full(len(r), weight)]
    return (np.concatenate(vals), np.concatenate([r, r, r]), np.concatenate(cols),
            len(r))


def solve_fields(shape: Tuple[int, int], step: int, dz: int, z_lo: int, d: np.ndarray,
                 weight: np.ndarray, smooth_y: float, smooth_z: float, ridge: float,
                 kappa: Optional[np.ndarray] = None, smooth_o: float = 0.0,
                 ridge_o: float = 0.0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Solve for the log gain ``phi`` (z, y) and, with ``kappa``, the offset ``psi``.

    ``d[l, i, k]`` is the mismatch at level ``l`` of first-copy z index ``z_lo + i`` and
    overlap row ``k``; ``weight`` has the same shape. The equation is
    ``phi(z - dz, k) - phi(z, step + k) + kappa * (psi(z - dz, k) - psi(z, step + k))
    = -d``; without ``kappa`` there is no ``psi``. ``psi`` is in units of ``q_ref``.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    sz, sy = shape
    n = sz * sy
    n_fields = 1 if kappa is None else 2
    total = n * n_fields
    nlev, nzo, h = d.shape
    iz, ik = np.meshgrid(np.arange(nzo), np.arange(h), indexing="ij")
    z_a = (z_lo + iz).ravel()
    z_b = z_a - dz
    k = ik.ravel()
    inside = (z_b >= 0) & (z_b < sz) & (z_a >= 0) & (z_a < sz)

    blocks, rhs = [], []
    for lev in range(nlev):
        w = np.sqrt(weight[lev].ravel() / nlev)
        keep = (w > 0) & inside
        za, zb, kk, ww = z_a[keep], z_b[keep], k[keep], w[keep]
        rows = np.arange(len(kk))
        ib, ia = zb * sy + kk, za * sy + step + kk
        r_all = [rows, rows]
        c_all = [ib, ia]
        v_all = [ww, -ww]
        if kappa is not None:
            kap = np.clip(kappa[lev].ravel()[keep], 0, MAX_KAPPA)
            r_all += [rows, rows]
            c_all += [n + ib, n + ia]
            v_all += [ww * kap, -ww * kap]
        blocks.append(sp.coo_matrix((np.concatenate(v_all), (np.concatenate(r_all),
                                                             np.concatenate(c_all))),
                                    shape=(len(kk), total)))
        rhs.append((-d[lev].ravel()[keep]) * ww)

    fields = [(0, smooth_y, smooth_z, ridge)]
    if kappa is not None:
        fields.append((n, smooth_o, smooth_o, ridge_o))
    for base, s_y, s_z, rdg in fields:
        for axis, weight_s, enabled in ((1, s_y, s_y > 0 and sy > 2),
                                        (0, s_z, s_z > 0 and sz > 2)):
            if enabled:
                vals, r_idx, c_idx, nrows = _second_difference(sz, sy, axis, weight_s,
                                                              base)
                blocks.append(sp.coo_matrix((vals, (r_idx, c_idx)),
                                            shape=(nrows, total)))
                rhs.append(np.zeros(nrows))
        if rdg > 0:
            blocks.append(sp.coo_matrix(
                (np.full(n, rdg), (np.arange(n), base + np.arange(n))), shape=(n, total)))
            rhs.append(np.zeros(n))
    a = sp.vstack(blocks).tocsr()
    b = np.concatenate(rhs)
    solution = np.asarray(spsolve((a.T @ a).tocsc(), a.T @ b), dtype=np.float64)
    phi = solution[:n].reshape(sz, sy)
    psi = solution[n:].reshape(sz, sy) if kappa is not None else None
    return phi, psi


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
    p.add_argument("--min_x", type=float, help="full-res px")
    p.add_argument("--max_x", type=float)
    p.add_argument("--min_y", type=float,
                   help="only pairs whose overlap lies in this y range (full-res px); "
                        "default: all pairs of the slice")
    p.add_argument("--max_y", type=float)
    p.add_argument("--smooth_y", type=float, default=30.0,
                   help="weight of the second-difference penalty along y")
    p.add_argument("--smooth_z", type=float, default=30.0, help="same along z")
    p.add_argument("--ridge", type=float, default=0.05, help="pull of phi towards 0")
    p.add_argument("--offset_term", action="store_true",
                   help="also fit a smooth additive offset (written into the noise map)")
    p.add_argument("--levels", type=float, nargs="+", default=[20, 35, 50, 65, 80],
                   help="tissue percentiles compared in the offset mode")
    p.add_argument("--smooth_offset", type=float, default=30.0,
                   help="second-difference penalty of the offset, along y and z")
    p.add_argument("--ridge_offset", type=float, default=0.2,
                   help="pull of the offset towards 0 (larger = offset used less)")
    p.add_argument("--max_offset", type=float, default=0.5,
                   help="largest |offset| allowed, as a share of the tissue level")
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
    if args.offset_term and len(args.levels) < 2:
        raise SystemExit("--offset_term needs at least two --levels")

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
    x0 = 0 if args.min_x is None else max(0, int(round(args.min_x / scale)))
    x1 = sx if args.max_x is None else min(sx, int(round(args.max_x / scale)))
    h = sy - step
    bg_len = max(1, args.background_length // scale)
    mode = "gain + offset" if args.offset_term else "gain only"
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
        tail_a = np.asarray(open_chunk(i)[a_z, a_rows, sx - bg_len:sx].compute(),
                            dtype=np.float32)
        tail_b = np.asarray(open_chunk(i + 1)[b_z, b_rows, sx - bg_len:sx].compute(),
                            dtype=np.float32)
        tissue = (block_tissue(va, noise[a_z, a_rows],
                               background_sigma(tail_a, noise[a_z, a_rows]),
                               1, 1, args.block, args.tissue_sigma)
                  & block_tissue(vb, noise[b_z, b_rows],
                                 background_sigma(tail_b, noise[b_z, b_rows]),
                                 1, 1, args.block, args.tissue_sigma))
        ca = np.clip(va - noise[a_z, a_rows][:, :, None], 0, None) \
            * scaler[a_z, a_rows][:, :, None]
        cb = np.clip(vb - noise[b_z, b_rows][:, :, None], 0, None) \
            * scaler[b_z, b_rows][:, :, None]
        if args.offset_term:
            res = pair_quantiles(ca, cb, args.block, args.levels, tissue)
        else:
            res = pair_log_ratios(ca, cb, args.block, tissue)
        print(f"  pair {i + 1}-{i + 2} ({n}/{len(selected)}): tissue "
              f"{100 * tissue.mean():4.1f}% of blocks"
              f"{'  -> no tissue, skipped' if res is None else ''}", flush=True)
        if res is not None:
            data.append(res)
            used.append(i)
    if len(used) < max(args.min_pairs, 2):
        raise SystemExit(f"only {len(used)} pairs have tissue; need at least "
                         f"{max(args.min_pairs, 2)}")
    z_lo = a_z.start
    levels = np.asarray(args.levels, dtype=float)

    def fit(sel: List[int]):
        """phi, psi (in units of q_ref), the data d (levels, z, rows), weight, q_ref."""
        if args.offset_term:
            d, level, w = aggregate_quantiles([data[j][0] for j in sel],
                                              [data[j][1] for j in sel],
                                              [data[j][2] for j in sel],
                                              args.min_blocks, args.min_pairs)
            w = w * valid[a_z, a_rows]
            mid = int(np.argmin(np.abs(levels - 50)))
            q_ref = float(np.median(level[mid][w > 0])) if (w > 0).any() else 1.0
            with np.errstate(divide="ignore", invalid="ignore"):
                kappa = np.where(level > 0, q_ref / level, 0.0)
            wl = np.broadcast_to(w, d.shape)
            phi, psi = solve_fields((sz, sy), step, dz, z_lo, d, wl, args.smooth_y,
                                    args.smooth_z, args.ridge, kappa,
                                    args.smooth_offset, args.ridge_offset)
            return phi, psi, d, w, q_ref, kappa
        d, w = aggregate([data[j][0] for j in sel], [data[j][1] for j in sel],
                         args.min_blocks, args.min_pairs)
        w = w * valid[a_z, a_rows]
        phi, _ = solve_fields((sz, sy), step, dz, z_lo, d[None],
                              np.broadcast_to(w, d.shape)[None], args.smooth_y,
                              args.smooth_z, args.ridge)
        return phi, None, d[None], w, 1.0, None

    def after_mismatch(d, phi, psi, kappa):
        """The data ``d`` (levels, z, rows) after the fields are applied."""
        out = d + implied_change(phi, step, dz, z_lo, (sz, sy))[None]
        if psi is not None:
            out = out + kappa * implied_change(psi, step, dz, z_lo, (sz, sy))[None]
        return out

    def level_text(name, values, good):
        return (f"{name}: " + "  ".join(
            f"p{int(lv)} {np.exp(np.median(v[good])):.3f}" if not args.offset_term else
            f"p{int(lv)} {1 + np.median(v[good]):.3f}" for lv, v in zip(levels, values)))

    if not args.no_validate and len(used) >= 2 * args.min_pairs:
        train = list(range(0, len(used), 2))
        test = list(range(1, len(used), 2))
        phi_tr, psi_tr, _, _, q_tr, _ = fit(train)
        print(f"\nheld-out check: fit on {len(train)} pairs, test on {len(test)} others")
        if args.offset_term:
            print("  pair   ratio second/first at the tissue percentiles "
                  f"{[int(v) for v in levels]}, before -> after")
            for j in test:
                qa, qb, cnt = data[j]
                with np.errstate(divide="ignore", invalid="ignore"):
                    mean = 0.5 * (qa + qb)
                    rel = np.where(mean > 0, (qb - qa) / mean, np.nan)
                good = cnt >= args.min_blocks
                with np.errstate(divide="ignore", invalid="ignore"):
                    kap = np.where(mean > 0, q_tr / mean, 0.0)
                aft = after_mismatch(np.nan_to_num(rel), phi_tr, psi_tr, kap)
                b4 = " ".join(f"{1 + np.nanmedian(rel[l][good]):.3f}"
                              for l in range(len(levels)))
                af = " ".join(f"{1 + np.median(aft[l][good]):.3f}"
                              for l in range(len(levels)))
                print(f"  {used[j] + 1:>2}-{used[j] + 2:<2}  {b4}  ->  {af}")
            print("  (1.000 is perfect; a gain error moves all percentiles by the same "
                  "factor, an offset moves the low ones more)")
        else:
            print("  pair   median ratio   row scatter (%)")
            for j in test:
                before, cnt = data[j]
                after = before + implied_change(phi_tr, step, dz, z_lo, (sz, sy))
                good = cnt >= args.min_blocks
                print(f"  {used[j] + 1:>2}-{used[j] + 2:<2}  "
                      f"{np.exp(np.median(before[good])):.3f} -> "
                      f"{np.exp(np.median(after[good])):.3f}   "
                      f"{100 * robust_sd(before[good]):.1f} -> "
                      f"{100 * robust_sd(after[good]):.1f}")
            print("  (ratio = second / first copy of the overlap, 1.000 is perfect; "
                  "row scatter is the spread of the log ratio over the rows)")

    phi, psi, d, w, q_ref, kappa = fit(list(range(len(used))))
    if not (w > 0).any():
        raise SystemExit("no (z, row) has enough tissue in enough pairs "
                         f"(--min_blocks {args.min_blocks}, --min_pairs {args.min_pairs})")
    phi = phi - float(np.median(phi[valid]))
    clipped = float(np.mean(np.abs(phi) > args.max_correction))
    phi = np.clip(phi, -args.max_correction, args.max_correction)
    new_scaler = np.where(valid, scaler * np.exp(phi), scaler).astype(np.float32)
    new_noise = noise
    if psi is not None:
        psi = psi - float(np.median(psi[valid]))
        psi = np.clip(psi, -args.max_offset, args.max_offset)
        offset_counts = psi * q_ref                       # corrected-count units
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = np.where(valid, offset_counts / new_scaler, 0.0)
        new_noise = (noise - shift).astype(np.float32)

    good = w > 0
    after = after_mismatch(d, phi, psi, kappa)
    print(f"\nfit on all {len(used)} pairs: rows with data {int(good.sum())} of {good.size}")
    if args.offset_term:
        print("  " + level_text("overlap mismatch before", d, good))
        print("  " + level_text("overlap mismatch after ", after, good))
        print(f"  tissue level q_ref {q_ref:.1f} counts; offset range "
              f"{offset_counts[valid].min():+.1f} .. {offset_counts[valid].max():+.1f} counts")
    else:
        print(f"  overlap mismatch (median ratio): {np.exp(np.median(d[0][good])):.3f} -> "
              f"{np.exp(np.median(after[0][good])):.3f}; row scatter "
              f"{100 * robust_sd(d[0][good]):.1f}% -> {100 * robust_sd(after[0][good]):.1f}%")
    print(f"  phi: min {phi[valid].min():+.3f}, max {phi[valid].max():+.3f} "
          f"(log factors; {100 * clipped:.1f}% of rows at the limit)")
    profile = np.array([phi[valid[:, y], y].mean() if valid[:, y].any() else np.nan
                        for y in range(sy)])
    offset_profile = (np.array([offset_counts[valid[:, y], y].mean()
                                if valid[:, y].any() else np.nan for y in range(sy)])
                      if psi is not None else None)
    print("  mean correction by chunk row (factor on the multiplier"
          + (", added counts" if psi is not None else "") + "):")
    for y0 in range(0, sy, max(1, sy // 12)):
        y1 = min(sy, y0 + max(1, sy // 12))
        tag = "  <- overlap with the previous chunk" if y1 <= h else (
            "  <- overlap with the next chunk" if y0 >= step else "")
        extra = (f"  {np.nanmean(offset_profile[y0:y1]):+7.1f}"
                 if offset_profile is not None else "")
        print(f"    rows {y0:>3}-{y1 - 1:<3} {np.exp(np.nanmean(profile[y0:y1])):.3f}"
              f"{extra}{tag}")

    if args.dry_run:
        print("dry run: nothing written")
        return
    if not os.path.exists(scaler_backup):
        shutil.copyfile(scaler_file, scaler_backup)
        shutil.copyfile(noise_file, noise_backup)
        print(f"kept the originals at {scaler_backup} and {noise_backup}")
    write_tiff(scaler_file, new_scaler)
    write_tiff(noise_file, new_noise.astype(np.float32))
    print(f"wrote {scaler_file} and {noise_file}"
          + ("" if psi is not None else " (noise map unchanged)"))


if __name__ == "__main__":
    main()