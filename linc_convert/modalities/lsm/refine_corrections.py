"""Refine the multiplier map of a channel with the overlaps between neighbouring chunks.

``create_corrections`` finds the multiplier of every (z, y) row from tissue statistics.
That cannot tell whether the first rows of a chunk and its last rows are at the right
level *relative to each other*, because anatomy hides a smooth tilt. The overlap can:
chunk ``i + 1`` images row ``step + k`` of chunk ``i`` again as its row ``k`` (z shifted
by ``dz``), so after a perfect correction the two copies of the same tissue agree.

The refinement keeps the existing multipliers and multiplies them by a *smooth* factor
``exp(phi(z, y))``. ``phi`` is the solution of one sparse least-squares problem:

    data        phi(z - dz, k) - phi(z, step + k)  =  -L(z, k)
                (L = median over tissue of log(second copy / first copy), taken
                 over every pair of chunks, so anatomy and random chunk gains
                 average out; weighted by how much tissue went into it)
    smoothness  second differences of phi along y and along z are small
                (so no stripes can be added; row-to-row detail stays as it was)
    ridge       phi stays small where the overlap says nothing (the rows in the
                middle of a chunk keep the original multipliers as far as the
                smoothness allows)

Only differences ``phi(step + k) - phi(k)`` are observed. The smoothness prior decides
how the correction is spread over the chunk, so a smooth tilt or bow is recovered
well, an isolated bump in the middle of a chunk is not (nothing sees it).

    python -m linc_convert.modalities.lsm.refine_corrections \\
        --input_path .../sample-slice060 --yaml_path .../LSM_metadata.yaml \\
        --camera_id 2 --slice_number 60 --channel 561 \\
        --coords .../slice060_561.yaml --corrections_dir .../60_corrections_noreg_L2 \\
        --layer 2

The pair data are split into two halves first: the fit on one half is tested on the
other, and the before/after agreement of the held-out pairs is printed. Then the fit
uses all pairs and the scaler TIFF of the channel is replaced. The original is kept
once as ``<channel>_alt_zy_scaler_prior.tiff`` and always used as the starting point,
so running again does not stack corrections; ``--restore`` puts it back.
"""

import argparse
import os
import shutil
import warnings
from typing import List, Optional, Tuple

import numpy as np
import yaml

TISSUE_FRACTION = 0.3        # tissue = x-block mean above this x the 90th percentile
MAX_TISSUE_SHARE = 0.9       # more tissue blocks than this: the pair lies in background
VALID_SCALER = 1e-6          # rows with a smaller multiplier are invalid


def prior_path(scaler_path: str) -> str:
    """Where the original multiplier map is kept."""
    return scaler_path.replace("_alt_zy_scaler.tiff", "_alt_zy_scaler_prior.tiff")


def pair_log_ratios(ca: np.ndarray, cb: np.ndarray, block: int
                    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Per (z, row) median of log(second / first) over tissue, and the tissue count.

    ``ca`` and ``cb`` are the corrected overlap of the first and the second chunk,
    shape (z, rows, x). Values are first averaged in blocks of ``block`` along x. Returns
    None if the pair is mostly background.
    """
    nz, h, nx = ca.shape
    nb = nx // block
    if nb == 0:
        return None
    a = ca[:, :, :nb * block].reshape(nz, h, nb, block).mean(axis=3)
    b = cb[:, :, :nb * block].reshape(nz, h, nb, block).mean(axis=3)
    mean = 0.5 * (a + b)
    pos = mean[mean > 0]
    if pos.size == 0:
        return None
    mask = (mean > TISSUE_FRACTION * np.percentile(pos, 90)) & (a > 0) & (b > 0)
    if mask.mean() > MAX_TISSUE_SHARE:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        logr = np.where(mask, np.log(np.where(b > 0, b, 1) / np.where(a > 0, a, 1)),
                        np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(logr, axis=2)
    return med.astype(np.float32), mask.sum(axis=2).astype(np.float32)


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
    weight = total / (total + 300.0)
    weight = np.where(ok.sum(axis=0) >= min_pairs, weight, 0.0)
    return np.nan_to_num(med), weight.astype(np.float64)


def solve_profile(shape: Tuple[int, int], step: int, dz: int, z_lo: int, d: np.ndarray,
                  weight: np.ndarray, smooth_y: float, smooth_z: float, ridge: float
                  ) -> np.ndarray:
    """Solve for ``phi`` (z, y) from the overlap data ``d`` (log ratios, rows 0..h).

    ``d[i, k]`` belongs to first-copy z index ``z_lo + i`` and overlap row ``k``.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    sz, sy = shape
    n = sz * sy
    idx = lambda z, y: z * sy + y            # noqa: E731
    nzo, h = d.shape
    iz, ik = np.meshgrid(np.arange(nzo), np.arange(h), indexing="ij")
    z_a = (z_lo + iz).ravel()
    z_b = z_a - dz
    k = ik.ravel()
    w = np.sqrt(weight.ravel())
    keep = (w > 0) & (z_b >= 0) & (z_b < sz) & (z_a >= 0) & (z_a < sz)
    z_a, z_b, k, w = z_a[keep], z_b[keep], k[keep], w[keep]
    rhs_data = (-d.ravel())[keep] * w
    m = len(k)
    rows = np.arange(m)
    data = sp.coo_matrix(
        (np.concatenate([w, -w]),
         (np.concatenate([rows, rows]),
          np.concatenate([idx(z_b, k), idx(z_a, step + k)]))), shape=(m, n))

    blocks = [data]
    rhs = [rhs_data]
    if smooth_y > 0:        # second differences along y
        zz, yy = np.meshgrid(np.arange(sz), np.arange(1, sy - 1), indexing="ij")
        zz, yy = zz.ravel(), yy.ravel()
        r = np.arange(len(zz))
        blocks.append(sp.coo_matrix(
            (np.concatenate([np.full(len(r), smooth_y), np.full(len(r), -2 * smooth_y),
                             np.full(len(r), smooth_y)]),
             (np.concatenate([r, r, r]),
              np.concatenate([idx(zz, yy - 1), idx(zz, yy), idx(zz, yy + 1)]))),
            shape=(len(r), n)))
        rhs.append(np.zeros(len(r)))
    if smooth_z > 0 and sz > 2:
        zz, yy = np.meshgrid(np.arange(1, sz - 1), np.arange(sy), indexing="ij")
        zz, yy = zz.ravel(), yy.ravel()
        r = np.arange(len(zz))
        blocks.append(sp.coo_matrix(
            (np.concatenate([np.full(len(r), smooth_z), np.full(len(r), -2 * smooth_z),
                             np.full(len(r), smooth_z)]),
             (np.concatenate([r, r, r]),
              np.concatenate([idx(zz - 1, yy), idx(zz, yy), idx(zz + 1, yy)]))),
            shape=(len(r), n)))
        rhs.append(np.zeros(len(r)))
    if ridge > 0:
        blocks.append(ridge * sp.identity(n, format="coo"))
        rhs.append(np.zeros(n))
    a = sp.vstack(blocks).tocsr()
    b = np.concatenate(rhs)
    normal = (a.T @ a).tocsc()
    phi = spsolve(normal, a.T @ b)
    return np.asarray(phi, dtype=np.float64).reshape(sz, sy)


def implied_change(phi: np.ndarray, step: int, dz: int, z_lo: int, shape: Tuple[int, int]
                   ) -> np.ndarray:
    """``phi(z - dz, k) - phi(z, step + k)`` for the overlap (the change of log ratio)."""
    sz, sy = shape
    h = sy - step
    z_hi = min(sz, sz + dz)
    za = np.arange(z_lo, z_hi)
    return phi[za - dz][:, :h] - phi[za][:, step:step + h]


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
    p.add_argument("--min_blocks", type=int, default=50,
                   help="tissue blocks a (z, row) needs in a pair")
    p.add_argument("--min_pairs", type=int, default=3,
                   help="pairs a (z, row) needs")
    p.add_argument("--max_correction", type=float, default=0.5,
                   help="largest |log change| allowed (0.5 = a factor 1.65)")
    p.add_argument("--no_validate", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="do not write the TIFF")
    p.add_argument("--restore", action="store_true",
                   help="put the original multiplier map back and stop")
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
    )
    from linc_convert.modalities.lsm.stitching_utils.scan_metadata import (
        get_crops,
        get_vertical_flip,
    )

    paths_ = correction_paths(args.corrections_dir, args.channel)
    scaler_file, noise_file = paths_["scaler"], paths_["noise"]
    backup = prior_path(scaler_file)
    if args.restore:
        if not os.path.exists(backup):
            raise SystemExit(f"no original multiplier map at {backup}")
        shutil.copyfile(backup, scaler_file)
        print(f"restored {scaler_file}")
        return
    if os.path.exists(backup):
        print(f"starting from the original map {backup}")
        import tifffile
        scaler = tifffile.imread(backup).astype(np.float32)
        noise, _ = load_corrections(args.corrections_dir, args.channel)
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
    print(f"channel {args.channel}: {len(chunks)} chunks, chunk (Z,Y,X)=({sz},{sy},{sx}) at "
          f"layer {args.layer}, step {step}, overlap {h} rows, dz {dz} rows; "
          f"x {x0}:{x1}")

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
    logs, counts, used = [], [], []
    for n, i in enumerate(selected, 1):
        va = np.asarray(open_chunk(i)[a_z, a_rows, x0:x1].compute(), dtype=np.float32)
        vb = np.asarray(open_chunk(i + 1)[b_z, b_rows, x0:x1].compute(), dtype=np.float32)
        ca = np.clip(va - noise[a_z, a_rows][:, :, None], 0, None) \
            * scaler[a_z, a_rows][:, :, None]
        cb = np.clip(vb - noise[b_z, b_rows][:, :, None], 0, None) \
            * scaler[b_z, b_rows][:, :, None]
        res = pair_log_ratios(ca, cb, args.block)
        tag = "background, skipped" if res is None else "ok"
        print(f"  pair {i + 1}-{i + 2} ({n}/{len(selected)}): {tag}", flush=True)
        if res is not None:
            logs.append(res[0])
            counts.append(res[1])
            used.append(i)
    if len(used) < max(args.min_pairs, 2):
        raise SystemExit(f"only {len(used)} pairs have tissue; need at least "
                         f"{max(args.min_pairs, 2)}")
    z_lo = a_z.start

    def fit(sel: List[int]):
        d, w = aggregate([logs[j] for j in sel], [counts[j] for j in sel],
                         args.min_blocks, args.min_pairs)
        w = w * valid[a_z, a_rows]
        phi = solve_profile((sz, sy), step, dz, z_lo, d, w, args.smooth_y,
                            args.smooth_z, args.ridge)
        return phi, d, w

    if not args.no_validate and len(used) >= 2 * args.min_pairs:
        train = list(range(0, len(used), 2))
        test = list(range(1, len(used), 2))
        phi_tr, _, _ = fit(train)
        change = implied_change(phi_tr, step, dz, z_lo, (sz, sy))
        print(f"\nheld-out check: fit on {len(train)} pairs, test on {len(test)} others")
        print("  pair   median ratio   row scatter (%)")
        for j in test:
            before = logs[j]
            after = before + change
            good = counts[j] >= args.min_blocks
            print(f"  {used[j] + 1:>2}-{used[j] + 2:<2}  "
                  f"{np.exp(np.median(before[good])):.3f} -> "
                  f"{np.exp(np.median(after[good])):.3f}   "
                  f"{100 * robust_sd(before[good]):.1f} -> "
                  f"{100 * robust_sd(after[good]):.1f}")
        print("  (ratio = second / first copy of the overlap, 1.000 is perfect; "
              "row scatter is the spread of the log ratio over the rows)")

    phi, d, w = fit(list(range(len(used))))
    if not (w > 0).any():
        raise SystemExit("no (z, row) has enough tissue in enough pairs "
                         f"(--min_blocks {args.min_blocks}, --min_pairs {args.min_pairs})")
    shift = float(np.median(phi[valid]))
    phi = phi - shift
    clipped = float(np.mean(np.abs(phi) > args.max_correction))
    phi = np.clip(phi, -args.max_correction, args.max_correction)
    new_scaler = np.where(valid, scaler * np.exp(phi), scaler).astype(np.float32)

    change = implied_change(phi, step, dz, z_lo, (sz, sy))
    good = w > 0
    print(f"\nfit on all {len(used)} pairs: rows with data {int(good.sum())} of {good.size}")
    print(f"  overlap mismatch (median ratio): {np.exp(np.median(d[good])):.3f} -> "
          f"{np.exp(np.median((d + change)[good])):.3f}; row scatter "
          f"{100 * robust_sd(d[good]):.1f}% -> {100 * robust_sd((d + change)[good]):.1f}%")
    print(f"  phi: min {phi[valid].min():+.3f}, max {phi[valid].max():+.3f} "
          f"(log factors; {100 * clipped:.1f}% of rows at the limit)")
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
    if not os.path.exists(backup):
        shutil.copyfile(scaler_file, backup)
        print(f"kept the original at {backup}")
    write_tiff(scaler_file, new_scaler)
    print(f"wrote {scaler_file}")


if __name__ == "__main__":
    main()