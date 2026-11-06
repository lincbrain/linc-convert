"""Compare correction strategies on held-out tiles of one channel.

For one channel (default 561) this builds the noise and scaler maps with many
strategies from a set of REFERENCE tiles, then scores every strategy on EVALUATION
tiles that were not used to make the maps, and prints / writes a ranked table.

    python evaluate_corrections.py --input_path .../strips_raw/sample-slice060 \\
        --yaml_path .../LSM_metadata.yaml --camera_id 2 --slice_number 60 \\
        --channel 561 --layer 2 --out_dir .../evaluation

Strategies
----------
* ``old_median_t{thr}``  -- the original method: per tile the median of the
  background-subtracted signal pixels, then the median over tiles.
* ``pct{K}_{auto|zero}_t{thr}`` -- ``percentile_fit`` with K percentiles
  (``P2`` = 30/70, ``P3`` = 25/50/75, ``P5`` = 20..60, ``P5b`` = 25..75); ``auto``
  fits and (if significant) applies the offset, ``zero`` forces offset 0 and only
  fits the multipliers.
* ``P5b_*_n{N}`` -- the same with N reference tiles (4, 12, 16, 24 if available;
  the main strategies use 8, evenly spread over the reference tiles).
* ``sweep_off{o}`` -- multipliers fitted at fixed offsets across the search range;
  this shows which offset is best by the held-out score, independent of the
  estimator.

Evaluation metric (lower is better)
-----------------------------------
There is no ground truth on real data, so the score measures what the
correction is for: stripes left in held-out tiles. For each evaluation tile
and each strategy the corrected row brightness ``C_k(z, y) = max(Q_k - noise, 0) *
scaler`` is computed at 5 brightness levels ``k`` (the 20th..80th percentile of
the row's tissue pixels, at a common overall rank in every row; a monotone
correction commutes with percentiles, so ``Q_k`` is measured once). The
*stripe residual* is the robust spread (1.4826 x MAD) of

    C_k(z, y) - mean of C_k over the neighbouring rows

along y and along z (the neighbours are within ``reach`` rows but not within
``gap`` rows of the row, so a stripe is compared with its surroundings), divided
by the tile's typical contrast ``median(C_80 - C_20)``, in percent, averaged over
the 5 levels and both axes (``stripe%``); ``dim..bright`` lists the 5 levels.
The residual is deliberately linear, not relative (log): a relative measure
shrinks when a large positive offset is added to everything, which would reward
a wrong offset, whereas the contrast is unchanged by the offset. A wrong offset
shows up as a stripe left at dim or at bright levels. Anatomy that varies on the
scale of a few rows adds the same floor to every strategy; compare strategies,
not the absolute value. All strategies are scored on the rows that are valid for
all of them, so none gains by suppressing difficult rows; ``cover`` is the share
of tissue rows a strategy can correct, and ``clip%`` the share of tissue
percentile levels the background subtraction sets to 0 (signal lost).
``own_bg_only`` (subtract the tile's own background, scaler 1) is the
no-correction baseline.

Needs the same environment as ``create_corrections`` (and optionally matplotlib
for a plot of the offset sweep).
"""

import argparse
import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.ndimage
import yaml

from linc_convert.modalities.lsm.create_corrections import (
    _auto_reference_tiles,
    _level_settings,
)
from linc_convert.modalities.lsm.stitching_utils import percentile_fit as pf
from linc_convert.modalities.lsm.stitching_utils.corrections import calibrate_tile
from linc_convert.modalities.lsm.stitching_utils.io import (
    crop_and_flip,
    discover_tile_paths,
    open_tile_array,
)
from linc_convert.modalities.lsm.stitching_utils.scan_metadata import (
    get_channel_names,
    get_crops,
    get_vertical_flip,
)

EVAL_LEVELS = (10.0, 20.0, 35.0, 50.0, 65.0, 80.0)   # tissue percentiles measured
STRIPE_LEVELS = (1, 2, 3, 4, 5)                      # indices used by the stripe metric
MIN_EVAL_WEIGHT = 0.05                               # tissue fraction a row needs
LEVEL_SETS = {
    "P2": (30.0, 70.0),
    "P3": (25.0, 50.0, 75.0),
    "P5": (20.0, 30.0, 40.0, 50.0, 60.0),
    "P5b": (25.0, 37.5, 50.0, 62.5, 75.0),
}


@dataclass
class Strategy:
    """One way of building the maps."""
    name: str
    kind: str                     # "old" or "pct"
    threshold: float
    levels: str = "P2"            # key of LEVEL_SETS
    mode: str = "auto"            # "auto", "zero" or "fixed"
    offset: float = 0.0           # for mode "fixed"
    n_ref: Optional[int] = None   # use only the first n_ref reference tiles
    sweep_fraction: Optional[float] = None   # offset as fraction of the search limit


MAIN_REFERENCE_TILES = 8       # reference tiles of the main strategies


def spread(items: Sequence[int], count: int) -> List[int]:
    """``count`` items evenly spread over ``items`` (all of them if fewer)."""
    items = list(items)
    if count >= len(items):
        return items
    return [items[i] for i in np.unique(np.round(
        np.linspace(0, len(items) - 1, count)).astype(int))]


def build_strategies(n_available: int, quick: bool) -> List[Strategy]:
    """The strategies to compare; ``n_available`` reference tiles exist."""
    main = min(MAIN_REFERENCE_TILES, n_available)
    out: List[Strategy] = [Strategy("old_median_t1.1", "old", 1.1, n_ref=main)]
    if not quick:
        out.append(Strategy("old_median_t1.05", "old", 1.05, n_ref=main))
    level_keys = ["P2", "P5"] if quick else ["P2", "P3", "P5", "P5b"]
    for key in level_keys:
        out.append(Strategy(f"{key}_zero_t1.1", "pct", 1.1, key, "zero", n_ref=main))
        out.append(Strategy(f"{key}_auto_t1.1", "pct", 1.1, key, "auto", n_ref=main))
    if not quick:
        for thr in (1.05, 1.2):
            out.append(Strategy(f"P2_zero_t{thr}", "pct", thr, "P2", "zero", n_ref=main))
            out.append(Strategy(f"P2_auto_t{thr}", "pct", thr, "P2", "auto", n_ref=main))
    for count in (4, 12, 16, 24):          # other numbers of reference tiles
        if count != main and count <= n_available and (not quick or count in (4, 12)):
            out.append(Strategy(f"P5b_zero_t1.1_n{count}", "pct", 1.1, "P5b", "zero",
                                n_ref=count))
            out.append(Strategy(f"P5b_auto_t1.1_n{count}", "pct", 1.1, "P5b", "auto",
                                n_ref=count))
    fractions = (-0.8, -0.4, 0.4, 0.8) if quick else \
        (-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8)
    for f in fractions:
        out.append(Strategy(f"sweep_{f:+.1f}L", "pct", 1.1, "P2", "fixed",
                            n_ref=main, sweep_fraction=f))
    return out


# ---------------------------------------------------------------------------
# measuring tiles (with an on-disk cache)
# ---------------------------------------------------------------------------

class Data:
    """Access to the tiles of one channel with cached measurements."""

    def __init__(self, tile_paths, crop, vertical_flip, layer, bg_len, x_stride,
                 max_columns, cache_dir, channel):
        self.tile_paths, self.crop, self.flip = tile_paths, crop, vertical_flip
        self.layer, self.bg_len, self.x_stride = layer, bg_len, x_stride
        self.max_columns, self.cache_dir, self.channel = max_columns, cache_dir, channel
        self.memory: Dict[str, object] = {}
        os.makedirs(cache_dir, exist_ok=True)

    def volume(self, tile: int):
        tile_array = open_tile_array(self.tile_paths[tile - 1], self.layer,
                                     chunks=(128, 128, 128))
        return crop_and_flip(tile_array, self.crop, self.flip, self.layer)

    def _file(self, key: str) -> str:
        digest = hashlib.md5(key.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{self.channel}_{digest}.npz")

    def stats(self, tile: int, threshold: float, levels: Sequence[float]
              ) -> pf.TileStats:
        """``percentile_fit.sample_tile`` of a tile, cached."""
        key = (f"stats|{self.channel}|{self.tile_paths[tile - 1]}|{self.layer}|"
               f"{threshold}|{tuple(levels)}|{self.max_columns}|{self.bg_len}")
        if key in self.memory:
            return self.memory[key]
        path = self._file(key)
        if os.path.exists(path):
            z = np.load(path)
            value = pf.TileStats(z["noise_map"], z["percentiles"], z["weight"],
                                 float(z["fraction"]))
        else:
            value = pf.sample_tile(
                self.volume(tile), self.bg_len, threshold,
                pf.PercentileSettings(percentiles=tuple(levels),
                                      max_columns=self.max_columns))
            np.savez(path, noise_map=value.noise_map, percentiles=value.percentiles,
                     weight=value.weight, fraction=value.tissue_fraction)
        self.memory[key] = value
        return value

    def old(self, tile: int, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """``calibrate_tile`` (noise map, scaler map) of a tile, cached."""
        key = (f"old|{self.channel}|{self.tile_paths[tile - 1]}|{self.layer}|"
               f"{threshold}|{self.bg_len}|{self.x_stride}")
        if key in self.memory:
            return self.memory[key]
        path = self._file(key)
        if os.path.exists(path):
            z = np.load(path)
            value = (z["noise"], z["scaler"])
        else:
            value = calibrate_tile(self.volume(tile), self.bg_len, threshold,
                                   self.x_stride)
            np.savez(path, noise=value[0], scaler=value[1])
        self.memory[key] = value
        return value


# ---------------------------------------------------------------------------
# building the maps
# ---------------------------------------------------------------------------

def make_maps(strategy: Strategy, data: Data, references: Sequence[int],
              limit: Optional[float]) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Noise map, scaler map and a record of what was fitted."""
    refs = spread(references, strategy.n_ref) if strategy.n_ref else list(references)
    if strategy.kind == "old":
        pairs = [data.old(t, strategy.threshold) for t in refs]
        noise = np.median(np.stack([p[0] for p in pairs]), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            scaler = np.nanmedian(np.stack([p[1] for p in pairs]), axis=0)
        return noise, np.nan_to_num(scaler, nan=1e-9), {"offset": 0.0, "n_ref": len(refs)}

    levels = LEVEL_SETS[strategy.levels]
    settings = pf.PercentileSettings(percentiles=levels, max_columns=data.max_columns)
    stats = [data.stats(t, strategy.threshold, levels) for t in refs]
    fixed = None
    if strategy.mode == "zero":
        fixed = 0.0
    elif strategy.mode == "fixed":
        fixed = float(strategy.sweep_fraction) * float(limit)
    fit = pf.fit_offset(stats, settings, fixed_offset=fixed)
    noise = np.median(np.stack([s.noise_map for s in stats]), axis=0) - fit.offset
    scaler = pf.multiplier_to_scaler(fit.multiplier)
    record = {"offset": float(fit.offset), "estimate": float(fit.estimate),
              "standard_error": float(fit.standard_error),
              "search_limit": float(fit.search_limit), "n_ref": len(refs),
              "score_at_zero": float(fit.score_zero),
              "score_at_estimate": float(fit.score_estimate)}
    return noise, scaler, record


# ---------------------------------------------------------------------------
# the metric
# ---------------------------------------------------------------------------

def ring_mean(values: np.ndarray, axis: int, reach: int, gap: int) -> np.ndarray:
    """Mean of the neighbours within ``reach`` but not within ``gap`` along ``axis``.

    NaN values are ignored; NaN where fewer than 30% of the neighbours have data.
    """
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0).astype(np.float64)

    def box(a: np.ndarray, half: int) -> np.ndarray:
        size = 2 * half + 1
        return scipy.ndimage.uniform_filter1d(a, size=size, axis=axis, mode="constant",
                                              cval=0.0) * size

    total = box(filled, reach) - box(filled, gap)
    count = box(valid.astype(np.float64), reach) - box(valid.astype(np.float64), gap)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    return np.where(count >= 0.3 * 2 * (reach - gap), mean, np.nan)


def robust_spread(r: np.ndarray) -> float:
    """1.4826 x median absolute deviation of the finite values (NaN if < 50)."""
    r = r[np.isfinite(r)]
    if r.size < 50:
        return float("nan")
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def corrected_levels(percentiles: np.ndarray, noise: np.ndarray,
                     scaler: np.ndarray) -> np.ndarray:
    """``max(Q - noise, 0) * scaler`` for (K, Z, Y) raw percentiles."""
    return np.maximum(percentiles - noise[None], 0.0) * scaler[None]


def score_all(maps: Dict[str, Tuple[np.ndarray, np.ndarray]],
              eval_stats: List[pf.TileStats], reach: int, gap: int) -> Dict[str, dict]:
    """Stripe residual, coverage and clipping of every strategy on the evaluation tiles."""
    names = list(maps) + ["own_bg_only"]
    per_tile: Dict[str, List[dict]] = {n: [] for n in names}
    for tile in eval_stats:
        rows = tile.weight >= MIN_EVAL_WEIGHT
        corrected, valid = {}, {}
        for name in names:
            if name == "own_bg_only":
                noise, scaler = tile.noise_map, np.ones_like(tile.noise_map)
            else:
                noise, scaler = maps[name]
            c = corrected_levels(tile.percentiles, noise, scaler)
            corrected[name] = c
            ok = rows & np.all(np.isfinite(c) & (c > 1e-6), axis=0) & (scaler > 1e-6)
            valid[name] = ok
        common = rows.copy()
        for name in names:
            common &= valid[name]
        for name in names:
            c = corrected[name]
            lin = np.where(common[None], c, np.nan)
            contrast = np.nanmedian(lin[STRIPE_LEVELS[-1]] - lin[STRIPE_LEVELS[0]])
            sigma = np.full((len(STRIPE_LEVELS), 2), np.nan)
            if np.isfinite(contrast) and contrast > 0:
                for i, k in enumerate(STRIPE_LEVELS):
                    for a, axis in enumerate((1, 0)):       # y, then z
                        sigma[i, a] = robust_spread(
                            lin[k] - ring_mean(lin[k], axis, reach, gap)) / contrast
            noise = tile.noise_map if name == "own_bg_only" else maps[name][0]
            lost = (tile.percentiles - noise[None] <= 0)[:, rows]
            per_tile[name].append({
                "sigma": sigma,
                "coverage": float(valid[name].sum() / max(rows.sum(), 1)),
                "clipped": float(lost.mean()) if lost.size else float("nan"),
                "common_rows": int(common.sum())})
    result = {}
    for name in names:
        sig = np.nanmean(np.stack([t["sigma"] for t in per_tile[name]]), axis=0) * 100
        by_level = sig.mean(axis=1)
        result[name] = {
            "stripe_resid": float(np.nanmean(sig)),
            "y_resid": float(np.nanmean(sig[:, 0])),
            "z_resid": float(np.nanmean(sig[:, 1])),
            "by_level": [float(v) for v in by_level],
            "coverage": float(np.mean([t["coverage"] for t in per_tile[name]])),
            "clipped": float(np.nanmean([t["clipped"] for t in per_tile[name]]) * 100),
            "common_rows": int(np.mean([t["common_rows"] for t in per_tile[name]]))}
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def split_tiles(chosen: List[int], n_ref: int, n_eval: int
                ) -> Tuple[List[int], List[int]]:
    """Evaluation tiles evenly spread among the chosen ones, the rest are references."""
    total = len(chosen)
    n_eval = min(n_eval, max(2, total // 3))
    positions = np.unique(np.round(np.linspace(0, total - 1, n_eval + 2)[1:-1]).astype(int))
    evals = [chosen[i] for i in positions]
    refs = [t for t in chosen if t not in evals][:n_ref]
    return refs, evals


def format_table(results: Dict[str, dict], records: Dict[str, dict]) -> str:
    """Ranked text table."""
    base = results["own_bg_only"]["stripe_resid"]
    order = sorted(results, key=lambda n: results[n]["stripe_resid"])
    lines = ["rank strategy                    stripe%  y%    z%   dim..bright (%)"
             "                  cover  clip%  offset (est +- se)    improve"]
    for i, name in enumerate(order, 1):
        r, rec = results[name], records.get(name, {})
        levels = " ".join(f"{v:5.2f}" for v in r["by_level"])
        offset = ""
        if rec:
            offset = f"{rec['offset']:+8.1f}"
            if "estimate" in rec and np.isfinite(rec.get("standard_error", np.nan)):
                offset += f" ({rec['estimate']:+.1f} +- {rec['standard_error']:.1f})"
        lines.append(
            f"{i:>4} {name:<26} {r['stripe_resid']:6.2f} {r['y_resid']:5.2f} "
            f"{r['z_resid']:5.2f}  {levels}  {r['coverage']:6.1%} "
            f"{r['clipped']:5.1f}  {offset:<22} {1 - r['stripe_resid'] / base:+6.1%}")
    return "\n".join(lines)


def try_plot(results: Dict[str, dict], records: Dict[str, dict], path: str) -> None:
    """Stripe residual against offset for the sweep (skipped without matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    pts = sorted((records[n]["offset"], results[n]) for n in results
                 if n.startswith("sweep_"))
    if not pts:
        return
    xs = [p[0] for p in pts]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, [p[1]["stripe_resid"] for p in pts], "o-", label="all levels")
    ax.plot(xs, [p[1]["by_level"][0] for p in pts], "s--", label="dimmest level")
    ax.plot(xs, [p[1]["by_level"][-1] for p in pts], "^--", label="brightest level")
    for name, style in (("P2_auto_t1.1", "k:"), ("P5_auto_t1.1", "r:")):
        if name in records and results.get(name):
            ax.axvline(records[name]["estimate"], color=style[0], linestyle=":",
                       label=f"{name} estimate")
    ax.set_xlabel("offset (counts)")
    ax.set_ylabel("stripe residual (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input_path", required=True)
    p.add_argument("--yaml_path", required=True)
    p.add_argument("--camera_id", type=int, default=2)
    p.add_argument("--slice_number", type=int, required=True)
    p.add_argument("--channel", default="561")
    p.add_argument("--layer", type=int, default=2)
    p.add_argument("--background_length", type=int, default=5000)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_reference_tiles", type=int, default=16,
                   help="reference tiles to pick; the main strategies use 8 of them "
                        "(evenly spread), others 4 / 12 / 16 / 24")
    p.add_argument("--n_eval_tiles", type=int, default=6)
    p.add_argument("--reference_tiles", type=int, nargs="*")
    p.add_argument("--eval_tiles", type=int, nargs="*")
    p.add_argument("--min_tissue_fraction", type=float, default=0.9)
    p.add_argument("--edge_margin", type=int, default=3)
    p.add_argument("--auto_scan_layer", type=int, default=3)
    p.add_argument("--max_columns", type=int, default=512)
    p.add_argument("--reach", type=int, default=32,
                   help="neighbourhood half-width in rows at full resolution")
    p.add_argument("--gap", type=int, default=4,
                   help="rows around a row left out of its neighbourhood (full res)")
    p.add_argument("--quick", action="store_true", help="fewer strategies")
    args = p.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.yaml_path) as f:
        scan = yaml.safe_load(f)
    channels = get_channel_names(scan, args.camera_id)
    if args.channel not in channels:
        raise SystemExit(f"channel {args.channel} not in camera {args.camera_id}: {channels}")
    tile_paths = discover_tile_paths(args.input_path, args.camera_id)
    crops = get_crops(scan, args.camera_id, args.slice_number)
    flip = get_vertical_flip(scan, args.camera_id)
    bg_len, x_stride = _level_settings(args.background_length, args.layer)

    if args.reference_tiles and args.eval_tiles:
        refs, evals = list(args.reference_tiles), list(args.eval_tiles)
    else:
        chosen = _auto_reference_tiles(
            tile_paths, crops, [args.channel], flip, args.background_length, 1.1,
            args.auto_scan_layer, args.n_reference_tiles + args.n_eval_tiles,
            args.min_tissue_fraction, args.edge_margin,
            os.path.join(args.out_dir, "tiles.yaml"))
        refs, evals = split_tiles(sorted(chosen), args.n_reference_tiles,
                                  args.n_eval_tiles)
    print(f"reference tiles {refs}\nevaluation tiles {evals}", flush=True)
    if len(refs) < 3 or len(evals) < 1:
        raise SystemExit("need at least 3 reference and 1 evaluation tile")

    data = Data(tile_paths, crops[args.channel], flip, args.layer, bg_len, x_stride,
                args.max_columns, os.path.join(args.out_dir, "cache"), args.channel)
    eval_stats = [data.stats(t, 1.1, EVAL_LEVELS) for t in evals]
    print(f"measured {len(eval_stats)} evaluation tiles ({time.time() - t0:.0f}s)",
          flush=True)

    strategies = build_strategies(len(refs), args.quick)
    maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    records: Dict[str, dict] = {}
    limit = None
    ordered = [s for s in strategies if s.mode != "fixed"] + \
              [s for s in strategies if s.mode == "fixed"]
    for s in ordered:
        if s.mode == "fixed" and limit is None:
            print(f"  skipping {s.name}: no search limit known")
            continue
        noise, scaler, record = make_maps(s, data, refs, limit)
        if s.name == "P2_auto_t1.1":
            limit = record["search_limit"]
        maps[s.name], records[s.name] = (noise, scaler), record
        shown = f"offset {record['offset']:+.1f}"
        if "estimate" in record:
            shown += f" (estimate {record['estimate']:+.1f} +- {record['standard_error']:.1f})"
        print(f"built {s.name:<24} {shown} ({time.time() - t0:.0f}s)", flush=True)

    reach = max(3, args.reach // 2 ** args.layer)
    gap = min(max(1, args.gap // 2 ** args.layer), reach - 1)
    results = score_all(maps, eval_stats, reach, gap)
    table = format_table(results, records)
    print("\n" + table)
    print(f"\nstripe% = stripe residual (lower is better); improve = reduction vs "
          f"own_bg_only; reach/gap used: {reach}/{gap} rows at layer {args.layer}")

    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(f"channel {args.channel} camera {args.camera_id} slice {args.slice_number} "
                f"layer {args.layer}\nreference tiles {refs}\nevaluation tiles {evals}\n\n"
                + table + "\n")
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"reference_tiles": refs, "eval_tiles": evals, "layer": args.layer,
                   "results": results, "records": records}, f, indent=1)
    try_plot(results, records, os.path.join(args.out_dir, "offset_sweep.png"))
    print(f"wrote {args.out_dir}/summary.txt, results.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()