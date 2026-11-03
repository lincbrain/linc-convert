"""Shared background offset and zy multipliers from two tissue percentiles.

Every (z, y) row of a tile is one line of x values. Per row and tile this
module keeps

* ``b``  -- the row's background (median of its trailing background columns),
* ``q30``, ``q70`` -- the 30th and 70th percentile (configurable) over x of the
  RAW values of the row's tissue pixels (raw value >= ``median(b) *
  threshold_multiplier``),
* ``w``  -- the row's tissue fraction (share of sampled x columns that are
  tissue).

A corrected row is ``(value - (b - offset)) * m`` with ONE ``offset`` shared by
every row and one multiplier ``m`` per (z, y) shared by every tile. With
``a = max(q - b + offset, 0)`` the corrected percentile of tile ``c`` is
``a * m``, and it should equal a level ``T[c, p]`` that is the same for all
rows of the tile, i.e. ``a[c, p, z, y] = T[c, p] * g[z, y]`` with ``g = 1 / m``.
For a given ``offset`` the fit minimises the weighted squared residual of that
rank-1 model

    sum over tiles c, rows (z, y), percentiles p of
        w[c, z, y] * ((a[c, p, z, y] - T[c, p] * g[z, y]) / s[c]) ** 2

by alternating closed-form updates of ``g`` and ``T``; ``w`` is the row's
tissue fraction and ``s[c]`` the tile's median ``q_high - q_low`` (independent
of the offset). The residual is in raw counts, so it does not shrink when the
offset gets large (a relative residual would) or small. The offset is found by
a 1-D search of the resulting objective, and ``m = 1 / g`` is scaled to a
median of 1 over the rows with tissue.

Why two percentiles can tell the offset: ``m`` multiplies the signal, so a row
with a larger gain has both percentiles scaled the same way, but an offset
shifts both by the same amount. If the offset is wrong, the ratio of the
corrected percentiles changes with the row's gain; the right offset makes the
ratio the same for all rows of a tile. Anatomy also changes that ratio from row
to row; averaging over rows and tiles and the tissue weights keep it small, and
the delete-one-tile jackknife standard error says whether what is left is
noise. The offset is only applied if it is significantly non-zero.
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import dask.array as da
import numpy as np

from linc_convert.modalities.lsm.stitching_utils.corrections import (
    MIN_SCALER,
    compute_noise_map,
)

BLOCK_ELEMENTS = 2 * 10 ** 7   # values read at a time while measuring the percentiles
BOUNDARY_FRACTION = 0.95       # an estimate beyond this share of the search range is not used
GRID_POINTS = 9                # coarse offset search points
REFINE_STEPS = 10              # golden-section steps after the coarse search
ALS_ITERATIONS = 6             # alternating updates of the row multipliers and tile levels


@dataclass
class PercentileSettings:
    """Settings of the percentile fit.

    Parameters
    ----------
    low, high : float
        Percentiles (0-100) of the tissue pixels of each row.
    max_columns : int
        At most this many x columns (evenly strided) are read per row.
    min_pixels : int
        A row needs at least this many tissue pixels among them.
    max_offset_fraction : float
        The offset is searched within +- this fraction of the typical corrected
        high percentile.
    min_significance : float
        Standard errors the estimate must differ from 0 to be applied.
    """
    low: float = 30.0
    high: float = 70.0
    max_columns: int = 512
    min_pixels: int = 8
    max_offset_fraction: float = 0.5
    min_significance: float = 2.0


@dataclass
class TileStats:
    """Per-row measurements of one tile and channel.

    ``noise_map`` is (Z, Y); ``percentiles`` is (2, Z, Y) of RAW values
    (low, high; NaN where the row has too few tissue pixels); ``weight`` is
    (Z, Y), the tissue fraction (0 where the percentiles are NaN).
    """
    noise_map: np.ndarray
    percentiles: np.ndarray
    weight: np.ndarray
    tissue_fraction: float


@dataclass
class PercentileFit:
    """Result of ``fit_offset``.

    ``offset`` is the applied value (0 if not significant); ``estimate`` and
    ``standard_error`` describe the fit; ``tile_estimates`` are the
    delete-one-tile estimates; ``multiplier`` is (Z, Y), NaN where no tile has
    usable tissue in the row; ``objective_*`` are the row-effect F statistic of the
    percentile log-ratio at offset 0 and at the estimate (about 1 when no row
    effect is left).
    """
    offset: float
    estimate: float
    standard_error: float
    multiplier: np.ndarray
    tile_estimates: List[float] = field(default_factory=list)
    objective_zero: float = float("nan")
    objective_estimate: float = float("nan")
    rows_used: List[int] = field(default_factory=list)
    rows_total: int = 0
    search_limit: float = float("nan")


def sample_tile(vol: da.Array, background_length: int, threshold_multiplier: float,
                settings: PercentileSettings) -> TileStats:
    """Measure background, tissue percentiles and tissue fraction of every row.

    The "p-th percentile of the tissue pixels" is taken at the same overall
    rank in every row of the tile, ``1 - f * (1 - p / 100)`` of all the row's
    pixels, where ``f`` is the tile's typical tissue fraction. A row with a
    larger gain has more pixels above the tissue threshold; counting ranks
    among that row's own tissue pixels would shift its percentiles for that
    reason alone, which looks like an offset. At a common overall rank a gain
    scales the percentile exactly. For rows with the tile's typical tissue
    fraction this is the plain percentile of their tissue pixels.

    Parameters
    ----------
    vol : dask.array.Array
        Cropped (and flipped) channel volume, shape (Z, Y, X).
    background_length : int
        Trailing x columns used for each row's background (left out of the
        percentiles).
    threshold_multiplier : float
        A pixel is tissue if its raw value is at least
        ``median(noise_map) * threshold_multiplier``. This sets the tile's
        tissue fraction, a row's weight, and which rows have a usable
        percentile (the percentile must itself be tissue).
    settings : PercentileSettings
        Fit settings.

    Returns
    -------
    TileStats
        Percentiles are NaN for a row whose percentile is not tissue or that has
        fewer than ``min_pixels`` tissue pixels; the weight is the row's own
        tissue fraction.
    """
    vol = vol.astype(np.float32)
    n_z, n_y, n_x = vol.shape
    noise_map = compute_noise_map(vol, background_length)
    threshold = float(np.median(noise_map)) * threshold_multiplier
    usable = max(n_x - background_length, 1)
    step = max(1, int(np.ceil(usable / settings.max_columns)))
    columns = vol[:, :, :usable:step]
    n_cols = columns.shape[2]
    percentiles = np.full((2, n_z, n_y), np.nan, dtype=np.float32)
    weight = np.zeros((n_z, n_y), dtype=np.float32)

    # the tile's typical tissue fraction, from a thin subsample of the rows
    coarse = np.asarray(columns[::8, ::8].compute(), dtype=np.float32)
    per_row = (coarse >= threshold).mean(axis=2)
    present = per_row[per_row >= 0.02]
    if present.size == 0:
        return TileStats(noise_map, percentiles, weight, 0.0)
    fraction = float(np.clip(np.median(present), 0.02, 0.98))
    ranks = 1.0 - fraction * (1.0 - np.array([settings.low, settings.high]) / 100.0)

    per_slab = max(1, BLOCK_ELEMENTS // max(1, n_y * n_cols))
    for z0 in range(0, n_z, per_slab):
        z1 = min(z0 + per_slab, n_z)
        block = np.asarray(columns[z0:z1].compute(), dtype=np.float32)
        count = (np.isfinite(block) & (block >= threshold)).sum(axis=2)
        values = np.quantile(block, ranks, axis=2).astype(np.float32)
        good = (count >= settings.min_pixels) & np.all(values >= threshold, axis=0)
        percentiles[:, z0:z1] = np.where(good[None], values, np.nan)
        weight[z0:z1] = np.where(good, count / n_cols, 0.0)
    return TileStats(noise_map, percentiles, weight, fraction)


def _stack(tiles: List[TileStats]) -> Tuple[np.ndarray, np.ndarray]:
    """Stack ``q - b`` as (C, 2, Z, Y) and the weights as (C, Z, Y)."""
    lines = np.stack([t.percentiles - t.noise_map[None] for t in tiles])
    weights = np.stack([t.weight for t in tiles])
    return lines.astype(np.float32), weights.astype(np.float32)


def _solve(lines: np.ndarray, weights: np.ndarray, offset: float,
           iterations: int = ALS_ITERATIONS) -> Tuple[np.ndarray, float]:
    """Fit multipliers and tile levels for a fixed offset; score the offset.

    ``a = q - b + offset`` is the corrected percentile before multiplying. The
    model is ``a * m = T`` (``T`` per tile and percentile, ``m`` per row), fitted
    by alternating least squares on the relative residual ``a * m / T - 1`` with
    ``m`` scaled to median 1. Because the relative residual shrinks when the
    offset is large, the score is the negative log-likelihood of a model whose
    noise is proportional to the predicted value ``T / m`` (relative noise level
    profiled out):

        0.5 * log(weighted mean of residual ** 2)
            + weighted mean of log(T / m).

    Both terms change by the same amount in opposite directions when only the
    overall scale of ``a`` changes, so neither a large nor a small offset is
    favoured just for its scale.

    Parameters
    ----------
    lines : numpy.ndarray
        ``q - b`` of shape (C, 2, Z, Y), NaN where the row has no tissue.
    weights : numpy.ndarray
        Weight of every value (C, 2, Z, Y): the row's tissue fraction, 0 for
        values left out. Values must stay positive after adding ``offset``.
    offset : float
        The offset.
    iterations : int
        Alternating least-squares updates.

    Returns
    -------
    multiplier : numpy.ndarray
        ``m`` (Z, Y), median 1 over the rows used; NaN where no value of the row
        is used.
    score : float
        The score above (``inf`` if a tile has no values or a value is not
        positive).
    """
    n_z, n_y = weights.shape[2:]
    nan_m = np.full((n_z, n_y), np.nan, dtype=np.float32)
    used = np.isfinite(lines) & (weights > 0)
    w = np.where(used, weights, 0.0).astype(np.float32)
    a = np.where(used, lines + np.float32(offset), 1.0).astype(np.float32)
    if np.any(a[used] <= 0) or not np.all(w.sum(axis=(1, 2, 3)) > 0):
        return nan_m, float("inf")
    rows = w.sum(axis=(0, 1)) > 0
    m = np.ones((n_z, n_y), dtype=np.float32)
    target = (w * a).sum(axis=(2, 3)) / np.maximum(w.sum(axis=(2, 3)), 1e-12)
    target = np.where(w.sum(axis=(2, 3)) > 0, target, target[target > 0].mean())
    for _ in range(iterations):
        ratio = a / target[:, :, None, None]
        num = (w * ratio).sum(axis=(0, 1))
        den = (w * ratio * ratio).sum(axis=(0, 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(den > 0, num / den, 1.0).astype(np.float32)
        median = float(np.median(m[rows]))
        if not median > 0:
            return nan_m, float("inf")
        m = m / median
        wm = w * m
        target = (wm * a).sum(axis=(2, 3)) / np.maximum(w.sum(axis=(2, 3)), 1e-12)
        target = np.where(w.sum(axis=(2, 3)) > 0, target, target[target > 0].mean())
    resid = a * m[None, None] / target[:, :, None, None] - 1.0
    total = float(w.sum())
    mean_sq = float((w * resid ** 2).sum() / total)
    log_term = float((w * (np.log(target)[:, :, None, None]
                           - np.log(np.maximum(m, 1e-12))[None, None])).sum() / total)
    score = 0.5 * np.log(max(mean_sq, 1e-30)) + log_term
    return np.where(rows, m, np.nan).astype(np.float32), float(score)


def _ratio_f(lines: np.ndarray, weights: np.ndarray, offset: float) -> float:
    """Row-effect F statistic of the percentile log-ratio at a given offset.

    For tile ``c`` and row ``(z, y)`` let ``y = log((q_high - b + offset) /
    (q_low - b + offset))``. Anatomy that brightens or darkens both percentiles
    alike cancels in ``y``. With the right offset ``y`` has no dependence on the
    row's gain, so after taking out each tile's mean, a row's tiles scatter
    around zero by the anatomy noise only; with a wrong offset the gain (shared
    by all tiles of the row) moves ``y`` of all its tiles the same way, which
    shows up as a row effect. The statistic is the weighted between-row mean
    square (of the tile-averaged ``y``) over the weighted within-row mean
    square, a ratio that stays scale-free as the offset changes: about 1 at the
    right offset, larger otherwise.

    Parameters
    ----------
    lines : numpy.ndarray
        ``q - b`` (C, 2, Z, Y).
    weights : numpy.ndarray
        Weight per tile and row (C, Z, Y), or per value (C, 2, Z, Y) of which
        the smaller is used; 0 for values left out. Both percentiles must stay
        positive after adding ``offset``.
    offset : float
        The offset.

    Returns
    -------
    float
        ``inf`` if there are too few rows seen in at least two tiles.
    """
    if weights.ndim == 4:
        weights = weights.min(axis=1)
    used = weights > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.log((lines[:, 1] + offset) / (lines[:, 0] + offset))
    used &= np.isfinite(y)
    w = np.where(used, weights, 0.0).astype(np.float64)
    y = np.where(used, y, 0.0)
    tile_w = w.sum(axis=(1, 2))
    if not np.all(tile_w > 0):
        return float("inf")
    y = y - ((w * y).sum(axis=(1, 2)) / tile_w)[:, None, None]
    y = np.where(used, y, 0.0)
    n_seen = used.sum(axis=0)
    row_w = w.sum(axis=0)
    multi = n_seen >= 2
    if multi.sum() < 3:
        return float("inf")
    row_mean = np.where(multi, (w * y).sum(axis=0) / np.maximum(row_w, 1e-12), 0.0)
    between = float((row_w * row_mean ** 2)[multi].sum()) / (multi.sum() - 1)
    within_ss = float((w * (y - row_mean[None]) ** 2)[:, multi].sum())
    within_df = float((n_seen[multi] - 1).sum())
    if within_df <= 0 or within_ss <= 0:
        return float("inf")
    return between / (within_ss / within_df)


def _search(lines: np.ndarray, weights: np.ndarray, low: float, high: float) -> float:
    """Offset in ``[low, high]`` that minimises the objective.

    Coarse grid, then golden-section refinement around the best grid point.
    """
    def objective(o: float) -> float:
        return _ratio_f(lines, weights, o)

    grid = np.linspace(low, high, GRID_POINTS)
    values = [objective(o) for o in grid]
    best = int(np.argmin(values))
    lo = grid[max(best - 1, 0)]
    hi = grid[min(best + 1, GRID_POINTS - 1)]
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fc, fd = objective(c), objective(d)
    for _ in range(REFINE_STEPS):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - phi * (hi - lo)
            fc = objective(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + phi * (hi - lo)
            fd = objective(d)
    return float(0.5 * (lo + hi))


def fit_offset(tiles: List[TileStats], settings: Optional[PercentileSettings] = None
               ) -> PercentileFit:
    """Fit the shared offset and the (z, y) multipliers.

    Parameters
    ----------
    tiles : list of TileStats
        The reference tiles of one channel (at least 2; 3 or more give a
        standard error).
    settings : PercentileSettings, optional
        Fit settings.

    Returns
    -------
    PercentileFit
        The offset is applied (``offset``) only if ``|estimate| >=
        min_significance * standard_error``; the multipliers are always solved
        at the applied offset.
    """
    settings = settings or PercentileSettings()
    lines, tissue = _stack(tiles)
    n_tiles = len(tiles)
    _, n_z, n_y = tissue.shape

    valid_high = lines[:, 1][tissue > 0]
    typical = float(np.median(valid_high)) if valid_high.size else float("nan")
    if not np.isfinite(typical) or typical <= 0:
        nan_m = np.full((n_z, n_y), np.nan, dtype=np.float32)
        return PercentileFit(0.0, float("nan"), float("nan"), nan_m)
    valid_low = lines[:, 0][tissue > 0]
    limit = settings.max_offset_fraction * float(np.median(valid_low))
    if not limit > 0:
        nan_m = np.full((n_z, n_y), np.nan, dtype=np.float32)
        return PercentileFit(0.0, float("nan"), float("nan"), nan_m)
    # Values that would be clipped at the most negative offset of the search are
    # left out, so the objective does not gain from the clipping itself.
    weights = np.where(lines > limit, tissue[:, None], 0.0).astype(np.float32)

    estimate = _search(lines, weights, -limit, limit)
    objective_zero = _ratio_f(lines, weights, 0.0)
    objective_estimate = _ratio_f(lines, weights, estimate)

    tile_estimates: List[float] = []
    se = float("nan")
    if n_tiles >= 3:
        for leave in range(n_tiles):
            keep = [i for i in range(n_tiles) if i != leave]
            tile_estimates.append(_search(lines[keep], weights[keep], -limit, limit))
        e = np.asarray(tile_estimates)
        se = float(np.sqrt((n_tiles - 1) / n_tiles * np.sum((e - e.mean()) ** 2)))

    # An estimate at the end of the search range means the objective had no
    # minimum inside it (not identified), and a zero spread of the delete-one
    # estimates is the same boundary hit; neither is applied.
    interior = abs(estimate) < BOUNDARY_FRACTION * limit
    significant = (interior and np.isfinite(se) and se > 0
                   and abs(estimate) >= settings.min_significance * se)
    applied = float(estimate) if significant else 0.0
    final_weights = np.where(lines + applied > MIN_SCALER, tissue[:, None], 0.0)
    multiplier, _ = _solve(lines, final_weights.astype(np.float32), applied)
    rows_used = [int((final_weights[i].min(axis=0) > 0).sum()) for i in range(n_tiles)]
    return PercentileFit(
        applied, float(estimate), se, multiplier, tile_estimates,
        float(objective_zero), float(objective_estimate),
        rows_used, n_z * n_y, float(limit))


def multiplier_to_scaler(multiplier: np.ndarray) -> np.ndarray:
    """Scaler map for the correction TIFF: the multiplier, 1e-9 where invalid."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ok = np.isfinite(multiplier) & (multiplier > 0)
    return np.where(ok, multiplier, 1e-9).astype(np.float32)