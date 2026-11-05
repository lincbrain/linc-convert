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
    keep_columns,
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
        Percentiles (0-100) of the tissue pixels of each row (used when
        ``percentiles`` is not given).
    percentiles : tuple of float, optional
        Two or more increasing percentiles (0-100) instead of ``low`` / ``high``.
    max_columns : int
        At most this many x columns (evenly strided) are read per row.
    min_pixels : int
        A row needs at least this many tissue pixels among them.
    max_offset_fraction : float
        The offset is searched within +- this fraction of the typical corrected
        high percentile.
    min_significance : float
        Standard errors the estimate must differ from 0 to be applied.
    column_mask_fraction : float, optional
        If given, every row of a tile is measured on the same x columns: a column
        is left out for all rows if more than this share of the tile's rows are
        below the tissue threshold there, and kept for all rows otherwise (see
        ``corrections.keep_columns``). The percentiles are then plain percentiles
        of the kept columns. None: judge every pixel on its own, with the percentile
        ranks adjusted for each tile's tissue fraction (the older method).
    exclude_outlier_levels : bool
        Leave out percentile levels whose values do not follow the same row
        multipliers as the other levels (see ``outlier_levels``).
    outlier_k : float
        A level is an outlier if its residual is more than this many robust standard
        deviations (1.4826 * MAD) above the median residual of the levels, at least
        twice that median and above 0.01.
    min_levels : int
        Outlier exclusion never leaves fewer levels than this.
    """
    low: float = 30.0
    high: float = 70.0
    max_columns: int = 512
    min_pixels: int = 8
    max_offset_fraction: float = 0.5
    min_significance: float = 2.0
    percentiles: Optional[Tuple[float, ...]] = None
    column_mask_fraction: Optional[float] = None
    exclude_outlier_levels: bool = True
    outlier_k: float = 4.0
    min_levels: int = 5

    def levels(self) -> Tuple[float, ...]:
        """The percentiles used: ``percentiles`` if given, else ``(low, high)``."""
        levels = tuple(self.percentiles) if self.percentiles else (self.low, self.high)
        if len(levels) < 2 or list(levels) != sorted(set(levels)) \
                or not 0 <= levels[0] or not levels[-1] <= 100:
            raise ValueError("percentiles must be at least two distinct values in "
                             f"increasing order within 0-100, got {levels}")
        return levels


@dataclass
class TileStats:
    """Per-row measurements of one tile and channel.

    ``noise_map`` is (Z, Y); ``percentiles`` is (K, Z, Y) of RAW values
    (one per percentile, increasing; NaN where the row has too few tissue pixels); ``weight`` is
    (Z, Y), the tissue fraction (0 where the percentiles are NaN).
    """
    noise_map: np.ndarray
    percentiles: np.ndarray
    weight: np.ndarray
    tissue_fraction: float
    columns_kept: int = 0
    columns_total: int = 0


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
    levels: List[float] = field(default_factory=list)
    excluded_levels: List[float] = field(default_factory=list)
    level_residuals: List[float] = field(default_factory=list)


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

    With ``settings.column_mask_fraction`` set, all rows use the same columns
    (``corrections.keep_columns``) and the percentiles are plain percentiles of
    those columns (no rank adjustment is needed, every row has the same pixels).

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
    levels = settings.levels()
    percentiles = np.full((len(levels), n_z, n_y), np.nan, dtype=np.float32)
    weight = np.zeros((n_z, n_y), dtype=np.float32)

    if settings.column_mask_fraction is not None:
        # one set of columns for every row: see corrections.keep_columns
        allcols = np.asarray(columns.compute(), dtype=np.float32)
        keep = keep_columns(allcols, threshold, settings.column_mask_fraction)
        n_keep = int(keep.sum())
        if n_keep < settings.min_pixels:
            return TileStats(noise_map, percentiles, weight, 0.0, n_keep, n_cols)
        kept = np.ascontiguousarray(allcols[:, :, keep])
        del allcols
        per_slab = max(1, BLOCK_ELEMENTS // max(1, n_y * n_keep))
        for z0 in range(0, n_z, per_slab):
            z1 = min(z0 + per_slab, n_z)
            block = kept[z0:z1]
            count = (np.isfinite(block) & (block >= threshold)).sum(axis=2)
            values = np.quantile(block, np.array(levels) / 100.0,
                                 axis=2).astype(np.float32)
            good = (count >= settings.min_pixels) & np.all(values >= threshold, axis=0)
            percentiles[:, z0:z1] = np.where(good[None], values, np.nan)
            weight[z0:z1] = np.where(good, count / n_keep, 0.0)
        return TileStats(noise_map, percentiles, weight, n_keep / n_cols,
                         n_keep, n_cols)

    # the tile's typical tissue fraction, from a thin subsample of the rows
    coarse = np.asarray(columns[::8, ::8].compute(), dtype=np.float32)
    per_row = (coarse >= threshold).mean(axis=2)
    present = per_row[per_row >= 0.02]
    if present.size == 0:
        return TileStats(noise_map, percentiles, weight, 0.0)
    fraction = float(np.clip(np.median(present), 0.02, 0.98))
    ranks = 1.0 - fraction * (1.0 - np.array(levels) / 100.0)

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
    """Stack ``q - b`` as (C, K, Z, Y) and the weights as (C, Z, Y)."""
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
        ``q - b`` of shape (C, K, Z, Y), NaN where the row has no tissue.
    weights : numpy.ndarray
        Weight of every value (C, K, Z, Y): the row's tissue fraction, 0 for
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


def _row_effect_f(y: np.ndarray, used: np.ndarray, weights: np.ndarray) -> float:
    """Weighted between-row over within-row mean square of ``y`` (C, Z, Y)."""
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


def _ratio_f(lines: np.ndarray, weights: np.ndarray, offset: float) -> float:
    """Row-effect F statistic of the percentile log-ratios at a given offset.

    For tile ``c``, row ``(z, y)`` and every percentile ``k`` above the lowest let
    ``y_k = log((q_k - b + offset) / (q_0 - b + offset))``. Anatomy that
    brightens or darkens all percentiles alike cancels in ``y_k``. With the right
    offset ``y_k`` has no dependence on the row's gain, so after taking out each
    tile's mean, a row's tiles scatter around zero by the anatomy noise only;
    with a wrong offset the gain (shared by all tiles of the row) moves ``y_k``
    of all its tiles the same way, which shows up as a row effect. For each
    ``k`` the statistic is the weighted between-row mean square (of the
    tile-averaged ``y_k``) over the weighted within-row mean square, a ratio that
    stays scale-free as the offset changes: about 1 at the right offset, larger
    otherwise. The result is the mean over ``k``.

    Parameters
    ----------
    lines : numpy.ndarray
        ``q - b`` (C, K, Z, Y).
    weights : numpy.ndarray
        Weight per tile and row (C, Z, Y), or per value (C, K, Z, Y) of which
        the smallest is used; 0 for values left out. All percentiles must stay
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
    base = lines[:, 0] + offset
    scores = []
    for k in range(1, lines.shape[1]):
        with np.errstate(invalid="ignore", divide="ignore"):
            y = np.log((lines[:, k] + offset) / base)
        scores.append(_row_effect_f(y, (weights > 0) & np.isfinite(y), weights))
    return float(np.mean(scores))


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


def level_residuals(lines: np.ndarray, weights: np.ndarray, offset: float,
                    multiplier: np.ndarray) -> np.ndarray:
    """Weighted mean ``|a * m / T - 1|`` of every percentile level, shape (K,).

    ``a = q - b + offset``; ``T`` is the weighted tile level of each percentile for the
    given multipliers. A level whose rows do not follow the same multipliers as the
    others (for example one that still contains background) has a large value.
    NaN for a level with no weight.
    """
    used = np.isfinite(lines) & (weights > 0)
    w = np.where(used, weights, 0.0).astype(np.float64)
    a = np.where(used, lines + offset, 1.0).astype(np.float64)
    m = np.where(np.isfinite(multiplier), multiplier, 1.0).astype(np.float64)
    wm = w * m[None, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        target = (wm * a).sum(axis=(2, 3)) / w.sum(axis=(2, 3))
        resid = np.abs(a * m[None, None] / target[:, :, None, None] - 1.0)
        out = (w * resid).sum(axis=(0, 2, 3)) / w.sum(axis=(0, 2, 3))
    return out


def outlier_levels(lines: np.ndarray, weights: np.ndarray, offset: float,
                   k: float = 4.0, min_levels: int = 5
                   ) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Find percentile levels that disagree with the others about the multipliers.

    For every active level the multipliers are fitted with the OTHER active levels and
    the level's own residual (``level_residuals``) is measured; the level with the
    largest one is dropped if it is more than ``k`` robust standard deviations
    (1.4826 * MAD, over the active levels) above their median, at least twice the
    median and above 0.01 (relative). Then the fit is repeated, until nothing is flagged or only
    ``min_levels`` are left.

    Returns
    -------
    (excluded, weights, residuals)
        Indices of the excluded levels, the weights with those levels set to 0, and
        the residuals of all levels at the last fit (NaN for excluded ones after
        they were dropped is not applied; they keep the value they had when dropped).
    """
    weights = weights.copy()
    active = [i for i in range(lines.shape[1]) if (weights[:, i] > 0).any()]
    excluded: List[int] = []
    shown = np.full(lines.shape[1], np.nan)
    while True:
        # Residual of every level when the multipliers are fitted WITHOUT it: a level
        # that is itself bad cannot pull the multipliers towards itself.
        res = np.full(lines.shape[1], np.nan)
        for i in active:
            w_i = weights.copy()
            w_i[:, i] = 0.0
            m, _ = _solve(lines, w_i, offset)
            only = np.zeros_like(weights)
            only[:, i] = weights[:, i]
            res[i] = level_residuals(lines, only, offset, m)[i]
        for i in active:
            shown[i] = res[i]
        if len(active) <= min_levels:
            break
        r = res[active]
        med = float(np.nanmedian(r))
        mad = float(np.nanmedian(np.abs(r - med)))
        worst = active[int(np.nanargmax(r))]
        if not (res[worst] > med + k * 1.4826 * mad and res[worst] > 2.0 * med
                and res[worst] > 0.01):
            break
        excluded.append(worst)
        active.remove(worst)
        weights[:, worst] = 0.0
    return excluded, weights, shown


def fit_offset(tiles: List[TileStats], settings: Optional[PercentileSettings] = None,
               fixed_offset: Optional[float] = None) -> PercentileFit:
    """Fit the shared offset and the (z, y) multipliers.

    Parameters
    ----------
    tiles : list of TileStats
        The reference tiles of one channel (at least 2; 3 or more give a
        standard error).
    settings : PercentileSettings, optional
        Fit settings.
    fixed_offset : float, optional
        Skip the search and the significance test and solve the multipliers at
        this offset (used to compare offsets); ``estimate`` is then this value
        and the standard error is NaN.

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

    valid_high = lines[:, -1][tissue > 0]
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

    estimate = (float(fixed_offset) if fixed_offset is not None
                else _search(lines, weights, -limit, limit))
    excluded: List[int] = []
    shown = np.full(lines.shape[1], np.nan)
    if settings.exclude_outlier_levels:
        excluded, weights, shown = outlier_levels(
            lines, weights, estimate, settings.outlier_k, settings.min_levels)
        if excluded and fixed_offset is None:
            estimate = _search(lines, weights, -limit, limit)
    objective_zero = _ratio_f(lines, weights, 0.0)
    objective_estimate = _ratio_f(lines, weights, estimate)

    tile_estimates: List[float] = []
    se = float("nan")
    if n_tiles >= 3 and fixed_offset is None:
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
    applied = (float(fixed_offset) if fixed_offset is not None
               else float(estimate) if significant else 0.0)
    final_weights = np.where(lines + applied > MIN_SCALER, tissue[:, None], 0.0)
    if excluded:
        final_weights[:, excluded] = 0.0
    multiplier, _ = _solve(lines, final_weights.astype(np.float32), applied)
    rows_used = [int((final_weights[i].min(axis=0) > 0).sum()) for i in range(n_tiles)]
    return PercentileFit(
        applied, float(estimate), se, multiplier, tile_estimates,
        float(objective_zero), float(objective_estimate),
        rows_used, n_z * n_y, float(limit), list(settings.levels()),
        [float(settings.levels()[i]) for i in excluded],
        [float(v) for v in shown])


def multiplier_to_scaler(multiplier: np.ndarray) -> np.ndarray:
    """Scaler map for the correction TIFF: the multiplier, 1e-9 where invalid."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ok = np.isfinite(multiplier) & (multiplier > 0)
    return np.where(ok, multiplier, 1e-9).astype(np.float32)