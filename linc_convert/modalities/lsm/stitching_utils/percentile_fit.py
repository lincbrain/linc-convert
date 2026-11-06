"""Shared background offset and zy multipliers from full-chunk row percentiles.

Every (z, y) row of a reference tile (a whole chunk, after the stitching crop) is one
line of x values. Per row and tile this module keeps

* ``b``  -- the row's background (median of its trailing background columns),
* ``q_p`` -- percentiles ``p`` (default 10 15 20 30 40 50 60 70 80 90 95) over x of the
  RAW values of the row's tissue pixels (the same x columns for every row of the tile),
* ``w``  -- the row's tissue fraction.

A corrected row is ``(value - (b - offset)) * m`` with ONE ``offset`` shared by every
row and one multiplier ``m`` per (z, y) shared by every tile. With
``a = q_p - b + offset`` the corrected percentile is ``a * m`` and it should be the same
for all rows of a tile: ``a[c, p, z, y] * m[z, y] = T[c, p]``.

The offset and the multipliers are found by ALTERNATING (no nested search):

1. Multipliers for the current offset (alternating least squares on the relative
   residual ``a * m / T - 1``), with the HIGH percentiles weighted heavily
   (``weight_p = (p / 100) ** multiplier_weight_power``). The high percentiles are
   bright tissue, where an offset error is a small share of the value, so they tell
   the row gains almost independently of the offset.
2. Offset for the current multipliers: the value that minimises, over all tiles and
   percentiles, the differences between the rows' corrected percentiles (the same
   score as above, with ``m`` held fixed), with the LOW percentiles weighted heavily
   (``weight_p = (1 - p / 100) ** offset_weight_power``). The low percentiles are dim
   tissue, where an offset error is a large share of the value, so they tell the
   offset best.

Steps 1 and 2 repeat until the offset changes by less than ``offset_tol`` counts or
``max_iterations`` rounds are made. The score is the negative log-likelihood of a model
whose noise is proportional to the predicted value (relative noise level profiled out),
so neither a large nor a small offset is favoured for its scale alone:

    0.5 * log(weighted mean of residual ** 2) + weighted mean of log(T / m).

``m`` is scaled to a median of 1 over the rows with tissue. An estimate at the edge of
the search range (+- ``max_offset_fraction`` of the typical low percentile above the
background) means the score had no minimum inside it; it is then not applied.
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
        low percentile.
    column_mask_fraction : float, optional
        If given, every row of a tile is measured on the same x columns: a column
        is left out for all rows if more than this share of the tile's rows are
        below the tissue threshold there, and kept for all rows otherwise (see
        ``corrections.keep_columns``). The percentiles are then plain percentiles
        of the kept columns. None: judge every pixel on its own, with the percentile
        ranks adjusted for each tile's tissue fraction (the older method).
    multiplier_weight_power : float
        Weight of percentile ``p`` in the multiplier fit is ``(p / 100) ** power``:
        higher percentiles count more.
    offset_weight_power : float
        Weight of percentile ``p`` in the offset search is ``(1 - p / 100) ** power``:
        lower percentiles count more.
    max_iterations : int
        Rounds of (multipliers, offset) at most.
    offset_tol : float
        The offset is stable when it changes by less than this many counts in a round
        (or flips between two values that close).
    """
    low: float = 30.0
    high: float = 70.0
    max_columns: int = 512
    min_pixels: int = 8
    max_offset_fraction: float = 0.5
    percentiles: Optional[Tuple[float, ...]] = None
    column_mask_fraction: Optional[float] = None
    multiplier_weight_power: float = 3.0
    offset_weight_power: float = 3.0
    max_iterations: int = 8
    offset_tol: float = 0.1

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

    ``offset`` is the applied value (0 if the estimate was not inside the search
    range); ``estimate`` is what the alternation converged to; ``multiplier`` is
    (Z, Y), NaN where no tile has usable tissue in the row. ``offset_history`` is
    the offset after every round, ``converged`` whether the last round moved it by
    less than ``offset_tol``. ``score_zero`` / ``score_estimate`` are the offset
    score (see the module docstring, low percentiles weighted) of the final
    multipliers at offset 0 and at the estimate. ``standard_error`` and
    ``tile_estimates`` are not computed (NaN / empty).
    """
    offset: float
    estimate: float
    standard_error: float
    multiplier: np.ndarray
    tile_estimates: List[float] = field(default_factory=list)
    score_zero: float = float("nan")
    score_estimate: float = float("nan")
    rows_used: List[int] = field(default_factory=list)
    rows_total: int = 0
    search_limit: float = float("nan")
    levels: List[float] = field(default_factory=list)
    offset_history: List[float] = field(default_factory=list)
    converged: bool = False
    multiplier_level_weights: List[float] = field(default_factory=list)
    offset_level_weights: List[float] = field(default_factory=list)


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




def level_weights(levels: Tuple[float, ...], power: float, high: bool) -> np.ndarray:
    """Weight of every percentile level, scaled to a mean of 1.

    ``(p / 100) ** power`` if ``high`` (higher percentiles count more), else
    ``(1 - p / 100) ** power`` (lower percentiles count more).
    """
    p = np.asarray(levels, dtype=np.float64) / 100.0
    w = p ** power if high else (1.0 - p) ** power
    return (w / w.mean()).astype(np.float32)


def _offset_score(lines: np.ndarray, weights: np.ndarray, multiplier: np.ndarray,
                  offset: float) -> float:
    """Score of an offset for FIXED multipliers: how far the corrected percentiles
    of the rows are from the tile's level, relative, low percentiles heavy.

    ``a = q - b + offset``; the tile level of percentile ``p`` is the weighted mean
    ``T[c, p]`` of ``a * m`` over the rows, and the score is

        0.5 * log(weighted mean of (a * m / T - 1) ** 2) + weighted mean of log(T / m)

    (see the module docstring). ``weights`` (C, K, Z, Y) carry the row tissue weights
    times the level weights; the offset must keep every weighted ``a`` positive.

    Returns
    -------
    float
        ``inf`` if a weighted value is not positive or a tile has no weight.
    """
    used = np.isfinite(lines) & (weights > 0)
    w = np.where(used, weights, 0.0).astype(np.float32)
    a = np.where(used, lines + np.float32(offset), 1.0).astype(np.float32)
    if np.any(a[used] <= 0) or not np.all(w.sum(axis=(1, 2, 3)) > 0):
        return float("inf")
    m = np.where(np.isfinite(multiplier) & (multiplier > 0), multiplier, 1.0
                 ).astype(np.float32)
    w_ck = w.sum(axis=(2, 3))
    am = a * m[None, None]
    target = (w * am).sum(axis=(2, 3)) / np.maximum(w_ck, 1e-12)
    target = np.where(w_ck > 0, target, 1.0)
    if np.any(target <= 0):
        return float("inf")
    resid = am / target[:, :, None, None] - 1.0
    total = float(w.sum())
    mean_sq = float((w * resid ** 2).sum() / total)
    log_term = float((w * (np.log(target)[:, :, None, None]
                           - np.log(np.maximum(m, 1e-12))[None, None])).sum() / total)
    return float(0.5 * np.log(max(mean_sq, 1e-30)) + log_term)


def _search(objective, low: float, high: float) -> float:
    """Offset in ``[low, high]`` that minimises ``objective``.

    Coarse grid, then golden-section refinement around the best grid point; returns
    the best point evaluated.
    """
    tried = []

    def evaluate(o: float) -> float:
        value = objective(o)
        tried.append((float(o), value))
        return value

    grid = np.linspace(low, high, GRID_POINTS)
    values = [evaluate(o) for o in grid]
    best = int(np.argmin(values))
    lo = grid[max(best - 1, 0)]
    hi = grid[min(best + 1, GRID_POINTS - 1)]
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fc, fd = evaluate(c), evaluate(d)
    for _ in range(REFINE_STEPS):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - phi * (hi - lo)
            fc = evaluate(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + phi * (hi - lo)
            fd = evaluate(d)
    evaluate(0.5 * (lo + hi))
    finite = [t for t in tried if np.isfinite(t[1])]
    return float(min(finite, key=lambda t: t[1])[0]) if finite else float("nan")


def fit_offset(tiles: List[TileStats], settings: Optional[PercentileSettings] = None,
               fixed_offset: Optional[float] = None) -> PercentileFit:
    """Fit the shared offset and the (z, y) multipliers by alternating.

    Parameters
    ----------
    tiles : list of TileStats
        The reference tiles of one channel (full chunks after the stitching crop).
    settings : PercentileSettings, optional
        Fit settings.
    fixed_offset : float, optional
        Skip the alternation and solve the multipliers at this offset;
        ``estimate`` is then this value.

    Returns
    -------
    PercentileFit
        Round by round: multipliers for the current offset (high percentiles
        heavy), then the offset for those multipliers (low percentiles heavy), see
        the module docstring. The offset is applied (``offset``) when the estimate
        lies inside the search range; the multipliers are always solved at the applied
        offset.
    """
    settings = settings or PercentileSettings()
    lines, tissue = _stack(tiles)
    n_tiles = len(tiles)
    _, n_z, n_y = tissue.shape
    levels = settings.levels()
    mult_levels = level_weights(levels, settings.multiplier_weight_power, True)
    off_levels = level_weights(levels, settings.offset_weight_power, False)

    def failed() -> PercentileFit:
        nan_m = np.full((n_z, n_y), np.nan, dtype=np.float32)
        return PercentileFit(0.0, float("nan"), float("nan"), nan_m, levels=list(levels))

    valid_high = lines[:, -1][tissue > 0]
    typical = float(np.median(valid_high)) if valid_high.size else float("nan")
    if not np.isfinite(typical) or typical <= 0:
        return failed()
    valid_low = lines[:, 0][tissue > 0]
    limit = settings.max_offset_fraction * float(np.median(valid_low))
    if not limit > 0:
        return failed()
    # Values that would be clipped at the most negative offset of the search are
    # left out, so the score does not gain from the clipping itself.
    weights = np.where(lines > limit, tissue[:, None], 0.0).astype(np.float32)
    w_mult = weights * mult_levels[None, :, None, None]
    w_off = weights * off_levels[None, :, None, None]

    history: List[float] = []
    converged = False
    if fixed_offset is not None:
        estimate = float(fixed_offset)
        history.append(estimate)
        converged = True
    else:
        offset = 0.0
        for _ in range(settings.max_iterations):
            m, _ = _solve(lines, w_mult, offset)
            if not np.isfinite(m).any():
                return failed()
            new = _search(lambda o: _offset_score(lines, w_off, m, o), -limit, limit)
            if not np.isfinite(new):
                return failed()
            seq = [0.0] + history + [new]
            if len(seq) >= 4:
                # the rounds approach the answer geometrically (the multipliers absorb
                # part of an offset error each round): extrapolate (Aitken)
                d1, d2 = seq[-2] - seq[-3], seq[-1] - seq[-2]
                if d1 != 0 and 0.1 < d2 / d1 < 0.9:
                    new = float(np.clip(seq[-1] + d2 * (d2 / d1) / (1.0 - d2 / d1),
                                        -0.99 * limit, 0.99 * limit))
            history.append(new)
            step = abs(new - offset)
            offset = new
            if step < settings.offset_tol:
                converged = True
                break
            if len(history) >= 3 and abs(history[-1] - history[-3]) < settings.offset_tol:
                # flip-flop between two values closer than the tolerance: stable
                offset = 0.5 * (history[-1] + history[-2])
                converged = True
                break
        estimate = float(offset)

    interior = abs(estimate) < BOUNDARY_FRACTION * limit
    applied = float(fixed_offset) if fixed_offset is not None else (
        float(estimate) if interior else 0.0)
    final_tissue = np.where(lines + applied > MIN_SCALER, tissue[:, None], 0.0)
    multiplier, _ = _solve(
        lines, (final_tissue * mult_levels[None, :, None, None]).astype(np.float32),
        applied)
    score_zero = _offset_score(lines, w_off, multiplier, 0.0)
    score_estimate = _offset_score(lines, w_off, multiplier, estimate)
    rows_used = [int((final_tissue[i].min(axis=0) > 0).sum()) for i in range(n_tiles)]
    return PercentileFit(
        applied, estimate, float("nan"), multiplier, [],
        score_zero, score_estimate, rows_used, n_z * n_y, float(limit), list(levels),
        history, converged, [float(v) for v in mult_levels],
        [float(v) for v in off_levels])


def multiplier_to_scaler(multiplier: np.ndarray) -> np.ndarray:
    """Scaler map for the correction TIFF: the multiplier, 1e-9 where invalid."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ok = np.isfinite(multiplier) & (multiplier > 0)
    return np.where(ok, multiplier, 1e-9).astype(np.float32)