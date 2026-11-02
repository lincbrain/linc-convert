"""Fit one shared background offset together with the per-row zy multipliers.

The usual correction is ``(vol - noise).clip(0) * scaler``, with ``noise`` the
row background measured in the empty trailing columns. Here the subtracted
background is ``noise - offset`` instead, with a single ``offset`` for every row:

    corrected = (value - (background - offset)) * multiplier

``background`` and ``value`` are measured, so only ``offset`` (one number) and
``multiplier`` (one per (z, y) row) are unknown. They are found by least squares.
A positive ``offset`` subtracts *less* than the background, a negative one more.

Why brightness levels, not single x slices
------------------------------------------
A (z, y) row has one pixel in each x slice, so a few slices cannot fix its
multiplier, and comparing single pixels with their neighbours is biased: the
neighbours only partly predict the pixel (regression dilution), which looks
exactly like an offset. Instead every row is summarised by its *brightness
curve*: the ``n_levels`` quantiles of its background-subtracted values over x.
The stripe acts on a curve as ``value = gain * T + offset_true`` for the true
curve ``T``, so curves of neighbouring rows differ by a gain (and an offset),
and the quantile levels play the part of the "several slices": same level, same
rank, no noisy pixel-to-pixel matching. Levels are chosen from the tile's
tissue fraction and are the same for every row, so no row is selected by its own
values (a threshold on the row's own values would favour rows with a high gain).

Offset
------
A reference curve ``r`` for a row is the mean of the curves of the rows around
it (``|dz|, |dy| <= reach``) leaving out its own row and column band
(``|dz| <= gap`` or ``|dy| <= gap``), so a stripe in y or in z never reaches its
own reference. If the tissue sits on a constant baseline, ``value = gain * T -
offset``, a row and its reference obey ``value = gamma * r + c`` with
``c = offset * (gamma - 1)``: a row whose gain is 5% above its neighbours sits
``0.05 * offset`` above the diagonal. The offset is therefore the slope of
``c`` against ``gamma - 1``.

``gamma`` and ``c`` of one row are noisy and anatomy also moves them (a row with
brighter tissue has a larger ``gamma``, and the shape change shifts ``c`` too),
which would bias a plain regression. Stripes are the same in every tile and
anatomy is not, so ``gamma - 1`` is *instrumented* with the median ``gamma`` of
the other tiles (a two-stage / instrumental-variable fit): the offset is
``sum(c * (z - 1)) / sum((gamma - 1) * (z - 1))`` over rows and tiles, with ``z``
the other tiles' consensus. This needs at least two tiles (three for a standard
error, which is a delete-one-tile jackknife). The reference of this step is built
from the raw curves; rebuilding it from the corrected curves would make the
offset a free direction (any offset reproduces itself).

Multipliers
-----------
With the offset fixed, the reference is rebuilt from the corrected curves
(``(value + offset).clip(0) * multiplier``) a few times; each row's
``multiplier = 1 / gamma`` follows from the regression through the origin of
``value + offset`` on the reference, normalised to median 1 per tile.

The offset only becomes visible through the stripes themselves, so it is
determined well only when the stripes are clearly above the noise and with many
rows and tiles. ``OffsetFit.offset_se`` and ``tile_offsets`` show how well; an
offset that is not significantly different from 0 is set to 0.
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import dask.array as da
import numpy as np
import scipy.ndimage

from linc_convert.modalities.lsm.stitching_utils.corrections import (
    MIN_SCALER,
    compute_noise_map,
)

HUBER_K = 1.345          # Huber tuning constant (in robust standard deviations)
MIN_RING_FRACTION = 0.3  # share of the reference ring that must hold data
LEVEL_RANGE = (0.25, 0.95)   # span of the levels inside the tissue distribution
BLOCK_ELEMENTS = 2 * 10 ** 7  # values read at a time while measuring the curves


@dataclass
class OffsetSettings:
    """Tuning of the offset fit (lengths are full-resolution pixels).

    Attributes
    ----------
    n_levels : int
        Brightness levels (quantiles) per row.
    iterations : int
        Times the reference is rebuilt from the latest correction.
    reach : int
        Half-width of the reference neighbourhood in z and y.
    gap : int
        Half-width of the excluded row/column band around a row; at least a
        stripe width.
    max_columns : int
        At most this many x columns per row are used (evenly strided).
    min_levels : int, optional
        Valid levels a row needs to get a multiplier (default: half of them).
    max_offset_fraction : float
        The offset is limited to this fraction of the typical signal.
    min_significance : float
        The offset is set to 0 unless ``|offset| / standard error`` reaches this
        (0 disables the check).
    """

    n_levels: int = 12
    iterations: int = 4
    reach: int = 64
    gap: int = 8
    max_columns: int = 512
    min_levels: Optional[int] = None
    max_offset_fraction: float = 0.5
    min_significance: float = 2.0

    def at_layer(self, layer: int) -> Tuple[int, int]:
        """Return ``(reach, gap)`` in pixels of pyramid level ``layer``."""
        scale = 2 ** layer
        reach = max(3, self.reach // scale)
        gap = min(max(1, self.gap // scale), reach - 1)
        return reach, gap


@dataclass
class TileSample:
    """The brightness curves of one tile.

    Attributes
    ----------
    noise_map : numpy.ndarray
        Row background, shape (Z, Y).
    signal : numpy.ndarray
        Background-subtracted quantile of every row at every level, shape
        (n_levels, Z, Y), unclipped.
    probabilities : numpy.ndarray
        Quantile probability of each level.
    tissue_fraction : float
        Share of the tile's values above the signal threshold.
    min_ref : float
        Smallest reference value that counts as tissue.
    """

    noise_map: np.ndarray
    signal: np.ndarray
    probabilities: np.ndarray
    tissue_fraction: float
    min_ref: float


@dataclass
class OffsetFit:
    """Result of ``fit_offset``.

    Attributes
    ----------
    offset : float
        The shared offset that is applied (positive subtracts less; 0 if the
        estimate was not significant).
    estimate : float
        The estimate before the significance check.
    standard_error : float
        Delete-one-tile jackknife standard error of the estimate (NaN with fewer
        than 3 tiles). An estimate without a standard error is not applied
        unless the significance check is disabled.
    tile_offsets : list of float
        The estimate of each tile (with the others as instrument); only for
        information, as the tiles share their instrument.
    multipliers : list of numpy.ndarray
        Per tile, shape (Z, Y), median 1 over valid rows, NaN for rows without
        enough signal.
    rows_used, rows_total : list of int
        Per tile, rows that got a multiplier and all rows.
    """

    offset: float
    estimate: float
    standard_error: float
    tile_offsets: List[float]
    multipliers: List[np.ndarray]
    rows_used: List[int] = field(default_factory=list)
    rows_total: List[int] = field(default_factory=list)


def _brightness_levels(tissue_fraction: float, n_levels: int) -> np.ndarray:
    """Quantile probabilities: a fixed span of the top ``tissue_fraction``."""
    inside = np.linspace(LEVEL_RANGE[0], LEVEL_RANGE[1], n_levels)
    return 1.0 - tissue_fraction * (1.0 - inside)


def sample_tile(vol: da.Array, background_length: int, threshold_multiplier: float,
                settings: OffsetSettings) -> TileSample:
    """Measure the row background and every row's brightness curve.

    Parameters
    ----------
    vol : dask.array.Array
        Cropped (and flipped) channel volume, shape (Z, Y, X).
    background_length : int
        Trailing x columns used for each row's background (they are left out of
        the curves).
    threshold_multiplier : float
        Signal threshold multiplier; ``median(noise_map) * multiplier`` marks
        tissue when the tile's tissue fraction is measured, and
        ``median(noise_map) * (multiplier - 1)`` is the smallest reference value
        used.
    settings : OffsetSettings
        Fit settings.

    Returns
    -------
    TileSample
        With NaN signal if the tile has no tissue at all.
    """
    vol = vol.astype(np.float32)
    n_z, n_y, n_x = vol.shape
    noise_map = compute_noise_map(vol, background_length)
    threshold = float(np.median(noise_map))
    usable = max(n_x - background_length, 1)
    step = max(1, usable // settings.max_columns)
    columns = vol[:, :, :usable:step]
    min_ref = max(threshold * (threshold_multiplier - 1.0), MIN_SCALER)

    # tissue fraction from a thin subsample of the rows
    coarse = np.asarray(columns[::8, ::8].compute(), dtype=np.float32)
    per_row = (coarse >= threshold * threshold_multiplier).mean(axis=2)
    present = per_row[per_row >= 0.02]
    if present.size == 0:
        signal = np.full((settings.n_levels, n_z, n_y), np.nan, dtype=np.float32)
        return TileSample(noise_map, signal, np.full(settings.n_levels, np.nan),
                          0.0, float(min_ref))
    fraction = float(np.clip(np.median(present), 0.02, 0.95))
    levels = _brightness_levels(fraction, settings.n_levels)

    signal = np.empty((settings.n_levels, n_z, n_y), dtype=np.float32)
    per_slab = max(1, BLOCK_ELEMENTS // max(1, n_y * columns.shape[2]))
    for z0 in range(0, n_z, per_slab):
        z1 = min(z0 + per_slab, n_z)
        block = np.asarray(columns[z0:z1].compute(), dtype=np.float32)
        block -= noise_map[z0:z1, :, None]
        signal[:, z0:z1] = np.quantile(block, levels, axis=2)
    return TileSample(noise_map, signal, levels, fraction, float(min_ref))


def cross_excluded_mean(values: np.ndarray, reach: int, gap: int) -> np.ndarray:
    """Mean of the values around each cell, leaving out its row and column band.

    The neighbourhood of cell ``(z, y)`` is ``|dz| <= reach, |dy| <= reach``
    without ``|dz| <= gap`` and without ``|dy| <= gap``. NaN values are ignored.

    Parameters
    ----------
    values : numpy.ndarray
        2-D array (z, y).
    reach, gap : int
        See above; ``reach > gap``.

    Returns
    -------
    numpy.ndarray
        float64 means; NaN where fewer than ``MIN_RING_FRACTION`` of the
        neighbourhood holds data.
    """
    if reach <= gap:
        raise ValueError(f"reach ({reach}) must exceed gap ({gap})")
    valid = np.isfinite(values)
    total_in = np.where(valid, values, 0.0).astype(np.float64)
    count_in = valid.astype(np.float64)

    def box(a: np.ndarray, hz: int, hy: int) -> np.ndarray:
        size = (2 * hz + 1, 2 * hy + 1)
        return scipy.ndimage.uniform_filter(
            a, size=size, mode="constant", cval=0.0) * (size[0] * size[1])

    def ring(a: np.ndarray) -> np.ndarray:
        return (box(a, reach, reach) - box(a, gap, reach)
                - box(a, reach, gap) + box(a, gap, gap))

    total, count = ring(total_in), ring(count_in)
    full = (2 * (reach - gap)) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    return np.where(count >= MIN_RING_FRACTION * full, mean, np.nan)


def _raw_reference(sample: TileSample, reach: int, gap: int) -> np.ndarray:
    """Reference curves from the uncorrected curves, shape (levels, Z, Y)."""
    ref = np.empty(sample.signal.shape, dtype=np.float32)
    for k in range(ref.shape[0]):
        ref[k] = cross_excluded_mean(sample.signal[k].astype(np.float64), reach, gap)
    return ref


def _sums(signal: np.ndarray, ref: np.ndarray, ok: np.ndarray,
          weight: np.ndarray) -> np.ndarray:
    """Per-row weighted sums ``n, sum d, sum d^2, sum r, sum r^2, sum r*d``."""
    w = np.where(ok, weight, 0.0)
    d = np.where(ok, signal, 0.0).astype(np.float64)
    r = np.where(ok, ref, 0.0).astype(np.float64)
    return np.stack([w.sum(0), (w * d).sum(0), (w * d * d).sum(0),
                     (w * r).sum(0), (w * r * r).sum(0), (w * r * d).sum(0)])


def _line_fits(signal: np.ndarray, ref: np.ndarray, ok: np.ndarray,
               weight: np.ndarray, min_levels: int
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted line ``value = gamma * ref + c`` of every row (NaN if unusable)."""
    n, s_d, _, s_r, s_rr, s_rd = _sums(signal, ref, ok, weight)
    spread = n * s_rr - s_r * s_r
    good = (n >= min_levels) & (spread > 1e-6 * s_r * s_r)
    safe = np.where(good, spread, 1.0)
    gamma = np.where(good, (n * s_rd - s_r * s_d) / safe, np.nan)
    gamma = np.where((gamma > 0.2) & (gamma < 5.0), gamma, np.nan)
    c = (s_d - gamma * s_r) / np.maximum(n, 1e-12)
    return gamma, c


def _huber_weights(signal: np.ndarray, ref: np.ndarray, ok: np.ndarray,
                   predicted: np.ndarray, floor: float) -> np.ndarray:
    """Huber weights of one level from the relative residual of a fit."""
    rel = np.where(ok, (signal - predicted) / np.maximum(np.abs(predicted), floor),
                   0.0)
    w = np.ones_like(rel)
    if ok.any():
        centre = np.median(rel[ok])
        tau = HUBER_K * 1.4826 * np.median(np.abs(rel[ok] - centre))
        if tau > 0:
            mag = np.maximum(np.abs(rel), 1e-12)
            w = np.where(mag <= tau, 1.0, tau / mag)
    return w


def estimate_offset(samples: List[TileSample], reach: int, gap: int,
                    min_levels: int, max_offset_fraction: float
                    ) -> Tuple[float, float, List[float]]:
    """Estimate the shared offset from the raw curves (see the module docstring).

    Parameters
    ----------
    samples : list of TileSample
        One per reference tile; at least two.
    reach, gap : int
        Reference neighbourhood in pixels of the tiles' level.
    min_levels : int
        Weighted number of valid levels a row needs.
    max_offset_fraction : float
        Limit on ``|offset|`` as a fraction of the typical signal.

    Returns
    -------
    estimate : float
        Shared offset (0 if the stripes carry no usable information).
    standard_error : float
        From the spread of the per-tile estimates.
    tile_offsets : list of float
        Estimate of each tile with the others as instrument.

    Raises
    ------
    ValueError
        With fewer than two tiles.
    """
    n_tiles = len(samples)
    if n_tiles < 2:
        raise ValueError("estimating the offset needs at least two reference "
                         "tiles (stripes are told from anatomy by repeating "
                         "across tiles)")
    refs = [_raw_reference(s, reach, gap) for s in samples]
    oks = [np.isfinite(r) & (r >= s.min_ref) & np.isfinite(s.signal)
           for s, r in zip(samples, refs)]
    weights = [np.ones(s.signal.shape) for s in samples]
    for _ in range(3):            # fit the lines, then reweight with the residuals
        gammas, cs = [], []
        for s, r, ok, w in zip(samples, refs, oks, weights):
            g, c = _line_fits(s.signal, r, ok, w, min_levels)
            gammas.append(g)
            cs.append(c)
        weights = [np.stack([_huber_weights(
            s.signal[k], r[k], ok[k], g * r[k] + c, s.min_ref)
            for k in range(r.shape[0])])
            for s, r, ok, g, c in zip(samples, refs, oks, gammas, cs)]
    gammas, cs = [], []
    for s, r, ok, w in zip(samples, refs, oks, weights):
        g, c = _line_fits(s.signal, r, ok, w, min_levels)
        gammas.append(g)
        cs.append(c)

    # rows near the border have a lopsided reference, the same in every tile, which
    # would pass for a stripe
    n_z, n_y = gammas[0].shape
    interior = np.zeros((n_z, n_y), dtype=bool)
    interior[reach:max(n_z - reach, reach), reach:max(n_y - reach, reach)] = True
    with_data = [s.signal[np.isfinite(s.signal)] for s in samples]
    with_data = [v for v in with_data if v.size]
    if not with_data:
        return 0.0, float("nan"), [float("nan")] * n_tiles
    typical = float(np.median(with_data[0]))
    limit = max(max_offset_fraction * abs(typical), 1.0)

    def pooled(members: List[int]) -> Tuple[float, List[float]]:
        """IV estimate over ``members`` (tile indices) and each member's own."""
        stack = np.stack([gammas[t] for t in members])
        nums, dens = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for i, t in enumerate(members):
                z = np.nanmedian(np.delete(stack, i, axis=0), axis=0)
                use = (np.isfinite(gammas[t]) & np.isfinite(z)
                       & np.isfinite(cs[t]) & interior)
                nums.append(float(np.sum((cs[t] * (z - 1.0))[use])))
                dens.append(float(np.sum(((gammas[t] - 1.0) * (z - 1.0))[use])))
        total = sum(dens)
        value = float(np.clip(sum(nums) / total, -limit, limit)) if total > 0 \
            else float("nan")
        return value, [n / d if d > 0 else float("nan") for n, d in zip(nums, dens)]

    estimate, tile_offsets = pooled(list(range(n_tiles)))
    if not np.isfinite(estimate):
        return 0.0, float("nan"), tile_offsets
    # delete-one-tile jackknife: the tiles share the instrument, so their own
    # spread would understate the uncertainty
    se = float("nan")
    if n_tiles >= 3:
        leave = np.array([pooled([t for t in range(n_tiles) if t != j])[0]
                          for j in range(n_tiles)])
        if np.isfinite(leave).all():
            se = float(np.sqrt((n_tiles - 1) / n_tiles
                               * np.sum((leave - leave.mean()) ** 2)))
    return estimate, se, tile_offsets


def _row_gains(sample: TileSample, multiplier: np.ndarray, offset: float,
               reach: int, gap: int, min_levels: int) -> np.ndarray:
    """One refinement of the multipliers at a fixed offset (NaN if invalid)."""
    n_levels, n_z, n_y = sample.signal.shape
    gain = np.where(np.isfinite(multiplier), 1.0 / multiplier, 1.0)
    num, den, count = (np.zeros((n_z, n_y)) for _ in range(3))
    for k in range(n_levels):
        d = sample.signal[k].astype(np.float64)
        ref = cross_excluded_mean(np.clip(d + offset, 0, None) * multiplier,
                                  reach, gap)
        ok = np.isfinite(ref) & (ref >= sample.min_ref) & np.isfinite(d)
        predicted = gain * np.where(ok, ref, 1.0)
        w = np.where(ok, _huber_weights(d + offset, ref, ok, predicted,
                                        sample.min_ref), 0.0)
        r0 = np.where(ok, ref, 0.0)
        num += w * r0 * np.where(ok, d + offset, 0.0)
        den += w * r0 * r0
        count += w
    valid = (count >= min_levels) & (den > 0)
    gamma = np.where(valid, num / np.where(den > 0, den, 1.0), np.nan)
    return np.where(gamma > 0, gamma, np.nan)


def fit_offset(samples: List[TileSample], reach: int, gap: int, iterations: int = 4,
               fixed_offset: Optional[float] = None,
               min_levels: Optional[int] = None,
               max_offset_fraction: float = 0.5,
               min_significance: float = 2.0) -> OffsetFit:
    """Fit the shared offset and every tile's per-row multipliers.

    Parameters
    ----------
    samples : list of TileSample
        One per reference tile (all of the same channel).
    reach, gap : int
        Reference neighbourhood, in pixels of the tiles' level
        (``OffsetSettings.at_layer``).
    iterations : int, default=4
        Number of times the reference is rebuilt for the multipliers.
    fixed_offset : float, optional
        Use this offset instead of estimating one.
    min_levels : int, optional
        Weighted number of valid levels a row needs (default: half of them).
    max_offset_fraction : float, default=0.5
        Limit on ``|offset|`` as a fraction of the typical signal.
    min_significance : float, default=2.0
        The offset is set to 0 unless ``|estimate| / standard error`` reaches
        this (0 disables the check).

    Returns
    -------
    OffsetFit
    """
    n_levels = samples[0].signal.shape[0]
    needed = max(1, n_levels // 2) if min_levels is None else min_levels
    if fixed_offset is None:
        estimate, se, tile_offsets = estimate_offset(
            samples, reach, gap, needed, max_offset_fraction)
        significant = (min_significance <= 0 and np.isfinite(estimate)) or (
            np.isfinite(se) and abs(estimate) >= min_significance * se)
        offset = estimate if significant else 0.0
    else:
        estimate = offset = float(fixed_offset)
        se, tile_offsets = float("nan"), []

    multipliers = [np.ones(s.signal.shape[1:]) for s in samples]
    for _ in range(iterations):
        new = []
        for s, m in zip(samples, multipliers):
            gamma = _row_gains(s, m, offset, reach, gap, needed)
            with np.errstate(invalid="ignore", divide="ignore"):
                inv = 1.0 / gamma
            new.append(inv / np.nanmedian(inv) if np.isfinite(inv).any() else inv)
        multipliers = new
    return OffsetFit(
        offset, estimate, se, tile_offsets, multipliers,
        [int(np.isfinite(m).sum()) for m in multipliers],
        [int(m.size) for m in multipliers])


def multiplier_to_scaler(multiplier: np.ndarray) -> np.ndarray:
    """Turn a multiplier map into a scaler map (1e-9 marks invalid rows)."""
    return np.nan_to_num(multiplier, nan=1e-9).astype(np.float32)