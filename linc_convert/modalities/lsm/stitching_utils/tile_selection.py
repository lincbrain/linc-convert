"""Pick reference tiles for the corrections automatically."""

import warnings
from pathlib import Path
from typing import Dict, List, Union

import dask.array as da
import numpy as np
import zarr

from linc_convert.modalities.lsm.stitching_utils.corrections import (
    MIN_SCALER,
    compute_noise_map,
    subtract_noise,
)

PathLike = Union[str, Path]


def available_layer(path: PathLike, wanted: int) -> int:
    """Return the highest pyramid level <= ``wanted`` that the chunk has.

    Parameters
    ----------
    path : str or pathlib.Path
        Chunk ``.ome.zarr`` folder.
    wanted : int
        Preferred pyramid level.

    Returns
    -------
    int
        ``wanted`` if present, otherwise the next finer level that exists.

    Raises
    ------
    KeyError
        If the chunk has no pyramid level at all.
    """
    group = zarr.open_group(str(path), mode="r")
    for level in range(wanted, -1, -1):
        if str(level) in group:
            return level
    raise KeyError(f"no pyramid level <= {wanted} in {path}")


def tissue_row_fraction(vol: da.Array, background_length: int,
                        threshold_multiplier: float, x_stride: int,
                        min_signal_fraction: float = 0.05,
                        min_signal_count: int = 20,
                        min_snr: float = 5.0) -> float:
    """Fraction of (z, y) rows of a chunk that cross usable tissue.

    A row counts if enough of its sampled columns are above the signal
    threshold (the one ``calibrate_tile`` uses) and its median signal is at
    least ``min_snr`` robust noise standard deviations above the background.
    The count is deliberately low (default 5% of the columns): a chunk spans
    the whole sample width, so a tissue row can be mostly empty space.

    Parameters
    ----------
    vol : dask.array.Array
        Cropped, flipped channel volume, shape (Z, Y, X).
    background_length : int
        Trailing x columns used for the noise floor and its spread.
    threshold_multiplier : float
        Pixels below ``median(noise map) * threshold_multiplier`` are not signal.
    x_stride : int
        Every ``x_stride``-th x column is sampled.
    min_signal_fraction : float, default=0.05
        A row needs at least this fraction of its sampled columns above the
        threshold.
    min_signal_count : int, default=20
        ...and at least this many sampled columns.
    min_snr : float, default=5.0
        Required median signal in robust noise standard deviations.

    Returns
    -------
    float
        Fraction of rows with tissue, 0-1.
    """
    vol = vol.astype(np.float32)
    noise = compute_noise_map(vol, background_length)
    threshold = float(np.median(noise))
    edge = vol[:, :, -background_length:]
    noise_sd = 1.4826 * np.asarray(
        da.nanmedian(da.fabs(edge - noise[:, :, None]), axis=2).compute())
    signal = da.where((vol >= threshold * threshold_multiplier) & da.isfinite(vol),
                      subtract_noise(vol, noise), np.nan)
    sampled = np.asarray(signal[:, :, ::x_stride].compute())
    count = np.isfinite(sampled).sum(axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(sampled, axis=2)
    needed = max(min_signal_count, min_signal_fraction * sampled.shape[2])
    rows = (np.isfinite(median) & (median > MIN_SCALER) & (count >= needed)
            & (median >= min_snr * noise_sd))
    return float(rows.mean())


def choose_reference_tiles(fractions: Dict[int, float], count: int = 8,
                           min_tissue: float = 0.9, edge_margin: int = 3
                           ) -> List[int]:
    """Choose evenly spread reference tiles from the tissue-rich middle.

    Tiles with at least ``min_tissue`` of their rows in tissue are eligible.
    The first and last ``edge_margin`` eligible tiles are dropped (the slice
    edges have partial tissue and unusually strong stripes), and ``count``
    tiles are taken evenly across what is left: neighbouring tiles share
    anatomy, so widely spaced tiles average it out best.

    Parameters
    ----------
    fractions : dict of int to float
        Tissue-row fraction of every tile, keyed by 1-based tile number.
    count : int, default=8
        Number of tiles to pick. The stripes of one camera are explained by
        about 4-8 tiles; more adds nothing.
    min_tissue : float, default=0.9
        Minimum tissue-row fraction of an eligible tile.
    edge_margin : int, default=3
        Eligible tiles skipped at each end of the tissue run. Ignored (with the
        edge tiles kept) if fewer than ``2 * edge_margin + 1`` tiles are eligible.

    Returns
    -------
    list of int
        Sorted 1-based tile numbers; fewer than ``count`` if fewer are eligible.

    Raises
    ------
    ValueError
        If no tile reaches ``min_tissue``.
    """
    good = sorted(t for t, f in fractions.items() if f >= min_tissue)
    if not good:
        best = max(fractions, key=fractions.get)
        raise ValueError(
            f"no tile has >= {min_tissue:.0%} of its rows in tissue (best: tile "
            f"{best} with {fractions[best]:.0%}); lower min_tissue_fraction or "
            "pass reference_tiles")
    middle = good[edge_margin:len(good) - edge_margin]
    if len(good) <= 2 * edge_margin or not middle:
        middle = good
    if len(middle) <= count:
        return middle
    picks = np.round(np.linspace(0, len(middle) - 1, count)).astype(int)
    return [middle[i] for i in sorted(set(picks.tolist()))]
