"""Background, scalar, and sheer correction."""

import math
import os
import warnings
from typing import Dict, Optional, Tuple, Union

import dask.array as da
import numpy as np
import tifffile
from dask_image.ndinterp import affine_transform

X_STRIDE = 64            # every 64th x column feeds the per-row scaler median
MIN_SCALER = 1e-3        # a row whose median signal is <= this is invalid

Array = Union[np.ndarray, da.Array]


def compute_noise_map(vol: da.Array, background_length: int) -> np.ndarray:
    """Estimate the background of every (z, y) row."""
    edge = vol[:, :, -background_length:].astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        noise = np.asarray(da.nanmedian(edge, axis=2).compute())
    return np.nan_to_num(noise, nan=0.0)


def subtract_noise(vol: Array, noise: np.ndarray) -> Array:
    """Subtract the background from every row, clipping at 0."""
    xp = da if isinstance(vol, da.Array) else np
    return xp.clip(vol - noise[:, :, None], 0, None)


def apply_corrections(vol: da.Array, noise: np.ndarray,
                      scaler: np.ndarray) -> da.Array:
    """Apply ``(vol - noise).clip(0) * scaler`` and cast to uint16."""
    block = subtract_noise(vol.astype(np.float32), noise) * scaler[:, :, None]
    return da.clip(block, 0, 65535).astype(np.uint16)


def keep_columns(columns: np.ndarray, threshold: float,
                 max_below: float) -> np.ndarray:
    """Which x columns are used for every row of a tile.

    A pixel is "left out" (masked) if it is below ``threshold`` (or not finite).
    Judged per pixel, a dim row loses more of its tissue pixels than a bright one
    (same threshold, less signal), so the rows would be measured on different
    pixels. Judged per column instead: if more than ``max_below`` of all the rows
    of the tile are masked at an x column, that whole column is left out for every
    row; otherwise the whole column is kept for every row, whatever a single row's
    pixel there looks like. Every row is then measured on the same columns.

    Parameters
    ----------
    columns : numpy.ndarray
        Raw values, shape (Z, Y, n_columns).
    threshold : float
        Raw value below which a pixel is masked.
    max_below : float
        Largest share (0-1) of rows that may be masked in a kept column.

    Returns
    -------
    numpy.ndarray
        Boolean (n_columns,): True for the kept columns.
    """
    ok = np.isfinite(columns) & (columns >= threshold)
    below = 1.0 - ok.reshape(-1, columns.shape[2]).mean(axis=0)
    return below <= max_below


def calibrate_tile(vol: da.Array, background_length: int,
                   threshold_multiplier: float,
                   x_stride: int = X_STRIDE,
                   column_mask_fraction: Optional[float] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the noise map and zy scaler map for one chunk and channel.

    Steps: noise map (``compute_noise_map``); threshold = median of the noise
    map; a pixel is signal only if its raw value is at least
    ``threshold * threshold_multiplier``; raw scaler = median over x (every
    ``x_stride``-th column) of the noise-subtracted signal pixels, invalid
    where it is ``<= MIN_SCALER`` or undefined; scaler =
    ``1 / (raw scaler / median(raw scaler))``, with 1e-9 for invalid rows.

    Parameters
    ----------
    vol : dask.array.Array
        Cropped (and flipped) channel volume, shape (Z, Y, X).
    background_length : int
        Number of trailing x columns used for each row's noise floor.
    threshold_multiplier : float
        See above.
    x_stride : int, default=X_STRIDE
        Every ``x_stride``-th x column feeds the per-row scaler median (use a
        smaller value on lower-resolution levels).
    column_mask_fraction : float, optional
        If given, the columns used are the same for every row: an x column is
        left out for all rows if more than this share of the rows are below the
        signal threshold there, and kept for all rows otherwise (``keep_columns``);
        the row median is then taken over all kept columns. None keeps the pixel
        by pixel threshold.

    Returns
    -------
    noise_map : numpy.ndarray
        Background noise per (z, y) row, shape (Z, Y).
    scaler_map : numpy.ndarray
        Multiplier per (z, y) row, shape (Z, Y). Rows without usable signal
        are 1e-9.
    """
    vol = vol.astype(np.float32)
    noise_map = compute_noise_map(vol, background_length)

    threshold = float(np.median(noise_map))
    if column_mask_fraction is not None:
        raw = np.asarray(vol[:, :, ::x_stride].compute(), dtype=np.float32)
        keep = keep_columns(raw, threshold * threshold_multiplier,
                            column_mask_fraction)
        print(f"  columns used for every row: {int(keep.sum())} of {keep.size}",
              flush=True)
        denoised = np.where(np.isfinite(raw),
                            np.clip(raw - noise_map[:, :, None], 0, None), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            raw_scaler = (np.nanmedian(denoised[:, :, keep], axis=2) if keep.any()
                          else np.full(noise_map.shape, np.nan))
    else:
        denoised = subtract_noise(vol, noise_map)
        signal = da.where(
            (vol >= threshold * threshold_multiplier) & da.isfinite(vol),
            denoised, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            raw_scaler = np.asarray(
                da.nanmedian(signal[:, :, ::x_stride], axis=2).compute())

    # a row whose median signal is ~0 is as unusable as one with no data
    raw_scaler = np.where(np.isfinite(raw_scaler) & (raw_scaler > MIN_SCALER),
                          raw_scaler, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        target = np.nanmedian(raw_scaler)
    if not np.isfinite(target) or target == 0:
        target = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        reciprocal = 1.0 / (raw_scaler / target)
    return noise_map, np.nan_to_num(reciprocal, nan=1e-9)


def correction_paths(corrections_dir: str, channel: str,
                     chunk: Optional[int] = None) -> Dict[str, str]:
    """Locate the correction TIFFs of a channel.

    Parameters
    ----------
    corrections_dir : str
        Folder written by ``create_corrections``.
    channel : str
        Channel name.
    chunk : int, optional
        1-based chunk number for per-chunk corrections
        (``{ch}_chunk0003_alt_zy_*.tiff``); None for the shared maps.

    Returns
    -------
    dict of str to str
        Paths under the keys ``"noise"`` and ``"scaler"``.
    """
    folder = os.path.join(corrections_dir, channel)
    stem = channel if chunk is None else f"{channel}_chunk{chunk:04d}"
    return {"noise": os.path.join(folder, f"{stem}_alt_zy_noise.tiff"),
            "scaler": os.path.join(folder, f"{stem}_alt_zy_scaler.tiff")}


def load_corrections(
        corrections_dir: str,
        channel: str,
        chunk: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the noise and scaler maps made by ``create_corrections``.

    ``chunk`` is the 1-based chunk number for per-chunk maps (None = shared).
    """
    maps = []
    for path in correction_paths(corrections_dir, channel, chunk).values():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing correction file: {path}")
        arr = tifffile.imread(path).astype(np.float32)
        maps.append(arr)
    return maps[0], maps[1]


def refined_paths(corrections_dir: str, channel: str) -> Dict[str, str]:
    """Files written by ``refine_multipliers``."""
    folder = os.path.join(corrections_dir, channel)
    return {"background": os.path.join(folder, f"{channel}_background_subtraction.tiff"),
            "multiplier": os.path.join(folder, f"{channel}_multiplier_correction.tiff"),
            "noise": os.path.join(folder, f"{channel}_multiplier_noise.tiff")}


def load_refined_corrections(corrections_dir: str, channel: str
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Background map and scaler from the three files of ``refine_multipliers``.

    The scaler is ``multiplier_correction * multiplier_noise`` (about 0 where the
    multiplier noise marks the row invalid, so the blend gives it no weight).
    """
    maps = {}
    for key, path in refined_paths(corrections_dir, channel).items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing refined correction file: {path} (run refine_multipliers)")
        maps[key] = tifffile.imread(path).astype(np.float32)
    return maps["background"], maps["multiplier"] * maps["noise"]


def sheared_shape(shape: Tuple[int, int, int], shear_x: float,
                  shear_z: float) -> Tuple[int, int, int]:
    """Shape of a chunk after the x-by-z and z-by-y shears."""
    sz, sy, sx = shape
    return (sz + int(math.ceil(shear_z * (sy - 1))), sy,
            sx + int(math.ceil(shear_x * (sz - 1))))


def shear_volume(vol: da.Array, shear_x: float, shear_z: float) -> da.Array:
    """Lazily shear a chunk (nearest neighbour).

    Both shears use the *unsheared* coordinates: ``x' = x + shear_x * z`` and
    ``z' = z + shear_z * y``.

    Parameters
    ----------
    vol : dask.array.Array
        Chunk, shape (Z, Y, X).
    shear_x, shear_z : float
        See ``sheared_shape``.

    Returns
    -------
    dask.array.Array
        Sheared array of shape ``sheared_shape(vol.shape, ...)``; ``vol``
        itself if both shears are 0.
    """
    out_shape = sheared_shape(vol.shape, shear_x, shear_z)
    if out_shape == tuple(vol.shape):
        return vol
    forward = np.array([[1, shear_z, 0], [0, 1, 0], [
                       shear_x, 0, 1]], dtype=float)
    return affine_transform(vol, matrix=np.linalg.inv(forward), offset=0, order=0,
                            mode="constant", cval=0.0, output_shape=out_shape,
                            output_chunks=vol.chunksize)