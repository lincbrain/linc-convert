from typing import Dict, List, Optional

import dask
import dask.array as da
import numpy as np
from dask_image.ndinterp import affine_transform
from scipy.ndimage import convolve

# ---------------------------------------------------------------------
# Intensity correction
# ---------------------------------------------------------------------


def _corr_zy_postprocess(
    corr: np.ndarray,
    counts: np.ndarray,
    min_pixels: int,
    kernel_size: int,
) -> np.ndarray:
    """
    Numpy-only post-processing for the (Z, Y) correction map.

    This is the part of the correction-map computation that can't be
    expressed as plain dask-array operations: boolean-indexed in-place
    assignment, a scalar fallback derived from a global nanmedian, and
    2D convolution via `scipy.ndimage`. It's called through
    `dask.delayed` on the *lazy* `corr`/`counts` dask arrays (see
    `compute_corr_zy_from_pixel_mask`), so it becomes one node in the
    same task graph as the rest of the correction pipeline rather than
    a separate eager step -- this lets the caller compute the
    correction map and the corrected volume together, in a single pass
    over the source data, instead of two independent passes.

    Parameters
    ----------
    corr : np.ndarray
        (Z, Y) median intensity per row (already collapsed over X).
    counts : np.ndarray
        (Z, Y) count of valid (finite, above-threshold) pixels per row.
    min_pixels : int
        Minimum number of valid pixels required per row (rows below
        this are treated as low-quality and replaced with a fallback).
    kernel_size : int
        Size of the smoothing kernel.

    Returns
    -------
    np.ndarray
        Smoothed (Z, Y) correction map, scaled by 1/1000, float32.
    """
    corr = corr.copy()

    # -------------------------
    # Remove low-quality rows
    # -------------------------
    corr[counts < min_pixels] = np.nan

    valid_per_z = np.sum(np.isfinite(corr), axis=1)
    bad_rows = valid_per_z < 10

    fallback = np.nanmedian(corr)
    if not np.isfinite(fallback):
        fallback = 1.0

    corr[bad_rows] = fallback

    # -------------------------
    # 2D smoothing
    # -------------------------
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)

    valid = np.isfinite(corr)
    corr_filled = np.nan_to_num(corr, nan=0)

    num = convolve(corr_filled, kernel, mode="nearest")
    den = convolve(valid.astype(np.float32), kernel, mode="nearest")

    corr_smooth = num / (den + 1e-6)

    # Preserve invalid regions
    corr_smooth[~valid] = 99999999.0

    return (corr_smooth / 1000).astype(np.float32)


def compute_corr_zy_from_pixel_mask(
    vol: da.Array,
    mask: np.ndarray,
    tissue_frac_min: float,
    threshold: float,
    kernel_size: int = 5,
) -> da.Array:
    """
    Compute a (Z, Y) intensity correction map from a 3D volume, lazily.

    The workflow:
    1. Mask invalid pixels
    2. Remove low-intensity values
    3. Collapse along X using median
    4. Filter low-quality rows
    5. Apply 2D smoothing

    This stays entirely lazy: it returns a dask array, built from a
    `dask.delayed` node wrapping the small amount of numpy-only
    post-processing (boolean masking, the global-fallback nanmedian, and
    `scipy.ndimage.convolve`, none of which have a natural dask-array
    form). Nothing is read from `vol` until the caller actually calls
    `.compute()` -- and if the caller computes this correction map
    together with other outputs derived from the same `vol` (as
    `stripe_skew_corr` does), dask's scheduler shares the upstream
    read/mask/reduction work between them instead of repeating it.

    Parameters
    ----------
    vol : dask.array.Array
        Input volume with shape (Z, Y, X).
    mask : np.ndarray
        Binary mask of shape (Y, X) or (Z, Y, X).
    tissue_frac_min : float
        Minimum fraction of valid pixels required per row.
    threshold : float
        Intensity threshold below which values are ignored.
    kernel_size : int, default=5
        Size of smoothing kernel.

    Returns
    -------
    dask.array.Array
        Lazy correction map of shape (Z, Y), float32.
    """
    vol = vol.astype(np.float32)
    Z, Y, X = vol.shape

    # -------------------------
    # Broadcast mask
    # -------------------------
    mask_da = da.from_array(mask, chunks=vol.chunks[1:])

    if mask.shape == (Y, X):
        mask_da = da.broadcast_to(mask_da[None], (Z, Y, X))
    elif mask.shape != (Z, Y, X):
        raise ValueError(
            f"mask shape {mask.shape} != volume shape {(Z, Y, X)}")

    # -------------------------
    # Apply mask + threshold
    # -------------------------
    masked = da.where(mask_da, vol, np.nan)
    masked = da.where((masked < threshold) | ~
                      da.isfinite(masked), np.nan, masked)

    # Collapse along X -- stays lazy.
    corr = da.nanmedian(masked[:, :, ::8], axis=2)
    corr = da.where((corr < threshold*1.2), threshold*1.2, corr)
    counts = da.sum(da.isfinite(masked), axis=2)

    min_pixels = int(tissue_frac_min * X)

    corr_smooth = dask.delayed(_corr_zy_postprocess)(
        corr, counts, min_pixels, kernel_size
    )

    return da.from_delayed(corr_smooth, shape=(Z, Y), dtype=np.float32)


def apply_corr_zy_lazy(
    vol: da.Array,
    corr_zy: "da.Array | np.ndarray",
    eps: float = 1e-6,
) -> da.Array:
    """
    Apply a (Z, Y) correction map lazily to a volume.

    Parameters
    ----------
    vol : dask.array.Array
        Input volume (Z, Y, X).
    corr_zy : dask.array.Array or np.ndarray
        Correction map (Z, Y). May be a lazy dask array (the common case)
        or an already-computed numpy array.
    eps : float, default=1e-6
        Small value to avoid division by zero.

    Returns
    -------
    dask.array.Array
        Corrected volume (uint16).
    """
    vol = vol.astype(np.float32)

    if isinstance(corr_zy, np.ndarray):
        corr_da = da.from_array(corr_zy.astype(np.float32))
    else:
        corr_da = corr_zy.astype(np.float32)
    corr_da = corr_da[:, :, None]  # (Z, Y, 1)

    corrected = vol / (corr_da + eps)

    return da.clip(corrected, 0, 65535).astype(np.uint16)


# ---------------------------------------------------------------------
# Geometric transforms
# ---------------------------------------------------------------------

def skew_correction_affine_dask(
    vol_yzx: da.Array,
    conversion_factors: List[float],
    delta: float = 36.0,
) -> da.Array:
    """
    Apply shear-based skew correction to a volume.

    The shear couples only Z and X (output X depends on input Z; output
    Y and Z each depend only on their own input axis) -- so this is
    always safe to apply to any Y-range of a volume independently; it
    never needs neighboring Y data to produce correct output for the
    rows it's given.

    Parameters
    ----------
    vol_yzx : dask.array.Array
        Input volume in (Y, Z, X) order.
    conversion_factors : list[float]
        Pixel size scaling in (Y, Z, X).
    delta : float, default=36.0
        Skew angle in degrees.

    Returns
    -------
    dask.array.Array
        Skew-corrected volume (Y, Z, X_out).
    """
    shear = np.tan(np.deg2rad(delta)) * \
        conversion_factors[1] / conversion_factors[2]

    y, z, x = vol_yzx.shape
    x_out = int(np.ceil(shear * z)) + x

    affine = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, shear, 1],
    ])
    affine_inv = np.linalg.inv(affine)

    return affine_transform(
        vol_yzx,
        matrix=affine_inv,
        offset=0,
        order=1,
        mode="constant",
        cval=0.0,
        output_shape=(y, z, x_out),
        output_chunks=vol_yzx.chunksize,
    )


def apply_affine(vol_yzx: da.Array, affine: np.ndarray) -> da.Array:
    """
    Apply a 3D affine transform to a volume.

    Parameters
    ----------
    vol_yzx : dask.array.Array
        Input volume (Y, Z, X).
    affine : np.ndarray
        4x4 affine matrix.

    Returns
    -------
    dask.array.Array
        Transformed volume.
    """
    return affine_transform(
        vol_yzx,
        matrix=np.linalg.inv(affine),
        order=1,
        mode="constant",
        cval=0.0,
    )


def maybe_flip_z_lazy(vol: da.Array, flip: bool) -> da.Array:
    """
    Optionally flip volume along Z axis.

    Parameters
    ----------
    vol : dask.array.Array
        Input volume (Z, Y, X).
    flip : bool
        Whether to flip.

    Returns
    -------
    dask.array.Array
        Possibly flipped volume.
    """
    return vol[::-1] if flip else vol


def skew_correct_volume_lazy(
    vol: da.Array,
    scan_parameters: dict,
    camera_id: int,
    force_flip: Optional[bool] = None,
) -> da.Array:
    """
    Apply full skew correction pipeline to a volume.

    Steps:
    - optional Z flip
    - transpose to (Y,Z,X)
    - affine shear correction
    - transpose back

    This is correct on any Y-slice of a tile independently -- the shear
    never mixes Y with Z or X, so callers don't need to pass any
    position/offset information for it to behave correctly on a partial
    Y-range; whatever rows are in `vol` are corrected as themselves.

    Parameters
    ----------
    vol : dask.array.Array
        Input volume (Z, Y, X).
    scan_parameters : dict
        Scan metadata.
    camera_id : int
        Camera identifier.
    force_flip : bool, optional
        Override flip direction.

    Returns
    -------
    dask.array.Array
        Skew-corrected volume (Z, Y, X).
    """
    delta = scan_parameters["skewCorr"]["delta"]
    umps = scan_parameters["skewCorr"]["umPixelSize"]
    factors = [umps["y"], umps["z"], umps["x"]]

    if force_flip is None:
        flip = bool(scan_parameters["crop"]
                    [f"Camera{camera_id}"]["verticalFlip"])
    else:
        flip = force_flip

    vol = maybe_flip_z_lazy(vol, flip)
    vol = da.transpose(vol, (1, 0, 2))  # → (Y,Z,X)

    vol = skew_correction_affine_dask(vol, factors, delta)

    return da.transpose(vol, (1, 0, 2))  # → (Z,Y,X)


# ---------------------------------------------------------------------
# Channel utilities
# ---------------------------------------------------------------------

def crop_volume_channels(
    vol: da.Array,
    cam_info: List[dict],
    channels: Optional[str] = None,
) -> Dict[str, da.Array]:
    """
    Crop per-channel regions from a volume.

    Parameters
    ----------
    vol : dask.array.Array
        Input volume (Z, Y, X).
    cam_info : list of dict
        Channel crop metadata.
    channels : str, optional
        If provided, only extract this channel.

    Returns
    -------
    dict[str, dask.array.Array]
        Cropped volumes per channel.
    """
    out: Dict[str, da.Array] = {}

    for meta in cam_info:
        ch = meta["channel"]

        if channels is None or channels == ch:
            y1, y2 = meta["y_start"], meta["y_end"]
            out[ch] = vol[:, y1:y2, :]

    return out


def crop_mip_channels(
    mip: np.ndarray,
    cam_info: List[dict],
    x_crop: Optional[tuple[int, int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Crop 2D MIP image into per-channel regions.

    Parameters
    ----------
    mip : np.ndarray
        Input image (Y, X).
    cam_info : list of dict
        Channel crop metadata.
    x_crop : tuple[int, int], optional
        Additional X-axis cropping.

    Returns
    -------
    dict[str, np.ndarray]
        Cropped MIP per channel.
    """
    out: Dict[str, np.ndarray] = {}

    for meta in cam_info:
        y1, y2 = meta["y_start"], meta["y_end"]
        cropped = mip[y1:y2, :]

        if x_crop is not None:
            x1, x2 = x_crop
            cropped = cropped[:, x1:x2]

        out[meta["channel"]] = cropped

    return out


# ---------------------------------------------------------------------
# High-level preprocessing
# ---------------------------------------------------------------------

def stripe_skew_corr(
    vol: da.Array,
    mask: np.ndarray,
    threshold: float,
    camera_id: int,
    scan_parameters: dict,
    tissue_frac_min: float = 0.02,
) -> da.Array:
    """
    Apply stripe correction and skew correction to a volume.

    Pipeline:
    1. Compute (Z,Y) correction map
    2. Remove low-intensity pixels
    3. Apply correction
    4. Apply skew correction

    The correction map (step 1) and the corrected output (steps 2-4) are
    built as a single lazy dask graph sharing the same source `vol` --
    nothing is read from `vol` until the caller calls `.compute()` on
    the returned array, at which point the source is read once, not
    twice (the correction map doesn't force its own separate eager pass
    over the data).

    `vol` and `mask` may be the whole tile or any Y-slice of it -- none
    of these four steps mix Y with Z or X, so this is correct (not an
    approximation) on a partial Y-range, with one caveat: the "bad row"
    fallback value and edge-smoothing kernel inside the correction-map
    step use statistics over whatever Y-range they're given, so calling
    this per Y-chunk uses chunk-local statistics rather than whole-tile
    statistics for that part specifically.

    Parameters
    ----------
    vol : dask.array.Array
        Input volume (Z, Y, X), or a Y-slice of it.
    mask : np.ndarray
        Binary mask, matching `vol`'s Y-range.
    threshold : float
        Intensity threshold.
    camera_id : int
        Camera identifier.
    scan_parameters : dict
        Acquisition parameters.
    tissue_frac_min : float, default=0.02
        Minimum valid pixel fraction.

    Returns
    -------
    dask.array.Array
        Corrected volume.
    """
    corr_zy = compute_corr_zy_from_pixel_mask(
        vol,
        mask,
        tissue_frac_min,
        threshold,
    )

    # vol = da.where(vol < threshold, 0, vol)
    vol = apply_corr_zy_lazy(vol, corr_zy)
    vol = skew_correct_volume_lazy(vol, scan_parameters, camera_id)

    return vol
