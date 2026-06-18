from typing import Dict, List, Optional

import dask.array as da
import numpy as np
from dask_image.ndinterp import affine_transform
from scipy.ndimage import convolve

# ---------------------------------------------------------------------
# Intensity correction
# ---------------------------------------------------------------------


def compute_corr_zy_from_pixel_mask(
    vol: da.Array,
    mask: np.ndarray,
    tissue_frac_min: float,
    threshold: float,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Compute a (Z, Y) intensity correction map from a 3D volume.

    The workflow:
    1. Mask invalid pixels
    2. Remove low-intensity values
    3. Collapse along X using median
    4. Filter low-quality rows
    5. Apply 2D smoothing

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
    np.ndarray
        Correction map of shape (Z, Y).
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

    # Collapse along X
    corr = da.nanmedian(masked, axis=2)
    counts = da.sum(da.isfinite(masked), axis=2)

    corr, counts = da.compute(corr, counts)

    # -------------------------
    # Remove low-quality rows
    # -------------------------
    min_pixels = int(tissue_frac_min * X)
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


def apply_corr_zy_lazy(
    vol: da.Array,
    corr_zy: np.ndarray,
    eps: float = 1e-6,
) -> da.Array:
    """
    Apply a (Z, Y) correction map lazily to a volume.

    Parameters
    ----------
    vol : dask.array.Array
        Input volume (Z, Y, X).
    corr_zy : np.ndarray
        Correction map (Z, Y).
    eps : float, default=1e-6
        Small value to avoid division by zero.

    Returns
    -------
    dask.array.Array
        Corrected volume (uint16).
    """
    vol = vol.astype(np.float32)

    corr_da = da.from_array(corr_zy.astype(np.float32))
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
        order=3,
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
        order=3,
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

    Parameters
    ----------
    vol : dask.array.Array
        Input volume (Z, Y, X).
    mask : np.ndarray
        Binary mask.
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

    vol = da.where(vol < threshold, 0, vol)
    vol = apply_corr_zy_lazy(vol, corr_zy)
    vol = skew_correct_volume_lazy(vol, scan_parameters, camera_id)

    return vol
