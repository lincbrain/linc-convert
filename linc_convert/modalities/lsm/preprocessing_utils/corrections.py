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
    corr = corr.copy()
    corr[counts < min_pixels] = np.nan

    valid_per_z = np.sum(np.isfinite(corr), axis=1)
    bad_rows = valid_per_z < 10

    fallback = np.nanmedian(corr)
    if not np.isfinite(fallback):
        fallback = 1.0

    corr[bad_rows] = fallback

    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)

    valid = np.isfinite(corr)
    corr_filled = np.nan_to_num(corr, nan=0)

    num = convolve(corr_filled, kernel, mode="nearest")
    den = convolve(valid.astype(np.float32), kernel, mode="nearest")

    corr_smooth = num / (den + 1e-6)
    corr_smooth[corr_smooth < 5] = 99999999.0

    return (corr_smooth / 1000).astype(np.float32)


def compute_corr_zy(
    vol: da.Array,
    tissue_frac_min: float,
    threshold: float,
    kernel_size: int = 5,
) -> da.Array:
    vol = vol.astype(np.float32)
    Z, Y, X = vol.shape

    masked = da.where((vol < threshold) | ~
                      da.isfinite(vol), np.nan, vol)

    corr = da.nanmedian(masked[:, :, ::8], axis=2)
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
    vol = vol.astype(np.float32)

    if isinstance(corr_zy, np.ndarray):
        corr_da = da.from_array(corr_zy.astype(np.float32))
    else:
        corr_da = corr_zy.astype(np.float32)
    corr_da = corr_da[:, :, None]

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
    return affine_transform(
        vol_yzx,
        matrix=np.linalg.inv(affine),
        order=3,
        mode="constant",
        cval=0.0,
    )


def maybe_flip_z_lazy(vol: da.Array, flip: bool) -> da.Array:
    return vol[::-1] if flip else vol


def skew_correct_volume_lazy(
    vol: da.Array,
    scan_parameters: dict,
    camera_id: int,
    force_flip: Optional[bool] = None,
) -> da.Array:
    """
    Apply full skew correction pipeline to a volume.

    Parameters
    ----------
    vol : dask.array.Array
        Input volume (Z, Y, X).
    scan_parameters : dict
        Scan metadata. Expected keys:
        - `acquisitionSettings.skewCorrection.delta_deg`: skew angle.
        - `voxelSize_um.rawAcquisition`: dict with `x`/`y`/`z` pixel
          sizes in microns, *before* skew correction (this is the
          voxel geometry the shear itself is computed from).
        - `channelLayout.Camera{camera_id}.verticalFlip`: whether this
          camera's volumes should be flipped along Z before shearing.
    camera_id : int
        Camera identifier.
    force_flip : bool, optional
        Override flip direction.

    Returns
    -------
    dask.array.Array
        Skew-corrected volume (Z, Y, X).
    """
    delta = scan_parameters["acquisitionSettings"]["skewCorrection"]["delta_deg"]
    umps = scan_parameters["voxelSize_um"]["rawAcquisition"]
    factors = [umps["y"], umps["z"], umps["x"]]

    if force_flip is None:
        flip = bool(
            scan_parameters["channelLayout"]
            [f"Camera{camera_id}"]["verticalFlip"]
        )
    else:
        flip = force_flip

    vol = maybe_flip_z_lazy(vol, flip)
    vol = da.transpose(vol, (1, 0, 2))  # -> (Y,Z,X)

    vol = skew_correction_affine_dask(vol, factors, delta)

    return da.transpose(vol, (1, 0, 2))  # -> (Z,Y,X)


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

    Each channel is cropped along Y (lateral axis, splitting the raw
    dual-channel camera frame apart) using `y_start`/`y_end`, and along
    Z (depth axis) using `z_start`/`z_end` when those are provided. Z
    cropping is skipped (axis left as-is) for any channel where
    `z_start`/`z_end` are `None`.

    Parameters
    ----------
    vol : dask.array.Array
        Input volume (Z, Y, X).
    cam_info : list of dict
        Channel crop metadata (from `get_camera_info`).
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
            z1, z2 = meta.get("z_start"), meta.get("z_end")

            cropped = vol[:, y1:y2, :]
            if z1 is not None and z2 is not None:
                cropped = cropped[z1:z2, :, :]

            out[ch] = cropped

    return out


def crop_mip_channels(
    mip: np.ndarray,
    cam_info: List[dict],
    x_crop: Optional[tuple] = None,
) -> Dict[str, np.ndarray]:
    """
    Crop 2D MIP image into per-channel regions.

    The MIP is a YX max-intensity projection (already collapsed along
    Z), so only the Y axis is cropped here from `cam_info`
    (`y_start`/`y_end`, same lateral-axis channel split used by
    `crop_volume_channels`). There is no Z axis left to crop. `x_crop`,
    if given, crops the MIP's actual X axis (the strip/acquisition
    direction) -- unrelated to `cam_info`'s `z_start`/`z_end`.

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
    tissue_frac_min: float = 0.005,
) -> da.Array:
    Z, Y, X = vol.shape
    mask_da = da.from_array(mask, chunks=vol.chunks[1:])

    if mask.shape == (Y, X):
        mask_da = da.broadcast_to(mask_da[None], (Z, Y, X))
    elif mask.shape != (Z, Y, X):
        raise ValueError(
            f"mask shape {mask.shape} != volume shape {(Z, Y, X)}")

    masked = da.where(mask_da, vol, np.nan)

    corr_zy = compute_corr_zy(
        masked,
        tissue_frac_min,
        threshold,
    )

    masked = da.where(mask_da, vol, 0)

    vol = apply_corr_zy_lazy(masked, corr_zy)
    vol = skew_correct_volume_lazy(vol, scan_parameters, camera_id)

    return vol
