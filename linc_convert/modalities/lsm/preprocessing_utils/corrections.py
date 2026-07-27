from typing import Dict, List, Optional
import warnings

import dask
import dask.array as da
import numpy as np
import scipy.ndimage as ndi
from dask_image.ndinterp import affine_transform
from scipy.ndimage import convolve


def _corr_zy_postprocess(corr, counts, min_pixels, kernel_size):
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


def compute_corr_zy(vol, tissue_frac_min, threshold, kernel_size=5):
    vol = vol.astype(np.float32)
    Z, Y, X = vol.shape
    masked = da.where((vol < threshold*1.05) | ~da.isfinite(vol), np.nan, vol)
    corr = da.nanmedian(masked[:, :, ::8], axis=2)
    counts = da.sum(da.isfinite(masked), axis=2)
    min_pixels = int(tissue_frac_min * X)
    corr = da.where((corr < threshold*1.2), threshold*1.2, corr)
    corr_smooth = dask.delayed(_corr_zy_postprocess)(
        corr, counts, min_pixels, kernel_size)
    return da.from_delayed(corr_smooth, shape=(Z, Y), dtype=np.float32)


def apply_corr_zy_lazy(vol, corr_zy, eps=1e-6):
    vol = vol.astype(np.float32)
    if isinstance(corr_zy, np.ndarray):
        corr_da = da.from_array(corr_zy.astype(np.float32))
    else:
        corr_da = corr_zy.astype(np.float32)
    corr_da = corr_da[:, :, None]
    corrected = vol / (corr_da + eps)
    return da.clip(corrected, 0, 65535).astype(np.uint16)


def skew_correction_affine_dask(vol_yzx, conversion_factors, delta=36.0):
    shear = np.tan(np.deg2rad(delta)) * \
        conversion_factors[1] / conversion_factors[2]
    y, z, x = vol_yzx.shape
    x_out = int(np.ceil(shear * z)) + x
    affine = np.array([[1, 0, 0], [0, 1, 0], [0, shear, 1]])
    affine_inv = np.linalg.inv(affine)
    return affine_transform(
        vol_yzx, matrix=affine_inv, offset=0, order=3, mode="constant",
        cval=0.0, output_shape=(y, z, x_out), output_chunks=vol_yzx.chunksize,
    )


def apply_affine(vol_yzx, affine):
    return affine_transform(vol_yzx, matrix=np.linalg.inv(affine), order=3, mode="constant", cval=0.0)


def maybe_flip_z_lazy(vol, flip):
    return vol[::-1] if flip else vol


def embed_zy_affine_for_volume(affine_zy):
    affine_zy = np.asarray(affine_zy, dtype=np.float64)
    if affine_zy.shape != (3, 3):
        raise ValueError(
            f"Expected a 3x3 (Z,Y) affine matrix, got shape {affine_zy.shape}")
    affine_3d = np.eye(4)
    affine_3d[0:2, 0:2] = affine_zy[0:2, 0:2]
    affine_3d[0:2, 3] = affine_zy[0:2, 2]
    return affine_3d


def apply_channel_affine_volume(vol, affine_zy):
    affine_3d = embed_zy_affine_for_volume(affine_zy)
    return apply_affine(vol, affine_3d)


def apply_channel_affine_mask(mask, affine_zy, zy_cross_term_tol=1e-6):
    affine_zy = np.asarray(affine_zy, dtype=np.float64)
    if affine_zy.shape != (3, 3):
        raise ValueError(
            f"Expected a 3x3 (Z,Y) affine matrix, got shape {affine_zy.shape}")
    if abs(affine_zy[1, 0]) > zy_cross_term_tol:
        warnings.warn(
            "Registration affine has a non-negligible Z<->Y cross term "
            f"({affine_zy[1, 0]!r}), which can't be applied to the "
            "Z-collapsed tissue mask -- only the Y-scale/translation "
            "part of the affine is being applied to the mask.",
            stacklevel=2,
        )
    a_yy = affine_zy[1, 1]
    t_y = affine_zy[1, 2]
    affine_yx = np.array([[a_yy, 0.0, t_y], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    registered = ndi.affine_transform(
        mask.astype(np.float32), matrix=np.linalg.inv(affine_yx),
        order=0, mode="constant", cval=0.0,
    )
    return registered > 0.5


def skew_correct_volume_lazy(vol, scan_parameters, camera_id, force_flip=None):
    delta = scan_parameters["acquisitionSettings"]["skewCorrection"]["delta_deg"]
    umps = scan_parameters["voxelSize_um"]["rawAcquisition"]
    factors = [umps["y"], umps["z"], umps["x"]]
    if force_flip is None:
        flip = bool(scan_parameters["channelLayout"]
                    [f"Camera{camera_id}"]["verticalFlip"])
    else:
        flip = force_flip
    vol = maybe_flip_z_lazy(vol, flip)
    vol = da.transpose(vol, (1, 0, 2))
    vol = skew_correction_affine_dask(vol, factors, delta)
    return da.transpose(vol, (1, 0, 2))


def crop_volume_channels(vol, cam_info, channels=None):
    out = {}
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


def crop_mip_channels(mip, cam_info, x_crop=None, ch=None):
    out = {}
    for meta in cam_info:
        if ch is None or meta["channel"] == ch:
            y1, y2 = meta["y_start"], meta["y_end"]
            cropped = mip[y1:y2, :]
            if x_crop is not None:
                x1, x2 = x_crop
                cropped = cropped[:, x1:x2]
            out[meta["channel"]] = cropped
    return out


def stripe_skew_corr(vol, mask, threshold, camera_id, scan_parameters,
                     tissue_frac_min=0.02, force_flip=None):
    Z, Y, X = vol.shape
    mask_da = da.from_array(mask, chunks=vol.chunks[1:])
    if mask.shape == (Y, X):
        mask_da = da.broadcast_to(mask_da[None], (Z, Y, X))
    elif mask.shape != (Z, Y, X):
        raise ValueError(
            f"mask shape {mask.shape} != volume shape {(Z, Y, X)}")
    masked = da.where(mask_da, vol, np.nan)
    corr_zy = compute_corr_zy(masked, tissue_frac_min, threshold)
    masked = da.where(mask_da, vol, 0)
    vol = apply_corr_zy_lazy(masked, corr_zy)
    vol = skew_correct_volume_lazy(
        vol, scan_parameters, camera_id, force_flip=force_flip)
    return vol
