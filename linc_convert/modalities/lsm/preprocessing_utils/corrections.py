from typing import Dict, List, Optional, Tuple

import dask
import dask.array as da
import numpy as np
import warnings
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
    mask: np.ndarray,
    tissue_frac_min: float,
    threshold: float,
    kernel_size: int = 5,
) -> da.Array:
    vol = vol.astype(np.float32)
    Z, Y, X = vol.shape

    if mask.shape == (Y, X):
        mask_da = da.from_array(mask, chunks=vol.chunks[1:]) if isinstance(
            mask, np.ndarray) else mask
        mask_da = da.broadcast_to(mask_da[None], (Z, Y, X))
    elif mask.shape == (Z, Y, X):
        mask_da = da.from_array(mask, chunks=vol.chunks) if isinstance(
            mask, np.ndarray) else mask
    else:
        raise ValueError(
            f"mask shape {mask.shape} != volume shape {(Z, Y, X)} or {(Y, X)}")

    # Keep a pixel only if it's inside the tissue mask AND above
    # threshold AND finite -- same combined condition
    # stripe_skew_corr/apply_affine_split use elsewhere in this file,
    # just expressed as one da.where instead of two chained ones.
    masked = da.where(
        mask_da & (vol >= threshold * 1.05) & da.isfinite(vol),
        vol, np.nan,
    )
    corr = da.nanmedian(masked[:, :, ::64], axis=2)
    counts = da.sum(da.isfinite(masked), axis=2)
    min_pixels = int(tissue_frac_min * X)
    corr = da.where(counts < min_pixels, 9999999.0, corr)
    # Without this scaling, corr_zy sits at the same scale as the raw
    # intensity itself (e.g. ~200), so apply_corr_zy_lazy's vol/corr_zy
    # collapses to ~1.0 for typical pixels -- which rounds to 0/1 once
    # cast to uint16, destroying virtually all dynamic range. Dividing
    # by 1000 here keeps the corrected output at roughly the original
    # intensity scale instead.
    return corr / 1000
    # corr_smooth = dask.delayed(_corr_zy_postprocess)(
    #    corr, counts, min_pixels, kernel_size)


def compute_alt_zy_calibration_for_tile(
    vol: da.Array,
    mask: np.ndarray,
    threshold: float,
    background_length: int = 5000,
    x_stride: int = 64,
) -> Tuple[float, np.ndarray]:
    """
    Compute ONE reference tile's contribution to the alternate zy
    correction (see `pipeline`'s `use_alt_zy_correction`): a single
    background-noise scalar, and a per-(Z, Y) RECIPROCAL multiplier
    map (i.e. meant to be applied via multiplication, not division).

    Unlike `compute_corr_zy`, this:
    - First estimates the camera's own background noise as the median
      of the last `background_length` X columns (a single scalar, not
      per-row), and works on the noise-subtracted (clipped at 0)
      volume from that point on.
    - Normalizes the per-row scalers by their OWN median (not a fixed
      constant like 1000), computed from this tile alone.
    - Returns the RECIPROCAL of that normalized scaler, since the
      intended application is `(vol - noise).clip(0) * reciprocal`,
      not division.

    This is meant to be called once per reference tile (see
    `pipeline`'s `alt_zy_reference_tiles`), with the results from
    several reference tiles averaged together afterward -- averaging
    happens on these RECIPROCAL values, not on the pre-inversion
    scalers, since those are what actually gets applied.

    Parameters
    ----------
    vol : dask.array.Array
        Raw (Z, Y, X) volume for one reference tile, already flipped
        if applicable -- same frame it'll be applied in later.
    mask : np.ndarray
        Tissue mask, shape (Y, X) or (Z, Y, X).
    threshold : float
        Intensity threshold (same role as in `compute_corr_zy`).
    background_length : int, default=5000
        Width, in columns, of the background region (from the far
        edge of X) used to estimate the noise floor.
    x_stride : int, default=64
        Subsampling stride along X for the per-row median -- same
        role as `compute_corr_zy`'s hardcoded `::64`.

    Returns
    -------
    noise : float
        Background noise scalar for this tile.
    reciprocal_map : np.ndarray
        Shape (Z, Y). Apply via `(vol - noise).clip(0) * reciprocal_map`.
    """
    vol = vol.astype(np.float32)
    Z, Y, X = vol.shape

    edge_region = np.asarray(vol[:, :, -background_length:].compute())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        noise = float(np.nanmedian(edge_region))
    if not np.isfinite(noise):
        noise = 0.0

    vol_denoised = da.clip(vol - noise, 0, None)

    if mask.shape == (Y, X):
        mask_da = da.from_array(mask, chunks=vol.chunks[1:]) if isinstance(
            mask, np.ndarray) else mask
        mask_da = da.broadcast_to(mask_da[None], (Z, Y, X))
    elif mask.shape == (Z, Y, X):
        mask_da = da.from_array(mask, chunks=vol.chunks) if isinstance(
            mask, np.ndarray) else mask
    else:
        raise ValueError(
            f"mask shape {mask.shape} != volume shape {(Z, Y, X)} or {(Y, X)}")

    masked = da.where(
        mask_da & (vol >= threshold * 1.05) & da.isfinite(vol),
        vol_denoised, np.nan,
    )
    raw_scaler = da.nanmedian(masked[:, :, ::x_stride], axis=2)  # (Z, Y)
    raw_scaler_np = np.asarray(raw_scaler.compute())

    # A row whose post-noise-subtraction median comes out at (or very
    # near) zero -- e.g. tissue barely above background for that
    # row/depth -- is just as unreliable as a row with no tissue data
    # at all. Treat it the same way (NaN here), so it flows through
    # the same "insufficient data" fallback downstream, instead of
    # producing a 1/~0 = huge reciprocal that would blow out that row.
    raw_scaler_np = np.where(
        np.isfinite(raw_scaler_np) & (raw_scaler_np > 1e-3),
        raw_scaler_np, np.nan,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        scaler_median = np.nanmedian(raw_scaler_np)
    if not np.isfinite(scaler_median) or scaler_median == 0:
        scaler_median = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        normalized_scaler = raw_scaler_np / scaler_median
        reciprocal_map = 1.0 / normalized_scaler

    return noise, reciprocal_map

    # return da.from_delayed(corr_smooth, shape=(Z, Y), dtype=np.float32)


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
def generate_skew_affine(
    conversion_factors: List[float],
    delta: float = 36.0,
):
    shear = np.tan(np.deg2rad(delta)) * \
        conversion_factors[1] / conversion_factors[2]

    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [shear, 0, 1, 0],
        [0, 0, 0, 1]
    ])


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


def get_crop_values(size_z, cam_info_split: List[dict], cam_info_stitching, reference_channel, is_flipped):
    """
    Crop `vol` (already cropped to the split-crop bounds for
    `reference_channel`) down to the stitching-crop bounds, expressed
    relative to that split-crop frame.

    `vol.shape[0]` is taken as the split-cropped Z depth (D) -- valid
    as long as `vol` really is this same tile/channel's own
    split-cropped volume (every tile is assumed to share the same
    shape elsewhere in this codebase, so this should hold).
    """

    split_y0 = split_z0 = None
    for meta in cam_info_split:
        if meta["channel"] == reference_channel:
            split_y0 = meta["y_start"]
            split_z0 = meta.get("z_start")
            break
    if split_y0 is None:
        raise KeyError(
            f"reference_channel '{reference_channel}' not found in cam_info_split"
        )
    split_y0 = split_y0 if split_y0 is not None else 0
    split_z0 = split_z0 if split_z0 is not None else 0

    D = size_z

    for meta in cam_info_stitching:
        ch = meta["channel"]
        if ch != reference_channel:
            continue

        stitch_y0, stitch_y1 = meta["y_start"], meta["y_end"]
        stitch_z0, stitch_z1 = meta.get("z_start"), meta.get("z_end")

        y1 = stitch_y0 - split_y0
        y2 = stitch_y1 - split_y0

        local_tz0 = (stitch_z0 if stitch_z0 is not None else 0) - split_z0
        local_tz1 = (stitch_z1 if stitch_z1 is not None else 0) - split_z0

        if is_flipped:
            z1 = D - local_tz1
            z2 = D - local_tz0
        else:
            z1 = local_tz0
            z2 = local_tz1

        return z1, z2, y1, y2


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


def apply_affine_split(vol_zyx: da.Array, affine: np.ndarray, y_start: int, y_end: int, corr_zy, mask) -> da.Array:
    """Resample `vol_zyx` through `affine` and return only the requested region.

    Internally reads a padded bounding box (large enough to cover the
    requested output region after the affine is applied) rather than the
    whole volume, then crops down to exactly what was asked for.
    """
    z_start = 0
    z_end = vol_zyx.shape[0]
    x_start = 0
    x_end = vol_zyx.shape[2]

    def make_translate(z0, y0, x0):
        return np.array([
            [1, 0, 0, z0],
            [0, 1, 0, y0],
            [0, 0, 1, x0],
            [0, 0, 0, 1],
        ], dtype=float)

    inv_affine = np.linalg.inv(affine)

    # 1. Map the output region's corners back through the inverse affine to
    #    find the input-space bounding box needed to fill it.
    corners = [
        inv_affine @ np.array([z, y, x, 1.0])
        for z in (z_start, z_end - 1)
        for y in (y_start, y_end - 1)
        for x in (x_start, x_end - 1)
    ]
    input_coords = np.array(corners)[:, :3]

    in_min = np.floor(input_coords.min(axis=0)).astype(int)
    in_max = np.ceil(input_coords.max(axis=0)).astype(
        int) + 1  # exclusive upper bound

    # 2. Union with the originally requested region so it stays addressable.
    in_min[0], in_max[0] = min(in_min[0], z_start), max(in_max[0], z_end)
    in_min[1], in_max[1] = min(in_min[1], y_start), max(in_max[1], y_end)
    in_min[2], in_max[2] = min(in_min[2], x_start), max(in_max[2], x_end)

    # 3. Clip to the actual volume bounds so we never read out-of-bounds.
    vol_shape = vol_zyx.shape
    pad_start = np.maximum([0, 0, 0], in_min)
    pad_end = np.minimum(vol_shape, in_max)
    pad_z_start, pad_y_start, pad_x_start = pad_start
    pad_z_end, pad_y_end, pad_x_end = pad_end

    # 4. Slice the padded region and shift the affine into its local frame.
    translate = make_translate(pad_z_start, pad_y_start, pad_x_start)
    affine_local = np.linalg.inv(translate) @ affine @ translate

    padded_slice = vol_zyx[
        pad_z_start:pad_z_end, pad_y_start:pad_y_end, pad_x_start:pad_x_end
    ]
    Z, Y, X = padded_slice.shape

    # mask is a (Y, X) (or (Z, Y, X)) array matching the FULL vol_zyx's
    # extent, but padded_slice is windowed to [pad_*_start:pad_*_end] --
    # mask needs the same windowing before use, exactly like corr_zy
    # gets windowed by [pad_z_start:pad_z_end, pad_y_start:pad_y_end]
    # a few lines below.
    if mask.ndim == 2 and mask.shape == (vol_zyx.shape[1], vol_zyx.shape[2]):
        mask = mask[pad_y_start:pad_y_end, pad_x_start:pad_x_end]
    elif mask.ndim == 3 and mask.shape == tuple(vol_zyx.shape):
        mask = mask[pad_z_start:pad_z_end,
                    pad_y_start:pad_y_end, pad_x_start:pad_x_end]

    mask_da = da.from_array(mask, chunks=padded_slice.chunks[1:])

    if mask.shape == (Y, X):
        mask_da = da.broadcast_to(mask_da[None], (Z, Y, X))
    elif mask.shape != (Z, Y, X):
        raise ValueError(
            f"mask shape {mask.shape} != volume shape {(Z, Y, X)}")

    masked = da.where(mask_da, padded_slice, 0)
    corr_zy = corr_zy[pad_z_start:pad_z_end, pad_y_start:pad_y_end]

    padded_slice = apply_corr_zy_lazy(masked, corr_zy)

    transformed = affine_transform(
        padded_slice,
        matrix=np.linalg.inv(affine_local),
        order=0,
        mode="constant",
        cval=0.0,
    )

    # 5. Crop the transformed, padded result down to the requested region.
    crop_z0 = z_start - pad_z_start
    crop_y0 = y_start - pad_y_start
    crop_x0 = x_start - pad_x_start
    dz, dy, dx = z_end - z_start, y_end - y_start, x_end - x_start

    return transformed[
        crop_z0:crop_z0 + dz,
        crop_y0:crop_y0 + dy,
        crop_x0:crop_x0 + dx,
    ]

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
    Z, Y, X = vol.shape
    mask_da = da.from_array(mask, chunks=vol.chunks[1:])

    if mask.shape == (Y, X):
        mask_da = da.broadcast_to(mask_da[None], (Z, Y, X))
    elif mask.shape != (Z, Y, X):
        raise ValueError(
            f"mask shape {mask.shape} != volume shape {(Z, Y, X)}")

    # compute_corr_zy now does its own mask+threshold+finite filtering
    # internally, given the raw vol and mask directly -- no need to
    # pre-mask (NaN-fill) here first the way this used to.
    corr_zy = compute_corr_zy(
        vol,
        mask,
        tissue_frac_min,
        threshold,
    )

    masked = da.where(mask_da, vol, 0)

    vol = apply_corr_zy_lazy(masked, corr_zy)
    vol = skew_correct_volume_lazy(vol, scan_parameters, camera_id)

    return vol


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
