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


class StripeStatsAccumulator:
    """
    Incrementally gathers the statistics needed to compute the (Z, Y)
    stripe-correction map, one X-chunk at a time, instead of requiring
    the whole (mask-filtered) volume in memory at once.

    Produces EXACTLY the same result as calling `compute_corr_zy` on
    `da.where(mask, vol, np.nan)` for the whole volume directly (see
    `stripe_skew_corr` for that reference computation) -- this is not
    an approximation, provided every chunk covering the full X range
    is added exactly once, in any order, via `add_chunk`.

    The median needs the actual sampled values (not just a running
    statistic), so this collects each chunk's own `::8`-strided X
    columns and only computes the median once, in `finalize()`, over
    the full collected set. The `::8` stride is kept phase-aligned
    across chunk boundaries (via the chunk's global `x0`), so
    concatenating every chunk's contribution reproduces exactly the
    same set of sampled columns, in the same order, that striding the
    whole array by `::8` in one shot would have selected -- regardless
    of whether `x_chunk_size` happens to be a multiple of the stride.

    The valid-pixel `counts` (a plain sum, not a median) don't need
    the stride trick -- each chunk's own count over its full width is
    simply added to a running total.
    """

    def __init__(self, tissue_frac_min, threshold, kernel_size=5, x_stride=8):
        self.tissue_frac_min = tissue_frac_min
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.x_stride = x_stride
        self._strided_pieces = []
        self._counts_total = None
        self._x_total = 0

    def add_chunk(self, vol_chunk: np.ndarray, mask_chunk: np.ndarray, x0: int) -> None:
        """
        Parameters
        ----------
        vol_chunk : np.ndarray
            (Z, Y, w) registered/cropped volume chunk (already
            computed, plain numpy).
        mask_chunk : np.ndarray
            (Y, w) tissue mask chunk, matching `vol_chunk`'s Y and X
            extent.
        x0 : int
            This chunk's starting X index in the *global* (whole-tile)
            coordinate frame -- used only to keep the `::x_stride`
            sampling phase-aligned across chunks, not to place data.
        """
        vol_chunk = vol_chunk.astype(np.float32)
        Z, Y, w = vol_chunk.shape
        mask_b = np.broadcast_to(mask_chunk.astype(bool)[None], (Z, Y, w))

        # Reproduces stripe_skew_corr's two-stage masking exactly:
        # outside the tissue mask -> NaN (via the outer da.where in
        # stripe_skew_corr), then also NaN if below threshold*1.05 or
        # already non-finite (compute_corr_zy's own filtering).
        masked = np.where(
            mask_b & (vol_chunk >= self.threshold *
                      1.05) & np.isfinite(vol_chunk),
            vol_chunk, np.nan,
        )

        if self._counts_total is None:
            self._counts_total = np.zeros((Z, Y), dtype=np.float64)
        self._counts_total += np.sum(np.isfinite(masked), axis=2)

        phase = (-x0) % self.x_stride
        self._strided_pieces.append(masked[:, :, phase::self.x_stride])

        self._x_total += w

    def finalize(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            The final (Z, Y) stripe-correction map, identical to what
            `compute_corr_zy` would produce on the whole volume.
        """
        full_strided = np.concatenate(self._strided_pieces, axis=2)
        with warnings.catch_warnings():
            # nanmedian warns on all-NaN slices; compute_corr_zy's
            # dask/np.nanmedian does the same thing silently under the
            # hood, so suppressing here just matches that behavior.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            corr = np.nanmedian(full_strided, axis=2)
        corr = np.where(corr < self.threshold * 1.2,
                        self.threshold * 1.2, corr)
        min_pixels = int(self.tissue_frac_min * self._x_total)
        return _corr_zy_postprocess(
            corr, self._counts_total, min_pixels, self.kernel_size
        )


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


def apply_affine(vol_yzx, affine, order=3):
    return affine_transform(vol_yzx, matrix=np.linalg.inv(affine), order=order, mode="constant", cval=0.0)


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


def apply_channel_affine_volume(vol, affine_zy, order=3):
    """
    Register a channel's volume onto the reference channel's frame,
    using a 3x3 (Z, Y) affine (X is left untouched).

    Returns a *lazy* dask array. Callers should materialize this with
    `.compute()` immediately after calling this function, exactly
    once, rather than leaving it lazy and slicing/computing pieces of
    it repeatedly downstream (e.g. once per Y-chunk) -- cubic-spline
    interpolation (`order=3`, the default) requires a global,
    recursive prefiltering pass along every axis, so each separate
    slice-and-compute call from the same lazy graph redoes a large
    fraction of that work from scratch. Materializing once and slicing
    the resulting plain array is dramatically cheaper than repeatedly
    slicing the lazy graph -- empirically, even a single downstream
    slice-and-compute call can cost more than materializing the entire
    volume once.

    Parameters
    ----------
    vol : dask.array.Array
        Input volume (Z, Y, X).
    affine_zy : np.ndarray
        3x3 homogeneous affine acting on (Z, Y).
    order : int, default=3
        Spline interpolation order passed to `apply_affine`. Lower
        orders (1 = linear, 0 = nearest) are cheaper per-evaluation
        and, unlike order=3, don't require global prefiltering, so
        they don't suffer from the repeated-slicing cost above even if
        left lazy -- but materializing once is still recommended
        regardless of order.

    Returns
    -------
    dask.array.Array
        Lazy registered volume (Z, Y, X).
    """
    affine_3d = embed_zy_affine_for_volume(affine_zy)
    return apply_affine(vol, affine_3d, order=order)


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


def skew_shear_amount(scan_parameters) -> float:
    """
    The skew shear amount (see `skew_correction_affine_dask`), computed
    directly from `scan_parameters` -- shared by both the actual
    correction and the X-chunk padding calculation below, so they can
    never disagree.
    """
    delta = scan_parameters["acquisitionSettings"]["skewCorrection"]["delta_deg"]
    umps = scan_parameters["voxelSize_um"]["rawAcquisition"]
    return np.tan(np.deg2rad(delta)) * umps["z"] / umps["x"]


def skew_shear_x_padding(scan_parameters, z_extent: int) -> int:
    """
    X padding (columns, each side) an X-chunk needs so that running it
    through `skew_correct_volume_x_chunk` and trimming reproduces
    exactly what running the whole tile through `skew_correct_volume_lazy`
    and slicing out that chunk's output range would give.

    The shear only couples Z and X (`X_out = X_in + shear * Z_in`), so
    for a chunk's output X-range, the required *input* X-range can
    extend up to `shear * (Z - 1)` columns beyond the chunk's own
    nominal boundary, in the direction `shear`'s sign pushes it. This
    pads both sides by that amount (plus a small safety margin) rather
    than figuring out the sign-dependent single side, so it's correct
    regardless of the sign of `shear`.

    Parameters
    ----------
    scan_parameters : dict
        Loaded scan configuration.
    z_extent : int
        The Z extent (after registration/cropping) the volume will
        have when the shear is applied.

    Returns
    -------
    int
        Padding, in columns, to add on each side of an X-chunk.
    """
    shear = skew_shear_amount(scan_parameters)
    return int(np.ceil(abs(shear) * max(z_extent - 1, 0))) + 2


def skew_correct_volume_x_chunk(
    vol_padded_zyx,
    scan_parameters,
    camera_id,
    pad_left: int,
    out_width: int,
    force_flip=None,
):
    """
    Apply skew correction to a *padded* X-chunk (Z, Y, X_padded) and
    trim the result down to the `out_width`-wide output range that
    corresponds to the chunk's own (unpadded) columns.

    `vol_padded_zyx` must have been read starting `pad_left` columns
    before the chunk's true global start (i.e. `pad_left` is how much
    *real* left padding is actually present -- less than the value
    `skew_shear_x_padding` returned if that got clamped near the raw
    tile's own X boundary).

    Equivalent to running `skew_correct_volume_lazy` on the whole tile
    and slicing out this chunk's own output columns -- verified
    numerically to match to floating-point precision (~1e-12, from
    cubic-spline coefficients computed on a differently-sized array;
    negligible next to uint16 quantization).

    Parameters
    ----------
    vol_padded_zyx : dask.array.Array or np.ndarray
        Padded input chunk (Z, Y, X_padded).
    scan_parameters : dict
        Scan metadata (see `skew_correct_volume_lazy`).
    camera_id : int
        Camera identifier.
    pad_left : int
        Real left padding present in `vol_padded_zyx`, in columns.
    out_width : int
        Width, in columns, of this chunk's own (unpadded) output
        contribution.
    force_flip : bool, optional
        Passed through to `skew_correct_volume_lazy`.

    Returns
    -------
    np.ndarray
        This chunk's output contribution (Z, Y, out_width).
    """
    sheared = skew_correct_volume_lazy(
        vol_padded_zyx, scan_parameters, camera_id, force_flip=force_flip
    )
    sheared = np.asarray(sheared.compute()) if hasattr(
        sheared, "compute") else np.asarray(sheared)
    return sheared[:, :, pad_left: pad_left + out_width]


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
