"""Blend weights and overlap blending between neighbouring chunks."""

import dask.array as da
import numpy as np
import scipy.ndimage

from linc_convert.modalities.lsm.stitching_utils.corrections import MIN_SCALER

RAMP_POWER = 3               # 1 = linear; larger = longer plateau at 0.5


def sheared_weights(scaler: np.ndarray, shear_z: float, z_out: int) -> np.ndarray:
    """Per-(z, y) blending confidence of a chunk, in the sheared frame."""
    weight = np.where(scaler > MIN_SCALER,
                      1.0 / np.maximum(scaler, MIN_SCALER), 0.0).astype(np.float32)
    if z_out == weight.shape[0] and shear_z == 0:
        return weight
    return scipy.ndimage.affine_transform(
        weight, matrix=np.array([[1.0, -shear_z], [0.0, 1.0]]), offset=0,
        output_shape=(z_out, weight.shape[1]), order=0, mode="constant", cval=0.0)


def ramp_weights(n: int) -> np.ndarray:
    """Weight of the *new* chunk over an overlap of n rows."""
    u = 2 * np.linspace(0, 1, n) - 1
    return 0.5 + 0.5 * np.sign(u) * np.abs(u) ** RAMP_POWER


def blend_overlap(prev: da.Array, cur: da.Array, w_prev: np.ndarray,
                  w_cur: np.ndarray, z_prev: int, z_cur: int, z_total: int,
                  overlap: int) -> da.Array:
    """Blend the trailing rows of the previous chunk with the leading rows of this one.

    Both chunks are placed in the mosaic's z frame. Where both have data the
    weights are ``ramp * confidence`` and ``(1 - ramp) * confidence``,
    normalised to sum to 1; where only one has data it is used as is.

    Parameters
    ----------
    prev, cur : dask.array.Array
        Corrected, sheared chunks, shape (Z', Y, X').
    w_prev, w_cur : numpy.ndarray
        Their ``sheared_weights``.
    z_prev, z_cur : int
        z origin of each chunk in the mosaic.
    z_total : int
        Mosaic z size.
    overlap : int
        Number of overlapping rows.

    Returns
    -------
    dask.array.Array
        uint16 array of shape (z_total, overlap, X').
    """
    z_out = prev.shape[0]
    r = ramp_weights(overlap).astype(np.float32)[None, :]

    def place(arr: np.ndarray, z0: int) -> np.ndarray:
        full = np.zeros((z_total, overlap), dtype=np.float32)
        full[z0:z0 + z_out] = arr
        return full

    wp = place(w_prev[:, -overlap:], z_prev)
    wc = place(w_cur[:, :overlap], z_cur)
    cur_w = wc * np.where(wp > 0, r, 1.0)
    prev_w = wp * np.where(wc > 0, 1.0 - r, 1.0)
    den = np.where(cur_w + prev_w > 0, cur_w + prev_w, 1.0)

    def pad(arr: da.Array, z0: int) -> da.Array:
        pad_width = ((z0, z_total - z0 - z_out), (0, 0), (0, 0))
        return da.pad(arr.astype(np.float32), pad_width)

    out = (pad(cur[:, :overlap], z_cur) * cur_w[:, :, None]
           + pad(prev[:, -overlap:], z_prev) * prev_w[:, :, None]) / den[:, :, None]
    return da.clip(da.rint(out), 0, 65535).astype(np.uint16)
