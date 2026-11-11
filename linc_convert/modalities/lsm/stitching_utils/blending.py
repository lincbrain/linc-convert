"""Blend weights and overlap blending between neighbouring chunks."""

import dask.array as da
import numpy as np
import scipy.ndimage

from linc_convert.modalities.lsm.stitching_utils.corrections import MIN_SCALER

RAMP_POWER = 1               # linear ramp only: 1 = linear blend across the overlap; larger =
                             #   longer plateau at 0.5
RAMP_SHAPES = ("cosine", "linear")
CONFIDENCE_POWER = 0.2       # how much the size of a row's correction (its multiplier)
                             #   counts: the confidence is (1 / multiplier) ** this.
                             #   0 = not at all (pure ramp); it was 2.


def sheared_weights(scaler: np.ndarray, shear_z: float, z_out: int) -> np.ndarray:
    """Per-(z, y) blending confidence of a chunk, in the sheared frame.

    A row that needed a large multiplier (a dim row) counts a little less:
    ``(1 / multiplier) ** CONFIDENCE_POWER``. Rows that are invalid (multiplier at or
    below ``MIN_SCALER``) have no weight at all, so the other chunk is used there.
    """
    weight = np.where(scaler > MIN_SCALER,
                      (1.0 / np.maximum(scaler, MIN_SCALER))
                      ** CONFIDENCE_POWER, 0.0).astype(np.float32)
    if z_out == weight.shape[0] and shear_z == 0:
        return weight
    return scipy.ndimage.affine_transform(
        weight, matrix=np.array([[1.0, -shear_z], [0.0, 1.0]]), offset=0,
        output_shape=(z_out, weight.shape[1]), order=0, mode="constant", cval=0.0)


def ramp_weights(n: int, shape: str = "cosine") -> np.ndarray:
    """Weight of the *new* chunk over an overlap of n rows (0 at the first row, 1 at the last).

    ``"cosine"``: ``0.5 - 0.5 cos(pi t)`` for ``t`` from 0 to 1 (a raised cosine: the
    weights of the two chunks always add to 1, and the ramp is flat at both ends, so
    there is no visible kink where the blend starts and ends). ``"linear"``: ``t``
    (with ``RAMP_POWER`` 1).

    Raises
    ------
    ValueError
        If ``shape`` is unknown.
    """
    t = np.linspace(0, 1, n)
    if shape == "cosine":
        return 0.5 - 0.5 * np.cos(np.pi * t)
    if shape == "linear":
        u = 2 * t - 1
        return 0.5 + 0.5 * np.sign(u) * np.abs(u) ** RAMP_POWER
    raise ValueError(f"ramp shape must be one of {RAMP_SHAPES}, got {shape!r}")


def blend_overlap(prev: da.Array, cur: da.Array, w_prev: np.ndarray,
                  w_cur: np.ndarray, z_prev: int, z_cur: int, z_total: int,
                  overlap: int, ramp: str = "cosine") -> da.Array:
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
    ramp : {"cosine", "linear"}, default="cosine"
        Shape of the blend ramp (see ``ramp_weights``).

    Returns
    -------
    dask.array.Array
        uint16 array of shape (z_total, overlap, X').
    """
    z_out = prev.shape[0]
    r = ramp_weights(overlap, ramp).astype(np.float32)[None, :]

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