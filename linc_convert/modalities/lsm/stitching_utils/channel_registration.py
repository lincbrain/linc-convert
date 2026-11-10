"""Registration of a channel to a reference channel (488), folded into the skew shear.

The channel is split (its ``splitCrop`` y range, all z) and flipped like its camera.
Its brightness correction is applied there, in its own frame. Then ONE affine
resample takes it to the reference channel's stitching-cropped frame and shears it
(skew, and z by y) in the same step::

    forward:  q = S (A p + t - o) + pad
    sampling: p = A^-1 (S^-1 (q - pad) + o - t)

``p`` is a voxel (z, y, x) of the channel's split, flipped, corrected chunk; ``A`` and
``t`` are the channel -> reference registration (reference = also split and flipped);
``o`` is the corner of the reference's stitching crop in the reference's split,
flipped frame; ``S`` is the shear of ``shear_volume``; ``pad`` shifts z so that a
negative z shear starts at 0 (as the stitch does); ``q`` is a voxel of the sheared
chunk, i.e. of the same output the reference channel's chunks are made in.

All arrays are in (z, y, x) order.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import dask.array as da
import numpy as np
import scipy.ndimage
import yaml
from dask_image.ndinterp import affine_transform


@dataclass
class ChannelRegistration:
    """Registration of one channel to the reference channel, at one pyramid layer.

    Attributes
    ----------
    matrix : numpy.ndarray
        ``A``, 3 x 3, channel -> reference, (z, y, x) order.
    translation : numpy.ndarray
        ``t``, length 3, in voxels of the layer (the file's full-resolution value
        divided by ``2 ** layer``).
    origin : tuple of float
        ``o``: corner (z, y, 0) of the reference stitching crop in the reference's
        split, flipped frame, in voxels of the layer.
    order : int
        Interpolation order of the data (0 nearest, 1 linear).
    """
    matrix: np.ndarray
    translation: np.ndarray
    origin: Tuple[float, float, float]
    order: int = 1

    @property
    def is_identity(self) -> bool:
        """True if the registration is the identity (any integer translation of the
        crop is exact with nearest neighbour)."""
        return bool(np.allclose(self.matrix, np.eye(3))
                    and np.allclose(self.translation, np.round(self.translation)))


def load_channel_registration(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read ``A`` and ``t`` (channel -> reference, (z, y, x), full-resolution voxels).

    The YAML holds either ``affine`` (4 x 4 homogeneous matrix, last row 0 0 0 1) or
    ``matrix`` (3 x 3) and ``translation`` (3 values).

    Parameters
    ----------
    path : str
        YAML file.

    Returns
    -------
    matrix : numpy.ndarray
        3 x 3.
    translation : numpy.ndarray
        Length 3.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the content is not one of the formats above, has non-finite values, or
        is (nearly) singular.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Channel registration file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    try:
        if "affine" in data:
            full = np.asarray(data["affine"], dtype=float)
            if full.shape != (4, 4):
                raise ValueError(f"affine must be 4 x 4, got {full.shape}")
            if not np.allclose(full[3], [0, 0, 0, 1]):
                raise ValueError("the last row of affine must be 0 0 0 1")
            matrix, translation = full[:3, :3], full[:3, 3]
        elif "matrix" in data and "translation" in data:
            matrix = np.asarray(data["matrix"], dtype=float)
            translation = np.asarray(data["translation"], dtype=float)
            if matrix.shape != (3, 3) or translation.shape != (3,):
                raise ValueError("matrix must be 3 x 3 and translation 3 values")
        else:
            raise ValueError("expected 'affine' (4 x 4) or 'matrix' (3 x 3) and "
                             "'translation' (3)")
    except (TypeError, ValueError) as err:
        raise ValueError(f"{path}: {err}") from None
    if not (np.isfinite(matrix).all() and np.isfinite(translation).all()):
        raise ValueError(f"{path}: the registration has non-finite values")
    if abs(np.linalg.det(matrix)) < 1e-6:
        raise ValueError(f"{path}: the registration matrix is singular")
    return matrix, translation


def _shear_matrix(shear_x: float, shear_z: float) -> np.ndarray:
    """``S`` of ``shear_volume`` in (z, y, x): ``z' = z + shear_z y``, ``x' = x + shear_x z``."""
    return np.array([[1.0, shear_z, 0.0], [0.0, 1.0, 0.0], [shear_x, 0.0, 1.0]])


def sampling_transform(reg: ChannelRegistration, shear_x: float, shear_z: float,
                       z_in_ref: int, z_out: int) -> Tuple[np.ndarray, np.ndarray]:
    """Matrix ``M`` and offset ``b`` with ``p = M q + b`` (output voxel q -> input p).

    Parameters
    ----------
    reg : ChannelRegistration
        The registration.
    shear_x, shear_z : float
        Shears of the stitch.
    z_in_ref : int
        z size of the reference's stitching crop (before the shear).
    z_out : int
        z size after the shear. For a negative ``shear_z`` the output z is shifted by
        ``z_out - z_in_ref`` so that it starts at 0, as the stitch does.

    Returns
    -------
    M, b : numpy.ndarray
        3 x 3 and length 3.
    """
    shear = _shear_matrix(shear_x, shear_z)
    pad = np.array([(z_out - z_in_ref) if shear_z < 0 else 0.0, 0.0, 0.0])
    a_inv = np.linalg.inv(reg.matrix)
    s_inv = np.linalg.inv(shear)
    origin = np.asarray(reg.origin, dtype=float)
    matrix = a_inv @ s_inv
    offset = a_inv @ (-s_inv @ pad + origin - reg.translation)
    return matrix, offset


def _crop_box_sampling(shear_x: float, shear_z: float, z_in_ref: int,
                       z_out: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sampling transform of the shear alone: output voxel -> voxel of the reference
    stitching crop (before the shear). Used to keep only what lies inside that crop."""
    shear = _shear_matrix(shear_x, shear_z)
    pad = np.array([(z_out - z_in_ref) if shear_z < 0 else 0.0, 0.0, 0.0])
    s_inv = np.linalg.inv(shear)
    return s_inv, -s_inv @ pad


def register_tile(corrected: da.Array, reg: ChannelRegistration, shear_x: float,
                  shear_z: float, ref_shape: Tuple[int, int, int],
                  out_shape: Tuple[int, int, int],
                  out_chunks: Tuple[int, int, int]) -> da.Array:
    """Lazily resample a corrected, split, flipped chunk into the sheared reference frame.

    Parameters
    ----------
    corrected : dask.array.Array
        uint16 corrected chunk, shape (Z, Ys, X) (its own split frame).
    reg : ChannelRegistration
        Registration to the reference channel.
    shear_x, shear_z : float
        Shears of the stitch.
    ref_shape : tuple of int
        Shape (Z, Y, X) of the reference's stitching-cropped chunk.
    out_shape : tuple of int
        ``sheared_shape(ref_shape, shear_x, abs(shear_z))``.
    out_chunks : tuple of int
        Dask chunks of the result (y should be the whole chunk).

    Returns
    -------
    dask.array.Array
        uint16 array of shape ``out_shape``; zeros where the channel has no data.
    """
    matrix, offset = sampling_transform(reg, shear_x, shear_z, ref_shape[0],
                                        out_shape[0])
    order = 0 if reg.is_identity else reg.order
    data = affine_transform(corrected, matrix=matrix, offset=offset, order=order,
                            mode="constant", cval=0.0, output_shape=out_shape,
                            output_chunks=out_chunks)
    # the reference's stitching crop is a hard limit: keep only the voxels whose
    # place in the reference frame lies inside it (the same parallelogram the
    # reference channel's own chunks have), whatever data the channel has beyond it
    box_m, box_b = _crop_box_sampling(shear_x, shear_z, ref_shape[0], out_shape[0])
    inside = affine_transform(
        da.ones(ref_shape, dtype=np.uint8, chunks=out_chunks), matrix=box_m,
        offset=box_b, order=0, mode="constant", cval=0, output_shape=out_shape,
        output_chunks=out_chunks)
    return da.where(inside > 0, data, 0).astype(np.uint16)


def registered_weights(weight: np.ndarray, reg: ChannelRegistration, shear_x: float,
                       shear_z: float, ref_shape: Tuple[int, int, int],
                       out_shape: Tuple[int, int, int]) -> np.ndarray:
    """Blend confidence ``weight`` (Z, Ys) of the channel, in the sheared reference frame.

    The confidence only depends on (z, y) in the channel's frame. It is resampled
    with the (z, y) part of the same transform, evaluated at the middle of x
    (exact unless the registration couples z or y with x). Nearest neighbour; zero
    where the channel has no data.

    Returns
    -------
    numpy.ndarray
        float32, shape ``(out_shape[0], out_shape[1])``.
    """
    matrix, offset = sampling_transform(reg, shear_x, shear_z, ref_shape[0],
                                        out_shape[0])
    x_mid = 0.5 * (out_shape[2] - 1)
    matrix2 = matrix[:2, :2]
    offset2 = matrix[:2, 2] * x_mid + offset[:2]
    box_m, box_b = _crop_box_sampling(shear_x, shear_z, ref_shape[0], out_shape[0])
    inside = scipy.ndimage.affine_transform(
        np.ones(ref_shape[:2], dtype=np.uint8), matrix=box_m[:2, :2],
        offset=box_m[:2, 2] * x_mid + box_b[:2], output_shape=out_shape[:2], order=0,
        mode="constant", cval=0)
    out = scipy.ndimage.affine_transform(
        weight, matrix=matrix2, offset=offset2, output_shape=out_shape[:2], order=0,
        mode="constant", cval=0.0)
    return np.where(inside > 0, out, 0.0).astype(np.float32)


def reference_origin(ref_split_y: Tuple[int, int], ref_stitch_y: Tuple[int, int],
                     ref_stitch_z: Tuple[Optional[int], Optional[int]], ref_depth: int,
                     ref_flip: bool, layer: int) -> Tuple[float, float, float]:
    """Corner (z, y, 0) of the reference stitching crop in its split, flipped frame.

    Parameters
    ----------
    ref_split_y, ref_stitch_y : tuple of int
        ``(lo, hi)`` of the reference's split and stitching y ranges, in voxels of
        the layer (as ``level_range`` gives them).
    ref_stitch_z : tuple of int or None
        ``(lo, hi)`` of the stitching z range in the RAW (unflipped) z of the
        reference, in voxels of the layer; None for the whole depth.
    ref_depth : int
        z size of the reference's raw chunk at the layer.
    ref_flip : bool
        Whether the reference's camera flips z.
    layer : int
        Unused except for documentation: all values are already in layer voxels.

    Returns
    -------
    tuple of float
        (z, y, 0).
    """
    z_lo, z_hi = ref_stitch_z
    if z_lo is None or z_hi is None:
        z_lo, z_hi = 0, ref_depth
    oz = (ref_depth - z_hi) if ref_flip else z_lo
    return float(oz), float(ref_stitch_y[0] - ref_split_y[0]), 0.0


def scale_translation(translation: np.ndarray, layer: int) -> np.ndarray:
    """Full-resolution translation -> voxels of ``layer``."""
    return np.asarray(translation, dtype=float) / (2 ** layer)


def describe(reg: ChannelRegistration) -> str:
    """One-line description for the log."""
    kind = "identity" if reg.is_identity else f"order {reg.order}"
    t = ", ".join(f"{v:.2f}" for v in reg.translation)
    return f"registration to the reference ({kind}), translation (z,y,x) = ({t}) px"