"""Functions related to generation of downsampled layers in ome-zarr."""

import logging
import math
from typing import Callable, Optional, Sequence

import dask.array as da

logger = logging.getLogger(__name__)


def default_levels(
        spatial_shape: tuple, spatial_chunk: tuple, no_pyramid_axis: Optional[int]
) -> int:
    """
    Compute the default number of downsampling levels for a spatial pyramid.

    For each axis in `spatial_shape` (except the one indexed by
    `no_pyramid_axis`, if given), this computes how many times you can
    halve the dimension (from `spatial_shape[i]`) by the corresponding chunk
    size (`spatial_chunk[i]`) before reaching chunk ≤ 1, and returns the
    maximum of those halving‐counts (rounded up), with a lower bound of 0.

    Parameters
    ----------
    spatial_shape : tuple of int
        The full size of each spatial dimension.
    spatial_chunk : tuple of int
        The chunk size used for each spatial dimension.
    no_pyramid_axis : int or None
        If not None, that axis index will be excluded when computing levels.

    Returns
    -------
    int
        The number of pyramid levels (≥ 0) needed to reduce all applicable
        axes by repeated factors of two.
    """
    default_levels = max(
            int(math.ceil(math.log2(s / spatial_chunk[i])))
            for i, s in enumerate(spatial_shape)
            if no_pyramid_axis is None or i != no_pyramid_axis
    )
    levels = max(default_levels, 0)
    return levels


def next_level_shape(
        prev_shape: Sequence[int], no_pyramid_axis: Optional[int]
) -> list[int]:
    """
    Compute the shape of the next coarser level by halving each dimension.

    Each axis in `prev_shape` is divided by two (integer division),
    clamped to a minimum of 1, except for the axis indexed by
    `no_pyramid_axis`, which remains unchanged.

    Parameters
    ----------
    prev_shape : sequence of int
        Shape of the current level (e.g., [N1, N2, …]).
    no_pyramid_axis : int or None
        Axis index to leave unchanged; if None, all axes are halved.

    Returns
    -------
    list of int
        New shape for the next level, same length as `prev_shape`,
        with each entry equal to `max(1, prev_shape[i] // 2)` or
        unchanged if `i == no_pyramid_axis`.
    """
    new_shape = []
    for i, length in enumerate(prev_shape):
        if i == no_pyramid_axis:
            new_shape.append(length)
        else:
            new_shape.append(max(1, length // 2))
    return new_shape


def compute_next_level(
        arr: da.Array,
        ndim: int,
        no_pyramid_axis: int | None = None,
        window_func: Callable = da.mean,
) -> da.Array:
    """
    Compute the next level of a dask array pyramid.

    Parameters
    ----------
    arr : dask.array.Array
        Input array of shape (..., N1, N2, ..., Nndim).
    ndim : int
        Number of “pyramid” dimensions at the end of arr.ndim.
    no_pyramid_axis : int or None
        If not None, that axis (0 ≤ axis < ndim) will not be downsampled.
    window_func : callable
        A reduction function like da.mean or da.median.

    Returns
    -------
    dask.array.Array
        Array of shape (..., ceil(N1/2), ceil(N2/2), ...,ceil(Nndim/2))
        except on `no_pyramid_axis` where the length is unchanged.
    """
    # figure out which global axes we’re coarsening
    start = arr.ndim - ndim
    pyramid_axes = list(range(start, arr.ndim))

    # build the coarsening factors: 2 along each pyramid dim, except 1 if skip
    factors = {
        axis: (
            1
            if (no_pyramid_axis is not None and axis == pyramid_axes[no_pyramid_axis])
               or arr.shape[axis] == 1
            else 2
        )
        for axis in pyramid_axes
    }
    dtype = arr.dtype

    return da.coarsen(window_func, arr, factors, trim_excess=True).astype(dtype)
