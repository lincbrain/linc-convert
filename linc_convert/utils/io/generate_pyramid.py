import logging
import math
from typing import Optional

import dask.array as da

logger = logging.getLogger(__name__)


def default_levels(
        spatial_shape: tuple,
        spatial_chunk: tuple,
        no_pyramid_axis: Optional[int]
        ) -> int:
    default_levels = max(
            int(math.ceil(math.log2(s / spatial_chunk[i])))
            for i, s in enumerate(spatial_shape)
            if no_pyramid_axis is None or i != no_pyramid_axis
            )
    levels = max(default_levels, 0)
    return levels


def next_level_shape(prev_shape: tuple, no_pyramid_axis: Optional[int]) -> list:
    new_shape = []
    for i, length in enumerate(prev_shape):
        if i == no_pyramid_axis:
            new_shape.append(length)
        else:
            new_shape.append(max(1, length // 2))
    return new_shape


def compute_next_level(arr, ndim, no_pyramid_axis=None, window_func=da.mean):
    """
    Compute the next (half-resolution) level of a dask array pyramid along the
    last `ndim` dimensions, optionally skipping reduction along one axis.

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
            1 if (no_pyramid_axis is not None and axis == pyramid_axes[
                no_pyramid_axis]) or arr.shape[axis] == 1
            else 2)
        for axis in pyramid_axes
        }
    dtype = arr.dtype

    return da.coarsen(window_func, arr, factors, trim_excess=True).astype(dtype)
