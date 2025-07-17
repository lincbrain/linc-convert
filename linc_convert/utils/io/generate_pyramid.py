import itertools
import logging
import math
from typing import Literal, Optional

import dask
import dask.array as da
import numpy as np
import tensorstore as ts
import tqdm
import zarr
from dask.diagnostics import ProgressBar
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata

from linc_convert.utils.math import ceildiv

logger = logging.getLogger(__name__)


def generate_pyramid_old(
        omz: zarr.Group,
        levels: int = -1,
        ndim: int = 3,
        max_load: int = 512,
        mode: Literal["mean", "median"] = "median",
        no_pyramid_axis: Optional[int] = None,
) -> list[list[int]]:
    """
    Generate the levels of a pyramid in an existing Zarr.

    Parameters
    ----------
    omz : zarr.Group
        Zarr group object
    levels : int
        Number of additional levels to generate.
        By default, stop when all dimensions are smaller than their
        corresponding chunk size.
    ndim : int
        Number of spatial dimensions.
    max_load : int
        Maximum number of voxels to load along each dimension.
    mode : {"mean", "median"}
        Whether to use a mean or median moving window.
    no_pyramid_axis : int | None
        Axis that should not be downsampled. If None, downsample
        across all three dimensions.
    Returns
    -------
    shapes : list[list[int]]
        Shapes of all levels, from finest to coarsest, including the
        existing top level.
    """
    if max_load % 2:
        max_load += 1

    # Read properties from base level
    base_level = omz["0"]
    base_shape = list(base_level.shape)
    chunk_size = base_level.chunks
    opts = get_zarray_options(base_level)

    # Select windowing function
    window_func = {"median": np.median, "mean": np.mean}[mode]

    batch_shape, spatial_shape = base_shape[:-ndim], base_shape[-ndim:]
    all_shapes = [spatial_shape]

    # Compute default number of levels based on chunk size
    if levels == -1:
        levels = default_levels(spatial_shape, chunk_size[-ndim:], no_pyramid_axis)

    for lvl in tqdm.tqdm(range(1, levels + 1)):
        # Compute downsampled shape
        prev_shape = spatial_shape
        spatial_shape = next_level_shape(prev_shape, no_pyramid_axis)
        all_shapes.append(spatial_shape)
        logger.info("Compute level", lvl, "with shape", spatial_shape)

        arr = omz.create_array(str(lvl), shape=batch_shape + spatial_shape, **opts)

        # Iterate across `max_load` chunks
        # (note that these are unrelated to underlying zarr chunks)
        grid_shape = [ceildiv(n, max_load) for n in prev_shape]

        for chunk_index in tqdm.tqdm(
                itertools.product(*[range(x) for x in grid_shape])):

            # Read one chunk of data at the previous resolution
            slicer = [Ellipsis] + [
                slice(i * max_load, min((i + 1) * max_load, n))
                for i, n in zip(chunk_index, prev_shape)
            ]
            dat = omz[str(lvl - 1)][tuple(slicer)]

            # Discard the last voxel along odd dimensions
            # if one dimension has length 1, it should not be cropped
            crop = [
                0 if y == 1 else x % 2 for x, y in zip(dat.shape[-ndim:], prev_shape)
            ]
            # Only crop the axes that are downsampled
            if no_pyramid_axis is not None:
                crop[no_pyramid_axis] = 0
            slcr = [slice(-1) if x else slice(None) for x in crop]
            dat = dat[tuple([Ellipsis, *slcr])]

            # last strip had a single voxel and become empty after cropping, nothing to do
            if 0 in dat.shape:
                continue

            patch_shape = dat.shape[-ndim:]

            # Reshape into patches of shape 2x2x2
            windowed_shape = [
                x for n in patch_shape for x in (max(n // 2, 1), min(n, 2))
            ]
            if no_pyramid_axis is not None:
                windowed_shape[2 * no_pyramid_axis] = patch_shape[no_pyramid_axis]
                windowed_shape[2 * no_pyramid_axis + 1] = 1

            dat = dat.reshape(tuple(batch_shape + windowed_shape))
            # -> last `ndim` dimensions have shape 2x2x2
            dat = dat.transpose(
                list(range(len(batch_shape)))
                + list(range(len(batch_shape), len(batch_shape) + 2 * ndim, 2))
                + list(range(len(batch_shape) + 1, len(batch_shape) + 2 * ndim, 2))
            )
            # -> flatten patches
            smaller_shape = [max(n // 2, 1) for n in patch_shape]
            if no_pyramid_axis is not None:
                smaller_shape[no_pyramid_axis] = patch_shape[no_pyramid_axis]

            dat = dat.reshape(tuple(batch_shape + smaller_shape + [-1]))

            # Compute the median/mean of each patch
            dtype = dat.dtype
            dat = window_func(dat, axis=-1)
            dat = dat.astype(dtype)

            # Write output
            slicer = [Ellipsis] + [
                slice(i * max_load // 2, min((i + 1) * max_load // 2, n))
                if axis_index != no_pyramid_axis
                else slice(i * max_load, min((i + 1) * max_load, n))
                for i, axis_index, n in zip(chunk_index, range(ndim), spatial_shape)
            ]

            arr[tuple(slicer)] = dat

    return all_shapes


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


def get_zarray_options(base_level):
    from .zarr_io.drivers.zarr_python import dimension_separator_to_chunk_key_encoding

    opts = dict(
        dtype=base_level.dtype,
        chunks=base_level.chunks,
        shards=base_level.shards,
        filters=base_level.filters,
        compressors=base_level.compressors,
        fill_value=base_level.fill_value,
        order=base_level.order,
        attributes=base_level.metadata.attributes,
        overwrite=True,
    )
    if isinstance(base_level.metadata, ArrayV2Metadata):
        opts_extra = dict(
            chunk_key_encoding=dimension_separator_to_chunk_key_encoding(
                base_level.metadata.dimension_separator, 2)
        )
    elif isinstance(base_level.metadata, ArrayV3Metadata):
        opts_extra = dict(
            chunk_key_encoding=base_level.metadata.chunk_key_encoding,
            serializer=base_level.serializer,
            dimension_names=base_level.metadata.dimension_names, )
    else:
        opts_extra = {}
    opts.update(**opts_extra)
    return opts


class _TSAdapter:
    def __init__(self, ts):
        self._ts = ts

    @property
    def shape(self): return tuple(self._ts.shape)

    @property
    def ndim(self):  return self._ts.ndim

    @property
    def dtype(self):
        # Expose the NumPy dtype here:
        return self._ts.dtype.numpy_dtype

    def __getitem__(self, idx):
        return self._ts[idx].read().result()


def generate_pyramid(
        omz: zarr.Group,
        levels: int = -1,
        ndim: int = 3,
        mode: Literal["mean", "median"] = "median",
        no_pyramid_axis: Optional[int] = None,
) -> list[list[int]]:
    """
    Generate the levels of a pyramid in an existing Zarr.

    Parameters
    ----------
    omz : zarr.Group
        Zarr group object
    levels : int
        Number of additional levels to generate.
        By default, stop when all dimensions are smaller than their
        corresponding chunk size.
    ndim : int
        Number of spatial dimensions.
    mode : {"mean", "median"}
        Whether to use a mean or median moving window.
    no_pyramid_axis : int | None
        Axis that should not be downsampled. If None, downsample
        across all three dimensions.
    Returns
    -------
    shapes : list[list[int]]
        Shapes of all levels, from finest to coarsest, including the
        existing top level.
    """
    from .zarr_io.drivers.tensorstore import default_read_config
    from .zarr_io.drivers.tensorstore import default_write_config

    # Read properties from base level
    base_level = omz["0"]
    base_shape = list(base_level.shape)
    chunk_size = base_level.chunks
    opts = get_zarray_options(base_level)

    # Select windowing function
    window_func = {"median": da.median, "mean": da.mean}[mode]

    batch_shape, spatial_shape = base_shape[:-ndim], base_shape[-ndim:]
    all_shapes = [spatial_shape]

    # Compute default number of levels based on chunk size
    if levels == -1:
        levels = default_levels(spatial_shape, chunk_size[-ndim:], no_pyramid_axis)

    rconfig = default_read_config(str(omz["0"].store_path))
    reader = ts.open(rconfig).result()

    reader = _TSAdapter(reader)
    dat = da.from_array(reader, chunks=opts["chunks"])
    tasks = []
    for lvl in tqdm.tqdm(range(1, levels + 1)):
        prev_shape = spatial_shape
        spatial_shape = next_level_shape(prev_shape, no_pyramid_axis)
        all_shapes.append(spatial_shape)
        logger.info(f"Compute level {lvl} with shape {spatial_shape}")
        arr = omz.create_array(str(lvl), shape=batch_shape + spatial_shape, **opts)

        # dat = da.from_zarr(omz[str(lvl - 1)])

        wconfig = default_write_config(str(arr.store_path),
                                       shape=batch_shape + spatial_shape,
                                       dtype=dat.dtype, chunk=opts["chunks"],
                                       shard=opts["shards"],
                                       version=omz.info._zarr_format)
        wconfig["delete_existing"] = True
        wconfig["create"] = True
        writer = ts.open(wconfig).result()
        dat = compute_next_level(dat, ndim, no_pyramid_axis, window_func)
        # TODO: this exists even without sharding
        if arr.shards:
            dat = dat.rechunk(arr.shards)
        else:
            dat = dat.rechunk(arr.chunks)
        dat = dat.persist()
        with ProgressBar():
            dat.store(writer)
            # TODO: delay this task, write together
    return all_shapes


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


def compute_next_level_old(arr, ndim, no_pyramid_axis, window_func):
    batch_shape, prev_shape = arr.shape[:-ndim], arr.shape[-ndim:]
    batch_shape = list(batch_shape)
    crop = [
        0 if i == 1 else i % 2 for i in prev_shape]
    if no_pyramid_axis is not None:
        crop[no_pyramid_axis] = 0
    slcr = [slice(-1) if x else slice(None) for x in crop]
    arr = arr[tuple([Ellipsis, *slcr])]
    if 0 in arr.shape:
        return
    patch_shape = arr.shape[-ndim:]
    # Reshape into patches of shape 2x2x2
    windowed_shape = [
        x for n in patch_shape for x in (max(n // 2, 1), min(n, 2))
    ]
    if no_pyramid_axis is not None:
        windowed_shape[2 * no_pyramid_axis] = patch_shape[no_pyramid_axis]
        windowed_shape[2 * no_pyramid_axis + 1] = 1
    arr = arr.reshape(tuple(batch_shape + windowed_shape))
    # -> last `ndim` dimensions have shape 2x2x2
    arr = arr.transpose(
        list(range(len(batch_shape)))
        + list(range(len(batch_shape), len(batch_shape) + 2 * ndim, 2))
        + list(range(len(batch_shape) + 1, len(batch_shape) + 2 * ndim, 2))
    )
    # -> flatten patches
    smaller_shape = [max(n // 2, 1) for n in patch_shape]
    if no_pyramid_axis is not None:
        smaller_shape[no_pyramid_axis] = patch_shape[no_pyramid_axis]
    arr = arr.reshape(tuple(batch_shape + smaller_shape + [-1]))
    # Compute the median/mean of each patch
    dtype = arr.dtype
    arr = window_func(arr, axis=-1)
    arr = arr.astype(dtype)
    return arr
