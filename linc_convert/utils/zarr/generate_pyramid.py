import itertools
from typing import Literal

import numpy as np
import zarr
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata

from linc_convert.utils.math import ceildiv
from linc_convert.utils.zarr.create_array import \
    dimension_separator_to_chunk_key_encoding


def generate_pyramid(
    omz: zarr.Group,
    levels: int = -1,
    ndim: int = 3,
    max_load: int = 512,
    mode: Literal["mean", "median"] = "median",
    no_pyramid_axis: int | str | None = None,
) -> list[list[int]]:
    """
    Generate the levels of a pyramid in an existing Zarr.

    Parameters
    ----------
    omz : zarr.Group

    levels : int
        Number of additional levels to generate.
        By default, stop when all dimensions are smaller than their
        corresponding chunk size.
    shard : list[int] | bool | {"auto"} | None
        Shard size.
        * If `None`, use same shard size as the input array;
        * If `False`, no dot use sharding;
        * If `True` or `"auto"`, automatically find shard size;
        * Otherwise, use provided shard size.
    ndim : int
        Number of spatial dimensions.
    max_load : int
        Maximum number of voxels to load along each dimension.
    mode : {"mean", "median"}
        Whether to use a mean or median moving window.

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
    shape = list(base_level.shape)
    chunk_size = base_level.chunks
    opts = get_zarray_options(base_level)

    # Select windowing function
    if mode == "median":
        func = np.median
    else:
        assert mode == "mean"
        func = np.mean

    level = 0
    batch, shape = shape[:-ndim], shape[-ndim:]
    allshapes = [shape]

    while True:
        level += 1

        # Compute downsampled shape
        prev_shape, shape = shape, []
        for i, length in enumerate(prev_shape):
            if i == no_pyramid_axis:
                shape.append(length)
            else:
                shape.append(max(1, length // 2))

        # Stop if seen enough levels or level shape smaller than chunk size
        if levels == -1:
            if all(x <= c for x, c in zip(shape, chunk_size[-ndim:])):
                break
        elif level > levels:
            break

        print("Compute level", level, "with shape", shape)

        allshapes.append(shape)
        omz.create_array(str(level), shape=batch + shape, **opts)

        # Iterate across `max_load` chunks
        # (note that these are unrelared to underlying zarr chunks)
        grid_shape = [ceildiv(n, max_load) for n in prev_shape]
        for chunk_index in itertools.product(*[range(x) for x in grid_shape]):
            print(f"chunk {chunk_index} / {tuple(grid_shape)})", end="\r")

            # Read one chunk of data at the previous resolution
            slicer = [Ellipsis] + [
                slice(i * max_load, min((i + 1) * max_load, n))
                for i, n in zip(chunk_index, prev_shape)
            ]
            fullshape = omz[str(level - 1)].shape
            dat = omz[str(level - 1)][tuple(slicer)]

            # Discard the last voxel along odd dimensions
            crop = [
                0 if y == 1 else x % 2 for x, y in zip(dat.shape[-ndim:], fullshape)
            ]
            # Don't crop the axis not down-sampling
            # cannot do if not no_pyramid_axis since it could be 0
            if no_pyramid_axis is not None:
                crop[no_pyramid_axis] = 0
            slcr = [slice(-1) if x else slice(None) for x in crop]
            dat = dat[tuple([Ellipsis, *slcr])]

            if any(n == 0 for n in dat.shape):
                # last strip had a single voxel, nothing to do
                continue

            patch_shape = dat.shape[-ndim:]

            # Reshape into patches of shape 2x2x2
            windowed_shape = [
                x for n in patch_shape for x in (max(n // 2, 1), min(n, 2))
            ]
            if no_pyramid_axis is not None:
                windowed_shape[2 * no_pyramid_axis] = patch_shape[no_pyramid_axis]
                windowed_shape[2 * no_pyramid_axis + 1] = 1

            dat = dat.reshape(batch + windowed_shape)
            # -> last `ndim`` dimensions have shape 2x2x2
            dat = dat.transpose(
                list(range(len(batch)))
                + list(range(len(batch), len(batch) + 2 * ndim, 2))
                + list(range(len(batch) + 1, len(batch) + 2 * ndim, 2))
            )
            # -> flatten patches
            smaller_shape = [max(n // 2, 1) for n in patch_shape]
            if no_pyramid_axis is not None:
                smaller_shape[no_pyramid_axis] = patch_shape[no_pyramid_axis]

            dat = dat.reshape(batch + smaller_shape + [-1])

            # Compute the median/mean of each patch
            dtype = dat.dtype
            dat = func(dat, axis=-1)
            dat = dat.astype(dtype)

            # Write output
            slicer = [Ellipsis] + [
                slice(i * max_load // 2, min((i + 1) * max_load // 2, n))
                if axis_index != no_pyramid_axis
                else slice(i * max_load, min((i + 1) * max_load, n))
                for i, axis_index, n in zip(chunk_index, range(ndim), shape)
            ]

            omz[str(level)][tuple(slicer)] = dat

    print("")

    return allshapes


def get_zarray_options(base_level):
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
        chunk_key_encoding = dimension_separator_to_chunk_key_encoding(base_level.metadata.dimension_separator, 2)
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