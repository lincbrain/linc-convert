import itertools
import re
from typing import Any, Literal

import nibabel as nib
import numpy as np
import zarr

from linc_convert.utils.math import ceildiv
from linc_convert.utils.unit import convert_unit


def generate_pyramid(
    omz: zarr.Group,
    levels: int | None = None,
    ndim: int = 3,
    max_load: int = 512,
    mode: Literal["mean", "median"] = "median",
    no_pyramid_axis: int | None = None,
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

    Returns
    -------
    shapes : list[list[int]]
        Shapes of all levels, from finest to coarsest, including the
        existing top level.
    """
    if max_load % 2:
        max_load += 1

    # Read properties from base level
    shape = list(omz["0"].shape)
    chunk_size = omz["0"].chunks
    opt = {
        "dimension_separator": omz["0"]._dimension_separator,
        "order": omz["0"]._order,
        "dtype": omz["0"]._dtype,
        "fill_value": omz["0"]._fill_value,
        "compressor": omz["0"]._compressor,
        "chunks": omz["0"].chunks,
    }

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
        if levels is None:
            if all(x <= c for x, c in zip(shape, chunk_size[-ndim:])):
                break
        elif level > levels:
            break

        print("Compute level", level, "with shape", shape)

        allshapes.append(shape)
        omz.create_dataset(str(level), shape=batch + shape, **opt)

        # Iterate across `max_load` chunks
        # (note that these are unrelated to underlying zarr chunks)
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



