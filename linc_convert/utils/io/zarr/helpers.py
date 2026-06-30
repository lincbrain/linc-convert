"""Helper functions for Zarr I/O."""

import math

import numpy as np
from numpy.typing import DTypeLike

from linc_convert.utils.zarr_config import ZarrConfig

SHARD_FILE_SIZE_LIMIT = (
    2  # compression ratio
    * 2  # GB
    * 2 ** 30  # GB->Bytes
)


def _compute_zarr_layout(
    shape: tuple, dtype: DTypeLike, zarr_config: ZarrConfig
) -> tuple[tuple, tuple | None]:
    ndim = len(shape)
    if ndim == 5:
        if zarr_config.no_time:
            raise ValueError("no_time is not supported for 5D data")
        chunk_tc = (
            1 if zarr_config.chunk_time else shape[0],
            1 if zarr_config.chunk_channels else shape[1],
        )
        shard_tc = (
            chunk_tc[0] if zarr_config.shard_time else shape[0],
            chunk_tc[1] if zarr_config.shard_channels else shape[1],
        )

    elif ndim == 4:
        if zarr_config.no_time:
            chunk_tc = (1 if zarr_config.chunk_channels else shape[0],)
            shard_tc = (chunk_tc[0] if zarr_config.shard_channels else shape[0],)
        else:
            chunk_tc = (1 if zarr_config.chunk_time else shape[0],)
            shard_tc = (chunk_tc[0] if zarr_config.shard_time else shape[0],)
    elif ndim == 3 or ndim == 2:
        chunk_tc = tuple()
        shard_tc = tuple()
   
    else:
        raise ValueError("Zarr layout only supports 2+ dimensions.")

    chunk = zarr_config.chunk
    if len(chunk) > ndim:
        raise ValueError("Provided chunk size has more dimension than data")
    if len(zarr_config.chunk) != ndim:
        chunk = chunk_tc + chunk + chunk[-1:] * max(0, 3 - len(chunk))

    shard = zarr_config.shard

    if isinstance(shard, tuple) and len(shard) > ndim:
        raise ValueError("Provided shard size has more dimension than data")
    # If shard is not used or is fully specified, return early.
    if shard is None or (isinstance(shard, tuple) and len(shard) == ndim):
        return chunk, shard

    chunk_spatial = chunk[-3:]
    if shard == "auto":
        # Compute auto shard sizes based on the file size limit.
        itemsize = dtype.itemsize
        chunk_size = np.prod(chunk_spatial) * itemsize
        shard_size = np.prod(shard_tc) * chunk_size
        B_multiplier = SHARD_FILE_SIZE_LIMIT / shard_size
        multiplier = int(B_multiplier ** (1 / 3))
        if multiplier < 1:
            multiplier = 1

        shape_spatial = shape[-3:]
        # For each spatial dimension, the minimal multiplier needed to cover the data:
        L = [int(np.ceil(s / c)) for s, c in zip(shape_spatial, chunk_spatial)]
        dims = len(chunk_spatial)

        shard = tuple(int(c * multiplier) for c in chunk_spatial)
        m_uniform = int(B_multiplier ** (1 / dims))
        M = []
        free_dims = []
        for i in range(dims):
            # If the uniform guess already overshoots the data, clamp to the minimal
            # covering multiplier.
            if m_uniform * chunk_spatial[i] >= shape_spatial[i]:
                M.append(L[i])
            else:
                M.append(m_uniform)
                free_dims.append(i)

        # Iteratively try to increase free dimensions while keeping the overall
        # product ≤ B_multiplier.
        improved = True
        while improved and free_dims:
            improved = False
            for i in free_dims:
                candidate = M[i] + 1
                # If increasing would exceed the data size in this dimension,
                # clamp to the minimal covering multiplier.
                if candidate * chunk_spatial[i] >= shape_spatial[i]:
                    candidate = L[i]
                new_product = np.prod(
                    [candidate if j == i else M[j] for j in range(dims)]
                )
                if new_product <= B_multiplier and candidate > M[i]:
                    M[i] = candidate
                    improved = True
            # Remove dimensions that have reached or exceeded the data size.
            free_dims = [
                i for i in free_dims if M[i] * chunk_spatial[i] < shape_spatial[i]
            ]
        shard = tuple(M[i] * chunk_spatial[i] for i in range(dims))

    shard = shard_tc + shard + shard[-1:] * max(0, 3 - len(shard))
    return chunk, shard


def auto_shard_size(
    max_shape: list[int],
    itemsize: int | np.dtype | str,
    max_file_size: int = 2 * 1024 ** 4,
    compression_ratio: float = 2,
) -> list[int]:
    """
    Find maximal shard size that ensures file size below cap.

    Parameters
    ----------
    max_shape : list[int]
        Maximum shape along each dimension.
    itemsize : np.dtype or int
        Data type, or data type size
    max_file_size : int
        Maximum file size (default: 2TB).
        S3 has a 5TB/file limit, but given that we use an estimated
        compression factor, we aim for 2TB to leave some leeway.
    compression_ratio : float
        Estimated compression factor.
        I roughly found 2 for bosc-compressed LSM data, when compressing
        only across space and channels (5 channels).

    Returns
    -------
    shard : list[int]
        Estimated shard size along each dimension.
        Returned shards are either max_shape or powers of two.
    """
    if not isinstance(itemsize, int):
        itemsize = np.dtype(itemsize).itemsize

    # Maximum number of elements in the shard
    max_numel = max_file_size * compression_ratio / itemsize

    shard = [1] * len(max_shape)
    while True:
        # If shard larger than volume, we can stop
        if all(x >= s for x, s in zip(shard, max_shape)):
            break
        # Make shard one level larger
        new_shard = [min(2 * x, s) for x, s in zip(shard, max_shape)]
        # If shard is too large, stop and keep previous shard
        if np.prod(new_shard) > max_numel:
            break
        # Otherwise, use larger shard and recurse
        shard = new_shard

    # replace max size with larger power of two
    shard = [2 ** math.ceil(math.log2(x)) for x in shard]
    return shard


def fix_shard_chunk(
    shard: list[int],
    chunk: list[int],
    shape: list[int],
) -> tuple[list[int], list[int]]:
    """
    Fix incompatibilities between chunk and shard size.

    Parameters
    ----------
    shard : list[int]
    chunk : list[int]
    shape : list[int]

    Returns
    -------
    shard : list[int]
    chunk : list[int]
    """
    shard = list(shard)
    chunk = list(chunk)
    for i in range(len(chunk)):
        # if chunk spans the entire volume, match chunk and shard
        if chunk[i] == shape[i] and chunk[i] != shard[i]:
            chunk[i] = shard[i]
        # ensure that shard is a multiple of chunk
        if shard[i] % chunk[i]:
            shard[i] = chunk[i] * int(math.ceil(shard[i] / chunk[i]))
    return shard, chunk
