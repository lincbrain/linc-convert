import ast

import numpy as np
import zarr
from numpy._typing import DTypeLike
from zarr.core.chunk_key_encodings import ChunkKeyEncodingParams

from linc_convert.utils.zarr.compressor import make_compressor
from linc_convert.utils.zarr.zarr_config import ZarrConfig

SHARD_FILE_SIZE_LIMIT = (2 *  # compression ratio
                         2 *  # TB
                         2 ** 30  # TB->Bytes
                         # I use 2GB for now
                         )


def open_zarr_group(zarr_config:ZarrConfig):
    # TODO: check out is not none or empty
    store = zarr.storage.LocalStore(zarr_config.out)
    return zarr.group(store=store, overwrite=True, zarr_format=zarr_config.zarr_version)


def create_array(
        omz: zarr.Group,
        name: str,
        shape: tuple,
        zarr_config: ZarrConfig,
        dtype: DTypeLike = np.int32,
        data=None
) -> zarr.Array:
    compressor = zarr_config.compressor
    compressor_opt = zarr_config.compressor_opt
    chunk, shard = compute_zarr_layout(shape, dtype, zarr_config)

    if isinstance(compressor_opt, str):
        compressor_opt = ast.literal_eval(compressor_opt)

    opt = {
        "chunks": chunk,
        "shards": shard,
        "order": zarr_config.order,
        "dtype": np.dtype(dtype).str,
        "fill_value": None,
        "compressors": make_compressor(compressor, zarr_config.zarr_version, **compressor_opt),
    }

    chunk_key_encoding = dimension_separator_to_chunk_key_encoding(zarr_config.dimension_separator, zarr_config.zarr_version)
    if chunk_key_encoding:
        opt["chunk_key_encoding"] = chunk_key_encoding
    arr= omz.create_array(name=name,
                          shape=shape,
                          **opt)
    if data:
        arr[:] = data
    return arr


def dimension_separator_to_chunk_key_encoding(dimension_separator,zarr_version):
    dimension_separator = dimension_separator
    if dimension_separator == '.' and zarr_version == 2:
        pass
    elif dimension_separator == '/' and zarr_version == 3:
        pass
    else:
        dimension_separator = ChunkKeyEncodingParams(
            name="default" if zarr_version == 3 else "v2",
            separator=dimension_separator)
        return dimension_separator


def compute_zarr_layout(
        shape: tuple,
        dtype: DTypeLike,
        zarr_config: ZarrConfig
) -> tuple[tuple, tuple | None]:
    ndim = len(shape)
    if ndim == 5:
        if zarr_config.no_time:
            raise ValueError('no_time is not supported for 5D data')
        chunk_tc = (
            1 if zarr_config.chunk_time else shape[0],
            1 if zarr_config.chunk_channels else shape[1],
        )
        shard_tc = (
            chunk_tc[0] if zarr_config.shard_time else shape[0],
            chunk_tc[1] if zarr_config.shard_channels else shape[1]
        )

    elif ndim == 4:
        if zarr_config.no_time:
            chunk_tc = (1 if zarr_config.chunk_channels else shape[0],)
            shard_tc = (chunk_tc[0] if zarr_config.shard_channels else shape[0],)
        else:
            chunk_tc = (1 if zarr_config.chunk_time else shape[0],)
            shard_tc = (chunk_tc[0] if zarr_config.shard_time else shape[0],)
    elif ndim == 3:
        chunk_tc = tuple()
        shard_tc = tuple()
    else:
        raise ValueError("Zarr layout only supports 3+ dimensions.")

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
            # If the uniform guess already overshoots the data, clamp to the minimal covering multiplier.
            if m_uniform * chunk_spatial[i] >= shape_spatial[i]:
                M.append(L[i])
            else:
                M.append(m_uniform)
                free_dims.append(i)

        # Iteratively try to increase free dimensions while keeping the overall product ≤ B_multiplier.
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
            free_dims = [i for i in free_dims if
                         M[i] * chunk_spatial[i] < shape_spatial[i]]
        shard = tuple(M[i] * chunk_spatial[i] for i in range(dims))

    shard = shard_tc + shard + shard[-1:] * max(0, 3 - len(shard))
    return chunk, shard
