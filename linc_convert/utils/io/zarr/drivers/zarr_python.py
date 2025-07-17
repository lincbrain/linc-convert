import ast
from typing import Iterator, Union, Literal, Any

import numpy as np
import zarr
import zarr.codecs
from numpy.typing import DTypeLike
from zarr.core.chunk_key_encodings import ChunkKeyEncodingParams

from linc_convert.utils.zarr_config import ZarrConfig
from linc_convert.utils.io.zarr.abc import ZarrArray, ZarrGroup


class ZarrPythonArray(ZarrArray):
    def __init__(self, array: zarr.Array):
        super().__init__(str(array.store_path))
        self._array = array

    @property
    def attrs(self):
        return self._array.attrs

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def dtype(self) -> np.dtype:
        return self._array.dtype

    @property
    def chunks(self) -> tuple:
        return self._array.chunks

    @property
    def shards(self) -> tuple:
        return self._array.shards

    @property
    def zarr_version(self) -> int:
        return self._array.metadata.zarr_format

    @property
    def ndim(self) -> int:
        return self._array.ndim

    def __setitem__(self, key, value):
        self._array[key] = value

    def __getitem__(self, item):
        return self._array[item]

    def __getattr__(self, name):
        if name == "_array":
            return self._array
        if hasattr(self._array, name):
            return getattr(self._array, name)
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'")


class ZarrPythonGroup(ZarrGroup):
    @classmethod
    def from_config(cls, zarr_config: ZarrConfig) -> 'ZarrPythonGroup':
        store = zarr.storage.LocalStore(zarr_config.out)
        return cls(zarr.group(store=store,
                              # TODO: figure out overwrite
                              # overwrite=overwrite,
                              zarr_format=zarr_config.zarr_version))

    def __init__(self, zarr_group: zarr.Group):
        super().__init__(str(zarr_group.store_path))
        self._zgroup = zarr_group

    def create_group(self, *args, **kwargs):
        """
        Create a new subgroup in the group.
        """
        return ZarrPythonGroup(self._zgroup.create_group(*args, **kwargs))

    def keys(self):
        yield from self._zgroup.keys()

    @property
    def attrs(self):
        return self._zgroup.attrs

    def __iter__(self) -> Iterator[str]:
        yield from self.keys()

    def __getitem__(self, key) -> Union[ZarrPythonArray, 'ZarrPythonGroup']:
        if key not in self._zgroup:
            raise KeyError(f"Key '{key}' not found in group '{self.path}'")

        item = self._zgroup[key]
        if isinstance(item, zarr.Group):
            return ZarrPythonGroup(item)
        elif isinstance(item, zarr.Array):
            return ZarrPythonArray(item)
        else:
            raise TypeError(f"Unsupported item type: {type(item)}")

    def __getattr__(self, name):
        return getattr(self._zgroup, name)

    def create_array(self,
                     name: str,
                     shape: tuple,
                     zarr_config: ZarrConfig = None,
                     dtype: DTypeLike = np.int32,
                     data=None,
                     **kwargs
                     ) -> zarr.Array:
        if zarr_config is None:
            arr = self._zgroup.create_array(name, shape, dtype, **kwargs)
            if data is not None:
                arr[:] = data
            return ZarrPythonArray(arr)

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
            "compressors": make_compressor(compressor, zarr_config.zarr_version,
                                           **compressor_opt),
        }

        chunk_key_encoding = dimension_separator_to_chunk_key_encoding(
            zarr_config.dimension_separator, zarr_config.zarr_version)
        if chunk_key_encoding:
            opt["chunk_key_encoding"] = chunk_key_encoding
        arr = self._zgroup.create_array(name=name,
                                        shape=shape,
                                        **opt)
        if data:
            arr[:] = data
        return ZarrPythonArray(arr)

    def create_array_from_base(self, name: str, shape: tuple, data=None,
                               **kwargs) -> 'ZarrPythonArray':
        """
        Create a new array using the properties from a base_level object.
        """
        base_level = self['0']
        opts = dict(
            dtype=base_level.dtype,
            chunks=base_level.chunks,
            shards=getattr(base_level, "shards", None),
            filters=getattr(base_level._array, "filters", None),
            compressors=getattr(base_level._array, "compressors", None),
            fill_value=getattr(base_level._array, "fill_value", None),
            order=getattr(base_level._array, "order", None),
            attributes=getattr(getattr(base_level._array, "metadata", None),
                               "attributes", None),
            overwrite=True,
        )
        # Handle extra options based on metadata type
        meta = getattr(base_level, "metadata", None)
        if meta is not None:
            if hasattr(meta, "dimension_separator"):
                opts["chunk_key_encoding"] = dimension_separator_to_chunk_key_encoding(
                    meta.dimension_separator, 2)
            if hasattr(meta, "chunk_key_encoding"):
                opts["chunk_key_encoding"] = getattr(meta, "chunk_key_encoding", None)
            if hasattr(base_level, "serializer"):
                opts["serializer"] = getattr(base_level, "serializer", None)
            if hasattr(meta, "dimension_names"):
                opts["dimension_names"] = getattr(meta, "dimension_names", None)
        # Remove None values
        opts = {k: v for k, v in opts.items() if v is not None}
        opts.update(kwargs)
        arr = self._zgroup.create_array(name=name, shape=shape, **opts)
        if data is not None:
            arr[:] = data
        return ZarrPythonArray(arr)

    @property
    def zarr_version(self) -> Literal[2, 3]:
        return self._zgroup.metadata.zarr_format

    def __delitem__(self, key):
        del self._zgroup[key]

    def _get_zarr_python_group(self):
        return self._zgroup


def make_compressor(name: str | None, zarr_version: Literal[2, 3], **prm: dict) -> Any:
    """Build compressor object from name and options."""
    if not isinstance(name, str):
        return name

    if zarr_version == 2:
        import numcodecs
        compressor_map = {
            "blosc": numcodecs.Blosc,
            "zlib": numcodecs.Zstd,
        }
    elif zarr_version == 3:
        import zarr.codecs
        compressor_map = {
            "blosc": zarr.codecs.BloscCodec,
            "zlib": zarr.codecs.ZstdCodec,
        }
    else:
        raise ValueError()
    name = name.lower()

    if name not in compressor_map:
        raise ValueError('Unknown compressor', name)
    Compressor = compressor_map[name]

    return Compressor(**prm)


SHARD_FILE_SIZE_LIMIT = (2 *  # compression ratio
                         2 *  # TB
                         2 ** 30  # GB->Bytes
                         )


def open_zarr_group(zarr_config: ZarrConfig):
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
        "compressors": make_compressor(compressor, zarr_config.zarr_version,
                                       **compressor_opt),
    }

    chunk_key_encoding = dimension_separator_to_chunk_key_encoding(
        zarr_config.dimension_separator, zarr_config.zarr_version)
    if chunk_key_encoding:
        opt["chunk_key_encoding"] = chunk_key_encoding
    arr = omz.create_array(name=name,
                           shape=shape,
                           **opt)
    if data:
        arr[:] = data
    return arr


def dimension_separator_to_chunk_key_encoding(dimension_separator, zarr_version):
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
