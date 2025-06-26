import ast
from typing import Iterator, Union, Literal

import numpy as np
import zarr
from numpy._typing import DTypeLike

from linc_convert.utils.zarr import ZarrConfig, make_compressor
from linc_convert.utils.zarr.create_array import compute_zarr_layout, \
    dimension_separator_to_chunk_key_encoding
from linc_convert.utils.zarr.zarr_io.abc import ZarrArray, ZarrGroup


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
