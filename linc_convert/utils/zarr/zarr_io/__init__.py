import ast
import logging
from abc import ABC, abstractmethod, ABCMeta
from os import PathLike
from typing import Iterator, Literal, Optional, Tuple, Any, Union, Mapping

import dask.array as da
import numpy as np
import tqdm
import zarr
from dask.diagnostics import ProgressBar
from numpy._typing import DTypeLike

from linc_convert.utils.zarr import ZarrConfig
from linc_convert.utils.zarr.compressor import make_compressor
from linc_convert.utils.zarr.create_array import compute_zarr_layout, \
    dimension_separator_to_chunk_key_encoding
from linc_convert.utils.zarr.generate_pyramid import compute_next_level, default_levels, \
    next_level_shape

logger = logging.getLogger(__name__)


class ZarrNode(ABC):
    """
    Base class for any Zarr-like object (group or array).
    """

    def __init__(self, store_path: Union[str, PathLike]):
        # record where on disk (or out‐of core store) this node lives
        self._store_path = store_path

    @property
    def store_path(self) -> str:
        return self._store_path

    @abstractmethod
    def attrs(self):
        """Access metadata/attributes for this node."""
        ...

    @property
    @abstractmethod
    def zarr_version(self) -> int:
        """
        Return the Zarr format version (e.g., 2 or 3).
        """
        ...


class ZarrArray(ZarrNode):
    """
    Abstract interface for a Zarr array (n-dimensional data).
    """

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of the array.
        """
        ...

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """
        Data type of the array.
        """
        ...

    @property
    @abstractmethod
    def chunks(self) -> Tuple[int, ...]:
        """
        Chunk shape for the array.
        """
        ...

    @property
    @abstractmethod
    def shards(self) -> Optional[Tuple[int, ...]]:
        """
        Shard shape, if supported; otherwise None.
        """
        ...

    @property
    @abstractmethod
    def ndim(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, key: Any) -> Any:
        """
        Read data from the array.
        """
        ...

    @abstractmethod
    def __setitem__(self, key: Any, value: Any) -> None:
        """
        Write data to the array.
        """
        ...


class ZarrGroup(ZarrNode, metaclass=ABCMeta):
    """
    Abstract interface for a Zarr group (container of arrays and/or subgroups).
    """

    @classmethod
    @abstractmethod
    def from_config(cls, zarr_config: ZarrConfig, overwrite=False) -> 'ZarrGroup':
        ...

    @abstractmethod
    def __getitem__(self, name: str) -> ZarrNode:
        """
        Get a subgroup or array by name within this group.
        """
        ...

    @abstractmethod
    def create_group(self, name: str, overwrite: bool = False) -> 'ZarrGroup':
        """
        Create or open a subgroup within this group.
        """
        ...

    @abstractmethod
    def create_array(
            self,
            name: str,
            shape: Tuple[int, ...],
            **kwargs
    ) -> 'ZarrArray':
        """
        Create a new array within this group.
        """
        ...

    @abstractmethod
    def create_array_from_base(
            self,
            name: str,
            shape: Tuple[int, ...],
            data: Optional[Any] = None,
            **kwargs
    ) -> 'ZarrArray':
        """
        Create a new array using metadata of an existing base-level array.
        """
        ...

    @abstractmethod
    def generate_pyramid(
            self,
            levels: int = -1,
            ndim: int = 3,
            mode: str = "median",
            no_pyramid_axis: Optional[int] = None,
    ) -> list[list[int]]:
        """
        Generate a multiresolution pyramid across spatial dimensions.
        Returns list of shapes for each level.
        """
        ...

    @abstractmethod
    def __delitem__(self, key):
        ...


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

    # def __setattr__(self, name, value):
    #     if name == "_array" and self._array is not None:
    #         raise AttributeError("Cannot set '_array' attribute directly.")
    #     elif hasattr(self._array, name):
    #         setattr(self._array, name, value)
    #     else:
    #         raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class ZarrPythonGroup(ZarrGroup):
    @classmethod
    def from_config(cls, zarr_config: ZarrConfig) -> 'ZarrPythonGroup':
        store = zarr.storage.LocalStore(zarr_config.out)
        return cls(zarr.group(store=store,
                              #TODO: figure out overwrite
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

    def generate_pyramid(
            self,
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

        base_level = self['0']
        base_shape = list(base_level.shape)
        chunk_size = base_level.chunks

        window_func = {"median": da.median, "mean": da.mean}[mode]

        batch_shape, spatial_shape = base_shape[:-ndim], base_shape[-ndim:]
        all_shapes = [spatial_shape]

        if levels == -1:
            levels = default_levels(spatial_shape, chunk_size[-ndim:], no_pyramid_axis)

        for lvl in tqdm.tqdm(range(1, levels + 1)):
            prev_shape = spatial_shape
            spatial_shape = next_level_shape(prev_shape, no_pyramid_axis)
            all_shapes.append(spatial_shape)
            logger.info(f"Compute level {lvl} with shape {spatial_shape}")
            arr = self.create_array_from_base(
                str(lvl), shape=batch_shape + spatial_shape)
            dat = da.from_zarr(self[str(lvl - 1)]._array)
            dat = compute_next_level(dat, ndim, no_pyramid_axis, window_func)
            if arr.shards:
                dat = dat.rechunk(arr.shards)
            else:
                dat = dat.rechunk(arr.chunks)
            with ProgressBar():
                dat.store(arr)
        return all_shapes

    @property
    def zarr_version(self) -> Literal[2, 3]:
        return self._zgroup.metadata.zarr_format

    def __delitem__(self, key):
        del self._zgroup[key]

class ZarrTSArray(ZarrArray):
    import tensorstore
    def __init__(self, ts: tensorstore.TensorStore):
        super().__init__(ts.kvstore.path)
        self._ts = ts
    @property
    def shape(self): return self._ts.shape
    @property
    def ndim(self):  return self._ts.ndim
    @property
    def dtype(self): return self._ts.dtype.numpy_dtype
    @property
    def chunks(self): return self._ts.chunk_layout.read_chunk.shape
    @property
    def shards(self) -> Optional[Tuple[int, ...]]:
        if self._ts.chunk_layout.read_chunk.shape == self._ts.chunk_layout.write_chunk.shape:
            return None
        else:
            return self._ts.chunk_layout.write_chunk.shape

    def __getitem__(self, idx): return self._ts[idx].read().result()
    def __setitem__(self, idx, val): self._ts[idx] = val

def ZarrTSGroup(ZarrGroup):
    def __init__(self, path: Union[str, PathLike]):
        from upath import UPath
        if not isinstance(path, UPath):
            path = UPath(path)
        self._path = path
        super.__init__(path)

    @property
    def attrs(self) -> Mapping[str, Any]:
        return {}

    def __getitem__(self, key) -> Union[ZarrTSArray, 'ZarrTSGroup']:
        pass
    def keys(self):
        yield from self._zgroup.keys()


def open():
    pass

def open_group(backend=None):
    pass


def from_config(zarr_config: ZarrConfig) -> 'ZarrGroup':
    """
    Create a ZarrGroup from a ZarrConfig.
    
    Parameters
    ----------
    zarr_config : ZarrConfig
        Configuration for the Zarr group.
    
    Returns
    -------
    ZarrGroup
        An instance of ZarrGroup based on the configuration.
    """
    if zarr_config.driver == "zarr-python":
        return ZarrPythonGroup.from_config(zarr_config)
    elif zarr_config.driver == "tensorstore":
        return ZarrGroup.from_config(zarr_config)
    elif zarr_config.driver == "zarrita":
        raise NotImplementedError(f"{zarr_config.driver} is not yet supported")
    else:
        raise NotImplementedError(f"Driver '{zarr_config.driver}' is not supported.")
