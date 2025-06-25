import ast
import json
import logging
from abc import ABC, abstractmethod, ABCMeta
from os import PathLike
import os
from typing import Iterator, Literal, Optional, Tuple, Any, Union, Mapping

import dask.array as da
import numpy as np
import tqdm
import zarr
from dask.diagnostics import ProgressBar
from numpy._typing import DTypeLike
from upath import UPath

from linc_convert.utils.zarr import ZarrConfig
from linc_convert.utils.zarr._zarr import make_kvstore, auto_shard_size, \
    fix_shard_chunk, make_compressor_v2, make_compressor_v3
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
    def from_config(cls, zarr_config: ZarrConfig) -> 'ZarrGroup':
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

    # @abstractmethod
    # def generate_pyramid(
    #         self,
    #         levels: int = -1,
    #         ndim: int = 3,
    #         mode: str = "median",
    #         no_pyramid_axis: Optional[int] = None,
    # ) -> list[list[int]]:
    #     """
    #     Generate a multiresolution pyramid across spatial dimensions.
    #     Returns list of shapes for each level.
    #     """
    #     ...
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
            # dat = da.from_zarr(self[str(lvl - 1)]._array)
            dat = da.from_array(self[str(lvl - 1)], chunks = self[str(lvl - 1)].chunks)
            dat = compute_next_level(dat, ndim, no_pyramid_axis, window_func)
            if arr.shards:
                dat = dat.rechunk(arr.shards)
            else:
                dat = dat.rechunk(arr.chunks)
            dat = dat.persist()
            with ProgressBar():
                dat.store(arr)
        return all_shapes

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

    # def generate_pyramid(
    #         self,
    #         levels: int = -1,
    #         ndim: int = 3,
    #         mode: Literal["mean", "median"] = "median",
    #         no_pyramid_axis: Optional[int] = None,
    # ) -> list[list[int]]:
    #     """
    #     Generate the levels of a pyramid in an existing Zarr.

    #     Parameters
    #     ----------
    #     omz : zarr.Group
    #         Zarr group object
    #     levels : int
    #         Number of additional levels to generate.
    #         By default, stop when all dimensions are smaller than their
    #         corresponding chunk size.
    #     ndim : int
    #         Number of spatial dimensions.
    #     mode : {"mean", "median"}
    #         Whether to use a mean or median moving window.
    #     no_pyramid_axis : int | None
    #         Axis that should not be downsampled. If None, downsample
    #         across all three dimensions.
    #     Returns
    #     -------
    #     shapes : list[list[int]]
    #         Shapes of all levels, from finest to coarsest, including the
    #         existing top level.
    #     """

    #     base_level = self['0']
    #     base_shape = list(base_level.shape)
    #     chunk_size = base_level.chunks

    #     window_func = {"median": da.median, "mean": da.mean}[mode]

    #     batch_shape, spatial_shape = base_shape[:-ndim], base_shape[-ndim:]
    #     all_shapes = [spatial_shape]

    #     if levels == -1:
    #         levels = default_levels(spatial_shape, chunk_size[-ndim:], no_pyramid_axis)

    #     for lvl in tqdm.tqdm(range(1, levels + 1)):
    #         prev_shape = spatial_shape
    #         spatial_shape = next_level_shape(prev_shape, no_pyramid_axis)
    #         all_shapes.append(spatial_shape)
    #         logger.info(f"Compute level {lvl} with shape {spatial_shape}")
    #         arr = self.create_array_from_base(
    #             str(lvl), shape=batch_shape + spatial_shape)
    #         dat = da.from_zarr(self[str(lvl - 1)]._array)
    #         dat = compute_next_level(dat, ndim, no_pyramid_axis, window_func)
    #         if arr.shards:
    #             dat = dat.rechunk(arr.shards)
    #         else:
    #             dat = dat.rechunk(arr.chunks)
    #         with ProgressBar():
    #             dat.store(arr)
    #     return all_shapes

    @property
    def zarr_version(self) -> Literal[2, 3]:
        return self._zgroup.metadata.zarr_format

    def __delitem__(self, key):
        del self._zgroup[key]


import tensorstore as ts
class ZarrTSArray(ZarrArray):
    def __init__(self, ts: ts.TensorStore):
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

    @property
    def zarr_version(self) -> int:
        driver_map = {"zarr3": 3, "zarr": 2}
        return driver_map[self._ts.schema.codec.to_json()['driver']]

    def __getitem__(self, idx): return self._ts[idx].read().result()
    def __setitem__(self, idx, val): self._ts[idx] = val

    @property
    def attrs(self):
        return {}

    @classmethod
    def open(cls, path: PathLike, zarr_version: Literal[2, 3] = 3):

        conf = {
            "kvstore": make_kvstore(path),
            "driver": "zarr3" if zarr_version == 3 else "zarr",
            "open": True,
            "create": False,
            "delete_existing": False,
        }
        return cls(ts.open(conf).result())

# tswriter.schema.codec.to_json()
# tswriter.schema.to_json()
def _is_array(path):
    zarr2_array_file = path / ".zarray"
    if zarr2_array_file.is_file():
        content = zarr2_array_file.read_text()
        content = json.loads(content)
        assert content["zarr_format"] == 2
        return True
    zarr3_array_file = path / "zarr.json"
    if zarr3_array_file.is_file():
        content = zarr3_array_file.read_text()
        content = json.loads(content)
        assert content["zarr_format"] == 3
        if content.get("node_type", None) == "array":
            return True
    return False


def _is_group(path):
    zarr2_group_file = path / ".zgroup"
    if zarr2_group_file.is_file():
        content = zarr2_group_file.read_text()
        content = json.loads(content)
        assert content["zarr_format"] == 2
        return True
    zarr3_group_file = path / "zarr.json"
    if zarr3_group_file.is_file():
        content = zarr3_group_file.read_text()
        content = json.loads(content)
        assert content["zarr_format"] == 3
        if content.get("node_type", None) == "group":
            return True
    return False


def _detect_metadata(path: PathLike) -> Optional[Tuple[str, int]]:
    """
    Look for Zarr metadata files in `path` and return (node_type, version).
    Checks zarr.json (v3), then .zarray/.zgroup (v2).
    """
    # Zarr v3
    z3 = path / 'zarr.json'
    if z3.is_file():
        try:
            meta = json.loads(z3.read_text())
            fmt = meta.get('zarr_format')
            if fmt == 3:
                node = meta.get('node_type', 'array')
                if node in ('array', 'group'):
                    return node, 3
        except json.JSONDecodeError:
            pass
    # Zarr v2
    for fname, ntype in (('.zarray', 'array'), ('.zgroup', 'group')):
        f = path / fname
        if f.is_file():
            try:
                meta = json.loads(f.read_text())
                if meta.get('zarr_format') == 2:
                    return ntype, 2
            except json.JSONDecodeError:
                pass
    return None


def default_write_config(
        path: os.PathLike | str,
        shape: list[int],
        dtype: np.dtype | str,
        chunk: list[int] = [32],
        shard: list[int] | Literal["auto"] | None = None,
        compressor: str = "blosc",
        compressor_opt: dict | None = None,
        version: int = 3,
) -> dict:
    """
    Generate a default TensorStore configuration.

    Parameters
    ----------
    chunk : list[int]
        Chunk size.
    shard : list[int], optional
        Shard size. No sharding if `None`.
    compressor : str
        Compressor name
    version : int
        Zarr version

    Returns
    -------
    config : dict
        Configuration
    """
    from upath import UPath
    path = UPath(path)
    if not path.protocol:
        path = "file://" / path

    # Format compressor
    if version == 3 and compressor == "zlib":
        compressor = "gzip"
    if version == 2 and compressor == "gzip":
        compressor = "zlib"
    compressor_opt = compressor_opt or {}

    # Prepare chunk size
    if isinstance(chunk, int):
        chunk = [chunk]
    chunk = chunk[:1] * max(0, len(shape) - len(chunk)) + chunk

    # Prepare shard size
    if shard:
        if shard == "auto":
            shard = auto_shard_size(shape, dtype)
        if isinstance(shard, int):
            shard = [shard]
        shard = shard[:1] * max(0, len(shape) - len(shard)) + shard

        # Fix incompatibilities
        shard, chunk = fix_shard_chunk(shard, chunk, shape)

    # ------------------------------------------------------------------
    #   Zarr 3
    # ------------------------------------------------------------------
    if version == 3:
        if compressor and compressor != "raw":
            compressor = [make_compressor_v3(compressor, **compressor_opt)]
        else:
            compressor = []

        codec_little_endian = {"name": "bytes", "configuration": {"endian": "little"}}

        if shard:
            chunk_grid = {
                "name": "regular",
                "configuration": {"chunk_shape": shard},
            }

            sharding_codec = {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": chunk,
                    "codecs": [
                        codec_little_endian,
                        *compressor,
                    ],
                    "index_codecs": [
                        codec_little_endian,
                        {"name": "crc32c"},
                    ],
                    "index_location": "end",
                },
            }
            codecs = [sharding_codec]

        else:
            chunk_grid = {"name": "regular", "configuration": {"chunk_shape": chunk}}
            codecs = [
                codec_little_endian,
                *compressor,
            ]

        metadata = {
            "chunk_grid": chunk_grid,
            "codecs": codecs,
            "data_type": np.dtype(dtype).name,
            "fill_value": 0,
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": r"/"},
            },
        }
        config = {
            "driver": "zarr3",
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    #   Zarr 2
    # ------------------------------------------------------------------
    else:
        if compressor and compressor != "raw":
            compressor = make_compressor_v2(compressor, **compressor_opt)
        else:
            compressor = None

        metadata = {
            "chunks": chunk,
            "order": "F",
            "dtype": np.dtype(dtype).str,
            "fill_value": 0,
            "compressor": compressor,
        }
        config = {
            "driver": "zarr",
            "metadata": metadata,
            "key_encoding": r"/",
        }

    # Prepare store
    config["metadata"]["shape"] = shape
    config["kvstore"] = make_kvstore(path)

    return config


def _init_group(group_path: PathLike, version: int):
    group_path.mkdir(parents=True, exist_ok=True)
    if version == 3:
        (group_path / 'zarr.json').write_text(
            json.dumps({'zarr_format': 3, 'node_type': 'group'})
        )
    else:
        (group_path / '.zgroup').write_text(
            json.dumps({'zarr_format': 2})
        )


class ZarrTSGroup(ZarrGroup):

    def __init__(self, store_path: Union[str, PathLike]):
        super().__init__(store_path)
        from upath import UPath
        if not isinstance(store_path, UPath):
            store_path = UPath(store_path)
        self._path = store_path
        metadata = _detect_metadata(store_path)
        assert metadata is not None
        assert metadata[0] == "group"
        self._zarr_version = metadata[1]
        self._zarr_config = None

    @property
    def zarr_version(self):
        return self._zarr_version

    @property
    def attrs(self) -> Mapping[str, Any]:
        return {}

    def keys(self):
        return (
            sub_dir.name for sub_dir in self._path.iterdir()
            if sub_dir.is_dir() and _detect_metadata(sub_dir)
        )

    def __getitem__(self, key) -> Union[ZarrTSArray, 'ZarrTSGroup']:
        metadata = _detect_metadata(self._path / key)
        if metadata is None:
            raise KeyError(key)
        if metadata[0] == "group":
            return ZarrTSGroup(self._path / key)
        if metadata[0] == "array":
            return ZarrTSArray.open(self._path / key, metadata[1])

    def __contains__(self, key) -> bool:
        if not (self._path / key).exists():
            return False
        if _detect_metadata(self._path / key) is None:
            return False
        return True

    def __delitem__(self, key):
        if key in self:
            (self._path / key).rmdir(recursive=True)

    # def generate_pyramid(self, levels: int = -1, ndim: int = 3, mode: str = "median",
    #                      no_pyramid_axis: Optional[int] = None) -> list[list[int]]:
    #     pass

    def create_array_from_base(self, name: str, shape: Tuple[int, ...],
                               data: Optional[Any] = None, **kwargs) -> 'ZarrArray':
        
        spec = self['0']._ts.spec().to_json()
        spec['metadata']['shape'] = shape
        kvstore = make_kvstore(self._path / name)
        spec['kvstore'] = kvstore
        spec['delete_existing'] = True
        spec['create'] = True
        arr = ts.open(spec).result()
        if data is not None:
            arr[:] = data
        return ZarrTSArray(arr)
    

    def create_array(self,
                     name: str,
                     shape: tuple,
                     zarr_config: ZarrConfig = None,
                     dtype: DTypeLike = np.int32,
                     data=None,
                     **kwargs
                    ) -> zarr.Array:
        if zarr_config is None:
            conf = default_write_config(
                self._path / name,
                shape=shape,
                dtype=dtype,
                **kwargs
            )
        else:
            compressor_opt = ast.literal_eval(zarr_config.compressor_opt)
            conf = default_write_config(
                self._path / name,
                shape=shape,
                dtype=dtype,
                chunk=zarr_config.chunk,
                shard=zarr_config.shard,
                compressor=zarr_config.compressor,
                # compressor_opt=,
                version=zarr_config.zarr_version,
            )
        conf['delete_existing'] = True
        conf['create'] = True
        arr = ts.open(conf).result()
        if data is not None:
            arr[:] = data
        return ZarrTSArray(arr)

    def create_group(self, name: str, *, overwrite: bool = False,
                     zarr_version: Literal[2, 3] = None) -> 'ZarrGroup':
        if zarr_version is None:
            zarr_version = self.zarr_version
        if overwrite:
            mode = 'w'
        else:
            mode = 'w-'
        return self.open(self._path / name, mode=mode, zarr_version=zarr_version)

    @classmethod
    def from_config(cls, zarr_config: ZarrConfig) -> 'ZarrGroup':
        return cls.open(zarr_config.out, zarr_version=zarr_config.zarr_version)

    @classmethod
    def open(cls, path: Union[str, PathLike], mode="a", *,
             zarr_version: Literal[2, 3] = 3) -> 'ZarrTSGroup':
        """
        Open a Zarr group from a path.
        
        Parameters
        ----------
        path : Union[str, PathLike]
            Path to the Zarr group.
        mode : str
            Persistence mode:
            'r' means read only (must exist);
            'r+' means read/write (must exist);
            'a' means read/write (create if doesn't exist);
            'w' means create (overwrite if exists);
            'w-' means create (fail if exists).
        
        Returns
        -------
        ZarrTSGroup
            An instance of ZarrTSGroup.
        """
        p = UPath(path)

        if mode in ['r', 'r+']:
            if not p.exists() or not p.is_dir():
                raise FileNotFoundError(f"Group path '{p}' does not exist")
        elif mode == 'w-':
            if p.exists():
                raise FileExistsError(f"Group path '{p}' already exists")
        elif mode == 'a':
            if not p.exists():
                _init_group(p, zarr_version)
        elif mode == 'w':
            if p.exists():
                p.rmdir(recursive=True)
            _init_group(p, zarr_version)
        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Use 'r', 'r+', 'a', 'w', or 'w-' ")

        return cls(p)

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
        return ZarrTSGroup.from_config(zarr_config)
    elif zarr_config.driver == "zarrita":
        raise NotImplementedError(f"{zarr_config.driver} is not yet supported")
    else:
        raise NotImplementedError(f"Driver '{zarr_config.driver}' is not supported.")
