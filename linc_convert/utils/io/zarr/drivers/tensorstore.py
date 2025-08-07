"""TensorStore driver for Zarr arrays and groups."""

import os
from numbers import Number
from os import PathLike
from typing import (
    Any,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    Unpack,
)
from urllib.parse import urlparse

import numpy as np
import tensorstore as ts
import zarr
from numpy.typing import ArrayLike, DTypeLike
from upath import UPath

from linc_convert.utils.io.zarr import ZarrPythonArray, ZarrPythonGroup
from linc_convert.utils.io.zarr.abc import ZarrArray, ZarrArrayConfig
from linc_convert.utils.io.zarr.helpers import auto_shard_size, fix_shard_chunk
from linc_convert.utils.zarr_config import ZarrConfig


class ZarrTSArray(ZarrArray):
    """Zarr array backed by TensorStore."""

    def __init__(self, ts_array: ts.TensorStore) -> None:
        """
        Initialize the ZarrTSArray with a TensorStore array.

        Parameters
        ----------
        ts_array : tensorstore.TensorStore
            Underlying TensorStore array.
        """
        super().__init__(str(ts_array.kvstore.path))
        self._ts = ts_array

    @classmethod
    def from_zarr_python_array(cls, zarray: ZarrPythonArray) -> "ZarrTSArray":
        """Convert a ZarrPythonArray into a ZarrTSArray."""
        return cls.open(zarray.store_path, zarr_version=zarray.zarr_version)

    @property
    def ndim(self) -> int:
        """Number of dimensions of the array."""
        return self._ts.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the array."""
        return self._ts.shape

    @property
    def dtype(self) -> np.dtype:
        """Data type of the array."""
        return self._ts.dtype.numpy_dtype

    @property
    def chunks(self) -> Tuple[int, ...]:
        """Chunk shape for the array."""
        return self._ts.chunk_layout.read_chunk.shape

    @property
    def shards(self) -> Optional[Tuple[int, ...]]:
        """Shard shape, if supported; otherwise None."""
        read_shape = self._ts.chunk_layout.read_chunk.shape
        write_shape = self._ts.chunk_layout.write_chunk.shape
        return None if read_shape == write_shape else write_shape

    @property
    def attrs(self) -> Mapping[str, Any]:
        """Access metadata/attributes for this node."""
        # TODO: TensorStore currently doesn’t expose arbitrary attrs, so return empty.
        return {}

    @property
    def zarr_version(self) -> Literal[2, 3]:
        """Get the Zarr format version."""
        driver = self._ts.schema.codec.to_json().get("driver", "")
        return 3 if driver == "zarr3" else 2

    def __getitem__(self, key: str) -> ArrayLike:
        """Read data from the array."""
        return self._ts[key].read().result()

    def __setitem__(self, key: str, value: ArrayLike | Number) -> None:
        """Write data to the array."""
        self._ts[key] = value

    @classmethod
    def open(
        cls,
        path: Union[str, PathLike],
        *,
        zarr_version: Literal[2, 3] | int = 3,
        create: bool = False,
        delete_exsisting: bool = False,
    ) -> "ZarrTSArray":
        """
        Open an existing Zarr array.

        Parameters
        ----------
        path : Union[str, PathLike]
            Path to the array’s directory.
        zarr_version : {2, 3}
            Zarr format version to use.
        mode : {'r','r+','a','w','w-'}
            Access mode; see TensorStore docs.

        Returns
        -------
        ZarrTSArray
        """
        spec = {
            "kvstore": make_kvstore(path),
            "driver": "zarr3" if zarr_version == 3 else "zarr",
            "open": True,
            "create": create,
            "delete_existing": delete_exsisting,
        }
        ts_array = ts.open(spec).result()
        return cls(ts_array)


class ZarrTSGroup(ZarrPythonGroup):
    """Zarr Group implementation using TensorStore as backend."""

    def __init__(self, zarr_group: zarr.Group) -> None:
        """
        Initialize the ZarrTSGroup.

        Parameters
        ----------
        zarr_group : zarr.Group
            Underlying Zarr Python group.
        """
        super().__init__(zarr_group)

    @classmethod
    def from_zarr_python_group(cls, zarr_group: ZarrPythonGroup) -> "ZarrTSGroup":
        """Convert a ZarrPythonGroup into a ZarrTSGroup."""
        assert isinstance(zarr_group, ZarrPythonGroup)
        return cls(zarr_group._get_zarr_python_group())

    @classmethod
    def from_config(cls, zarr_config: ZarrConfig) -> "ZarrTSGroup":
        """Create a Zarr group from a configuration object."""
        return cls.from_zarr_python_group(ZarrPythonGroup.from_config(zarr_config))

    def __getitem__(self, key: str) -> Union[ZarrTSArray, "ZarrTSGroup"]:
        """Get a subgroup or array by name within this group."""
        node = super().__getitem__(key)
        if isinstance(node, ZarrPythonGroup):
            return self.from_zarr_python_group(node)
        elif isinstance(node, ZarrPythonArray):
            return ZarrTSArray.from_zarr_python_array(node)
        else:
            raise ValueError(
                "Unsupported node type in Zarr group: {}".format(type(node))
            )

    def __setitem__(self, key: str, value: Union[ZarrTSArray, "ZarrTSGroup"]) -> None:
        """Set a subgroup or array by name within this group."""
        raise NotImplementedError(
            "Assigning to zarr group is not supported with tensorstore."
        )

    def create_array(
        self,
        name: str,
        shape: Sequence[int],
        dtype: DTypeLike = np.int32,
        *,
        zarr_config: Optional[ZarrConfig] = None,
        data: Optional[ArrayLike] = None,
        **kwargs: Unpack[ZarrArrayConfig],
    ) -> ZarrTSArray:
        """
        Create a new array within this group.

        Parameters
        ----------
        name : str
        shape : Sequence[int]
        dtype : DTypeLike
        zarr_config : ZarrConfig | None
        data : ArrayLike | None

        Returns
        -------
        ZarrTSArray
        """
        arr = super().create_array(
            name=name,
            shape=shape,
            dtype=dtype,
            zarr_config=zarr_config,
            data=data,
            **kwargs,
        )
        arr = ZarrTSArray.from_zarr_python_array(arr)
        if data is not None:
            arr[:] = data
        return arr

    def create_array_from_base(
        self,
        name: str,
        shape: Sequence[int],
        data: Optional[ArrayLike] = None,
        **kwargs: Unpack[ZarrArrayConfig],
    ) -> ZarrTSArray:
        """
        Create a new array using metadata of an existing base-level array.

        Parameters
        ----------
        name : str
        shape : Sequence[int]
        data : ArrayLike | None

        Returns
        -------
        ZarrTSArray
        """
        arr = super().create_array_from_base(name=name, shape=shape, **kwargs)
        arr = ZarrTSArray.from_zarr_python_array(arr)
        if data is not None:
            arr[:] = data
        return arr


def make_compressor_v2(name: str | None, **prm: dict) -> dict:
    """Build compressor dictionary for Zarr v2."""
    name = name.lower()
    if name not in ("blosc", "zlib", "bz2", "zstd"):
        raise ValueError("Unknown compressor", name)
    return {"id": name, **prm}


def make_compressor_v3(name: str | None, **prm: dict) -> dict:
    """Build compressor dictionary for Zarr v3."""
    name = name.lower()
    if name not in ("blosc", "gzip", "zstd"):
        raise ValueError("Unknown compressor", name)
    return {"name": name, "configuration": prm}


def make_kvstore(path: str | os.PathLike) -> dict:
    """Transform a URI into a kvstore JSON object."""
    path = UPath(path)
    if path.protocol in ("file", ""):
        return {"driver": "file", "path": path.path}
    if path.protocol == "gcs":
        url = urlparse(str(path))
        return {"driver": "gcs", "bucket": url.netloc, "path": url.path}
    if path.protocol in ("http", "https"):
        url = urlparse(str(path))
        base_url = f"{url.scheme}://{url.netloc}"
        if url.params:
            base_url += ";" + url.params
        if url.query:
            base_url += "?" + url.query
        if url.fragment:
            base_url += "#" + url.fragment
        return {"driver": "http", "base_url": base_url, "path": url.path}
    if path.protocol == "memory":
        return {"driver": "memory", "path": path.path}
    if path.protocol == "s3":
        url = urlparse(str(path))
        path = {"path": url.path} if url.path else {}
        return {"driver": "s3", "bucket": url.netloc, **path}
    raise ValueError("Unsupported protocol:", path.protocol)


def default_read_config(path: os.PathLike | str) -> dict:
    """
    Generate a TensorStore configuration to read an existing Zarr.

    Parameters
    ----------
    path : PathLike | str
        Path to zarr array.
    """
    path = UPath(path)
    if not path.protocol:
        path = "file://" / path
    if (path / "zarr.json").exists():
        zarr_version = 3
    elif (path / ".zarray").exists():
        zarr_version = 2
    else:
        raise ValueError("Cannot find zarr.json or .zarray file")
    return {
        "kvstore": make_kvstore(path),
        "driver": "zarr3" if zarr_version == 3 else "zarr",
        "open": True,
        "create": False,
        "delete_existing": False,
    }


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
    else:
        chunk = list(chunk)
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
    else:
        for i, c in enumerate(chunk):
            chunk[i] = min(chunk[i], shape[i])
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
        for i in range(len(shape)):
            if shape[i] < chunk[i]:
                chunk[i] = shape[i]
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
