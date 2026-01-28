"""TensorStore driver for Zarr arrays and groups."""
import json
import os
from numbers import Number
from os import PathLike
from typing import (
    Iterator,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    Unpack,
)
from urllib.parse import urlparse

import numpy as np
import tensorstore as ts
from numpy.typing import ArrayLike, DTypeLike
from upath import UPath

from linc_convert.utils.io.zarr.abc import ZarrArray, ZarrArrayConfig, ZarrGroup
from linc_convert.utils.io.zarr.attributes import Attributes
from linc_convert.utils.io.zarr.helpers import auto_shard_size, fix_shard_chunk
from linc_convert.utils.io.zarr.metadata import GroupMetadata
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
        self._attrs: Optional[Attributes] = None

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
    def attrs(self) -> Attributes:
        """Access metadata/attributes for this node."""
        if self._attrs is None:
            self._attrs = Attributes(self, write_through=True)
        return self._attrs

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
        mode: Literal["r", "r+", "a", "w", "w-"] = "a",
        *,
        zarr_version: Literal[2, 3] | int = 3,
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
            Persistence mode: 'r' means read only (must exist); 'r+' means
            read/write (must exist); 'a' means read/write (create if doesn't
            exist); 'w' means create (overwrite if exists); 'w-' means create
            (fail if exists).

        Returns
        -------
        ZarrTSArray
        """
        create = False
        delete_existing = False
        if mode == "w":
            delete_existing = True
            create = True
        elif mode == "w-" or mode == "a":
            create = True

        spec = {
            "kvstore": make_kvstore(path),
            "driver": "zarr3" if zarr_version == 3 else "zarr",
            "open": True,
            "create": create,
            "delete_existing": delete_existing,
        }
        ts_array = ts.open(spec).result()
        return cls(ts_array)


class ZarrTSGroup(ZarrGroup):
    """Zarr Group implementation using TensorStore as backend."""

    def __init__(self, store_path: Union[str, PathLike]) -> None:
        """
        Initialize the ZarrTSGroup.

        Parameters
        ----------
        store_path : Union[str, PathLike]
            Path to the group’s directory.
        """
        super().__init__(store_path)
        self._path = UPath(store_path)
        meta = _detect_metadata(self._path)
        assert meta and meta[0] == "group"
        self._zarr_version = meta[1]
        self._attrs: Optional[Attributes] = None
        self._metadata: Optional[GroupMetadata] = None

    @classmethod
    def from_config(
        cls, out: str | PathLike[str], zarr_config: ZarrConfig
        ) -> "ZarrTSGroup":
        """
        Create a ZarrTSGroup from a configuration object.

        Parameters
        ----------
        zarr_config : ZarrConfig
            Configuration with .out and .zarr_version.

        Returns
        -------
        ZarrTSGroup
        """
        return cls.open(
            out, mode="a", zarr_version=zarr_config.zarr_version
        )

    @classmethod
    def open(
        cls,
        path: Union[str, PathLike],
        mode: Literal["r", "r+", "a", "w", "w-"] = "a",
        *,
        zarr_version: Literal[2, 3] = 3,
    ) -> "ZarrTSGroup":
        """
        Open or create a Zarr group backed by TensorStore.

        Parameters
        ----------
        path : Union[str, PathLike]
            Path to the Zarr group.
        mode : {'r','r+','a','w','w-'}
            Persistence mode: 'r' means read only (must exist); 'r+' means
            read/write (must exist); 'a' means read/write (create if doesn't
            exist); 'w' means create (overwrite if exists); 'w-' means create
            (fail if exists).
        zarr_version : {2,3}
            Zarr format version.

        Returns
        -------
        ZarrTSGroup
        """
        p = UPath(path)
        if mode in ("r", "r+"):
            if not p.exists() or not p.is_dir():
                raise FileNotFoundError(f"Group path '{p}' does not exist")
        elif mode == "w-":
            if p.exists():
                raise FileExistsError(f"Group path '{p}' already exists")
        elif mode == "a":
            if not p.exists():
                _init_group(p, zarr_version)
        elif mode == "w":
            if p.exists():
                p.rmdir(recursive=True)
            _init_group(p, zarr_version)
        else:
            raise ValueError(f"Invalid mode '{mode}'")
        return cls(p)

    @property
    def attrs(self) -> Attributes:
        """Access attributes for this node."""
        if self._attrs is None:
            self._attrs = Attributes(self, write_through=True)
        return self._attrs

    @property
    def metadata(self) -> GroupMetadata:
        """Access metadata for this node."""
        if self._metadata is None:
            self._metadata = GroupMetadata.from_files(self._path)
        return self._metadata

    @property
    def zarr_version(self) -> Literal[2, 3]:
        """Get the Zarr format version."""
        return self._zarr_version

    def __getitem__(self, key: str) -> Union[ZarrTSArray, "ZarrTSGroup"]:
        """Get a subgroup or array by name within this group."""
        meta = _detect_metadata(self._path / key)
        if not meta:
            raise KeyError(f"Key '{key}' not found")
        if meta[0] == "group":
            return ZarrTSGroup(self._path / key)
        return ZarrTSArray.open(self._path / key, zarr_version=meta[1])

    def __setitem__(self, key: str, value: Union[ZarrTSArray, "ZarrTSGroup"]) -> None:
        """Set a subgroup or array by name within this group."""
        raise NotImplementedError(
            "Assign to zarr group is not supported with tensorstore."
        )

    def __delitem__(self, key: str) -> None:
        """Delete a subgroup or array by name within this group."""
        target = self._path / key
        if target.exists():
            target.rmdir(recursive=True)

    def keys(self) -> Iterator[str]:
        """Get the names of all subgroups and arrays in this group."""
        return (
            p.name for p in self._path.iterdir() if p.is_dir() and _detect_metadata(p)
        )

    def __contains__(self, name: str) -> bool:
        """Check whether a subgroup or array exists in this group."""
        p = self._path / name
        return p.exists() and bool(_detect_metadata(p))

    def create_group(self, name: str, overwrite: bool = False) -> "ZarrTSGroup":
        """
        Create or open a subgroup within this group.

        Parameters
        ----------
        name : str
        overwrite : bool
            If True, delete existing before creating.

        Returns
        -------
        ZarrTSGroup
        """
        mode = "w" if overwrite else "w-"
        return self.open(self._path / name, mode=mode, zarr_version=self._zarr_version)

    def create_array(
        self,
        name: str,
        shape: Sequence[int],
        dtype: DTypeLike = np.int32,
        *,
        overwrite: bool = True,
        data: Optional[ArrayLike] = None,
        zarr_config: Optional[ZarrConfig] = None,
        **kwargs: Unpack[ZarrArrayConfig],
    ) -> ZarrTSArray:
        """
        Create a new array within this group.

        Parameters
        ----------
        name : str
        shape : Sequence[int]
        dtype : DTypeLike
        overwrite: bool
        zarr_config : ZarrConfig | None
        data : ArrayLike | None

        Returns
        -------
        ZarrTSArray
        """

        def _normalize_keys(d: dict) -> dict:
            # map plural/common variants -> canonical keys
            mapping = {
                "chunks": "chunk",
                "shards": "shard",
                "compressors": "compressor",
                "compressor_opts": "compressor_opt",
            }
            out = {}
            for k, v in d.items():
                if k in ("chunk_key_encoding", "fill_value"):
                    # explicitly unsupported/ignored
                    continue
                out[mapping.get(k, k)] = v
            # drop Nones so we don't pass them through
            return {k: v for k, v in out.items() if v is not None}

        # Start with defaults from zarr_config (if provided)
        base: dict = {}
        if zarr_config is not None:
            base = _normalize_keys({
                "chunk": getattr(zarr_config, "chunk", None),
                "shard": getattr(zarr_config, "shard", None),
                "compressor": getattr(zarr_config, "compressor", None),
                "compressor_opt": getattr(zarr_config, "compressor_opt", None),
            })

        # Normalize kwargs and make them override zarr_config-provided defaults
        kw = _normalize_keys(kwargs)
        merged = {**base, **kw}  # kwargs win

        # Build the write config
        conf = default_write_config(
            self._path / name,
            shape=shape,
            dtype=dtype,
            version=self.zarr_version,
            **merged,
        )

        if overwrite:
            conf.update(delete_existing=True)
        conf.update(create=True)
        arr = ts.open(conf).result()
        if data is not None:
            arr[:] = data
        return ZarrTSArray(arr)

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
        base = self["0"]._ts.spec().to_json()
        base["metadata"]["shape"] = shape
        base["kvstore"] = make_kvstore(self._path / name)
        base.update(delete_existing=True, create=True)
        arr = ts.open(base).result()
        if data is not None:
            arr[:] = data
        return ZarrTSArray(arr)



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


def default_read_config(path: str | PathLike[str]) -> dict:
    """
    Generate a TensorStore configuration to read an existing Zarr.

    Parameters
    ----------
    path : PathLike | str
        Path to zarr array.
    """
    path = UPath(path)
    if not path.protocol:
        path = UPath("file://" + str(path))
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


def _detect_metadata(path: PathLike) -> Optional[Tuple[str, int]]:
    """
    Look for Zarr metadata files in `path` and return (node_type, version).

    Checks zarr.json (v3), then .zarray/.zgroup (v2).
    """
    # Zarr v3
    z3 = path / "zarr.json"
    if z3.is_file():
        try:
            meta = json.loads(z3.read_text())
            fmt = meta.get("zarr_format")
            if fmt == 3:
                node = meta.get("node_type", "array")
                if node in ("array", "group"):
                    return node, 3
        except json.JSONDecodeError:
            pass
    # Zarr v2
    for fname, ntype in ((".zarray", "array"), (".zgroup", "group")):
        f = path / fname
        if f.is_file():
            try:
                meta = json.loads(f.read_text())
                if meta.get("zarr_format") == 2:
                    return ntype, 2
            except json.JSONDecodeError:
                pass
    return None


def default_write_config(
    path: str | PathLike[str],
    shape: list[int],
    dtype: np.dtype | str,
    chunk: list[int] = [32],
    shard: list[int] | Literal["auto"] | None = None,
    compressor: str = "blosc",
    compressor_opt: dict | None = None,
    fill_value: Number | None = 0,
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
    fill_value:
    version : int
        Zarr version

    Returns
    -------
    config : dict
        Configuration
    """
    path = UPath(path)
    if not path.protocol:
        path = UPath("file://" + str(path))

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
        if compressor and compressor != "raw" and compressor != "none":
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
            "fill_value": fill_value,
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
        if compressor and compressor != "raw" and compressor != "none":
            compressor = make_compressor_v2(compressor, **compressor_opt)
        else:
            compressor = None
        for i in range(len(shape)):
            if shape[i] < chunk[i]:
                chunk[i] = shape[i]
        metadata = {
            "chunks": chunk,
            "order": "F" if len(shape) >= 2 else "C",
            "dtype": np.dtype(dtype).str,
            "fill_value": fill_value,
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


def _init_group(group_path: PathLike, version: int) -> None:
    group_path.mkdir(parents=True, exist_ok=True)
    if version == 3:
        (group_path / "zarr.json").write_text(
            json.dumps({"zarr_format": 3, "node_type": "group"})
        )
    else:
        (group_path / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
