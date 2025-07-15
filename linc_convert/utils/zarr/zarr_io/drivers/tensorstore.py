import ast
import json
import math
import os
from os import PathLike
from typing import Optional, Tuple, Literal, Union, Mapping, Any
from urllib.parse import urlparse

import numcodecs
import numcodecs.abc
import numpy as np
from numpy.typing import DTypeLike
from upath import UPath

import tensorstore as ts
from linc_convert.utils.zarr import ZarrConfig
from linc_convert.utils.zarr.zarr_io.abc import ZarrArray, ZarrGroup


class ZarrTSArray(ZarrArray):
    def __init__(self, ts: ts.TensorStore):
        super().__init__(ts.kvstore.path)
        self._ts = ts

    @property
    def shape(self):
        return self._ts.shape

    @property
    def ndim(self):
        return self._ts.ndim

    @property
    def dtype(self):
        return self._ts.dtype.numpy_dtype

    @property
    def chunks(self):
        return self._ts.chunk_layout.read_chunk.shape

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

    def __getitem__(self, idx):
        return self._ts[idx].read().result()

    def __setitem__(self, idx, val):
        self._ts[idx] = val

    # TODO: implement
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
                     ) -> ZarrArray:
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
                # TODO: implement this
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

    def _get_zarr_python_group(self):
        import zarr
        return zarr.open_group(self._path, mode='a')


def make_compressor(name: str, **prm: dict) -> numcodecs.abc.Codec:
    """Build compressor object from name and options."""
    # TODO: we should use `numcodecs.get_codec` instead`
    if not isinstance(name, str):
        return name
    name = name.lower()
    if name == "blosc":
        Compressor = numcodecs.Blosc
    elif name == "zlib":
        Compressor = numcodecs.Zlib
    else:
        raise ValueError("Unknown compressor", name)
    return Compressor(**prm)


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
