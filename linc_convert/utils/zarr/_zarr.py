"""Zarr utilities."""

# stdlib
import itertools
import json
import math
import os
from typing import Literal
from urllib.parse import urlparse

# externals
import nibabel as nib
import numcodecs
import numcodecs.abc
import numpy as np
import tensorstore as ts
from upath import UPath

from linc_convert.utils.math import ceildiv


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
    max_file_size: int = 2 * 1024**4,
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
    shard=  list(shard)
    chunk = list(chunk)
    for i in range(len(chunk)):
        # if chunk spans the entire volume, match chunk and shard
        if chunk[i] == shape[i] and chunk[i] != shard[i]:
            chunk[i] = shard[i]
        # ensure that shard is a multiple of chunk
        if shard[i] % chunk[i]:
            shard[i] = chunk[i] * int(math.ceil(shard[i] / chunk[i]))
    return shard, chunk


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
