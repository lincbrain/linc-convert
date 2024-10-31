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

# internals
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
    max_file_size: int = 2*1024**4,
    compression_ratio: float = 2
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
        new_shard = [min(2*x, s) for x, s in zip(shard, max_shape)]
        # If shard is too large, stop and keep previous shard
        if np.prod(new_shard) > max_numel:
            break
        # Otherwise, use larger shard and recurse
        shard = new_shard

    # replace max size with larger power of two
    shard = [2**math.ceil(math.log2(x)) for x in shard]
    return shard


def niftizarr_write_header(
    path: os.PathLike | str,
    shape: list[int],
    affine: np.ndarray,
    dtype: np.dtype | str,
    unit: Literal["micron", "mm"] | None = None,
    header: nib.Nifti1Header | nib.Nifti2Header | None = None,
    zarr_version: int = 3,
) -> None:
    """
    Write NIfTI header in a NIfTI-Zarr file.

    Parameters
    ----------
    path : PathLike | str
        Path to parent Zarr.
    affine : (4, 4) matrix
        Orientation matrix.
    shape : list[int]
        Array shape, in NIfTI order (x, y, z, t, c).
    dtype : np.dtype | str
        Data type.
    unit : {"micron", "mm"}, optional
        World unit.
    header : nib.Nifti1Header | nib.Nifti2Header, optional
        Pre-instantiated header.
    zarr_version : int, default=3
        Zarr version.
    """
    # TODO: we do not write the json zattrs, but it should be added in
    #       once the nifti-zarr package is released

    path = UPath(path)
    if not path.protocol:
        path = "file://" / path

    # If dimensions do not fit in a short (which is often the case), we
    # use NIfTI 2.
    if all(x < 32768 for x in shape):
        NiftiHeader = nib.Nifti1Header
    else:
        NiftiHeader = nib.Nifti2Header

    header = header or NiftiHeader()
    header.set_data_shape(shape)
    header.set_data_dtype(dtype)
    header.set_qform(affine)
    header.set_sform(affine)
    if unit:
        header.set_xyzt_units(nib.nifti1.unit_codes.code[unit])
    header = np.frombuffer(header.structarr.tobytes(), dtype="u1")

    if zarr_version == 3:
        metadata = {
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [len(header)]}
            },
            "codecs": [
                {"name": "bytes"},
                make_compressor_v3("gzip"),
            ],
            "data_type": "uint8",
            "fill_value": 0,
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": r"/"}
            },
        }
        tsconfig = {
            "driver": "zarr3",
            "metadata": metadata,
        }
    else:
        metadata = {
            "chunks": [len(header)],
            "order": "F",
            "dtype": "|u1",
            "fill_value": None,
            "compressor": make_compressor_v2("zlib"),
        }
        tsconfig = {
            "driver": "zarr",
            "metadata": metadata,
            "key_encoding":  r"/",
        }
    tsconfig["kvstore"] = make_kvstore(path / "nifti")
    tsconfig["metadata"]["shape"] = [len(header)]
    tsconfig["create"] = True
    tsconfig["delete_existing"] = True
    tswriter = ts.open(tsconfig).result()
    with ts.Transaction() as txn:
        tswriter.with_transaction(txn)[:] = header
    print("done.")


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

        codec_little_endian = {
            "name": "bytes",
            "configuration": {"endian": "little"}
        }

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
            chunk_grid = {
                "name": "regular",
                "configuration": {"chunk_shape": chunk}
            }
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
                "configuration": {"separator": r"/"}
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


def generate_pyramid(
    path: os.PathLike | str,
    levels: int | None = None,
    shard: list[int] | bool | Literal["auto"] | None = None,
    ndim: int = 3,
    max_load: int = 512,
    mode: Literal["mean", "median"] = "median",
) -> list[list[int]]:
    """
    Generate the levels of a pyramid in an existing Zarr.

    Parameters
    ----------
    path : PathLike | str
        Path to parent Zarr
    levels : int
        Number of additional levels to generate.
        By default, stop when all dimensions are smaller than their
        corresponding chunk size.
    shard : list[int] | bool | {"auto"} | None
        Shard size.
        * If `None`, use same shard size as the input array;
        * If `False`, no dot use sharding;
        * If `True` or `"auto"`, automatically find shard size;
        * Otherwise, use provided shard size.
    ndim : int
        Number of spatial dimensions.
    max_load : int
        Maximum number of voxels to load along each dimension.
    mode : {"mean", "median"}
        Whether to use a mean or median moving window.

    Returns
    -------
    shapes : list[list[int]]
        Shapes of all levels, from finest to coarsest, including the
        existing top level.
    """
    path = UPath(path)
    if not path.protocol:
        path = "file://" / path

    zarray2 = path / "0" / ".zarray"
    zarray3 = path / "0" / "zarr.json"

    if zarray2.exists():
        with zarray2.open("rb") as f:
            metadata = json.load(f)
        zarr_version = 2
    elif zarray3.exists():
        with zarray3.open("rb") as f:
            metadata = json.load(f)
        zarr_version = 3
    else:
        raise FileNotFoundError("Could not find initial zarr array")

    shape = list(metadata["shape"])

    if shard and zarr_version == 2:
        raise ValueError("Sharding requires zarr 3")

    # Find chunk size
    if zarr_version == 3:
        chunk_size = metadata["chunk_grid"]["configuration"]["chunk_shape"]
        chunk_size = list(chunk_size)
        for codec in metadata["codecs"]:
            if codec["name"] == "sharding_indexed":
                chunk_size = list(codec["configuration"]["chunk_shape"])
                break
    else:
        assert zarr_version == 2
        chunk_size = metadata["chunks"]

    shard_option = shard

    def update_metadata(metadata: dict, shape: list[int]) -> dict:
        metadata["shape"] = shape
        if shard_option is None:
            return metadata

        # find chunk size and compression codecs
        chunk = metadata["chunk_grid"]["configuration"]["chunk_shape"]
        chunk = list(chunk)
        codecs = metadata["codecs"]
        for codec in metadata["codecs"]:
            if codec["name"] == "sharding_indexed":
                chunk = list(codec["configuration"]["chunk_shape"])
                codecs = codec["configuration"]["codecs"]
                break

        # set appropriate chunking/sharding config
        if shard_option == "auto":
            shard = auto_shard_size(shape, metadata["data_type"])
            shard, chunk = fix_shard_chunk(shard, chunk, shape)
        else:
            shard = shard_option

        if shard_option is False:
            metadata["chunk_grid"]["configuration"]["chunk_shape"] = chunk
            metadata["codecs"] = codecs

        else:
            metadata["chunk_grid"] = {
                "name": "regular",
                "configuration": {"chunk_shape": shard},
            }
            metadata["codecs"] = [{
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": chunk,
                    "codecs": codecs,
                    "index_codecs": [
                        codecs[0],  # should be the bytes codec
                        {"name": "crc32c"},
                    ],
                    "index_location": "end",
                },
            }]

        return metadata

    # Select windowing function
    if mode == "median":
        func = np.median
    else:
        assert mode == "mean"
        func = np.mean

    level = 0
    batch, shape = shape[:-ndim], shape[-ndim:]
    allshapes = [shape]
    while True:
        level += 1

        # Compute downsampled shape
        prev_shape, shape = shape, [max(1, x//2) for x in shape]

        # Stop if seen enough levels or level shape smaller than chunk size
        if levels is None:
            if all(x <= c for x, c in zip(shape, chunk_size[-ndim:])):
                break
        elif level > levels:
            break

        print("Compute level", level, "with shape", shape)

        allshapes.append(shape)

        # Open input and output zarr
        rconfig = default_read_config(str(path / str(level-1)))
        wconfig = {
            "driver": "zarr3" if zarr_version == 3 else "zarr",
            "metadata": update_metadata(metadata, batch + shape),
            "kvstore": make_kvstore(path / str(level)),
            "create": True,
            "delete_existing": True,
        }
        tsreader = ts.open(rconfig).result()
        tswriter = ts.open(wconfig).result()

        # Iterate across `max_load` chunks
        # (note that these are unrelared to underlying zarr chunks)
        grid_shape = [ceildiv(n, max_load) for n in prev_shape]
        for chunk_index in itertools.product(*[range(x) for x in grid_shape]):

            print(f"chunk {chunk_index} / {tuple(grid_shape)})", end="\r")

            with ts.Transaction() as txn:

                # Read one chunk of data at the previous resolution
                slicer = [Ellipsis] + [
                    slice(i*max_load, min((i+1)*max_load, n))
                    for i, n in zip(chunk_index, prev_shape)
                ]
                dat = tsreader.with_transaction(txn)[*slicer].read().result()

                # Discard the last voxel along odd dimensions
                crop = [0 if x == 1 else x % 2 for x in dat.shape[-3:]]
                slcr = [slice(-1) if x else slice(None) for x in crop]
                dat = dat[Ellipsis, *slcr]

                patch_shape = dat.shape[-3:]

                # Reshape into patches of shape 2x2x2
                windowed_shape = [
                    x
                    for n in patch_shape
                    for x in (max(n//2, 1), min(n, 2))
                ]
                dat = dat.reshape(batch + windowed_shape)
                # -> last `ndim`` dimensions have shape 2x2x2
                dat = dat.transpose(
                    list(range(len(batch))) +
                    list(range(len(batch), len(batch) + 2*ndim, 2)) +
                    list(range(len(batch)+1, len(batch) + 2*ndim, 2))
                )
                # -> flatten patches
                smaller_shape = [max(n//2, 1) for n in patch_shape]
                dat = dat.reshape(batch + smaller_shape + [-1])

                # Compute the median of each patch
                dtype = dat.dtype
                dat = func(dat, axis=-1)
                dat = dat.astype(dtype)

                # Write output
                slicer = [Ellipsis] + [
                    slice(i*max_load//2, min((i+1)*max_load//2, n))
                    for i, n in zip(chunk_index, shape)
                ]
                tswriter.with_transaction(txn)[*slicer] = dat

    print("")

    return allshapes


def write_ome_metadata(
    path: str | os.PathLike,
    axes: list[str],
    space_scale: float | list[float] = 1,
    time_scale: float = 1,
    space_unit: str = "micrometer",
    time_unit: str = "second",
    name: str = "",
    pyramid_aligns: str | int | list[str | int] = 2,
    levels: int | None = None,
    zarr_version: int | None = None,
) -> None:
    """
    Write OME metadata into Zarr.

    Parameters
    ----------
    path : str | PathLike
        Path to parent Zarr.
    axes : list[str]
        Name of each dimension, in Zarr order (t, c, z, y, x)
    space_scale : float | list[float]
        Finest-level voxel size, in Zarr order (z, y, x)
    time_scale : float
        Time scale
    space_unit : str
        Unit of spatial scale (assumed identical across dimensions)
    space_time : str
        Unit of time scale
    name : str
        Name attribute
    pyramid_aligns : float | list[float] | {"center", "edge"}
        Whether the pyramid construction aligns the edges or the centers
        of the corner voxels. If a (list of) number, assume that a moving
        window of that size was used.
    levels : int
        Number of existing levels. Default: find out automatically.
    zarr_version : {2, 3} | None
        Zarr version. If `None`, guess from existing zarr array.

    """
    path = UPath(path)

    # Detect zarr version
    if not zarr_version:
        if (path / "0" / "zarr.json").exists():
            zarr_version = 3
        elif (path / "0" / ".zarray").exists():
            zarr_version = 2
        else:
            raise FileNotFoundError("No existing array to guess version from")

    # Read shape at each pyramid level
    zname = "zarr.json" if zarr_version == 3 else ".zarray"
    shapes = []
    level = 0
    while True:
        if levels is not None and level > levels:
            break

        zpath = path / str(level) / zname
        if not zpath.exists():
            levels = level
            break

        level += 1
        with zpath.open("rb") as f:
            zarray = json.load(f)
            shapes += [zarray["shape"]]

    axis_to_type = {
        "x": "space",
        "y": "space",
        "z": "space",
        "t": "time",
        "c": "channel",
    }

    # Number of spatial (s), batch (b) and total (n) dimensions
    ndim = len(axes)
    sdim = sum(axis_to_type[axis] == "space" for axis in axes)
    bdim = ndim - sdim

    if isinstance(pyramid_aligns, (int, str)):
        pyramid_aligns = [pyramid_aligns]
    pyramid_aligns = list(pyramid_aligns)
    if len(pyramid_aligns) < sdim:
        repeat = pyramid_aligns[:1] * (sdim-len(pyramid_aligns))
        pyramid_aligns = repeat + pyramid_aligns
    pyramid_aligns = pyramid_aligns[-sdim:]

    if isinstance(space_scale, (int, float)):
        space_scale = [space_scale]
    space_scale = list(space_scale)
    if len(space_scale) < sdim:
        repeat = space_scale[:1] * (sdim-len(space_scale))
        space_scale = repeat + space_scale
    space_scale = space_scale[-sdim:]

    multiscales = [{
        "version": "0.4",
        "axes": [
            {
                "name": axis,
                "type": axis_to_type[axis],

            }
            if axis_to_type[axis] == "channel" else
            {
                "name": axis,
                "type": axis_to_type[axis],
                "unit": (
                    space_unit if axis_to_type[axis] == "space" else
                    time_unit if axis_to_type[axis] == "time" else
                    None
                )
            }
            for axis in axes
        ],
        "datasets": [],
        "type": "median window " + "x".join(["2"] * sdim),
        "name": name,
    }]

    shape = shape0 = shapes[0]
    for n in range(len(shapes)):
        shape = shapes[n]
        multiscales[0]["datasets"].append({})
        level = multiscales[0]["datasets"][-1]
        level["path"] = str(n)

        scale = [1] * bdim + [
            (
                pyramid_aligns[i]**n
                if not isinstance(pyramid_aligns[i], str) else
                (shape0[bdim+i] / shape[bdim+i])
                if pyramid_aligns[i][0].lower() == "e" else
                ((shape0[bdim+i] - 1) / (shape[bdim+i] - 1))
            ) * space_scale[i]
            for i in range(sdim)
        ]
        translation = [0] * bdim + [
            (
                pyramid_aligns[i]**n - 1
                if not isinstance(pyramid_aligns[i], str) else
                (shape0[bdim+i] / shape[bdim+i]) - 1
                if pyramid_aligns[i][0].lower() == "e" else
                0
            ) * 0.5 * space_scale[i]
            for i in range(sdim)
        ]

        level["coordinateTransformations"] = [
            {
                "type": "scale",
                "scale": scale,
            },
            {
                "type": "translation",
                "translation": translation,
            }
        ]

    scale = [1] * ndim
    if "t" in axes:
        scale[axes.index("t")] = time_scale
    multiscales[0]["coordinateTransformations"] = [
        {
            "scale": scale,
            "type": "scale"
        }
    ]

    if zarr_version == 3:
        with (path / "zarr.json").open("wt") as f:
            json.dump({
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    # Zarr v2 way of doing it -- neuroglancer wants this
                    "multiscales": multiscales,
                    # Zarr RFC-2 recommended way
                    "ome": {
                        "version": "0.4",
                        "multiscales": multiscales
                    }
                }
            }, f, indent=4)
    else:
        multiscales[0]["version"] = "0.4"
        with (path / ".zgroup").open("wt") as f:
            json.dump({"zarr_format": 2}, f, indent=4)
        with (path / ".zattrs").open("wt") as f:
            json.dump({"multiscales": multiscales}, f, indent=4)
