"""Convert spool .dat file sets or ome.zarr files into a consolidated Zarr."""

# stdlib
import getpass
import logging
import os
import re
import time
import warnings
from collections import defaultdict, namedtuple
from glob import glob
from pathlib import Path, PurePosixPath
from typing import List, Literal, Optional, Tuple, Union

import dask

# externals
import dask.array as da
import numpy as np
import tifffile as tiff
import yaml
from dandi.dandiapi import DandiAPIClient
from dask.diagnostics import ProgressBar
from scipy.ndimage import map_coordinates

# internals
from linc_convert.utils.io.spool import SpoolSetInterpreter
from linc_convert.utils.io.zarr.abc import ZarrArray
from linc_convert.utils.io.zarr.drivers.zarr_python import (
    ZarrPythonGroup,
)
from linc_convert.utils.nifti_header import build_nifti_header
from linc_convert.utils.zarr_config import (
    GeneralConfig,
    NiftiConfig,
    ZarrConfig,
)

logger = logging.getLogger(__name__)


def format_to_regex(pattern: str) -> re.Pattern:
    """
    Convert a glob-like format string with {fields} into a regex.

    Example:
        "*y_val{y}*.ome.zarr"
    →   r".*y_val(?P<y>\\d+).*\\.ome\\.zarr"
    """

    # Escape regex special chars first
    regex = ""
    i = 0

    while i < len(pattern):
        c = pattern[i]

        # ✅ Handle wildcard *
        if c == "*":
            regex += ".*"
            i += 1

        # ✅ Handle placeholders {y}, {z}, etc.
        elif c == "{":
            j = pattern.index("}", i)
            key = pattern[i+1:j]

            # assume numeric fields by default
            regex += fr"(?P<{key}>\d+)"

            i = j + 1

        # ✅ Escape regex special characters
        elif c in ".[](){}+?^$\\|":
            regex += "\\" + c
            i += 1

        else:
            regex += c
            i += 1

    return re.compile("^" + regex + "$")


TileInfo = namedtuple(
    "TileInfo",
    ["y", "z", "filename", "reader", "delta_y", "delta_x"],
)


def _write_checkpoint(filename: str, x: int, y: int) -> None:
    with open(filename, "w") as f:
        f.write(f"{x},{y}\n")


def _read_checkpoint(filename: str, default_x: int, default_y: int) -> Tuple[int, int]:
    try:
        with open(filename, "r") as f:
            content = f.read().strip()
            x_str, y_str = content.split(",")
            return int(x_str), int(y_str)
    except (FileNotFoundError, ValueError):
        return default_x, default_y


def prompt_dandi_api_key() -> str:
    """Check for dandi api key and prompt user if key not found."""
    key = os.environ.get("DANDI_API_KEY")
    print(key)
    return key if key else getpass.getpass("Enter your DANDI API key: ")


def open_tile_reader(
    path: str,
    *,
    dandiset_id: Optional[str] = None,
    api_key: Optional[str] = None,
    chunks: Optional[Tuple[int, ...]] = None,
) -> da.Array:
    """Read a tile from the path and apply skew if needed."""
    if path.endswith(".ome.zarr"):
        if dandiset_id is None:
            return da.asarray(ZarrPythonGroup.open(path)["0"],
                              chunks=chunks if chunks is not None else (128, 128, 128))
        return da.asarray(ZarrPythonGroup.open_dandi(
            dandiset_id=dandiset_id,
            asset_path=path,
            api_key=api_key,
        )["0"], chunks=chunks if chunks is not None else (128, 128, 128))

    return da.asarray(
        SpoolSetInterpreter(path, f"{path}_info.mat").assemble_cropped(),
        chunks=chunks if chunks is not None else (128, 128, 128))


def discover_tile_paths(inp: str,
                        *,
                        dandiset_id: Optional[str],
                        api_key: Optional[str],
                        filename_pattern: Optional[str] = "*.ome.zarr",) -> List[str]:
    """Get all tiles from the folder specified."""
    if dandiset_id is None:
        paths = sorted(glob(os.path.join(inp, filename_pattern)))
        if not paths:
            raise ValueError(
                "No tile folders found in input directory")
        return paths

    with DandiAPIClient(
        api_url="https://api.dandiarchive.org/api",
        token=api_key,
    ) as client:
        dandiset = client.get_dandiset(dandiset_id, "draft")
        prefix = PurePosixPath(inp.rstrip("/") + "/")
        depth = len(prefix.parts)

        paths = [
            asset.path
            for asset in dandiset.get_assets_with_path_prefix(str(prefix))
            if len(PurePosixPath(asset.path).parts) == depth + 1
        ]

    if not paths:
        raise ValueError("No tile assets found in DANDI dataset")

    return paths


def convert_spool_or_zarr(
    inp: str,
    *,
    overlap: Union[int, str] = 192,
    delta_x: int = 0,
    voxel_size: List[float] = [1, 1, 1],
    general_config: GeneralConfig = None,
    zarr_config: ZarrConfig = None,
    nii_config: NiftiConfig = None,
    dandiset_id: Optional[str] = None,
    x_chunk_start: Optional[int] = None,
    x_chunk_end: Optional[int] = None,
    chunks_processed: int = 0,
    checkpoint_file: Optional[str] = None,
    filename_pattern: str = "*_run{y}*.ome.zarr"
) -> None:
    """
    Convert a collection of spool files or ome_zarr files into a large Zarr.

    Parameters
    ----------
    inp
        Path to the root directory, which contains a collection of
        subfolders named `*_y{:02d}_z{:02d}*_HR`, each containing a
        collection of files named `*spool.dat`. _z{:02d} is optional

        Or a Path to the root directory which contains a collection of
        ome.zarr files named `*_y{:02d}_z{:02d}*_HR.ome.zarr`. _z{:02d}
        is optional

        TODO: add instrution for metadata file and info file
    overlap
        Number of pixels between slices that are overlapped
    voxel_size
        Voxel size along the X, Y and Z dimensions, in microns.
    general_config
        General configuration
    zarr_config
        Zarr related configuration
    nii_config
        NIfTI header related configuration
    use_runs
        If True will use the run id instead of the y id for y value
    dandiset_id
        Dandiset_id that contains the ome.zarr files for inp
        (leave none if inp is local)
    x_end
        Max x values to crop or pad all tiles to
    z_start
        The minimum z value that should be read in each tile
    z_end
        The maximum z value that should be read in each tile
    allow_padding
        If true bad tiles with 0s along the x axis if any are too small
    number_workers
        The number of workers for dask.to_zarr
    threads_per_worker
        The number of threads each worker gets (only used if number_workers is set)
    skew_angle
        Angle that data is skewed and needs to be corrected
    chunks_processed
        The amount of chunks processed all at once
    blend
        Will blending be used across y layers
    stripes
        Directory that contains stripe correction files
    white_matter_intensity
        What the white matter intensity should be set to after stripe correction
    skip_first_layer
        Only do pyramid calculation and skip first layer
    """
    start_timer = time.time()

    voxel_size = list(map(float, reversed(voxel_size)))

    logger.info("Gathering files and metadata")

    api_key = prompt_dandi_api_key() if dandiset_id else None
    tile_paths = discover_tile_paths(
        inp, dandiset_id=dandiset_id, api_key=api_key, filename_pattern=re.sub(r"\{.*?\}", "*", filename_pattern))

    tiles = {}
    tiles_list = []
    regex = format_to_regex(filename_pattern)

    for path in tile_paths:
        name = os.path.basename(path.rstrip("/").replace(".ome.zarr", ""))
        match = regex.fullmatch(name)
        if not match:
            warnings.warn(f"Skipping unrecognized tile name: {name}")
            continue

        y_val = int(match.group("y"))

        z_val = int(match.groupdict().get("z") or 1)

        reader = open_tile_reader(
            path,
            dandiset_id=dandiset_id,
            api_key=api_key
        )
        delta_x_val = delta_x*y_val
        delta_y = 0

        if isinstance(overlap, str):
            with open(overlap, "r") as file:
                yaml_file = yaml.safe_load(file)
                delta_x_val = yaml_file["coordinates"][y_val]["x"]
                delta_y = yaml_file["coordinates"][y_val]["y"]

        tile = TileInfo(
            y_val,
            z_val,
            path,
            reader,
            delta_y,
            delta_x_val
        )

        key = (y_val, z_val)
        if key in tiles:
            raise ValueError(
                f"Duplicate tile: {path} conflicts with {tiles[key].filename}"
            )

        tiles[key] = tile
        tiles_list.append(tile)

    if not tiles_list:
        raise ValueError("No valid tiles detected")

    y_tiles = sorted({t.y for t in tiles_list})
    z_tiles = sorted({t.z for t in tiles_list})

    min_y, max_y = min(y_tiles), max(y_tiles)
    min_z, max_z = min(z_tiles), max(z_tiles)

    num_y, num_z = len(y_tiles), len(z_tiles)

    shapes = {}
    overlaps = {}
    dtypes = defaultdict(list)

    expected_sx = 0
    expected_sy = {}
    expected_overlap = {}
    expected_sz = {}
    # as it is zero, if a tile is missing, it will make dimension mismatch
    all_shapes = np.zeros((num_y, num_z, 3), dtype=int)
    all_overlaps = np.zeros((num_y, num_z), dtype=int)
    for y in range(min_y, max_y+1):
        for z in range(min_z, max_z+1):
            tile = tiles[(y, z)]
            reader = tile.reader
            sz, sy, sx = reader.shape

            shapes[(y, z)] = (sz, sy, sx)

            dtypes[reader.dtype].append((y, z))
            rel_y, rel_z = y - min_y, z - min_z
            all_shapes[rel_y, rel_z] = sz, sy, sx

            expected_sx = max(sx, expected_sx)
            expected_sy[y] = sy
            expected_sz[z] = sz
            overlap_value = overlap
            if y == min_y:
                overlap_value = 0
            elif not isinstance(overlap_value, int):
                tile_up = tiles[(y-1, z)]
                overlap_value = tile_up.delta_y + \
                    tile_up.reader.shape[1] - tile.delta_y
            overlaps[(y, z)] = overlap_value
            all_overlaps[rel_y, rel_z] = overlap_value
            expected_overlap[y] = overlap_value

    if len(dtypes) != 1:
        warnings.warn(f"Multiple dtypes detected: {dict(dtypes)}")

    dtype = next(iter(dtypes))

    logger.info("Ensureing compatiable tile shapes")
    diff_sx = all_shapes[:, :, 2] > expected_sx
    if diff_sx.any():
        y_idxs, z_idxs = np.where(diff_sx)
        raise ValueError(
            f"Inconsistent x shapes at indices: {list(zip(y_idxs, z_idxs))}"
        )
    for y_tile in range(min_y, max_y + 1):
        if y_tile not in expected_sy:
            raise ValueError(f"Missing y tile {y_tile}")
        diff_sy = all_shapes[:, :, 1] != expected_sy[y_tile]
        if diff_sy.any():
            y_idxs, z_idxs = np.where(diff_sy)
            raise ValueError(
                f"Inconsistent y shapes at tiles: {list(zip(y_idxs, z_idxs))}"
            )
        diff_overlap = all_overlaps[y_tile -
                                    min_y, :] != expected_overlap[y_tile]
        if diff_overlap.any():
            y_idxs, z_idxs = np.where(diff_overlap)
            raise ValueError(
                f"Inconsistent y overlaps at tiles: {list(zip(y_idxs, z_idxs))}"
            )

    for z_tile in range(min_z, max_z + 1):
        if z_tile not in expected_sz:
            raise ValueError(f"Missing z tile {z_tile}")
        diff_sz = all_shapes[:, :, 0] != expected_sz[z_tile]
        if diff_sz.any():
            y_idxs, z_idxs = np.where(diff_sz)
            raise ValueError(
                f"Inconsistent z shapes at tiles: {list(zip(y_idxs, z_idxs))}"
            )

    full_x = expected_sx
    full_y = sum(shapes[(y, z_tiles[0])][1]
                 for y in y_tiles) - sum(0 if y == min_y else overlaps[(y, z_tiles[0])]
                                         for y in y_tiles)
    full_z = sum(shapes[(y_tiles[0], z)][0] for z in z_tiles)
    fullshape = (full_z, full_y, full_x)

    chunks = zarr_config.chunk
    if len(chunks) == 1:
        chunks = [chunks[0]]*3

    omz = ZarrPythonGroup.from_config(general_config.out, zarr_config)
    start = 0 if x_chunk_start is None else x_chunk_start * \
        chunks[2]
    end = min(expected_sx, x_chunk_end *
              chunks[2] if x_chunk_end else expected_sx)
    checkpoint_x, checkpoint_y = start, min_y
    if checkpoint_file is not None:
        checkpoint_x, checkpoint_y = _read_checkpoint(
            checkpoint_file, start, min_y
        )
    if checkpoint_x == start and checkpoint_y == min_y:
        try:
            array = omz.create_array("0", shape=fullshape,
                                     zarr_config=zarr_config, dtype=dtype)
        except:
            array = omz["0"]
    else:
        array = omz["0"]

    x_chunks = array._array.chunks[2]*chunks_processed
    if x_chunks == 0:
        x_chunks = expected_sx
    logger.info("Writing level 0 array with shape %s", fullshape)
    bottom_overlap = None
    for z in z_tiles:
        for x in range(checkpoint_x, end,
                       x_chunks):
            x2 = min(expected_sx, x+x_chunks)
            for y in range(min_y, max_y+1):
                if checkpoint_file is not None:
                    _write_checkpoint(checkpoint_file, x)
                key = (y, z)
                if key in tiles:

                    tile = tiles[key]
                    rel_y, rel_z = tile.y - min_y, tile.z - min_z
                    data = open_tile_reader(
                        tile.filename,
                        dandiset_id=dandiset_id,
                        api_key=api_key,
                        chunks=array._array.chunks,
                    )

                    if data.shape[2] < expected_sx:
                        pad_width = expected_sx - data.shape[2]

                        data = da.pad(
                            data,
                            pad_width=((0, 0), (0, 0), (0, pad_width)),
                            mode="constant",
                            constant_values=0,
                        )

                    data_x = x-tile.delta_x
                    data_x2 = x2-tile.delta_x
                    data_x = min(data.shape[2], data_x)
                    data_x2 = max(0, data_x2)
                    if data_x2 < expected_sx:
                        data = data[:, :, :data_x2]
                    elif data_x2 > expected_sx:
                        data = da.pad(
                            data,
                            pad_width=((0, 0), (0, 0),
                                       (0, data_x2 - expected_sx)),
                            mode="constant",
                            constant_values=0,
                        )

                    if data_x > 0:
                        data = data[:, :, data_x:]
                    elif data_x < 0:
                        data = da.pad(
                            data,
                            pad_width=((0, 0), (0, 0), (-data_x, 0)),
                            mode="constant",
                            constant_values=0,
                        )

                    next_overlap = 0
                    if (y+1, z) in tiles:
                        next_overlap = overlaps[(y+1, z)]

                    if (overlaps[(y, z)] or next_overlap) and len(y_tiles) > 1:

                        if tile.y != min_y:
                            t = np.linspace(0, 1, overlaps[(y, z)])
                            ramp = (1 - np.cos(np.pi * t)) / 2
                            ramp_inverse = (1 + np.cos(np.pi * t)) / 2
                            ramp = ramp[None, :, None]
                            ramp_inverse = ramp_inverse[None, :, None]
                            top_overlap = data[:, :overlaps[(y, z)], :]
                            data = data[:, overlaps[(y, z)]:, :]

                            blended = bottom_overlap * ramp_inverse + \
                                top_overlap * ramp

                            data = da.concatenate(
                                [blended, data], axis=1)
                        if tile.y != max_y:
                            bottom_overlap = data[:, -next_overlap:, :]
                            data = data[:, :-next_overlap, :]

                    ystart = sum(
                        expected_sy[min_y + y_inner] for y_inner in range(rel_y)) - \
                        sum(expected_overlap[y_inner + min_y]
                            for y_inner in range(min(rel_y+1, max_y)))
                    zstart = sum(expected_sz[min_z + z_inner]
                                 for z_inner in range(rel_z))

                    data = da.rechunk(data, array._array.chunks)

                    print(data.shape)

                    logger.info(f"Storing Tile z:{z}, y:{y}, x:{x}-{x2}")
                    if x > checkpoint_x or y >= checkpoint_y:
                        logger.info(f"starting write {y}")
                        array[zstart:(
                            zstart + data.shape[0]), ystart:(ystart + data.shape[1]), x:x2] = data

                else:
                    raise ValueError(f"missing tile (z:{z}, y:{y})")
    for s in range(start, end, x_chunks):
        omz.generate_pyramid(levels=zarr_config.levels,
                             copy_config=general_config,
                             copy_zarr_config=zarr_config,
                             x_min=s,
                             x_max=s+x_chunks)
    omz.write_ome_metadata(
        axes=["z", "y", "x"],
        space_scale=voxel_size,
    )

    if nii_config and nii_config.nii:
        header = build_nifti_header(
            zgroup=omz,
            voxel_size_zyx=tuple(voxel_size),
            unit="micrometer",
            nii_config=nii_config,
        )
        omz.write_nifti_header(header)
    end_timer = time.time()
    length = end_timer - start_timer
    logger.info(f"Conversion completed in {length/60} minutes.")
