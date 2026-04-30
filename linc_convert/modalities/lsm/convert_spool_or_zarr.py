"""Convert spool .dat file sets or ome.zarr files into a consolidated Zarr."""

# stdlib
import getpass
import logging
import os
import re
import warnings
from collections import defaultdict, namedtuple
from glob import glob
from pathlib import PurePosixPath
from typing import List, Optional

# externals
import dask.array as da
import numpy as np
from dandi.dandiapi import DandiAPIClient
from dask.diagnostics import ProgressBar

# internals
from linc_convert.utils.io.spool import SpoolSetInterpreter
from linc_convert.utils.io.zarr.drivers.zarr_python import (
    ZarrPythonArray,
    ZarrPythonGroup,
)
from linc_convert.utils.nifti_header import build_nifti_header
from linc_convert.utils.zarr_config import (
    GeneralConfig,
    NiftiConfig,
    ZarrConfig,
)

logger = logging.getLogger(__name__)


_TILE_PATTERN = re.compile(
    r"^(?P<prefix>\w*)"
    r"_run(?P<run>[0-9]+)"
    r"_y(?P<y>[0-9]+)"
    r"(?:_z(?P<z>[0-9]+))?"
    r"(?P<suffix>\w*)$"
)


TileInfo = namedtuple(
    "TileInfo",
    ["prefix", "run", "y", "z", "suffix", "filename", "reader"],
)


def _prompt_dandi_api_key() -> str:
    key = os.environ.get("DANDI_API_KEY")
    return key if key else getpass.getpass("Enter your DANDI API key: ")


def _open_tile_reader(
    path: str,
    *,
    dandiset_id: Optional[str],
    api_key: Optional[str],
) -> SpoolSetInterpreter:
    if path.endswith(".ome.zarr"):
        if dandiset_id is None:
            return ZarrPythonGroup.open(path)["0"]
        return ZarrPythonGroup.open_dandi(
            dandiset_id=dandiset_id,
            asset_path=path,
            api_key=api_key,
        )["0"]

    return SpoolSetInterpreter(path, f"{path}_info.mat")


def _discover_tile_paths(inp: str,
                         *,
                         dandiset_id: Optional[str],
                         api_key: Optional[str]) -> List[str]:
    if dandiset_id is None:
        paths = sorted(glob(os.path.join(inp, "*_y*_HR/")))
        if not paths:
            paths = sorted(glob(os.path.join(inp, "*_y*_HR.ome.zarr")))
            if not paths:
                raise ValueError("No tile folders found in input directory")
        return paths

    with DandiAPIClient(
        api_url="https://api.lincbrain.org/api",
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
    overlap: int = 192,
    voxel_size: list[float] = (1, 1, 1),
    general_config: GeneralConfig = None,
    zarr_config: ZarrConfig = None,
    nii_config: NiftiConfig = None,
    use_runs: bool = False,
    dandiset_id: Optional[str] = None,
    max_x: Optional[int] = None
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
        dandiset_id that contains the ome.zarr files for inp
        (leave none if inp is local)
    max_x
        value to crop all x shapes to
    """
    logger.info("Gathering files and metadata")

    api_key = _prompt_dandi_api_key() if dandiset_id else None
    tile_paths = _discover_tile_paths(
        inp, dandiset_id=dandiset_id, api_key=api_key)

    tiles = {}
    tiles_list = []

    for path in tile_paths:
        name = os.path.basename(path.rstrip("/").replace(".ome.zarr", ""))
        match = _TILE_PATTERN.fullmatch(name)
        if not match:
            warnings.warn(f"Skipping unrecognized tile name: {name}")
            continue

        y_val = int(match.group("run") if use_runs else match.group("y"))
        z_val = int(match.group("z") or 1)

        reader = _open_tile_reader(
            path,
            dandiset_id=dandiset_id,
            api_key=api_key,
        )

        tile = TileInfo(
            match.group("prefix"),
            int(match.group("run")),
            y_val,
            z_val,
            match.group("suffix"),
            path,
            reader,
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
    dtypes = defaultdict(list)

    expected_sx = 0
    expected_sy = {}
    expected_sz = {}
    # TODO: as it is zero, if a tile is missing, it will make dimension mismatch
    all_shapes = np.zeros((num_y, num_z, 3), dtype=int)

    for (y, z), tile in tiles.items():
        reader = tile.reader
        if isinstance(reader, ZarrPythonArray):
            sz, sy, sx = reader.shape
        else:
            sz, sy, sx = reader.assembled_spool_shape
        if max_x is not None:
            sx = min(max_x, sx)

        shapes[(y, z)] = (sz, sy, sx)
        dtypes[reader.dtype].append((y, z))
        # Collect shapes and dtypes.
        rel_y, rel_z = y - min_y, z - min_z
        all_shapes[rel_y, rel_z] = sz, sy, sx
        expected_sx = sx
        expected_sy[y] = sy
        expected_sz[z] = sz

    if len(dtypes) != 1:
        warnings.warn(f"Multiple dtypes detected: {dict(dtypes)}")

    dtype = next(iter(dtypes))

    logger.info("Ensureing compatiable tile shapes")
    diff_sx = all_shapes[:, :, 2] != expected_sx
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
    for z_tile in range(min_z, max_z + 1):
        if z_tile not in expected_sz:
            raise ValueError(f"Missing z tile {z_tile}")
        diff_sz = all_shapes[:, :, 0] != expected_sz[z_tile]
        if diff_sz.any():
            y_idxs, z_idxs = np.where(diff_sz)
            raise ValueError(
                f"Inconsistent z shapes at tiles: {list(zip(y_idxs, z_idxs))}"
            )

    full_x = min(next(iter(shapes.values()))[2], max_x) if max_x else next(
        iter(shapes.values()))[2]
    full_y = sum(shapes[(y, z_tiles[0])][1]
                 for y in y_tiles) - (len(y_tiles) - 1) * overlap
    full_z = sum(shapes[(y_tiles[0], z)][0] for z in z_tiles)
    fullshape = (full_z, full_y, full_x)

    omz = ZarrPythonGroup.from_config(general_config.out, zarr_config)
    array = omz.create_array("0", shape=fullshape,
                             zarr_config=zarr_config, dtype=dtype)

    for z in z_tiles:

        for y in y_tiles:
            key = (y, z)

            if key in tiles:
                tile = tiles[key]
                reader = tile.reader
                data = (
                    da.array(_open_tile_reader(
                        tile.filename,
                        dandiset_id=dandiset_id,
                        api_key=api_key,
                    ))
                    if isinstance(reader, ZarrPythonArray)
                    else reader.assemble_cropped()
                )

                if overlap and len(y_tiles) > 1:
                    if tile.y != min_y:
                        data = data[:, overlap // 2:, :]
                    if tile.y != max_y:
                        data = data[:, : -(overlap // 2 + overlap % 2), :]
                if max_x is not None:
                    data = data[:, :, :min(data.shape[2], max_x)]

                ystart = sum(
                    expected_sy[min_y + y] - overlap for y in range(rel_y))
                zstart = sum(expected_sz[min_z + z] for z in range(rel_z))
                if rel_y != 0:
                    ystart += overlap // 2

                slicer = (
                    slice(zstart, zstart + data.shape[0]),
                    slice(ystart, ystart + data.shape[1]),
                    slice(None),
                )
                logger.info(f"Storing Tile z:{z}, y:{y}")

                with ProgressBar():
                    da.to_zarr(data, array._array, region=slicer)

            else:
                raise ValueError(f"missing tile (z:{z}, y:{y})")

    logger.info("Writing level 0 array with shape %s", fullshape)

    voxel_size = list(map(float, reversed(voxel_size)))

    omz.generate_pyramid(levels=zarr_config.levels)
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

    logger.info("Conversion complete.")
