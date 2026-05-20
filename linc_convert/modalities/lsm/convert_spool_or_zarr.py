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
from pathlib import PurePosixPath
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


_TILE_PATTERN = re.compile(
    r"^(?P<prefix>\w*)"
    r"_run(?P<run>[0-9]+)"
    r"_y(?P<y>[0-9]+)"
    r"(?:_z(?P<z>[0-9]+))?"
    r"(?P<suffix>\w*)$"
)

_TILE_PATTERN2 = re.compile(
    r"^(?P<prefix>[\w-]*)"
    r"_chunk-(?P<y>[0-9]+)"
    r"(?:_z(?P<z>[0-9]+))?"
    r"(?P<suffix>[\w-]*)$"
)


TileInfo = namedtuple(
    "TileInfo",
    ["y", "z", "filename", "reader", "delta_y", "delta_x"],
)


def write_checkpoint(filename: str, value: int) -> None:
    """
    Write a single integer to a checkpoint file.

    Parameters:
        filename (str): Path to checkpoint file
        value (int): Integer to save
    """
    with open(filename, "w") as f:
        f.write(f"{value}\n")


def read_checkpoint(filename: str, default: int) -> int:
    try:
        with open(filename, "r") as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return default


def prompt_dandi_api_key() -> str:
    """Check for dandi api key and prompt user if key not found."""
    key = os.environ.get("DANDI_API_KEY")
    print(key)
    return key if key else getpass.getpass("Enter your DANDI API key: ")


def open_tile_reader(
    path: str,
    *,
    dandiset_id: Optional[str],
    api_key: Optional[str],
    voxel_sizes: Tuple[float] = (1.0, 1.0, 1.0),
    skew_angle: float = 0.0,
    chunks: Optional[Tuple[int, ...]] = None,
    flip_vertically: bool = False,
) -> "Deskewed_Tile":
    """Read a tile from the path and apply skew if needed."""
    if path.endswith(".ome.zarr"):
        if dandiset_id is None:
            return Deskewed_Tile.wrap(ZarrPythonGroup.open(path)["0"],
                                      voxel_sizes,
                                      skew_angle,
                                      chunks,
                                      flip_vertically)
        return Deskewed_Tile.wrap(ZarrPythonGroup.open_dandi(
            dandiset_id=dandiset_id,
            asset_path=path,
            api_key=api_key,
        )["0"], voxel_sizes, skew_angle, chunks, flip_vertically)

    return Deskewed_Tile.wrap(
        SpoolSetInterpreter(path, f"{path}_info.mat").assemble_cropped(),
        voxel_sizes,
        skew_angle,
        chunks)


def discover_tile_paths(inp: str,
                        *,
                        dandiset_id: Optional[str],
                        api_key: Optional[str]) -> List[str]:
    """Get all tiles from the folder specified."""
    if dandiset_id is None:
        paths = sorted(glob(os.path.join(inp, "*_y*_HR/")))
        if not paths:
            paths = sorted(glob(os.path.join(inp, "*.ome.zarr")))
            if not paths:
                paths = sorted(glob(os.path.join(inp, "*_y*/")))
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


class Deskewed_Tile:
    """
    Read-only, on-the-fly deskewed view of SCAPE data.

    raw_data: (Z, Y, X) or (Z, Y, X, T)
    voxel_sizes: (z_size, y_size, x_size)
    Deskew: X <- Z shear (matches MATLAB code exactly)
    """

    def __init__(
        self,
        raw_data: da.Array,
        voxel_sizes: Tuple[float],
        skew_angle: float,
        bg_value: float = 0.0,
        flip_vertically: bool = False
    ) -> None:
        self.raw = raw_data
        self.z_size, self.y_size, self.x_size = voxel_sizes
        self.delta = skew_angle
        self.bg_value = bg_value
        self.flip_z = flip_vertically

        self.shps = (
            self.z_size
            * np.tan(np.deg2rad(self.delta))
            / self.x_size
        )

        self.has_time = (raw_data.ndim == 4)

        Z, Y, X = raw_data.shape[:3]
        self.extra_x = int(np.ceil(Z * self.shps))

        # Deskewed output shape (Z, Y, X_out[, T])
        if self.has_time:
            self._shape = (Z, Y, X + self.extra_x, raw_data.shape[3])
        else:
            self._shape = (Z, Y, X + self.extra_x)

    @property
    def shape(self) -> Tuple[int]:
        """Return shape after descewing is performed."""
        return self._shape

    @property
    def dtype(self) -> type:
        """Returns raw data data type."""
        return self.raw.dtype

    @property
    def ndim(self) -> int:
        """Returns raw data number of demensions."""
        return self.raw.ndim

    @property
    def chunks(self) -> Tuple[int]:
        """Returns raw data chunks."""
        return self.raw.chunks

    def __getitem__(self,
                    key: Union[Tuple[Union[int, slice]], int, slice]
                    ) -> np.ndarray:
        """Get a slice of the deskewed data."""
        if self.delta == 0.0:
            return self.raw[key]

        if not isinstance(key, tuple):
            key = (key,)
        while len(key) < len(self.shape):
            key += (slice(None),)

        z_key, y_key, x_key = key[:3]

        z_idx = self._key_to_idx(z_key, self.shape[0])
        y_idx = self._key_to_idx(y_key, self.shape[1])
        x_idx = self._key_to_idx(x_key, self.shape[2])

        Zg, Yg, Xg = np.meshgrid(
            z_idx, y_idx, x_idx, indexing="ij"
        )

        if self.flip_z:
            Z_eff = self.raw.shape[0] - 1 - Zg
            X_src = Xg + Z_eff * self.shps
        else:
            Z_eff = Zg
            X_src = Xg - Z_eff * self.shps

        Y_src = Yg
        Z_src = Z_eff

        # Interpolation margin (cubic)
        margin = 2

        # Compute required bounds in raw coordinates
        z0 = int(np.floor(Z_src.min())) - margin
        z1 = int(np.ceil(Z_src.max())) + margin + 1

        y0 = int(np.floor(Y_src.min())) - margin
        y1 = int(np.ceil(Y_src.max())) + margin + 1

        x0 = int(np.floor(X_src.min())) - margin
        x1 = int(np.ceil(X_src.max())) + margin + 1

        # Clamp to raw array bounds
        z0 = max(z0, 0)
        y0 = max(y0, 0)
        x0 = max(x0, 0)

        z1 = min(z1, self.raw.shape[0])
        y1 = min(y1, self.raw.shape[1])
        x1 = min(x1, self.raw.shape[2])

        # Slice raw data
        raw_slice = self._as_numpy(self.raw[z0:z1, y0:y1, x0:x1])

        # Shift coordinates into sliced array space
        Zs = Z_src - z0
        Ys = Y_src - y0
        Xs = X_src - x0

        coords = np.vstack([
            Zs.ravel(),
            Ys.ravel(),
            Xs.ravel(),
        ])

        # Interpolate
        sampled = map_coordinates(
            raw_slice,
            coords,
            order=1,
            mode="constant",
            cval=self.bg_value,
        )

        return sampled.reshape(Zg.shape)

    @staticmethod
    def _key_to_idx(key: Union[Tuple[Union[int, slice]], int, slice],
                    size: int
                    ) -> np.ndarray:
        if isinstance(key, slice):
            return np.arange(*key.indices(size))
        elif np.isscalar(key):
            return np.array([key])
        else:
            return np.asarray(key)

    @staticmethod
    def _as_numpy(arr: Union[da.Array, np.ndarray]) -> np.ndarray:
        if isinstance(arr, da.Array):
            return arr.compute()
        return np.asarray(arr)

    @classmethod
    def wrap(cls,
             arr: Union[da.Array, np.ndarray, ZarrArray],
             voxel_sizes: Tuple[float, ...],
             skew_angle: float,
             chunks: Optional[Tuple[int, ...]] = None,
             flip_vertically: bool = False
             ) -> Union[da.Array, np.ndarray, ZarrArray, "Deskewed_Tile"]:
        """Apply deskew wrapper if necessary otherwise convert to dask array."""
        if skew_angle == 0:
            if not isinstance(arr, da.Array):
                arr = da.array(arr)
            if chunks is not None:
                arr = da.rechunk(arr, chunks=chunks)
            return arr

        return da.from_array(cls(arr, voxel_sizes, skew_angle, flip_vertically=flip_vertically),
                             chunks=(arr.chunks if chunks is None else chunks))


def convert_spool_or_zarr(
    inp: str,
    *,
    overlap: Union[int, str] = 192,
    voxel_size: list[float] = (1, 1, 1),
    general_config: GeneralConfig = None,
    zarr_config: ZarrConfig = None,
    nii_config: NiftiConfig = None,
    use_runs: bool = False,
    dandiset_id: Optional[str] = None,
    x_chunk_start: Optional[int] = None,
    x_chunk_end: Optional[int] = None,
    x_end: Optional[int] = None,
    z_start: Optional[int] = None,
    z_end: Optional[int] = None,
    y_start: Optional[int] = None,
    y_end: Optional[int] = None,
    allow_padding: bool = False,
    number_workers: Optional[int] = None,
    threads_per_worker: int = 1,
    skew_angle: float = 0.0,
    chunks_processed: int = 0,
    blend: bool = False,
    stripes: Optional[str] = None,
    white_matter_intensity: float = 1000.0,
    skip_first_layer: bool = False,
    background_threshold: Optional[Union[float, Literal["auto"]]] = None,
    checkpoint_file: Optional[str] = None,
    alternate_pattern: bool = False,
    flip_vertically: bool = False
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
        inp, dandiset_id=dandiset_id, api_key=api_key)

    tiles = {}
    tiles_list = []

    for path in tile_paths:
        name = os.path.basename(path.rstrip("/").replace(".ome.zarr", ""))
        match = (
            _TILE_PATTERN2 if alternate_pattern else _TILE_PATTERN).fullmatch(name)
        if not match:
            warnings.warn(f"Skipping unrecognized tile name: {name}")
            continue

        y_val = int(match.group("y"))
        if use_runs:
            y_val = match.group("run")
        z_val = int(match.group("z") or 1)

        reader = open_tile_reader(
            path,
            dandiset_id=dandiset_id,
            api_key=api_key,
            voxel_sizes=voxel_size,
            skew_angle=skew_angle
        )
        delta_x = 0
        delta_y = 0

        if isinstance(overlap, str):
            with open(overlap, "r") as file:
                yaml_file = yaml.safe_load(file)
                delta_x = yaml_file["coordinates"][y_val]["x"]
                delta_y = yaml_file["coordinates"][y_val]["y"]

        tile = TileInfo(
            y_val,
            z_val,
            path,
            reader,
            delta_y,
            delta_x
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
            if x_end is not None:
                sx = min(x_end, sx)

            if z_end is not None:
                sz = min(z_end, sz)
            if z_start is not None:
                sz -= min(z_start, sz)

            if y_end is not None:
                sy = min(y_end, sy)
            if y_start is not None:
                sy -= min(y_start, sy)

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
    diff_sx = all_shapes[:, :, 2] != expected_sx
    if allow_padding:
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

    full_x = min(expected_sx, x_end) if x_end else expected_sx
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
    if not skip_first_layer:
        checkpoint = min_y
        if checkpoint_file is not None:
            checkpoint = read_checkpoint(checkpoint_file, min_y)
        if checkpoint == min_y:
            array = omz.create_array("0", shape=fullshape,
                                     zarr_config=zarr_config, dtype=dtype)
        else:
            array = omz["0"]

        x_chunks = array._array.chunks[2]*chunks_processed
        if x_chunks == 0:
            x_chunks = expected_sx
        logger.info("Writing level 0 array with shape %s", fullshape)
        bottom_overlap = None
        for z in z_tiles:
            for x in range(start, end,
                           x_chunks):
                x2 = min(expected_sx, x+x_chunks)
                for y in range(min_y, max_y+1):
                    if checkpoint_file is not None:
                        write_checkpoint(checkpoint_file, y)
                    key = (y, z)
                    if key in tiles:

                        tile = tiles[key]
                        rel_y, rel_z = tile.y - min_y, tile.z - min_z
                        data = open_tile_reader(
                            tile.filename,
                            dandiset_id=dandiset_id,
                            api_key=api_key,
                            voxel_sizes=voxel_size,
                            skew_angle=skew_angle,
                            chunks=array._array.chunks,
                            flip_vertically=flip_vertically
                        )

                        threshold = background_threshold
                        if background_threshold == "auto":
                            threshold = data[:, :, -300:].max().compute()

                        if z_end is not None:
                            data = data[:z_end, :, :]
                        if z_start is not None:
                            data = data[z_start:, :, :]

                        if y_end is not None:
                            data = data[:, :y_end, :]
                        if y_start is not None:
                            data = data[:, y_start:, :]

                        if allow_padding and data.shape[2] < expected_sx:
                            pad_width = expected_sx - data.shape[2]

                            data = da.pad(
                                data,
                                pad_width=((0, 0), (0, 0), (0, pad_width)),
                                mode="constant",
                                constant_values=0,
                            )
                        if x_end is not None:
                            data = data[:, :, :min(data.shape[2], x_end)]

                        data_x = x-tile.delta_x
                        data_x2 = x2-tile.delta_x
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
                        if threshold is not None:
                            data = data*(data < threshold)

                        if stripes is not None:

                            name = os.path.basename(
                                tile.filename.rstrip("/").replace(".ome.zarr", ""))

                            correction = tiff.imread(
                                f"{stripes}/{name}.tiff")
                            correction[correction == 0.0] = 1.0
                            if flip_vertically:
                                correction = correction[::-1, :]

                            correction = white_matter_intensity / correction

                            correction = correction[:, :, None]

                            data = data * correction

                        next_overlap = 0
                        if (y+1, z) in tiles:
                            next_overlap = overlaps[(y+1, z)]

                        if (overlaps[y, z] or next_overlap) and len(y_tiles) > 1:

                            if blend:
                                if tile.y != min_y:
                                    t = np.linspace(0, 1, overlaps[y, z])
                                    ramp = (1 - np.cos(np.pi * t)) / 2
                                    ramp_inverse = (1 + np.cos(np.pi * t)) / 2
                                    ramp = ramp[None, :, None]
                                    ramp_inverse = ramp_inverse[None, :, None]
                                    top_overlap = data[:, :overlaps[y, z], :]
                                    data = data[:, overlaps[y, z]:, :]

                                    blended = bottom_overlap * ramp_inverse + \
                                        top_overlap * ramp

                                    data = da.concatenate(
                                        [blended, data], axis=1)
                                if tile.y != max_y:
                                    bottom_overlap = data[:, -next_overlap:, :]
                                    data = data[:, :-next_overlap, :]

                            else:
                                if tile.y != min_y:
                                    data = data[:, overlaps[y, z] // 2:, :]
                                if tile.y != max_y:
                                    data = data[:, : -
                                                (next_overlap // 2 + next_overlap % 2), :]

                        ystart = sum(
                            expected_sy[min_y + y_inner] for y_inner in range(rel_y)) - \
                            sum(expected_overlap[y_inner + min_y]
                                for y_inner in range(min(rel_y+1, max_y)))
                        zstart = sum(expected_sz[min_z + z_inner]
                                     for z_inner in range(rel_z))
                        if rel_y != 0 and not blend:
                            ystart += overlaps[y, z] // 2

                        data = da.rechunk(data, array._array.chunks)

                        print(data.shape)

                        slicer = (
                            slice(zstart, zstart + data.shape[0]),
                            slice(ystart, ystart + data.shape[1]),
                            slice(x, x2),
                        )

                        logger.info(f"Storing Tile z:{z}, y:{y}, x:{x}-{x2}")
                        if y >= checkpoint:
                            if number_workers is not None:
                                with dask.config.set(number_workers=number_workers,
                                                     threads_per_worker=threads_per_worker):
                                    with ProgressBar():
                                        da.to_zarr(data, array._array,
                                                   region=slicer)
                            else:
                                with ProgressBar():
                                    da.to_zarr(data, array._array,
                                               region=slicer)

                    else:
                        raise ValueError(f"missing tile (z:{z}, y:{y})")
    omz.generate_pyramid(levels=zarr_config.levels,
                         copy_config=general_config,
                         copy_zarr_config=zarr_config,
                         x_min=start,
                         x_max=end)
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
