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
from typing import List, Optional, Tuple, Union

import dask

# externals
import dask.array as da
import numpy as np
from dandi.dandiapi import DandiAPIClient
from dask.diagnostics import ProgressBar
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

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
    voxel_sizes: Tuple[float] = (1.0, 1.0, 1.0),
    skew_angle: float = 0.0
) -> "DeskewedSCAPE_ZYX":
    if path.endswith(".ome.zarr"):
        if dandiset_id is None:
            return DeskewedSCAPE_ZYX(da.array(ZarrPythonGroup.open(path)["0"]), voxel_sizes, skew_angle, 0.0)
        return DeskewedSCAPE_ZYX(da.array(ZarrPythonGroup.open_dandi(
            dandiset_id=dandiset_id,
            asset_path=path,
            api_key=api_key,
        )["0"]), voxel_sizes, skew_angle, 0.0)

    return DeskewedSCAPE_ZYX(
        SpoolSetInterpreter(path, f"{path}_info.mat").assemble_cropped(),
        voxel_sizes,
        skew_angle,
        0.0)


class DeskewedSCAPE_ZYX:
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
    ) -> None:
        self.raw = raw_data
        self.z_size, self.y_size, self.x_size = voxel_sizes
        self.delta = skew_angle
        self.bg_value = bg_value

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

        X_src = Xg - Zg * self.shps
        Y_src = Yg
        Z_src = Zg

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
            order=3,
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
    x_end: Optional[int] = None,
    z_start: Optional[int] = None,
    z_end: Optional[int] = None,
    allow_padding: bool = False,
    number_workers: Optional[int] = None,
    threads_per_worker: int = 1,
    skew_angle: Optional[float] = 42.0,
    background_removal: bool = False
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
            voxel_sizes=voxel_size,
            skew_angle=skew_angle
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
        sz, sy, sx = reader.shape
        if x_end is not None:
            sx = min(x_end, sx)
        if z_end is not None:
            sz = min(z_end, sz)
        if z_start is not None:
            sz -= min(z_start, sz)

        shapes[(y, z)] = (sz, sy, sx)
        dtypes[reader.dtype].append((y, z))
        # Collect shapes and dtypes.
        rel_y, rel_z = y - min_y, z - min_z
        all_shapes[rel_y, rel_z] = sz, sy, sx
        expected_sx = max(sx, expected_sx)
        expected_sy[y] = sy
        expected_sz[z] = sz

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
    for z_tile in range(min_z, max_z + 1):
        if z_tile not in expected_sz:
            raise ValueError(f"Missing z tile {z_tile}")
        diff_sz = all_shapes[:, :, 0] != expected_sz[z_tile]
        if diff_sz.any():
            y_idxs, z_idxs = np.where(diff_sz)
            raise ValueError(
                f"Inconsistent z shapes at tiles: {list(zip(y_idxs, z_idxs))}"
            )

    full_x = min(next(iter(shapes.values()))[2], x_end) if x_end else next(
        iter(shapes.values()))[2]
    full_y = sum(shapes[(y, z_tiles[0])][1]
                 for y in y_tiles) - (len(y_tiles) - 1) * overlap
    full_z = sum(shapes[(y_tiles[0], z)][0] for z in z_tiles)
    fullshape = (full_z, full_y, full_x)

    omz = ZarrPythonGroup.from_config(general_config.out, zarr_config)
    array = omz.create_array("0", shape=fullshape,
                             zarr_config=zarr_config, dtype=dtype)

    X_CHUNKS = array._array.chunks[2]*40
    logger.info("Writing level 0 array with shape %s", fullshape)
    for z in z_tiles:
        for y in y_tiles:
            key = (y, z)
            if key in tiles:
                for x in range(0, expected_sx, X_CHUNKS):
                    x2 = min(expected_sx, x+X_CHUNKS)

                    tile = tiles[key]
                    rel_y, rel_z = tile.y - min_y, tile.z - min_z
                    reader = tile.reader
                    data = (
                        da.from_array(_open_tile_reader(
                            tile.filename,
                            dandiset_id=dandiset_id,
                            api_key=api_key,
                            voxel_sizes=voxel_size,
                            skew_angle=skew_angle
                        ), chunks=array._array.chunks)
                        if tile.filename.endswith(".ome.zarr")
                        else da.from_array(reader, chunks=array._array.chunks)
                    )

                    if overlap and len(y_tiles) > 1:
                        if tile.y != min_y:
                            data = data[:, overlap // 2:, :]
                        if tile.y != max_y:
                            data = data[:, : -(overlap // 2 + overlap % 2), :]
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
                    if z_end is not None:
                        data = data[:z_end, :, :]
                    if z_start is not None:
                        data = data[z_start:, :, :]

                    ystart = sum(
                        expected_sy[min_y + y] - overlap for y in range(rel_y))
                    zstart = sum(expected_sz[min_z + z] for z in range(rel_z))
                    if rel_y != 0:
                        ystart += overlap // 2

                    data = data[:, :, x:x2]

                    # data = skew_correction_shift_dask(
                    #    data, 0.0, voxel_size, skew_angle)

                    print(data.shape)

                    # data = da.from_array(data.compute(), chunks=(256, 256, 256))

                    slicer = (
                        slice(zstart, zstart + data.shape[0]),
                        slice(ystart, ystart + data.shape[1]),
                        slice(x, x2),
                    )
                    logger.info(f"Storing Tile z:{z}, y:{y}, x:{x}-{x2}")

                    data = da.from_array(
                        data.compute(), chunks=array._array.chunks)

                    if number_workers is not None:
                        with dask.config.set(number_workers=number_workers,
                                             threads_per_worker=threads_per_worker):
                            with ProgressBar():
                                da.to_zarr(data, array._array, region=slicer)
                    else:
                        with ProgressBar():
                            da.to_zarr(data, array._array, region=slicer)

            else:
                raise ValueError(f"missing tile (z:{z}, y:{y})")

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
