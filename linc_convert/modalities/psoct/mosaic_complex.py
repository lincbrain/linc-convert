"""
This module includes components that are based on the
https://github.com/CarolineMagnain/OCTAnalysis repository (currently private).  We
have permission from the owner to include the code here.
"""

import logging
import os.path as op
import warnings
from typing import Annotated

import cyclopts
import dask.array as da
import nibabel as nib
import numpy as np
import yaml
from cyclopts import Parameter
from dask.diagnostics import ProgressBar

from linc_convert.modalities.psoct.cli import psoct
from linc_convert.modalities.psoct.single_tile import process_complex3d
from linc_convert.utils.io.matlab_array_wrapper import as_arraywrapper
from linc_convert.utils.io.zarr import from_config
from linc_convert.utils.io.zarr.helpers import (
    _compute_zarr_layout as compute_zarr_layout,
)
from linc_convert.utils.nifti_header import build_nifti_header
from linc_convert.utils.stitch import MosaicInfo, TileInfo
from linc_convert.utils.unit import to_ome_unit
from linc_convert.utils.zarr_config import (
    GeneralConfig,
    NiftiConfig,
    ZarrConfig,
    autoconfig,
)

logger = logging.getLogger(__name__)

mosaic_complex = cyclopts.App(name="mosaic_complex", help_format="markdown")
psoct.command(mosaic_complex)

# 3d data has pixdim incorrectly set and cause nibabel keep logging warning
nib.imageglobals.logger.setLevel(40)


def _load_tile_info_yaml(yaml_file: str) -> dict:
    """Load tile information from YAML file."""
    with open(yaml_file, "r") as f:
        return yaml.safe_load(f)


def _load_complex_tile(file_path: str, key: str = None) -> da.Array:
    """Load complex 3D data from a file."""
    if file_path.endswith(".mat"):
        wrapper = as_arraywrapper(file_path, key)
        if not hasattr(wrapper, "dtype"):
            raise ValueError(f"Could not load array from {file_path}")
        data = wrapper
    elif file_path.endswith(".nii") or file_path.endswith(".nii.gz"):
        data = nib.load(file_path).dataobj
    return da.from_array(data, chunks=data.shape)


def _shift_focus(tile: da.Array, focus_plane: np.ndarray, s_max: int) -> da.Array:
    """Shift tile depth based on focus plane."""
    if len(focus_plane.shape) == 2:
        focus_plane = focus_plane[..., None]

    # This function will be applied per block using map_blocks
    def shift_block(block, block_info=None):
        nonlocal focus_plane
        if block_info is None:
            return block
        # Get the spatial coordinates for this block
        # For simplicity, we'll shift the entire tile
        # pad the sources at the END so indices up to Nz-1 are valid
        pad_width = ((0, 0), (0, 0), (s_max, s_max))
        block_padded = np.pad(block, pad_width, mode="constant", constant_values=np.nan)
        # build indices for the expanded depth
        z = np.arange(block.shape[-1] + s_max, dtype=np.int32)[None, None, :]
        # Use a subset of focus_plane corresponding to this block
        # For now, use the full focus_plane (assuming it matches tile spatial
        # dimensions)

        idx = z + focus_plane
        result = np.take_along_axis(block_padded, idx, axis=2)
        return result

    # Apply shift using map_blocks
    result_shape = (tile.shape[0], tile.shape[1], tile.shape[2] + s_max)
    return tile.map_blocks(shift_block, dtype=tile.dtype, chunks=result_shape)


@mosaic_complex.default
@autoconfig
def mosaic_complex(
    tile_info_file: str,
    *,
    dbi_output: Annotated[str, Parameter(name=["--dBI", "-d"])],
    o3d_output: Annotated[str, Parameter(name=["--O3D", "-o"])],
    r3d_output: Annotated[str, Parameter(name=["--R3D", "-r"])],
    focus_plane: str = None,
    zarr_config: ZarrConfig = None,
    general_config: GeneralConfig = None,
    nifti_config: NiftiConfig = None,
) -> None:
    """
    Create 3D mosaic from tile information in YAML file.

    Parameters
    ----------
    tile_info_file : str
        Path to YAML file containing tile information.
    dbi_output : str
        Output path for dBI volume.
    o3d_output : str
        Output path for O3D volume.
    r3d_output : str
        Output path for R3D volume.
    focus_plane : str, optional
        Path to focus plane NIfTI file for depth shifting.
    zarr_config : ZarrConfig, optional
        Zarr configuration.
    general_config : GeneralConfig, optional
        General configuration.
    nifti_config : NiftiConfig, optional
        NIfTI configuration.
    """
    logger.info("Started mosaic3d")

    # Load tile information from YAML
    tile_info = _load_tile_info_yaml(tile_info_file)

    # Extract configuration from YAML
    tiles_config = tile_info.get("tiles", [])
    if not tiles_config:
        raise ValueError("No tiles found in YAML file")

    # Get metadata
    metadata = tile_info.get("metadata", {})
    base_dir = metadata.get("base_dir")
    tile_width = metadata.get("tile_width")
    tile_height = metadata.get("tile_height")
    depth = metadata.get("depth")
    scan_resolution = metadata.get("scan_resolution", [1.0, 1.0, 1.0])
    unit = metadata.get("unit", "millimeter")
    flip_z = metadata.get("flip_z", False)
    clip_x = metadata.get("clip_x", 0)
    clip_y = metadata.get("clip_y", 0)
    raw_tile_width = metadata.get("raw_tile_width")
    file_key = metadata.get("file_key")  # Key for mat file array

    if tile_width is None or tile_height is None or depth is None:
        raise ValueError(
            "tile_width, tile_height, and depth must be specified in metadata"
        )

    # Load focus plane if provided
    focus_plane_data = None
    if focus_plane is not None:
        focus_plane_data = nib.load(focus_plane).get_fdata().astype(np.uint16)
        min_focus = np.min(focus_plane_data)
        max_focus = np.max(focus_plane_data)
        depth_increase = max_focus - min_focus
        focus_plane_data = focus_plane_data - min_focus
        depth += depth_increase

    # Collect tile coordinates and file paths
    x_coords_list = []
    y_coords_list = []
    tile_files = []

    for tile in tiles_config:
        x = tile.get("x")
        y = tile.get("y")
        file_path = tile.get("filepath")
        if x is None or y is None or file_path is None:
            logger.warning(f"Skipping incomplete tile: {tile}")
            continue
        x_coords_list.append(x)
        y_coords_list.append(y)
        tile_files.append(op.join(base_dir, file_path))

    if not tile_files:
        raise ValueError("No valid tiles found in YAML file")

    x_coords = np.array(x_coords_list)
    y_coords = np.array(y_coords_list)

    # Compute full mosaic dimensions
    full_width = int(np.nanmax(x_coords) + tile_width)
    full_height = int(np.nanmax(y_coords) + tile_height)

    # Compute zarr layout
    chunk, shard = compute_zarr_layout(
        (depth, full_height, full_width), np.float32, zarr_config
    )

    # Process each tile and collect results
    dbi_tiles = []
    r3d_tiles = []
    o3d_tiles = []
    coords = []

    logger.info(f"Loading and processing {len(tile_files)} tiles")
    for i, (x0, y0, file_path) in enumerate(zip(x_coords, y_coords, tile_files)):
        if not op.exists(file_path):
            logger.warning(f"Tile file not found: {file_path}, skipping")
            continue

        logger.info(f"Processing tile {i + 1}/{len(tile_files)}: {file_path}")

        # Load complex 3D data
        try:
            complex3d = _load_complex_tile(file_path, file_key)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}, skipping")
            continue

        # Validate shape
        if raw_tile_width and complex3d.shape[0] > 4 * raw_tile_width:
            warnings.warn(
                f"Complex3D shape {complex3d.shape} is larger than expected "
                f"{4 * raw_tile_width}. Trimming to 4*raw_tile_width."
            )
            complex3d = complex3d[: 4 * raw_tile_width, :, :]
        elif raw_tile_width and complex3d.shape[0] < 4 * raw_tile_width:
            raise ValueError(
                f"Complex3D shape {complex3d.shape} is smaller than expected "
                f"{4 * raw_tile_width}. Check the input file."
            )

        # Process complex data to get dBI, R3D, O3D
        dBI3D, R3D, O3D = process_complex3d(complex3d, offset=100, flip_phi=False)

        # Apply transformations
        if flip_z:
            dBI3D = da.flip(dBI3D, axis=2)
            R3D = da.flip(R3D, axis=2)
            O3D = da.flip(O3D, axis=2)
        # Handle focus plane shifting
        if focus_plane_data is not None:
            s_max = int(np.max(focus_plane_data))
            dBI3D = _shift_focus(dBI3D, focus_plane_data, s_max)
            R3D = _shift_focus(R3D, focus_plane_data, s_max)
            O3D = _shift_focus(O3D, focus_plane_data, s_max)

        # Clip tiles
        results = da.stack([dBI3D, R3D, O3D], axis=3)
        results = results[clip_x:, clip_y:, :, :]

        dbi_tiles.append(results[:, :, :, 0])
        r3d_tiles.append(results[:, :, :, 1])
        o3d_tiles.append(results[:, :, :, 2])
        coords.append((int(x0), int(y0)))

    if not coords:
        raise ValueError("No valid tiles were processed")

    # Stitch tiles for each modality using MosaicInfo
    logger.info("Stitching tiles")

    # Get tile_overlap from metadata (defaults to "auto")
    tile_overlap = metadata.get("tile_overlap", 0.2)

    # Create MosaicInfo for each modality - dimensions and coordinates extracted from
    # tiles
    dbi_mosaic = MosaicInfo.from_tiles(
        tiles=[TileInfo(x=c[0], y=c[1], image=t) for c, t in zip(coords, dbi_tiles)],
        depth=depth,
        chunk_size=None,  # Will use tile dimensions
        circular_mean=False,
        tile_overlap=tile_overlap,
    )

    r3d_mosaic = MosaicInfo.from_tiles(
        tiles=[TileInfo(x=c[0], y=c[1], image=t) for c, t in zip(coords, r3d_tiles)],
        depth=depth,
        chunk_size=None,  # Will use tile dimensions
        circular_mean=False,
        tile_overlap=tile_overlap,
    )

    o3d_mosaic = MosaicInfo.from_tiles(
        tiles=[TileInfo(x=c[0], y=c[1], image=t) for c, t in zip(coords, o3d_tiles)],
        depth=depth,
        chunk_size=None,  # Will use tile dimensions
        circular_mean=True,
        tile_overlap=tile_overlap,
    )

    # Stitch using lazy dask operations
    dBI_result = dbi_mosaic.stitch()
    R3D_result = r3d_mosaic.stitch()
    O3D_result = o3d_mosaic.stitch()

    # Transpose to (z, y, x) for output
    dBI_result = dBI_result.transpose(2, 1, 0)
    R3D_result = R3D_result.transpose(2, 1, 0)
    O3D_result = O3D_result.transpose(2, 1, 0)

    # Prepare outputs
    writers = []
    results = []
    zgroups = []

    for out, res in zip(
        [dbi_output, r3d_output, o3d_output], [dBI_result, R3D_result, O3D_result]
    ):
        if shard:
            res = da.rechunk(res, chunks=shard)
        else:
            res = da.rechunk(res, chunks=chunk)

        zgroup = from_config(out, zarr_config)
        zgroups.append(zgroup)

        writer = zgroup.create_array(
            "0", shape=res.shape, dtype=np.float32, zarr_config=zarr_config
        )
        writers.append(writer)
        results.append(res)

    # Store results
    task = da.store(results, writers, compute=False)
    with ProgressBar():
        da.compute(task)

    scan_resolution = scan_resolution[:3][::-1]  # Reverse to (z, y, x)
    logger.info("Finished stitching, generating pyramid and metadata")

    for zgroup in zgroups:
        zgroup.generate_pyramid(
            mode="mean", no_pyramid_axis=zarr_config.no_pyramid_axis
        )
        logger.info("Write OME-Zarr multiscale metadata")
        zgroup.write_ome_metadata(
            axes=["z", "y", "x"],
            space_scale=scan_resolution,
            space_unit=to_ome_unit(unit),
        )

        if not nifti_config.nii:
            continue

        header = build_nifti_header(
            zgroup=zgroup,
            voxel_size_zyx=tuple(scan_resolution),
            unit=unit,
            nii_config=nifti_config,
        )
        zgroup.write_nifti_header(header)

    logger.info("Finished generating pyramid")
