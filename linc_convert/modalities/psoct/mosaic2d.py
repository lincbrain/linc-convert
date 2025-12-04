"""
Create 2D mosaic from tile information in YAML file.
"""

import logging
import os
import os.path as op
from typing import Annotated, Literal, Optional

import cyclopts
import dask.array as da
import nibabel as nib
import numpy as np
import yaml
from cyclopts import Parameter
from dask.diagnostics import ProgressBar
from PIL import Image

from linc_convert.modalities.psoct.cli import psoct
from linc_convert.modalities.psoct.stitch import MosaicInfo, TileInfo
from linc_convert.utils.io.matlab import as_arraywrapper
from linc_convert.utils.io.zarr import from_config
from linc_convert.utils.io.zarr.helpers import \
    _compute_zarr_layout as compute_zarr_layout
from linc_convert.utils.nifti_header import build_nifti_header
from linc_convert.utils.unit import to_ome_unit
from linc_convert.utils.zarr_config import (
    GeneralConfig,
    NiftiConfig,
    ZarrConfig,
    autoconfig,
)

logger = logging.getLogger(__name__)

mosaic2d = cyclopts.App(name="mosaic2d", help_format="markdown")
psoct.command(mosaic2d)


def _load_tile_info_yaml(yaml_file: str) -> dict:
    """Load tile information from YAML file."""
    with open(yaml_file, "r") as f:
        return yaml.safe_load(f)


def _load_image_tile(file_path: str, key: str = None) -> da.Array:
    """
    Load 2D image from a file with lazy loading support.
    
    Supports:
    - .mat files (MATLAB)
    - Zarr archives (groups or arrays)
    - NIfTI files (with mmap for lazy loading)
    - Other formats via dask-image
    """
    import os.path as op
    
    # Check for .mat files
    if file_path.endswith('.mat'):
        wrapper = as_arraywrapper(file_path, key)
        if not hasattr(wrapper, "dtype"):
            raise ValueError(f"Could not load array from {file_path}")
        data = wrapper
        # Handle 3D data - take middle slice
        if data.ndim == 3:
            data = data[:, :, data.shape[2] // 2]
        elif data.ndim > 3:
            data = data.reshape(data.shape[0], data.shape[1], -1)[:, :, 0]
        return da.from_array(data, chunks="auto")
    
    # Check for zarr archives
    # Check if path looks like zarr (directory with zarr metadata files)
    is_zarr = False
    if op.isdir(file_path):
        # Check for zarr v3 (zarr.json) or v2 (.zgroup or .zarray)
        if op.exists(op.join(file_path, 'zarr.json')) or \
           op.exists(op.join(file_path, '.zgroup')) or \
           op.exists(op.join(file_path, '.zarray')):
            is_zarr = True
    elif file_path.endswith('.zarr') or '.zarr' in file_path:
        is_zarr = True
    
    if is_zarr:
        try:
            from linc_convert.utils.io.zarr import open_array, open_group
            
            # Try to open as zarr group first
            try:
                zarr_group = open_group(file_path, mode="r")
                # It's a group, try to get array '0'
                if '0' in zarr_group.keys():
                    zarr_array_wrapper = zarr_group['0']
                    # Get underlying zarr array for dask conversion
                    # ZarrPythonArray has _array attribute with the actual zarr.Array
                    if hasattr(zarr_array_wrapper, '_array'):
                        zarr_python_array = zarr_array_wrapper._array
                    elif hasattr(zarr_array_wrapper, '_zarr_array'):
                        zarr_python_array = zarr_array_wrapper._zarr_array
                    else:
                        # Fallback: try to use the wrapper directly
                        zarr_python_array = zarr_array_wrapper
                    data = da.from_array(zarr_python_array, chunks=zarr_python_array.chunks)
                else:
                    raise ValueError(f"Zarr group at {file_path} does not contain array '0'")
            except (ValueError, KeyError):
                # Try as array
                zarr_array_wrapper = open_array(file_path, mode="r")
                if hasattr(zarr_array_wrapper, '_array'):
                    zarr_python_array = zarr_array_wrapper._array
                elif hasattr(zarr_array_wrapper, '_zarr_array'):
                    zarr_python_array = zarr_array_wrapper._zarr_array
                else:
                    zarr_python_array = zarr_array_wrapper
                data = da.from_array(zarr_python_array, chunks=zarr_python_array.chunks)
            
            # Handle 3D data - take middle slice
            if data.ndim == 3:
                data = data[:, :, data.shape[2] // 2]
            elif data.ndim > 3:
                data = data.reshape(data.shape[0], data.shape[1], -1)[:, :, 0]
            return data
        except Exception as e:
            logger.warning(f"Failed to load as zarr: {e}, trying other formats")
    
    # Check for NIfTI files
    if file_path.endswith(('.nii', '.nii.gz')):
        img = nib.load(file_path)
        # Use dataobj for lazy loading with mmap instead of get_fdata()
        dataobj = img.dataobj
        # Convert to dask array with lazy loading
        data = da.from_array(dataobj, chunks="auto")
        # Handle 3D data - take middle slice
        if data.ndim == 3:
            data = data[:, :, data.shape[2] // 2]
        elif data.ndim > 3:
            # Take first 2D slice
            data = data.reshape(data.shape[0], data.shape[1], -1)[:, :, 0]
        return data
    
    # Try dask-image as fallback
    try:
        import dask_image.imread  # noqa: F401
        data = dask_image.imread.imread(file_path)
        # Handle 3D data - take middle slice
        if data.ndim == 3:
            data = data[:, :, data.shape[2] // 2]
        elif data.ndim > 3:
            data = data.reshape(data.shape[0], data.shape[1], -1)[:, :, 0]
        return data
    except ImportError:
        raise ValueError(
            f"Could not load {file_path}. "
            "Supported formats: .mat, .zarr, .nii/.nii.gz, or formats supported by dask-image"
        )
    except Exception as e:
        raise ValueError(f"Failed to load {file_path} with dask-image: {e}")


def _save_jpeg(image: np.ndarray, output_path: str, quality: int = 95):
    """Save image as JPEG."""
    # Normalize to 0-255 range
    img_min = np.nanmin(image)
    img_max = np.nanmax(image)
    if img_max > img_min:
        normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(image, dtype=np.uint8)

    # Convert to PIL Image and save
    pil_image = Image.fromarray(normalized)
    pil_image.save(output_path, "JPEG", quality=quality)


def _save_tiff(image: np.ndarray, output_path: str):
    """Save image as TIFF without normalization - data is saved as-is."""
    try:
        import tifffile
        # Save data as-is without any normalization or scaling
        # Preserve original dtype and values
        tifffile.imwrite(output_path, image)
    except ImportError:
        # Fallback to PIL if tifffile not available
        # PIL requires uint8 or uint16, so we need to convert
        # But we don't normalize - just convert dtype preserving values as much as possible
        if image.dtype in (np.float32, np.float64):
            # For float, convert to uint16 without normalization
            # This will clip values outside [0, 65535] range
            image_uint16 = np.clip(image, 0, 65535).astype(np.uint16)
        elif image.dtype == np.uint8:
            image_uint16 = image
        else:
            # For other types, convert to uint16
            image_uint16 = np.clip(image, 0, 65535).astype(np.uint16)
        pil_image = Image.fromarray(image_uint16)
        pil_image.save(output_path, "TIFF")


@mosaic2d.default
@autoconfig
def mosaic2d(
    tile_info_file: str,
    *,
    jpeg_output: Annotated[Optional[str], Parameter(name=["--jpeg", "-j"])] = None,
    tiff_output: Annotated[Optional[str], Parameter(name=["--tiff", "-t"])] = None,
    tile_overlap: float | Literal["auto"] = "auto",
    circular_mean: bool = False,
    zarr_config: ZarrConfig = None,
    general_config: GeneralConfig = None,
    nifti_config: NiftiConfig = None,
) -> None:
    """
    Create 2D mosaic from tile information in YAML file.

    Parameters
    ----------
    tile_info_file : str
        Path to YAML file containing tile information.
    jpeg_output : str, optional
        Path to save JPEG preview image.
    tiff_output : str, optional
        Path to save TIFF image.
    tile_overlap : float | Literal["auto"]
        Tile overlap in pixels. If "auto", compute from tile coordinates.
    circular_mean : bool
        Whether to use circular mean for blending.
    zarr_config : ZarrConfig, optional
        Zarr configuration.
    general_config : GeneralConfig, optional
        General configuration.
    nifti_config : NiftiConfig, optional
        NIfTI configuration.
    """
    logger.info("Started mosaic2d")

    # Load tile information from YAML
    tile_info = _load_tile_info_yaml(tile_info_file)

    # Extract configuration from YAML
    tiles_config = tile_info.get("tiles", [])
    if not tiles_config:
        raise ValueError("No tiles found in YAML file")

    # Get metadata
    metadata = tile_info.get("metadata", {})
    tile_width = metadata.get("tile_width")
    tile_height = metadata.get("tile_height")
    scan_resolution = metadata.get("scan_resolution", [1.0, 1.0])
    file_key = metadata.get("file_key")  # Key for mat file array
    
    # Use tile_overlap from function parameter (defaults to "auto")
    # If "auto" and metadata has tile_overlap, use that instead
    if tile_overlap == "auto" and "tile_overlap" in metadata:
        tile_overlap = metadata.get("tile_overlap", "auto")

    if tile_width is None or tile_height is None:
        raise ValueError("tile_width and tile_height must be specified in metadata")

    # Collect tile coordinates and file paths
    x_coords_list = []
    y_coords_list = []
    tile_files = []

    for tile in tiles_config:
        x = tile.get("x")
        y = tile.get("y")
        file_path = tile.get("file_path")
        if x is None or y is None or file_path is None:
            logger.warning(f"Skipping incomplete tile: {tile}")
            continue
        x_coords_list.append(x)
        y_coords_list.append(y)
        tile_files.append(file_path)

    if not tile_files:
        raise ValueError("No valid tiles found in YAML file")

    x_coords = np.array(x_coords_list)
    y_coords = np.array(y_coords_list)

    # Compute full mosaic dimensions (will be used by MosaicInfo)
    full_width = int(np.nanmax(x_coords) + tile_width)
    full_height = int(np.nanmax(y_coords) + tile_height)

    # Process each tile and collect results
    tiles = []
    coords = []

    logger.info(f"Loading and processing {len(tile_files)} tiles")
    for i, (x0, y0, file_path) in enumerate(zip(x_coords, y_coords, tile_files)):
        if not op.exists(file_path):
            logger.warning(f"Tile file not found: {file_path}, skipping")
            continue

        logger.info(f"Processing tile {i + 1}/{len(tile_files)}: {file_path}")

        # Load 2D image
        try:
            image = _load_image_tile(file_path, file_key)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}, skipping")
            continue

        # Ensure 2D
        if image.ndim != 2:
            if image.ndim == 3 and image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                logger.warning(f"Image {file_path} is not 2D, skipping")
                continue

        # Validate and crop tile to expected size
        if image.shape[0] >= tile_height and image.shape[1] >= tile_width:
            # Crop to expected size if needed
            if image.shape[0] != tile_height or image.shape[1] != tile_width:
                image = image[:tile_height, :tile_width]
        else:
            logger.warning(
                f"Tile {file_path} has unexpected size {image.shape}, "
                f"expected at least ({tile_height}, {tile_width})"
            )
            continue

        tiles.append(image)
        coords.append((int(x0), int(y0)))

    if not coords:
        raise ValueError("No valid tiles were processed")

    # Stitch tiles using MosaicInfo
    logger.info("Stitching tiles")
    tile_size = (tile_width, tile_height)
    
    # Create MosaicInfo for 2D mosaic
    mosaic = MosaicInfo.from_tiles_and_coords(
        tiles=[TileInfo(x=c[0], y=c[1], image=t) for c, t in zip(coords, tiles)],
        tile_width=tile_width,
        tile_height=tile_height,
        x_coords=x_coords,
        y_coords=y_coords,
        depth=None,  # 2D mosaic
        chunk_size=tile_size,
        circular_mean=circular_mean,
        tile_overlap=tile_overlap,
    )
    
    # Stitch using lazy dask operations
    result = mosaic.stitch()
    
    # Compute the result
    result = result.compute()

    # Set default output name if not provided
    general_config.set_default_name(op.splitext(op.basename(tile_info_file))[0])

    # Save to Zarr
    out = general_config.out
    logger.info(f"Saving to Zarr: {out}")

    # Result is (width, height), but Zarr expects (height, width) for 2D
    # Transpose to get (height, width)
    result_2d = result.T

    # Compute zarr layout for 2D
    chunk, shard = compute_zarr_layout((full_height, full_width), np.float32,
                                       zarr_config)

    zarr_config.out = out
    zgroup = from_config(zarr_config)

    if shard:
        result_dask = da.rechunk(da.from_array(result_2d, chunks=chunk), chunks=shard)
    else:
        result_dask = da.rechunk(da.from_array(result_2d, chunks=chunk), chunks=chunk)

    writer = zgroup.create_array("0", shape=result_2d.shape, dtype=np.float32,
                                 zarr_config=zarr_config)

    task = da.store(result_dask, writer, compute=False)
    with ProgressBar():
        task.compute()

    # Generate pyramid and metadata
    logger.info("Generating pyramid and metadata")
    zgroup.generate_pyramid()

    scan_resolution_2d = scan_resolution[:2] if len(scan_resolution) >= 2 else [1.0,
                                                                                1.0]
    zgroup.write_ome_metadata(["y", "x"], space_scale=scan_resolution_2d,
                              space_unit="millimeter")

    if nifti_config and nifti_config.nii:
        from niizarr import default_nifti_header

        nii_header = default_nifti_header(
            zgroup["0"], zgroup._get_zarr_python_group().attrs["multiscales"]
        )
        nii_header.set_xyzt_units("mm")
        zgroup.write_nifti_header(nii_header)

    logger.info("Finished generating pyramid")

    # Save JPEG if requested
    if jpeg_output:
        logger.info(f"Saving JPEG preview: {jpeg_output}")
        _save_jpeg(result_2d, jpeg_output)

    # Save TIFF if requested
    if tiff_output:
        logger.info(f"Saving TIFF: {tiff_output}")
        _save_tiff(result_2d, tiff_output)

    logger.info("Finished mosaic2d")
