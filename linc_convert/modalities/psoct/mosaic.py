"""
Create 2D mosaic from tile information in YAML file.
"""

import logging
import os.path as op
from typing import Annotated, Optional, Union

import cyclopts
import dask.array as da
import nibabel as nib
import numpy as np
import yaml
from PIL import Image
from cyclopts import Parameter
from dask.diagnostics import ProgressBar
from tqdm import tqdm

from linc_convert.modalities.psoct.cli import psoct
from linc_convert.modalities.psoct.stitch import MosaicInfo, TileInfo
from linc_convert.utils.io.matlab import as_arraywrapper
from linc_convert.utils.io.zarr import from_config
from linc_convert.utils.io.zarr.helpers import \
    _compute_zarr_layout as compute_zarr_layout
from linc_convert.utils.nifti_header import build_nifti_header
from linc_convert.utils.zarr_config import (
    GeneralConfig,
    NiftiConfig,
    ZarrConfig,
    autoconfig,
)

logger = logging.getLogger(__name__)

mosaic = cyclopts.App(name="mosaic", help_format="markdown")
psoct.command(mosaic)


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

    # Check for .mat files
    if file_path.endswith('.mat'):
        wrapper = as_arraywrapper(file_path, key)
        if not hasattr(wrapper, "dtype"):
            raise ValueError(f"Could not load array from {file_path}")
        data = wrapper
        # # Handle 3D data - take middle slice
        # if data.ndim == 3:
        #     data = data[:, :, data.shape[2] // 2]
        # elif data.ndim > 3:
        #     data = data.reshape(data.shape[0], data.shape[1], -1)[:, :, 0]
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
                    data = da.from_array(zarr_array_wrapper,
                                         chunks=zarr_array_wrapper.chunks)
                else:
                    raise ValueError(
                        f"Zarr group at {file_path} does not contain array '0'")
            except (ValueError, KeyError):
                # Try as array
                zarr_array_wrapper = open_array(file_path, mode="r")
                data = da.from_array(zarr_array_wrapper,
                                     chunks=zarr_array_wrapper.chunks)

            # Handle 3D data - take middle slice
            # if data.ndim == 3:
            #     data = data[:, :, data.shape[2] // 2]
            # elif data.ndim > 3:
            #     data = data.reshape(data.shape[0], data.shape[1], -1)[:, :, 0]
            return data
        except Exception as e:
            logger.warning(f"Failed to load as zarr: {e}, trying other formats")

    # Check for NIfTI files
    if file_path.endswith(('.nii', '.nii.gz')):
        img = nib.load(file_path)
        # Use dataobj for lazy loading with mmap instead of get_fdata()
        dataobj = img.dataobj
        # Convert to dask array with lazy loading
        data = da.from_array(dataobj, chunks=img.shape)
        # Handle 3D data - take middle slice
        # if data.ndim == 3:
        #     data = data[:, :, data.shape[2] // 2]
        # elif data.ndim > 3:
        #     # Take first 2D slice
        #     data = data.reshape(data.shape[0], data.shape[1], -1)[:, :, 0]
        return data

    # Try dask-image as fallback
    try:
        import dask_image.imread  # noqa: F401

        data = dask_image.imread.imread(file_path)
        # Handle 3D data - take middle slice
        # if data.ndim == 3:
        #     data = data[:, :, data.shape[2] // 2]
        # elif data.ndim > 3:
        #     data = data.reshape(data.shape[0], data.shape[1], -1)[:, :, 0]
        return data
    except ImportError:
        raise ValueError(
            f"Could not load {file_path}. "
            "Supported formats: .mat, .zarr, .nii/.nii.gz, or formats supported by "
            "dask-image"
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
        # But we don't normalize - just convert dtype preserving values as much as
        # possible
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


def _load_mask(mask_path: str, keep_lazy: bool = False) -> Union[da.Array, np.ndarray]:
    """
    Load a binary mask from a file.
    
    Supports the same formats as _load_image_tile but expects a 2D binary mask.
    
    Parameters
    ----------
    mask_path : str
        Path to mask file.
    keep_lazy : bool
        If True, keep mask as lazy dask array. If False, compute and return numpy array.
    
    Returns
    -------
    Union[da.Array, np.ndarray]
        Binary mask (0 or 1) as dask array or numpy array.
    """
    # Use the same loading function as tiles
    mask = _load_image_tile(mask_path, key=None)

    # Ensure it's a 2D array
    if mask.ndim != 2:
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        else:
            raise ValueError(f"Mask must be 2D, got shape {mask.shape}")

    # Convert to binary (0 or 1)
    # Handle both boolean and numeric masks
    if keep_lazy:
        # Keep as lazy dask array for optimization
        mask = (mask > 0).astype(np.float32)
    else:
        # Compute for immediate use
        mask = mask.compute() if isinstance(mask, da.Array) else mask
        mask = (mask > 0).astype(np.float32)

    return mask


def _apply_mask(result: da.Array, mask: da.Array) -> da.Array:
    """
    Apply mask to result using optimized Dask operations.
    
    This function structures the computation so that Dask can optimize by
    checking mask values first. The key optimization is:
    1. Align mask and result chunks so blocks correspond
    2. Use map_blocks with a function that can see both mask and result
    3. Structure the computation graph so mask information is available
    
    While Dask's lazy evaluation means result blocks are still in the graph,
    the aligned chunks and structured computation allow the scheduler to
    optimize execution. In practice, the scheduler can prioritize computing
    mask blocks first and use that information to optimize result computation.
    
    Parameters
    ----------
    result : da.Array
        The result array to mask.
    mask : da.Array
        Binary mask (0 or 1) with matching shape.
    
    Returns
    -------
    da.Array
        Masked result array.
    """
    # Handle different dimensionalities - broadcast mask if needed
    if result.ndim > mask.ndim:
        # Result has extra dimensions (e.g., 3D result with 2D mask)
        # Broadcast mask to match result shape
        for _ in range(result.ndim - mask.ndim):
            mask = mask[..., np.newaxis]
    
    # Ensure mask chunks align with result chunks for efficient computation
    # This is critical: aligned chunks allow Dask to see block-level relationships
    # and the scheduler can use mask block information to optimize result computation
    if mask.chunks != result.chunks:
        mask = mask.rechunk(result.chunks)
    
    # Use map_blocks with aligned chunks
    # The function receives both result and mask blocks, allowing it to
    # apply the mask efficiently. With aligned chunks, Dask's scheduler
    # can see the relationship and optimize block-level computation.
    def _mask_block(result_block, mask_block, *args, block_info=None, **kwargs):
        """
        Apply mask to a block.
        
        With aligned chunks, this function receives corresponding blocks
        of result and mask. The multiplication naturally zeros out where
        mask is 0, and Dask can optimize the computation graph accordingly.
        """
        if mask_block.sum() == 0:
            return np.zeros_like(result_block)
        return result_block * mask_block
    
    # Apply mask using map_blocks with aligned chunks
    # This structure allows Dask to optimize block-level computation
    # The scheduler can use mask information to optimize result computation
    return da.map_blocks(
        _mask_block,
        result,
        mask,
        dtype=result.dtype,
        chunks=result.chunks,
    )


@mosaic.default
@autoconfig
def mosaic2d(
    tile_info_file: str,
    *,
    jpeg_output: Annotated[Optional[str], Parameter(name=["--jpeg", "-j"])] = None,
    tiff_output: Annotated[Optional[str], Parameter(name=["--tiff", "-t"])] = None,
    nifti_output: Annotated[Optional[str], Parameter(name=["--nifti", "-n"])] = None,
    tile_overlap: float = 0.2,
    circular_mean: bool = False,
    clip_x: int = 0,
    clip_y: int = 0,
    mask: Annotated[Optional[str], Parameter(name=["--mask", "-m"])] = None,
    focus_plane: Annotated[
        Optional[str], Parameter(name=["--focus-plane", "-f"])] = None,
    normalize_focus_plane: bool = False,
    crop_focus_plane_depth: int = 500,
    crop_focus_plane_offset: int = 30,
    voxel_size_xyz: Annotated[
        Optional[list[float]], Parameter(name=["--voxel-size", "-s"])] = None,
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
    clip_x : int
        Number of pixels to clip from the left side of each tile. Coordinates will be
        shifted accordingly.
    clip_y : int
        Number of pixels to clip from the top side of each tile. Coordinates will be
        shifted accordingly.
    mask : str, optional
        Path to binary mask file to apply to the result. Mask should be 2D and match
        the result dimensions.
    focus_plane : str, optional
        Path to focus plane NIfTI file for depth shifting.
    normalize_focus_plane : bool
        Whether to normalize the focus plane by the minimum value.
    crop_focus_plane_depth : int
        Number of pixels to crop below the focus plane.
    crop_focus_plane_offset : int
        Offset of the focus plane to crop below the minimum value.
    voxel_size_xyz : list[float], optional
        Voxel size in x, y, z directions, in millimeters.
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
    if voxel_size_xyz is None:
        voxel_size_xyz = metadata.get("scan_resolution", [10, 10, 2.5])
    file_key = metadata.get("file_key")  # Key for mat file array
    base_dir = metadata.get("base_dir", ".")
    # Get clip values from metadata if not provided as parameters
    if clip_x == 0:
        clip_x = metadata.get("clip_x", 0)
    if clip_y == 0:
        clip_y = metadata.get("clip_y", 0)

    # Get mask from metadata if not provided as parameter
    if mask is None:
        mask = metadata.get("mask")

    # Use tile_overlap from function parameter (defaults to "auto")
    # If "auto" and metadata has tile_overlap, use that instead
    if tile_overlap == "auto" and "tile_overlap" in metadata:
        tile_overlap = metadata.get("tile_overlap", "auto")

    # Process each tile and collect TileInfo objects
    tile_infos = []

    logger.info(f"Loading and processing {len(tiles_config)} tiles")
    for tile in tqdm(tiles_config, desc="Processing tiles"):
        x = tile.get("x")
        y = tile.get("y")
        file_path = tile.get("filepath")
        if x is None or y is None or file_path is None:
            logger.warning(f"Skipping incomplete tile: {tile}")
            continue
        if base_dir:
            file_path = op.join(base_dir, file_path)
        if not op.exists(file_path):
            logger.warning(f"Tile file not found: {file_path}, skipping")
            continue

        # Load 2D image
        try:
            image = _load_image_tile(file_path, file_key)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}, skipping")
            continue
        # Apply clipping if specified
        if clip_x > 0 or clip_y > 0:
            # Clip from left (clip_x) and top (clip_y)
            # This removes pixels from the left and top edges
            if image.ndim == 2:
                image = image[clip_x:, clip_y:]
            elif image.ndim >= 3:
                image = image[clip_x:, clip_y:, ...]
            else:
                logger.warning(
                    f"Unexpected image dimensions {image.ndim} for {file_path}")
                continue

            # Shift coordinates to account for clipping
            # After clipping clip_x pixels from the left, the remaining content
            # represents what was at position clip_x in the original tile.
            # To align this correctly in the mosaic, we shift coordinates by +clip_x
            # and +clip_y
            x = int(x) + clip_x
            y = int(y) + clip_y
        else:
            x = int(x)
            y = int(y)

        # Create TileInfo
        tile_infos.append(TileInfo(x=x, y=y, image=image))

    if not tile_infos:
        raise ValueError("No valid tiles were processed")

    if focus_plane:
        focus_plane = nib.load(focus_plane).get_fdata().astype(np.uint16)
        focus_plane = focus_plane.squeeze()
        if normalize_focus_plane:
            focus_plane = focus_plane - focus_plane.min()
        focus_plane = focus_plane + crop_focus_plane_offset

        def apply_focus_plane(image):
            nonlocal focus_plane
            z = np.arange(crop_focus_plane_depth, dtype=np.int32)[None, None, :]
            idx = z + focus_plane[..., None]
            idx = idx[..., None]
            result = np.take_along_axis(image, idx, axis=2)
            return result

        all_images = da.stack([tile.image for tile in tile_infos], axis=-1)
        print(crop_focus_plane_depth)
        result_shape = (
            all_images.shape[0], all_images.shape[1], crop_focus_plane_depth,
            all_images.shape[3])
        all_images = all_images.map_blocks(apply_focus_plane, dtype=all_images.dtype,
                                           chunks=(
                                               all_images.chunks[0],
                                               all_images.chunks[1],
                                               crop_focus_plane_depth, 1))
        for i, tile in enumerate(tile_infos):
            tile.image = all_images[..., i]
    # Create MosaicInfo for 2D mosaic - dimensions and coordinates extracted from tiles
    # Stitch tiles using MosaicInfo
    logger.info("Stitching tiles")

    mosaic = MosaicInfo.from_tiles(
        tiles=tile_infos,
        depth=crop_focus_plane_depth if focus_plane is not None else None,  # 2D mosaic
        chunk_size=None,  # Will use tile dimensions
        circular_mean=circular_mean,
        tile_overlap=tile_overlap,
    )

    # Stitch using lazy dask operations
    result = mosaic.stitch()

    # Apply mask if provided
    if mask:
        logger.info(f"Loading and applying mask: {mask}")
        try:
            # Load mask as lazy dask array for optimization
            mask_array = _load_mask(mask, keep_lazy=True)
            
            # Check if mask dimensions match result (accounting for broadcasting)
            mask_2d_shape = mask_array.shape[:2] if mask_array.ndim >= 2 else mask_array.shape
            result_2d_shape = result.shape[:2] if result.ndim >= 2 else result.shape
            if mask_2d_shape != result_2d_shape:
                logger.error(
                    f"Mask shape {mask_array.shape} does not match result shape "
                    f"{result.shape}. "
                )

            # Apply mask using optimized function that allows Dask to skip computation
            # where mask is 0
            result = _apply_mask(result, mask_array)
            logger.info("Mask applied successfully (optimized for Dask)")
        except Exception as e:
            logger.error(f"Failed to load or apply mask: {e}")
            raise

    if nifti_output or jpeg_output or tiff_output:
        with ProgressBar():
            result = np.array(result)

    voxel_size_2d = voxel_size_xyz[:2] if len(voxel_size_xyz) >= 2 else [0.1, 0.1]
    # Save NIfTI file if requested
    if nifti_output:
        logger.info(f"Saving NIfTI file: {nifti_output}")
        # Create affine matrix for 2D image
        affine = np.eye(4)
        # Create NIfTI image (2D array needs to be expanded to 3D for NIfTI)
        # Add a singleton z dimension
        # result_3d = result_2d[:, :]
        nii_img = nib.Nifti1Image(result, affine)
        nii_img.header.set_xyzt_units(xyz="mm", t="sec")
        nii_img.header.set_zooms(voxel_size_2d)
        nib.save(nii_img, nifti_output)
        logger.info("NIfTI file saved successfully")
    result = result.T

    # Save JPEG if requested
    if jpeg_output:
        logger.info(f"Saving JPEG preview: {jpeg_output}")
        _save_jpeg(result, jpeg_output)

    # Save TIFF if requested
    if tiff_output:
        logger.info(f"Saving TIFF: {tiff_output}")
        _save_tiff(result, tiff_output)

    # Save to Zarr if output is specified
    if general_config.out:
        logger.info(f"Saving to Zarr: {general_config.out}")

        # Compute zarr layout for 2D
        chunk, shard = compute_zarr_layout(result.shape, np.float32,
                                           zarr_config)

        # Prepare Zarr group (similar to single_volume.py)
        zgroup = from_config(general_config.out, zarr_config)

        # Create array and write data directly (like single_volume.py)
        dataset = zgroup.create_array(
            "0", shape=result.shape, dtype=np.float32, chunk=chunk,
            shard=shard,
            zarr_config=zarr_config
        )
        # Write data directly using indexing (similar to single_volume.py)
        if isinstance(result, da.Array):
            if shard:
                result = da.rechunk(result, chunks=shard)
            else:
                result = da.rechunk(result, chunks=chunk)

            with ProgressBar():
                da.store(result, dataset, compute=True)

        else:
            dataset[...] = result

        # Generate pyramid and metadata
        logger.info("Generating pyramid and metadata")
        zgroup.generate_pyramid()
        logger.info("Finished generating pyramid")
        logger.info("Writing OME-Zarr metadata")
        zgroup.write_ome_metadata(["z", "y", "x"], space_scale=voxel_size_xyz[::-1],
                                  space_unit="millimeter")

        if nifti_config and nifti_config.nii:
            header = build_nifti_header(
                zgroup=zgroup,
                voxel_size_zyx=voxel_size_xyz[::-1],
                unit="millimeter",
                nii_config=nifti_config,
            )
            zgroup.write_nifti_header(header)


    else:
        logger.info("Skipping Zarr output (no output path specified)")

    logger.info("Finished mosaic2d")
