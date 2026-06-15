"""
Convert ome-zarr files set into a stitched together zarr file.

Example input files can be found at
https://lincbrain.org/dandiset/000056/draft/files?location=derivatives%2Fcompressed_camera1&page=1
"""

import logging
from typing import List, Literal, Optional, Union

# externals
import cyclopts

# internals
from linc_convert.modalities.lsm.cli import lsm
from linc_convert.modalities.lsm.convert_spool_or_zarr import convert_spool_or_zarr
from linc_convert.utils.zarr_config import (
    GeneralConfig,
    NiftiConfig,
    ZarrConfig,
    autoconfig,
)

logger = logging.getLogger(__name__)
stitch = cyclopts.App(name="stitch", help_format="markdown")
lsm.command(stitch)


@stitch.default
@autoconfig
def convert(
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
        Path to the root directory which contains a collection of
        ome.zarr files named `*_y{:02d}_z{:02d}*_HR.ome.zarr`. _z{:02d} 
        is optional
    overlap
        Number of pixels between slices that are overlapped
    delta_x
        The amount of displacemnt in the x direction each chunk has relative
        to the prior chunk.
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
    x_chunk_start
        The lowest x value chunk to output (inclusive)
    x_chunk_end
        The highest x value chunk to output (exclusive)
    x_end
        Max x values to crop or pad all tiles to
    z_start
        The minimum z value that should be read in each tile
    z_end
        The maximum z value that should be read in each tile
    y_start
        The minimum y value that should be read in each tile
    y_end
        The maximum y value that should be read in each tile
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
    checkpoint_file
        path to a file that can be used to store checkpoitns
    alternate_pattern
        use the alternate naming pattern for tiles instead of the the usual one
    flip_z
        flip the z axis
    """
    convert_spool_or_zarr(inp, overlap=overlap,
                          delta_x=delta_x,
                          voxel_size=voxel_size,
                          general_config=general_config,
                          zarr_config=zarr_config,
                          nii_config=nii_config,
                          dandiset_id=dandiset_id,
                          x_chunk_start=x_chunk_start,
                          x_chunk_end=x_chunk_end,
                          chunks_processed=chunks_processed,
                          )
