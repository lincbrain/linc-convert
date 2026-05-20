"""
Convert ome-zarr files set into a stitched together zarr file.

Example input files can be found at
https://lincbrain.org/dandiset/000056/draft/files?location=derivatives%2Fcompressed_camera1&page=1
"""

import logging
from typing import Literal, Optional, Union

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
    voxel_size: list[float] = (1, 1, 1),
    general_config: GeneralConfig = None,
    zarr_config: ZarrConfig = None,
    nii_config: NiftiConfig = None,
    use_runs: bool = True,
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
    skew_angle: float = 0,
    chunks_processed: int = 0,
    blend: bool = False,
    stripes: Optional[str] = None,
    white_matter_intensity: float = 1000.0,
    skip_first_layer: bool = False,
    background_threshold: Optional[Union[float, Literal["auto"]]] = None,
    checkpoint_file: Optional[str] = None,
    alternate_pattern: False,
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
    convert_spool_or_zarr(inp, overlap=overlap,
                          voxel_size=voxel_size,
                          general_config=general_config,
                          zarr_config=zarr_config,
                          nii_config=nii_config,
                          use_runs=use_runs,
                          dandiset_id=dandiset_id,
                          x_chunk_start=x_chunk_start,
                          x_chunk_end=x_chunk_end,
                          x_end=x_end,
                          z_start=z_start,
                          z_end=z_end,
                          y_start=y_start,
                          y_end=y_end,
                          allow_padding=allow_padding,
                          number_workers=number_workers,
                          threads_per_worker=threads_per_worker,
                          skew_angle=skew_angle,
                          chunks_processed=chunks_processed,
                          blend=blend,
                          stripes=stripes,
                          white_matter_intensity=white_matter_intensity,
                          skip_first_layer=skip_first_layer,
                          background_threshold=background_threshold,
                          checkpoint_file=checkpoint_file,
                          alternate_pattern=alternate_pattern
                          )
