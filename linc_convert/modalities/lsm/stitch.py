"""
Convert ome-zarr files set into a stitched together zarr file.

Example input files can be found at
https://lincbrain.org/dandiset/000056/draft/files?location=derivatives%2Fcompressed_camera1&page=1
"""

import logging
from typing import Optional

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
    overlap: int = 192,
    voxel_size: list[float] = (1, 1, 1),
    general_config: GeneralConfig = None,
    zarr_config: ZarrConfig = None,
    nii_config: NiftiConfig = None,
    use_runs: bool = True,
    dandiset_id: Optional[str] = None,
    max_x: Optional[int] = None,
    allow_padding: bool = False,
    number_workers: Optional[int] = None,
    threads_per_worker: int = 1,
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
        dandiset_id that contains the ome.zarr files for inp 
        (leave none if inp is local)
    max_x
        value to crop all x shapes to
    """
    convert_spool_or_zarr(inp, overlap=overlap,
                          voxel_size=voxel_size,
                          general_config=general_config,
                          zarr_config=zarr_config,
                          nii_config=nii_config,
                          use_runs=use_runs,
                          dandiset_id=dandiset_id,
                          max_x=max_x,
                          allow_padding=allow_padding,
                          number_workers=number_workers,
                          threads_per_worker=threads_per_worker)
