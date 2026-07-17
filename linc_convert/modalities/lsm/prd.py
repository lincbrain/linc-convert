"""Convert Kinetix .prd image(s) to OME-Zarr/NIfTI-Zarr.

A .prd file is a stack of uint16 frames with a fixed header
and inter-frame gap, and padding at the end. The input is a directory of
`ss_stack_*.prd` files, which are concatenated along the acquisition order.

Example input files can be found at
https://dandiarchive.org/dandiset/001769/draft/files?location=rawdata/HG9_Z1_Y1
"""

import logging
import os

import cyclopts
import dask.array as da

from linc_convert.modalities.lsm.cli import lsm
from linc_convert.utils.io.prd import PrdSetInterpreter
from linc_convert.utils.io.zarr import from_config
from linc_convert.utils.nifti_header import build_nifti_header
from linc_convert.utils.zarr_config import (
    GeneralConfig,
    NiftiConfig,
    ZarrConfig,
    autoconfig,
)

logger = logging.getLogger(__name__)
prd = cyclopts.App(name="prd", help_format="markdown")
lsm.command(prd)


@prd.default
@autoconfig
def convert(
    inp: str,
    *,
    voxel_size: list[float] = (1, 1, 1),
    general_config: GeneralConfig | None = None,
    zarr_config: ZarrConfig | None = None,
    nii_config: NiftiConfig | None = None,
) -> None:
    """
    Convert Kinetix .prd file(s) into a pyramidal OME-Zarr or NIfTI-Zarr.

    Parameters
    ----------
    inp
        Path to a directory of ss_stack_*.prd files.
    voxel_size
        Voxel size along the X, Y and Z dimensions, in microns.
    general_config
        General configuration
    zarr_config
        Zarr related configuration
    nii_config
        NIfTI header related configuration
    """
    reader = PrdSetInterpreter(inp)

    # Set default output path if not provided
    general_config.set_default_name(os.path.basename(inp))

    volume = reader.assemble()

    # Initialize Zarr group and array
    zgroup = from_config(general_config.out, zarr_config)
    array = zgroup.create_array(
        "0", shape=volume.shape, zarr_config=zarr_config, dtype=reader.dtype
    )
    logger.info(general_config.out)

    if array.shards:
        volume = da.rechunk(volume, array.shards)
    else:
        volume = da.rechunk(volume, array.chunks)

    logger.info(
        "Write level 0 for %d file(s) with shape: "
        "%d total_frames, %d height, %d width.",
        len(reader.prd_files),
        *volume.shape,
    )
    da.store(volume, array, compute=True)

    voxel_size = list(map(float, reversed(voxel_size)))

    # Generate Zarr pyramid and metadata.
    zgroup.generate_pyramid(levels=zarr_config.levels)
    logger.info("Write OME-Zarr multiscale metadata")
    zgroup.write_ome_metadata(axes=["z", "y", "x"], space_scale=voxel_size)

    if nii_config.nii:
        header = build_nifti_header(
            zgroup=zgroup,
            voxel_size_zyx=tuple(voxel_size),
            unit="micrometer",
            nii_config=nii_config,
        )
        zgroup.write_nifti_header(header)

    logger.info("Conversion complete.")
