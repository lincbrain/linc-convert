"""Converts TIFF to OME-Zarr or NIfTI-Zarr."""

import logging
import os

import cyclopts
import dask.array as da
import dask_image.imread
import numpy as np

from linc_convert.modalities.lsm.cli import lsm
from linc_convert.utils.io.zarr import from_config
from linc_convert.utils.nifti_header import build_nifti_header
from linc_convert.utils.unit import to_ome_unit
from linc_convert.utils.zarr_config import (
    GeneralConfig,
    NiftiConfig,
    ZarrConfig,
    autoconfig,
)

logger = logging.getLogger(__name__)
single_volume = cyclopts.App(name="single_volume", help_format="markdown")
lsm.command(single_volume)


@single_volume.default
@autoconfig
def convert(
    inp: str,
    *,
    voxel_size: list[float] = (1, 1, 1),
    general_config: GeneralConfig = None,
    zarr_config: ZarrConfig = None,
    nii_config: NiftiConfig = None,
) -> None:
    """
    Tiff to OME-Zarr.

    Convert tiff files
    into a pyramidal OME-ZARR (or NIfTI-Zarr) hierarchy.

    Parameters
    ----------
    inp
        Path to the input tiff file
    voxel_size
        Voxel size along the X, Y and Z dimensions, in microns.
    general_config
        General configuration
    zarr_config
        Zarr related configuration
    nii_config
        NIfTI header related configuration
    """
    general_config.set_default_name(os.path.splitext(inp)[0])

    inp_data = dask_image.imread.imread(inp)

    # Prepare Zarr group
    zgroup = from_config(general_config.out, zarr_config)

    if not hasattr(inp_data, "dtype"):
        raise Exception("Input is not a numpy array. This is unexpected.")

    dataset = zgroup.create_array(
        "0",
        shape=inp_data.shape,
        dtype=np.dtype(inp_data.dtype),
        zarr_config=zarr_config,
    )

    if dataset.shards:
        inp_data = da.rechunk(inp_data, dataset.shards)
    else:
        inp_data = da.rechunk(inp_data, dataset.chunks)

    da.store(inp_data, dataset)
    voxel_size = list(map(float, reversed(voxel_size)))
    # Generate Zarr pyramid and metadata
    zgroup.generate_pyramid(mode="mean", no_pyramid_axis=zarr_config.no_pyramid_axis)
    logger.info("Write OME-Zarr multiscale metadata")
    zgroup.write_ome_metadata(axes=["z", "y", "x"], space_unit=to_ome_unit("um"))

    if nii_config.nii:
        header = build_nifti_header(
            zgroup=zgroup,
            voxel_size_zyx=tuple(voxel_size),
            unit="micrometer",
            nii_config=nii_config,
        )
        zgroup.write_nifti_header(header)

    logger.info("Conversion complete.")
