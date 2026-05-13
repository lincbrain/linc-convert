"""
Convert ome-zarr files set into a stitched together zarr file.

Example input files can be found at
https://lincbrain.org/dandiset/000056/draft/files?location=derivatives%2Fcompressed_camera1&page=1
"""

import logging
import os
import time
from typing import Optional, Tuple

# externals
import cyclopts
import dask.array as da
import numpy as np
import tifffile as tiff

# internals
from linc_convert.modalities.lsm.cli import lsm
from linc_convert.modalities.lsm.convert_spool_or_zarr import (
    discover_tile_paths,
    open_tile_reader,
    prompt_dandi_api_key,
)
from linc_convert.utils.io.zarr.drivers.zarr_python import ZarrPythonGroup
from linc_convert.utils.zarr_config import (
    GeneralConfig,
    ZarrConfig,
    autoconfig,
)

logger = logging.getLogger(__name__)
pyramid = cyclopts.App(name="pyramid", help_format="markdown")
lsm.command(pyramid)


@pyramid.default
@autoconfig
def convert(
    inp: str,
    *,
    start_level: int,
    end_level: int,
    voxel_size: Tuple[float, ...],
    zarr_config: ZarrConfig = None
) -> None:
    ...
    voxel_size = list(map(float, reversed(voxel_size)))
    omz = ZarrPythonGroup.from_config(inp, zarr_config)

    omz.generate_pyramid(end_level, level_start=start_level)

    omz.write_ome_metadata(
        axes=["z", "y", "x"],
        space_scale=voxel_size,
    )
