"""
Convert ome-zarr files set into a stitched together zarr file.

Example input files can be found at
https://lincbrain.org/dandiset/000056/draft/files?location=derivatives%2Fcompressed_camera1&page=1
"""

import logging
import os
from typing import Optional

# externals
import cyclopts
import dask.array as da
import tifffile as tiff

# internals
from linc_convert.modalities.lsm.cli import lsm
from linc_convert.modalities.lsm.convert_spool_or_zarr import (
    discover_tile_paths,
    open_tile_reader,
    prompt_dandi_api_key,
)
from linc_convert.utils.zarr_config import (
    GeneralConfig,
    autoconfig,
)

logger = logging.getLogger(__name__)
mip = cyclopts.App(name="mip", help_format="markdown")
lsm.command(mip)


@mip.default
@autoconfig
def convert(
    inp: str,
    *,
    general_config: GeneralConfig = None,
    dandiset_id: Optional[str] = None,
    z_start: Optional[int] = None,
    z_end: Optional[int] = None,
) -> None:
    ...
    api_key = prompt_dandi_api_key() if dandiset_id else None
    tile_paths = discover_tile_paths(
        inp, dandiset_id=dandiset_id, api_key=api_key)

    for path in tile_paths:
        name = os.path.basename(path.rstrip("/").replace(".ome.zarr", ""))

        reader = open_tile_reader(
            path,
            dandiset_id=dandiset_id,
            api_key=api_key,
        )
        if z_start is not None:
            reader = reader[z_start:, :, :]
        if z_end is not None:
            reader = reader[:z_end, :, :]

        tiff.imwrite(f"{general_config.out}/{name}.tiff",
                     da.max(reader, axis=0).compute())
