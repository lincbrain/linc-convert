"""
Convert ome-zarr files set into a stitched together zarr file.

Example input files can be found at
https://lincbrain.org/dandiset/000056/draft/files?location=derivatives%2Fcompressed_camera1&page=1
"""

import logging
import os
import time
from typing import Optional

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
from linc_convert.utils.zarr_config import (
    GeneralConfig,
    autoconfig,
)

logger = logging.getLogger(__name__)
stripes = cyclopts.App(name="stripes", help_format="markdown")
lsm.command(stripes)


@stripes.default
@autoconfig
def convert(
    inp: str,
    *,
    general_config: GeneralConfig = None,
    dandiset_id: Optional[str] = None,
    z_start: Optional[int] = None,
    z_end: Optional[int] = None,
    y_start: Optional[int] = None,
    y_end: Optional[int] = None,
) -> None:
    ...
    api_key = prompt_dandi_api_key() if dandiset_id else None

    tile_paths = discover_tile_paths(
        inp, dandiset_id=dandiset_id, api_key=api_key)
    index = 0
    for path in tile_paths:
        logger.info(path)
        logger.info(index)
        index += 1
        start_time = time.time()
        name = os.path.basename(path.rstrip("/").replace(".ome.zarr", ""))

        reader = open_tile_reader(
            path,
            dandiset_id=dandiset_id,
            api_key=api_key,
        )
        if z_end is not None:
            reader = reader[:z_end, :, :]
        if z_start is not None:
            reader = reader[z_start:, :, :]
        if y_end is not None:
            reader = reader[:, :y_end, :]
        if y_start is not None:
            reader = reader[:, y_start:, :]
        output_name = f"{general_config.out}/{name}.tiff"
        if not os.path.exists(output_name):
            reader = da.where(reader >= 130, reader, da.nan)
            np_reader = da.nanmedian(reader, axis=2).compute()
            np_reader = np.nan_to_num(np_reader, nan=999999)

            tiff.imwrite(output_name + ".tmp",
                         np_reader)
            os.replace(output_name + ".tmp", output_name)
        print("--- %s secs ---" % (time.time() - start_time))
