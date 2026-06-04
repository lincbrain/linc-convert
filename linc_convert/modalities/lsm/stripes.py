import logging
import os
import time
from typing import Optional

import cyclopts
import numpy as np
import tifffile as tiff
from dask.diagnostics import ProgressBar
from pystripe import filter_streaks
from skimage.filters import threshold_otsu
from tqdm import tqdm
import dask.array as da

from linc_convert.modalities.lsm.cli import lsm
from linc_convert.modalities.lsm.convert_spool_or_zarr import (
    discover_tile_paths,
    open_tile_reader,
    prompt_dandi_api_key,
)
from linc_convert.utils.io.zarr.drivers.zarr_python import ZarrPythonGroup
from linc_convert.utils.zarr_config import GeneralConfig, ZarrConfig, autoconfig

logger = logging.getLogger(__name__)
stripes = cyclopts.App(name="stripes", help_format="markdown")
lsm.command(stripes)


@stripes.default
@autoconfig
def destripe_dask_pystripe_chunked(
    inp: str,
    *,
    mip_dir: str,
    general_config: GeneralConfig = None,
    sigma=(10, 40),
    level=0,
    wavelet="db3",
    crossover=10,
    dandiset_id: Optional[str] = None,
    z_start: Optional[int] = None,
    z_end: Optional[int] = None,
    y_start: Optional[int] = None,
    y_end: Optional[int] = None,
    zarr_config: ZarrConfig = None,
):
    """
    Apply PyStripe destriping to a Dask volume using Z-chunks.

    Parameters
    ----------
    dask_vol : dask.array (Z, Y, X)
    output_path : str (zarr store)
    """

    api_key = prompt_dandi_api_key() if dandiset_id else None

    tile_paths = discover_tile_paths(
        inp, dandiset_id=dandiset_id, api_key=api_key)

    for index, path in enumerate(tile_paths):
        logger.info(path)
        logger.info(index)

        start_time = time.time()
        name = os.path.basename(path.rstrip("/").replace(".ome.zarr", ""))
        output_name = f"{general_config.out}/{name}.ome.zarr"

        if not os.path.exists(output_name):

            dask_vol = open_tile_reader(
                path,
                dandiset_id=dandiset_id,
                api_key=api_key,
            )
            yx_path = os.path.join(
                mip_dir, f"{name}_proc-mip.tiff").replace("slice0", "slice")

            if not os.path.exists(yx_path):
                raise FileNotFoundError(f"Missing YX image: {yx_path}")

            img_yx = tiff.imread(yx_path).astype(np.float32)

            if y_end is not None:
                img_yx = img_yx[:y_end, :]
            if y_start is not None:
                img_yx = img_yx[y_start:, :]
            thr = threshold_otsu(img_yx)

            if z_end is not None:
                dask_vol = dask_vol[:z_end, :, :]
            if z_start is not None:
                dask_vol = dask_vol[z_start:, :, :]
            if y_end is not None:
                dask_vol = dask_vol[:, :y_end, :]
            if y_start is not None:
                dask_vol = dask_vol[:, y_start:, :]

            omz = ZarrPythonGroup.from_config(output_name+".tmp", zarr_config)
            out = omz.create_array("0", shape=dask_vol.shape,
                                   zarr_config=zarr_config, dtype=np.uint16)

            z_chunks = dask_vol.chunks[0]

            z_0 = 0

            for chunk_idx, z_len in enumerate(z_chunks):

                z_1 = z_0 + z_len

                print(f"\nProcessing chunk {chunk_idx}: Z[{z_0}:{z_1}]")

                # ✅ compute ONCE for this chunk
                chunk = dask_vol[z_0:z_1].compute()

                # ✅ process all slices in this chunk
                out_chunk = np.empty_like(chunk, dtype=np.uint16)

                for i in tqdm(range(z_len), desc=f"Chunk {chunk_idx}"):
                    img = chunk[i]

                    destriped = filter_streaks(
                        img,
                        sigma=list(sigma),
                        level=level,
                        wavelet=wavelet,
                        crossover=crossover,
                        threshold=thr,
                    )

                out_chunk[i] = destriped
                slicer = (
                    slice(z_0, z_1)
                )

                # ✅ write once
                # with ProgressBar():
                #    np.to_zarr(out_chunk, out._array,
                #               region=slicer)
                out._array[z_0:z_1] = out_chunk

                z_0 = z_1
            omz.generate_pyramid(levels=zarr_config.levels,
                                 copy_config=general_config,
                                 copy_zarr_config=zarr_config)
            os.replace(output_name + ".tmp", output_name)
            print("--- %s secs ---" % (time.time() - start_time))
