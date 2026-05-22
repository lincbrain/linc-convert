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
import numpy as np
import tifffile as tiff
from skimage.filters import threshold_otsu

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
    mip_dir: str,
    *,
    general_config: GeneralConfig = None,
    dandiset_id: Optional[str] = None,
    z_start: Optional[int] = None,
    z_end: Optional[int] = None,
    y_start: Optional[int] = None,
    y_end: Optional[int] = None,
) -> None:

    # -----------------------------
    # helpers (from your stripe code, simplified)
    # -----------------------------
    def compute_mask(img):
        img = img.astype(np.float32)
        hi = np.percentile(img, 99.9)
        img_c = np.minimum(img, hi)

        try:
            thr = threshold_otsu(img_c)
        except Exception:
            thr = np.percentile(img_c, 70)

        return img_c > thr

    # -----------------------------
    # main logic
    # -----------------------------
    api_key = prompt_dandi_api_key() if dandiset_id else None

    tile_paths = discover_tile_paths(
        inp, dandiset_id=dandiset_id, api_key=api_key)

    for index, path in enumerate(tile_paths):
        logger.info(path)
        logger.info(index)

        start_time = time.time()
        name = os.path.basename(path.rstrip("/").replace(".ome.zarr", ""))

        reader = open_tile_reader(
            path,
            dandiset_id=dandiset_id,
            api_key=api_key,
        )

        # optional cropping
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
            yx_path = os.path.join(mip_dir, f"{name}_proc-mip.tiff")

            if not os.path.exists(yx_path):
                raise FileNotFoundError(f"Missing YX image: {yx_path}")

            img_yx = tiff.imread(yx_path).astype(np.float32)

            if z_end is not None:
                img_yx = img_yx[:z_end, :, :]
            if z_start is not None:
                img_yx = img_yx[z_start:, :, :]
            if y_end is not None:
                img_yx = img_yx[:, :y_end, :]
            if y_start is not None:
                img_yx = img_yx[:, y_start:, :]

            # -----------------------------
            # 2) Compute mask + corr_y
            # -----------------------------
            mask = compute_mask(img_yx)

            # -----------------------------
            # 3) Build (Z,Y) correction image
            # -----------------------------
            vol_np = reader.compute()
            vol_np[~mask] = np.nan
            corr_zy = np.nanmedian(vol_np, axis=2)
            corr_zy = np.nan_to_num(corr_zy, nan=9999999.0)

            # -----------------------------
            # 4) Save
            # -----------------------------
            tiff.imwrite(output_name + ".tmp", corr_zy)
            os.replace(output_name + ".tmp", output_name)

        print("--- %s secs ---" % (time.time() - start_time))
