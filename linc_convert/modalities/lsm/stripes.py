"""Generate stripe .tff files from ome zarr and MIP projection."""

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


def _compute_mask(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    hi = np.percentile(img, 99.9)
    img_c = np.minimum(img, hi)

    try:
        thr = threshold_otsu(img_c)
    except Exception:
        thr = np.percentile(img_c, 70)

    return img_c > thr


@stripes.default
@autoconfig
def create(
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
    """
    Generate ZY projection TIFF images from volumetric tile data.

    This function discovers tile datasets from a local or DANDI source, 
    loads each volume, optionally crops it along the Z and Y axes, computes a mask 
    from a corresponding YX MIP image, and applies this mask to the volume before 
    generating a ZY projection using a NaN-aware median. The result is written to disk 
    as a TIFF file.

    Parameters
    ----------
    inp : str
        Input path specifying where to discover tile datasets. Can refer to a local path
        or a remote dataset when `dandiset_id` is provided.
    mip_dir : str
        Directory containing precomputed YX MIP TIFF images used for masking. Each tile
        must have a corresponding file named ``{tile_name}_proc-mip.tiff``.
    general_config : GeneralConfig, optional
        Configuration object containing output settings. Must include an ``out`` attribute
        specifying the output directory.
    dandiset_id : Optional[str], default None
        If provided, tiles are loaded from the specified DANDI dataset using an API key.
    z_start : Optional[int], default None
        Starting index for cropping along the Z dimension (inclusive).
    z_end : Optional[int], default None
        Ending index for cropping along the Z dimension (exclusive).
    y_start : Optional[int], default None
        Starting index for cropping along the Y dimension (inclusive).
    y_end : Optional[int], default None
        Ending index for cropping along the Y dimension (exclusive).

    Returns
    -------
    None
        This function writes output files to disk and does not return a value.

    Raises
    ------
    FileNotFoundError
        If the corresponding YX MIP image for a tile cannot be found.

    Notes
    -----
    - The mask is computed from the YX MIP image using `_compute_mask` and is applied
      across all Z slices.
    - Masked-out regions are set to NaN before computing the median projection.
    - NaN values are replaced with a large sentinel value (9999999.0).
    - Output files are written atomically using a temporary file and then renamed.
    """
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
            yx_path = os.path.join(
                mip_dir, f"{name}_proc-mip.tiff").replace("slice0", "slice")

            if not os.path.exists(yx_path):
                raise FileNotFoundError(f"Missing YX image: {yx_path}")

            img_yx = tiff.imread(yx_path).astype(np.float32)

            if y_end is not None:
                img_yx = img_yx[:y_end, :]
            if y_start is not None:
                img_yx = img_yx[y_start:, :]

            mask = _compute_mask(img_yx)

            vol_np = reader.compute()
            vol_np = vol_np.astype(float)
            vol_np[:, ~mask] = np.nan
            corr_zy = np.nanmedian(vol_np, axis=2)
            corr_zy = np.nan_to_num(corr_zy, nan=9999999.0)

            tiff.imwrite(output_name + ".tmp", corr_zy)
            os.replace(output_name + ".tmp", output_name)

        print("--- %s secs ---" % (time.time() - start_time))
