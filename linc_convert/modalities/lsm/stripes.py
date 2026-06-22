"""Generate stripe .tff files from ome zarr and MIP projection."""

import logging
import os
import time
from typing import Optional

# externals
import cyclopts
import dask.array as da
import numpy as np
from dask.diagnostics import ProgressBar

# internals
from linc_convert.modalities.lsm.cli import lsm
from linc_convert.modalities.lsm.convert_spool_or_zarr import (
    discover_tile_paths,
    open_tile_reader,
    prompt_dandi_api_key,
)
from linc_convert.modalities.lsm.preprocessing_utils.corrections import apply_affine
from linc_convert.modalities.lsm.preprocessing_utils.io import (
    camera_channel_map,
    get_camera_info,
    load_mask_and_thr,
    load_scan_parameters,
)
from linc_convert.modalities.lsm.preprocessing_utils.preprocessing import (
    crop_volume_channels,
    preprocess,
)
from linc_convert.modalities.lsm.preprocessing_utils.registration import get_all_affines
from linc_convert.utils.io.zarr.drivers.zarr_python import ZarrPythonGroup
from linc_convert.utils.zarr_config import (
    GeneralConfig,
    ZarrConfig,
    autoconfig,
)

logger = logging.getLogger(__name__)
stripes = cyclopts.App(name="stripes", help_format="markdown")
lsm.command(stripes)


@stripes.default
@autoconfig
def create(
    inp_cm1: str,
    inp_cm2: str,
    mip_dir: str,
    yaml_path: str,
    camera_id: int,
    file_num: int,
    *,
    general_config: GeneralConfig = None,
    zarr_config: ZarrConfig = None,
    dandiset_id: Optional[str] = None,
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
    scanParameters = load_scan_parameters(yaml_path)

    api_key = prompt_dandi_api_key() if dandiset_id else None

    tile_paths_1 = discover_tile_paths(
        inp_cm1, dandiset_id=dandiset_id, api_key=api_key)

    tile_paths_2 = discover_tile_paths(
        inp_cm2, dandiset_id=dandiset_id, api_key=api_key)

    affines = get_all_affines(tile_paths_1[file_num],
                              tile_paths_2[file_num], scanParameters, mip_dir)

    tile_paths = tile_paths_1 if camera_id == 1 else tile_paths_2

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

        if any(
            not os.path.exists(f"{general_config.out}/{i}/{name}.ome.zarr")
            for i in camera_channel_map[camera_id]
        ):

            yx_path = os.path.join(
                mip_dir, f"{name}_proc-mip.tiff").replace("slice0", "slice")

            if not os.path.exists(yx_path):
                raise FileNotFoundError(f"Missing YX image: {yx_path}")

            cam_info = get_camera_info(scanParameters, camera_id)
            vol_channels = crop_volume_channels(reader, cam_info)
            masks, thrs = load_mask_and_thr(path, mip_dir, cam_info)
            for i in camera_channel_map[camera_id]:
                output_name = f"{general_config.out}/{i}/{name}.ome.zarr"
                if not os.path.exists(output_name):
                    mask = masks[i]
                    thr = thrs[i]
                    chunk = zarr_config.chunk
                    if len(zarr_config.chunk) == 1:
                        chunk = tuple([zarr_config.chunk[0]]*3)

                    vol = vol_channels[i]
                    vol = preprocess(vol, mask, thr, camera_id, scanParameters)
                    vol = apply_affine(vol, affines[camera_id][i])

                    omz = ZarrPythonGroup.from_config(
                        output_name+".tmp", zarr_config)
                    out = omz.create_array("0", shape=vol.shape,
                                           zarr_config=zarr_config, dtype=np.uint16)
                    vol = da.rechunk(
                        vol, chunk)
                    with ProgressBar():
                        da.to_zarr(vol, out._array)
                    omz.generate_pyramid(levels=zarr_config.levels)
                    omz.write_ome_metadata(
                        axes=["z", "y", "x"],
                    )

                    os.replace(output_name + ".tmp", output_name)
                    print("--- %s secs ---" % (time.time() - start_time))

        print("--- %s secs ---" % (time.time() - start_time))
