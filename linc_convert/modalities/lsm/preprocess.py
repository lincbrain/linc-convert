"""preprocessing pipeline: background removal + apply stripe correction + skew correction + affine alignment and write OME-Zarr outputs."""

import logging
import os
import time
from typing import Optional

import cyclopts
import dask.array as da
import numpy as np
from dask.diagnostics import ProgressBar

from linc_convert.modalities.lsm.cli import lsm
from linc_convert.modalities.lsm.convert_spool_or_zarr import (
    discover_tile_paths,
    open_tile_reader,
    prompt_dandi_api_key,
)
from linc_convert.modalities.lsm.preprocessing_utils.corrections import (
    apply_affine,
    crop_volume_channels,
    stripe_skew_corr,
)
from linc_convert.modalities.lsm.preprocessing_utils.io import (
    camera_channel_map,
    get_camera_info,
    load_mask_and_thresholds,
    load_scan_parameters,
)
from linc_convert.modalities.lsm.preprocessing_utils.registration import (
    get_all_affines,
)
from linc_convert.utils.io.zarr.drivers.zarr_python import ZarrPythonGroup
from linc_convert.utils.zarr_config import GeneralConfig, ZarrConfig, autoconfig

logger = logging.getLogger(__name__)
preprocess = cyclopts.App(name="preprocess", help_format="markdown")
lsm.command(preprocess)


@preprocess.default
@autoconfig
def preprocess(
    inp_cm1: str,
    inp_cm2: str,
    mip_dir: str,
    yaml_path: str,
    camera_id: int,
    file_num: int,
    *,
    general_config: Optional[GeneralConfig] = None,
    zarr_config: Optional[ZarrConfig] = None,
    dandiset_id: Optional[str] = None,
) -> None:
    """
    Process volumetric stripe data and write aligned OME-Zarr outputs.

    This pipeline:
    1. Loads scan parameters and tile datasets
    2. Computes inter-channel affines between cameras
    3. Applies masking, correction, skew correction, and affine alignment
    4. Writes corrected volumes to OME-Zarr with pyramid levels

    Parameters
    ----------
    inp_cm1 : str
        Path (local or DANDI) to camera 1 tiles.
    inp_cm2 : str
        Path (local or DANDI) to camera 2 tiles.
    mip_dir : str
        Directory containing YX MIP TIFF files used for mask generation.
    yaml_path : str
        Path to scan parameter YAML file.
    camera_id : int
        Camera to process (1 or 2).
    file_num : int
        Index of tile used to compute cross-camera affine alignment.
    general_config : GeneralConfig, optional
        Output configuration (must define `.out` directory).
    zarr_config : ZarrConfig, optional
        Zarr storage configuration (chunking, pyramid levels, etc.).
    dandiset_id : str, optional
        If provided, tiles are loaded from DANDI instead of local disk.

    Raises
    ------
    FileNotFoundError
        If a required MIP file is missing.
    ValueError
        If camera_id is not 1 or 2.
    """
    if camera_id not in (1, 2):
        raise ValueError(f"camera_id must be 1 or 2, got {camera_id}")

    scan_parameters = load_scan_parameters(yaml_path)

    api_key = prompt_dandi_api_key() if dandiset_id else None

    # Discover tile paths
    tile_paths_1 = discover_tile_paths(
        inp_cm1, dandiset_id=dandiset_id, api_key=api_key)
    tile_paths_2 = discover_tile_paths(
        inp_cm2, dandiset_id=dandiset_id, api_key=api_key)

    # Compute inter-camera affines using selected tile
    affines = get_all_affines(
        tile_paths_1[file_num],
        tile_paths_2[file_num],
        scan_parameters,
        mip_dir,
    )

    # Select camera-specific tiles
    tile_paths = tile_paths_1 if camera_id == 1 else tile_paths_2

    cam_info = get_camera_info(scan_parameters, camera_id)

    for index, path in enumerate(tile_paths):
        start_time = time.time()

        name = os.path.basename(path.rstrip("/").replace(".ome.zarr", ""))

        logger.info(f"[{index}] Processing {name}")

        reader = open_tile_reader(
            path, dandiset_id=dandiset_id, api_key=api_key)

        # Skip if all outputs already exist
        outputs_exist = all(
            os.path.exists(f"{general_config.out}/{ch}/{name}.ome.zarr")
            for ch in camera_channel_map[camera_id]
        )
        if outputs_exist:
            logger.info(f"Skipping {name} (already processed)")
            continue

        # Validate MIP availability
        mip_path = os.path.join(
            mip_dir, f"{name}_proc-mip.tiff").replace("slice0", "slice")
        if not os.path.exists(mip_path):
            raise FileNotFoundError(f"Missing YX MIP file: {mip_path}")

        # Load per-channel volumes
        vol_channels = crop_volume_channels(reader, cam_info)
        masks, thrs = load_mask_and_thresholds(name, mip_dir, cam_info)

        for ch in camera_channel_map[camera_id]:
            output_path = f"{general_config.out}/{ch}/{name}.ome.zarr"

            if os.path.exists(output_path):
                continue

            mask = masks[ch]
            thr = thrs[ch]

            # Normalize chunk spec
            chunk = zarr_config.chunk
            if len(chunk) == 1:
                chunk = (chunk[0],) * 3

            # Processing pipeline
            vol = vol_channels[ch]
            vol = stripe_skew_corr(vol, mask, thr, camera_id, scan_parameters)
            vol = apply_affine(vol, affines[camera_id][ch])

            # Write Zarr
            tmp_path = output_path + ".tmp"
            omz = ZarrPythonGroup.from_config(tmp_path, zarr_config)

            out = omz.create_array(
                "0",
                shape=vol.shape,
                zarr_config=zarr_config,
                dtype=np.uint16,
            )

            vol = da.rechunk(vol, out._array.shards or chunk)

            with ProgressBar():
                da.to_zarr(vol, out._array)

            omz.generate_pyramid(levels=zarr_config.levels)
            omz.write_ome_metadata(axes=["z", "y", "x"])

            os.replace(tmp_path, output_path)

            logger.info(f"Wrote {output_path}")

        elapsed = time.time() - start_time
        logger.info(f"{name} completed in {elapsed:.2f}s")
