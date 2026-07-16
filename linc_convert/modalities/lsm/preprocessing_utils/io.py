import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tifffile
import yaml

from linc_convert.modalities.lsm.preprocessing_utils.corrections import (
    crop_mip_channels,
)
from linc_convert.modalities.lsm.preprocessing_utils.masks import (
    compute_tissue_mask,
    compute_tissue_mask_otsu,
)

# ---------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------


def load_scan_parameters(yaml_path: Path) -> dict:
    """
    Load scan/acquisition parameters from a YAML file.

    Parameters
    ----------
    yaml_path : Path
        Path to YAML configuration file.

    Returns
    -------
    dict
        Parsed scan parameters.
    """
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def load_mask_and_thresholds(
    name: str,
    mip_dir: str,
    cam_info: List[dict],
    *,
    downsample: int = 8,
    clip_high_percentile: float = 99.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Load MIP image and compute tissue masks + thresholds per channel.

    Parameters
    ----------
    name : str
        Tile basename (without extension).
    mip_dir : str
        Directory containing *_proc-mip.tiff files.
    cam_info : list of dict
        Camera cropping metadata (from `get_camera_info`).
    downsample : int, default=8
        Downsampling factor for mask computation.
    clip_high_percentile : float, default=99.9
        Percentile used to clip high-intensity outliers.

    Returns
    -------
    masks : dict[str, np.ndarray]
        Per-channel binary masks.
    thresholds : dict[str, float]
        Per-channel intensity thresholds.

    Raises
    ------
    FileNotFoundError
        If the corresponding MIP file is missing.
    """
    mip_path = os.path.join(
        mip_dir, f"{name}_proc-mip.tiff").replace("slice0", "slice")

    if not os.path.exists(mip_path):
        raise FileNotFoundError(f"Missing YX MIP file: {mip_path}")

    raw_mip = tifffile.imread(mip_path).astype(np.float32)

    # Crop MIP per channel
    mip_channels = crop_mip_channels(raw_mip, cam_info)

    masks: Dict[str, np.ndarray] = {}
    thresholds: Dict[str, float] = {}

    for ch, mip in mip_channels.items():
        mask, thr = compute_tissue_mask_otsu(
            mip,
            downsample=downsample,
            clip_high_percentile=clip_high_percentile,
        )
        masks[ch] = mask
        thresholds[ch] = thr

    return masks, thresholds


# ---------------------------------------------------------------------
# Camera metadata
# ---------------------------------------------------------------------

def _resolve_config_id(scan_parameters: dict, slice_number: int) -> str:
    """
    Find the `configEpoch` (from `scan_parameters["configEpochs"]`) whose
    `appliesToSlices` range contains `slice_number`.

    Parameters
    ----------
    scan_parameters : dict
        Loaded scan configuration.
    slice_number : int
        Physical slice number being processed.

    Returns
    -------
    str
        The matching epoch's `configID`.

    Raises
    ------
    ValueError
        If no epoch's slice range covers `slice_number`.
    """
    for epoch in scan_parameters.get("configEpochs", []):
        applies = epoch["appliesToSlices"]
        start = applies["start"]
        end = applies["end"]
        if end is None:
            end = math.inf
        if start <= slice_number <= end:
            return epoch["configID"]

    raise ValueError(
        f"No configEpoch in scan parameters covers slice {slice_number}"
    )


def get_channel_names(scan_parameters: dict, camera_id: int) -> List[str]:
    """
    Get the ordered channel names (e.g. ["594", "660"]) for a camera,
    from `scan_parameters["channelLayout"]`.

    Channel identity (which wavelengths live on which camera) is treated
    as fixed for the whole acquisition -- only the crop boundaries for
    each channel vary across configEpochs.

    Parameters
    ----------
    scan_parameters : dict
        Loaded scan configuration.
    camera_id : int
        Camera identifier (1 or 2).

    Returns
    -------
    list of str
        Channel names, ordered by their `Ch1`, `Ch2`, ... key.

    Raises
    ------
    KeyError
        If `channelLayout` is missing an entry for this camera.
    """
    cam_key = f"Camera{camera_id}"
    layout = scan_parameters.get("channelLayout", {})

    if cam_key not in layout:
        raise KeyError(f"Missing channelLayout for {cam_key}")

    cam_layout = layout[cam_key]
    ch_keys = sorted(
        (k for k in cam_layout if k.startswith("Ch")),
        key=lambda k: int(k[2:]),
    )
    return [cam_layout[k] for k in ch_keys]


def get_camera_info(
    scan_parameters: dict, camera_id: int, slice_number: int
) -> List[dict]:
    """
    Extract per-channel crop and metadata for a given camera, using the
    configEpoch whose slice range covers `slice_number`.

    Parameters
    ----------
    scan_parameters : dict
        Loaded scan configuration.
    camera_id : int
        Camera identifier (1 or 2).
    slice_number : int
        Physical slice number being processed by this run. Selects
        which configEpoch's `cropDefinitions` to use.

    Returns
    -------
    list of dict
        Each entry contains:
        - channel (str)
        - camera_id (int)
        - y_start, y_end (int)
            Lateral-axis ("y" in the acquisition YAML's axis
            convention) crop bounds, taken from this channel's
            `stitchingCrop.yRange`. This is the axis used by
            `crop_volume_channels`/`crop_mip_channels` to split a raw
            camera frame into per-channel volumes.
        - z_start, z_end (int or None)
            Depth-axis ("z" in the acquisition YAML) crop bounds,
            taken from `stitchingCrop.zRange`. Applied by
            `crop_volume_channels` to crop the raw volume's Z axis.
            Not applied to the YX MIP used for masking, since that MIP
            has already been max-projected along Z and has no Z axis
            left to crop. `None` if this epoch/channel doesn't define
            a zRange (in which case the Z axis is left uncropped).
        - vertical_flip (bool)
            For reference only: `skew_correct_volume_lazy` reads
            `verticalFlip` directly from
            `scan_parameters["channelLayout"]`, not from this dict.

    Raises
    ------
    ValueError
        If camera_id is invalid, or no configEpoch covers
        `slice_number`.
    KeyError
        If the resolved epoch has no crop/layout info for this camera,
        or a channel's `stitchingCrop.yRange` isn't defined.
    """
    if camera_id not in (1, 2):
        raise ValueError(f"Invalid camera_id: {camera_id}")

    cam_key = f"Camera{camera_id}"
    config_id = _resolve_config_id(scan_parameters, slice_number)

    crop_defs = scan_parameters.get("cropDefinitions", {})
    if config_id not in crop_defs or cam_key not in crop_defs[config_id]:
        raise KeyError(
            f"Missing cropDefinitions for {cam_key} in epoch '{config_id}'"
        )

    channels_crop = crop_defs[config_id][cam_key].get("channels", {})

    layout = scan_parameters.get("channelLayout", {})
    if cam_key not in layout:
        raise KeyError(f"Missing channelLayout for {cam_key}")
    vertical_flip = bool(layout[cam_key]["verticalFlip"])

    info: List[dict] = []

    for ch_name, ch_crop in sorted(
        channels_crop.items(), key=lambda item: item[1]["channelKey"]
    ):
        stitching = ch_crop.get("stitchingCrop") or {}
        y_range = stitching.get("yRange")
        z_range = stitching.get("zRange")

        if y_range is None:
            raise KeyError(
                f"stitchingCrop.yRange not defined for channel '{ch_name}' "
                f"on {cam_key} in epoch '{config_id}'"
            )

        info.append({
            "channel": ch_name,
            "camera_id": camera_id,
            "y_start": int(y_range[0]),
            "y_end": int(y_range[1]),
            "z_start": int(z_range[0]) if z_range is not None else None,
            "z_end": int(z_range[1]) if z_range is not None else None,
            "vertical_flip": vertical_flip,
        })

    return info
