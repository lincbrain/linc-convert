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
)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

camera_channel_map: Dict[int, List[str]] = {
    1: ["594", "660"],
    2: ["488", "561"],
}


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
    clip_high_percentile: float = 99.9,
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
        mask, thr = compute_tissue_mask(
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

def get_camera_info(scan_parameters: dict, camera_id: int) -> List[dict]:
    """
    Extract per-channel crop and metadata for a given camera.

    Parameters
    ----------
    scan_parameters : dict
        Loaded scan configuration.
    camera_id : int
        Camera identifier (1 or 2).

    Returns
    -------
    list of dict
        Each entry contains:
        - channel (str)
        - camera_id (int)
        - y_start, y_end (int)
        - x_start, x_end (int)
        - vertical_flip (bool)

    Raises
    ------
    KeyError
        If camera_id is not present in scan_parameters.
    ValueError
        If camera_id is invalid.
    """
    if camera_id not in camera_channel_map:
        raise ValueError(f"Invalid camera_id: {camera_id}")

    cam_key = f"Camera{camera_id}"

    if "crop" not in scan_parameters or cam_key not in scan_parameters["crop"]:
        raise KeyError(f"Missing crop information for {cam_key}")

    crop = scan_parameters["crop"][cam_key]
    channels = camera_channel_map[camera_id]

    info: List[dict] = []

    for idx, ch_name in enumerate(channels, start=1):
        ch_key = f"Ch{idx}"

        ch_crop = crop[ch_key]

        info.append({
            "channel": ch_name,
            "camera_id": camera_id,
            "y_start": int(ch_crop["yStart"]),
            "y_end": int(ch_crop["yEnd"]),
            "x_start": int(ch_crop["xStart"]),
            "x_end": int(ch_crop["xEnd"]),
            "vertical_flip": bool(crop["verticalFlip"]),
        })

    return info
