"""Read the scan metadata YAML: config epochs, crops, camera flip, skew."""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# (z0, z1, y0, y1); z may be None
Crop = Tuple[Optional[int], Optional[int], int, int]


def resolve_config_id(scan_parameters: Dict[str, Any], slice_number: int) -> str:
    """Find the key of the cropDefinitions block that covers a slice."""
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


def _channel_definitions(scan_parameters: Dict[str, Any], camera_id: int,
                         slice_number: int) -> Tuple[Dict[str, Any], str, str]:
    """Get the per-channel crop definitions of a camera."""
    cam_key = f"Camera{camera_id}"
    config_id = resolve_config_id(scan_parameters, slice_number)
    try:
        channels = scan_parameters["cropDefinitions"][config_id][cam_key]["channels"]
    except KeyError:
        raise KeyError(
            f"Missing cropDefinitions for {cam_key} in epoch '{config_id}'")
    return channels, cam_key, config_id


def _parse_crop(channel: str, definition: Dict[str, Any], kind: str,
                include_z: bool, where: str) -> Crop:
    crop = definition.get(kind) or {}
    if crop.get("yRange") is None:
        raise KeyError(
            f"{kind}.yRange not defined for channel '{channel}' on {where}")
    z = crop.get("zRange") if include_z else None
    return (int(z[0]) if z is not None else None,
            int(z[1]) if z is not None else None,
            int(crop["yRange"][0]), int(crop["yRange"][1]))


def get_crops(scan_parameters: Dict[str, Any], camera_id: int, slice_number: int,
              kind: str = "stitchingCrop", include_z: bool = True) -> Dict[str, Crop]:
    """Read the crop of every channel of a camera."""
    channels, cam_key, config_id = _channel_definitions(
        scan_parameters, camera_id, slice_number)
    where = f"{cam_key} in epoch '{config_id}'"
    return {str(ch): _parse_crop(str(ch), ch_def, kind, include_z, where)
            for ch, ch_def in channels.items()}


def get_crop(scan_parameters: Dict[str, Any], camera_id: int, slice_number: int,
             channel: str, kind: str = "stitchingCrop", include_z: bool = True) -> Crop:
    """Read the crop of one channel."""
    return get_crops(scan_parameters, camera_id, slice_number, kind, include_z)[channel]


def get_channel_names(scan_parameters: Dict[str, Any], camera_id: int) -> List[str]:
    """List the channel names of a camera."""
    layout = scan_parameters.get("channelLayout", {}).get(f"Camera{camera_id}")
    if layout is None:
        raise KeyError(f"Missing channelLayout for Camera{camera_id}")
    keys = sorted((k for k in layout if k.startswith("Ch")),
                  key=lambda k: int(k[2:]))
    return [str(layout[k]) for k in keys]


def get_vertical_flip(scan_parameters: Dict[str, Any], camera_id: int) -> bool:
    """Read whether a camera's volumes are flipped along z."""
    return bool(scan_parameters["channelLayout"][f"Camera{camera_id}"]["verticalFlip"])


def get_skew_shear(scan_parameters: Dict[str, Any]) -> float:
    """Compute the x shear per z step that undoes the oblique acquisition."""
    delta = scan_parameters["acquisitionSettings"]["skewCorrection"]["delta_deg"]
    umps = scan_parameters["voxelSize_um"]["rawAcquisition"]
    return float(np.tan(np.deg2rad(delta)) * umps["z"] / umps["x"])
