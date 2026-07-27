import math
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import dask.array as da
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
    """
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def load_channel_affines(
    affines_path: str, reference_channel: str
) -> Dict[str, np.ndarray]:
    """
    Load per-channel registration affines from a YAML file.
    """
    with open(affines_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    if reference_channel in raw:
        warnings.warn(
            f"Affines file specifies an affine for reference channel "
            f"'{reference_channel}'; ignoring it and using identity, "
            "since the reference channel is never registered.",
            stacklevel=2,
        )

    affines: Dict[str, np.ndarray] = {reference_channel: np.eye(3)}

    for ch_name, matrix in raw.items():
        if ch_name == reference_channel:
            continue

        arr = np.asarray(matrix, dtype=np.float64)
        if arr.shape != (3, 3):
            raise ValueError(
                f"Affine for channel '{ch_name}' must be 3x3, got "
                f"shape {arr.shape}"
            )
        affines[ch_name] = arr

    return affines


def load_mask_and_thresholds(
    name: str,
    mip_dir: Optional[str],
    cam_info: List[dict],
    *,
    reader: Optional["da.Array"] = None,
    downsample: int = 8,
    clip_high_percentile: float = 99.0,
    pre_split: bool = False,
    ch: Optional[str] = None,
    x_range=5000,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Load (or compute) a YX MIP image and compute tissue masks +
    thresholds per channel.

    If `mip_dir` is given, the MIP is loaded from a pre-generated
    `*_proc-mip.tiff` file. If `pre_split` is True, this reads a
    per-channel file named `{ch}_{name}_proc-mip.tiff`, which is
    expected to still span the FULL raw sensor Y range (same
    coordinate system `cam_info` uses) -- just organized as one file
    per channel rather than one combined file per camera. This keeps
    `crop_mip_channels` below correct and unconditional, regardless of
    whether `cam_info` reflects the tight `stitchingCrop` window or the
    broader `splitCrop` window (crop_stage="split", used during
    cross-channel registration) -- a per-channel file that was already
    cropped down to just its own local window would only be correct
    for one of those two cases, not both.

    If `mip_dir` is `None`, the MIP is instead computed at runtime as a
    max-intensity projection over Z of `reader` (the raw,
    not-yet-channel-split tile reader for this same tile).

    Parameters
    ----------
    name : str
        Tile basename (without extension).
    mip_dir : str, optional
        Directory containing MIP TIFF files. If `None`, the MIP is
        computed from `reader` instead.
    cam_info : list of dict
        Camera cropping metadata (from `get_camera_info`).
    reader : dask.array.Array, optional
        Raw, not-yet-channel-split lazy volume (Z, Y, X) for this
        tile, used to compute the MIP when `mip_dir` is `None`.
    downsample : int, default=8
        Downsampling factor for mask computation.
    clip_high_percentile : float, default=99.9
        Percentile used to clip high-intensity outliers.
    pre_split : bool, default=False
        If True (and `mip_dir` is given), read a per-channel MIP file
        named `{ch}_{name}_proc-mip.tiff` instead of the combined
        per-camera `{name}_proc-mip.tiff`. Requires `ch`.
    ch : str, optional
        Channel name. Required if `pre_split` is True; also used (in
        either case) to only crop/compute the mask for this one
        channel rather than every channel in `cam_info`.

    Returns
    -------
    masks : dict[str, np.ndarray]
        Per-channel binary masks.
    thresholds : dict[str, float]
        Per-channel intensity thresholds.

    Raises
    ------
    FileNotFoundError
        If `mip_dir` is given but the corresponding MIP file is
        missing.
    ValueError
        If `mip_dir` is `None` and `reader` isn't given, or if
        `pre_split` is True and `ch` isn't given.
    """
    if mip_dir is not None:
        if pre_split:
            if ch is None:
                raise ValueError(
                    "load_mask_and_thresholds: ch is required when "
                    "pre_split=True."
                )
            mip_path = os.path.join(
                mip_dir, f"{ch}_{name}_proc-mip.tiff").replace("slice0", "slice")

            if not os.path.exists(mip_path):
                raise FileNotFoundError(f"Missing YX MIP file: {mip_path}")

            raw_mip = tifffile.imread(mip_path).astype(np.float32)
        else:
            mip_path = os.path.join(
                mip_dir, f"{name}_proc-mip.tiff").replace("slice0", "slice")

            if not os.path.exists(mip_path):
                raise FileNotFoundError(f"Missing YX MIP file: {mip_path}")

            raw_mip = tifffile.imread(mip_path).astype(np.float32)
    else:
        if reader is None:
            raise ValueError(
                "load_mask_and_thresholds: either mip_dir or reader must "
                "be provided to obtain the YX MIP."
            )
        raw_mip = np.asarray(reader.max(axis=0).compute()).astype(np.float32)

    mip_channels = crop_mip_channels(raw_mip, cam_info, ch=ch)

    masks: Dict[str, np.ndarray] = {}
    thresholds: Dict[str, float] = {}

    for out_ch, mip in mip_channels.items():
        mask, thr = compute_tissue_mask(
            mip,
            downsample=downsample,
            clip_high_percentile=clip_high_percentile,
            x_range=x_range
        )
        masks[out_ch] = mask
        thresholds[out_ch] = thr

    return masks, thresholds


# ---------------------------------------------------------------------
# Camera metadata
# ---------------------------------------------------------------------

def _resolve_config_id(scan_parameters: dict, slice_number: int) -> str:
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
    scan_parameters: dict,
    camera_id: int,
    slice_number: int,
    crop_stage: str = "stitching",
) -> List[dict]:
    if camera_id not in (1, 2):
        raise ValueError(f"Invalid camera_id: {camera_id}")
    if crop_stage not in ("stitching", "split"):
        raise ValueError(
            f"crop_stage must be 'stitching' or 'split', got {crop_stage!r}"
        )

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
        if crop_stage == "stitching":
            stitching = ch_crop.get("stitchingCrop") or {}
            y_range = stitching.get("yRange")
            z_range = stitching.get("zRange")

            if y_range is None:
                raise KeyError(
                    f"stitchingCrop.yRange not defined for channel "
                    f"'{ch_name}' on {cam_key} in epoch '{config_id}'"
                )

            y_start, y_end = int(y_range[0]), int(y_range[1])
            z_start = int(z_range[0]) if z_range is not None else None
            z_end = int(z_range[1]) if z_range is not None else None
        else:
            split = ch_crop.get("splitCrop") or {}
            y_range = split.get("yRange")
            z_range = split.get("zRange")

            if y_range is None:
                raise KeyError(
                    f"splitCrop.yRange not defined for channel "
                    f"'{ch_name}' on {cam_key} in epoch '{config_id}'"
                )

            y_start, y_end = int(y_range[0]), int(y_range[1])
            z_start = int(z_range[0]) if z_range is not None else None
            z_end = int(z_range[1]) if z_range is not None else None

        info.append({
            "channel": ch_name,
            "camera_id": camera_id,
            "y_start": y_start,
            "y_end": y_end,
            "z_start": z_start,
            "z_end": z_end,
            "vertical_flip": vertical_flip,
        })

    return info


def find_camera_for_channel(scan_parameters: dict, channel: str) -> int:
    layout = scan_parameters.get("channelLayout", {})

    for camera_id in (1, 2):
        cam_key = f"Camera{camera_id}"
        cam_layout = layout.get(cam_key, {})
        ch_names = [cam_layout[k] for k in cam_layout if k.startswith("Ch")]
        if channel in ch_names:
            return camera_id

    raise KeyError(
        f"Channel '{channel}' not found in channelLayout for either camera"
    )


def get_reference_local_crop(
    scan_parameters: dict,
    reference_channel: str,
    slice_number: int,
    reference_split_z_depth: int,
) -> Dict[str, int]:
    camera_id = find_camera_for_channel(scan_parameters, reference_channel)
    cam_key = f"Camera{camera_id}"
    config_id = _resolve_config_id(scan_parameters, slice_number)

    crop_defs = scan_parameters.get("cropDefinitions", {})
    if config_id not in crop_defs or cam_key not in crop_defs[config_id]:
        raise KeyError(
            f"Missing cropDefinitions for {cam_key} in epoch '{config_id}'"
        )

    channels_crop = crop_defs[config_id][cam_key].get("channels", {})
    if reference_channel not in channels_crop:
        raise KeyError(
            f"No crop entry for reference channel '{reference_channel}' "
            f"on {cam_key} in epoch '{config_id}'"
        )

    ch_crop = channels_crop[reference_channel]
    split = ch_crop.get("splitCrop") or {}
    stitching = ch_crop.get("stitchingCrop") or {}

    split_y = split.get("yRange")
    split_z = split.get("zRange")
    stitch_y = stitching.get("yRange")
    stitch_z = stitching.get("zRange")

    if split_y is None:
        raise KeyError(
            f"splitCrop.yRange not defined for reference channel "
            f"'{reference_channel}' in epoch '{config_id}'"
        )
    if stitch_y is None or stitch_z is None:
        raise KeyError(
            f"stitchingCrop not fully defined for reference channel "
            f"'{reference_channel}' in epoch '{config_id}'"
        )

    layout = scan_parameters.get("channelLayout", {})
    if cam_key not in layout:
        raise KeyError(f"Missing channelLayout for {cam_key}")
    ref_vertical_flip = bool(layout[cam_key]["verticalFlip"])

    split_y0 = int(split_y[0])
    split_z0 = int(split_z[0]) if split_z is not None else 0
    stitch_z0, stitch_z1 = int(stitch_z[0]), int(stitch_z[1])

    local_tz0 = stitch_z0 - split_z0
    local_tz1 = stitch_z1 - split_z0

    if ref_vertical_flip:
        local_z_start = reference_split_z_depth - local_tz1
        local_z_end = reference_split_z_depth - local_tz0
    else:
        local_z_start = local_tz0
        local_z_end = local_tz1

    return {
        "y_start": int(stitch_y[0]) - split_y0,
        "y_end": int(stitch_y[1]) - split_y0,
        "z_start": local_z_start,
        "z_end": local_z_end,
    }
