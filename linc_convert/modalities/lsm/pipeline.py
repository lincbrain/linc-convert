"""Preprocess tiles (background removal, stripe/skew correction) and stream
the corrected volumes directly into a single blended OME-Zarr mosaic along
the y axis, without writing per-tile intermediates to disk.
"""

import gc
import getpass
import logging
import os
import time
from dataclasses import replace
from glob import glob
from pathlib import PurePosixPath
from typing import List, Optional, Tuple

import cyclopts
import dask.array as da
import numpy as np
import yaml
from dandi.dandiapi import DandiAPIClient
from dask.diagnostics import ProgressBar

from linc_convert.modalities.lsm.cli import lsm
from linc_convert.modalities.lsm.preprocessing_utils.corrections import (
    apply_channel_affine_mask,
    apply_channel_affine_volume,
    crop_volume_channels,
    maybe_flip_z_lazy,
    stripe_skew_corr,
)
from linc_convert.modalities.lsm.preprocessing_utils.io import (
    find_camera_for_channel,
    get_camera_info,
    get_channel_names,
    get_reference_local_crop,
    load_channel_affines,
    load_mask_and_thresholds,
    load_scan_parameters,
)
from linc_convert.utils.io.spool import SpoolSetInterpreter
from linc_convert.utils.io.zarr.drivers.zarr_python import ZarrPythonGroup
from linc_convert.utils.nifti_header import build_nifti_header
from linc_convert.utils.zarr_config import (
    GeneralConfig,
    NiftiConfig,
    ZarrConfig,
    autoconfig,
)

logger = logging.getLogger(__name__)
pipeline = cyclopts.App(name="pipeline", help_format="markdown")
lsm.command(pipeline)


def prompt_dandi_api_key() -> str:
    key = os.environ.get("DANDI_API_KEY")
    return key if key else getpass.getpass("Enter your DANDI API key: ")


def open_tile_reader(
    path: str,
    *,
    dandiset_id: Optional[str] = None,
    api_key: Optional[str] = None,
    chunks: Optional[Tuple[int, ...]] = None,
) -> da.Array:
    if path.endswith(".ome.zarr"):
        if dandiset_id is None:
            return da.asarray(
                ZarrPythonGroup.open(path)["0"],
                chunks=chunks if chunks is not None else (128, 128, 128),
            )
        return da.asarray(
            ZarrPythonGroup.open_dandi(
                dandiset_id=dandiset_id, asset_path=path, api_key=api_key,
            )["0"],
            chunks=chunks if chunks is not None else (128, 128, 128),
        )
    return da.asarray(
        SpoolSetInterpreter(path, f"{path}_info.mat").assemble_cropped(),
        chunks=chunks if chunks is not None else (128, 128, 128),
    )


def discover_tile_paths(inp, camera_id, *, dandiset_id, api_key):
    """Get all tile paths for one camera from the input location, in
    on-disk order.

    Tile names are expected to contain a camera token of either form:
    ``acq-camera-01`` (hyphen before the zero-padded number, e.g. the
    original MF283 dataset) or ``acq-camera01`` (no hyphen, e.g. the
    MF282 dataset). Both are checked, since different datasets use
    different conventions.
    """
    camera_tokens = [
        f"acq-camera-{camera_id:02d}",
        f"acq-camera{camera_id:02d}",
    ]

    if dandiset_id is None:
        paths = sorted({
            p
            for token in camera_tokens
            for p in glob(os.path.join(inp, f"*{token}*.ome.zarr"))
        })
        if not paths:
            raise ValueError(
                f"No tile folders found for camera {camera_id} "
                f"(tried tokens {camera_tokens!r}) in input directory"
            )
        return paths
    with DandiAPIClient(api_url="https://api.dandiarchive.org/api", token=api_key) as client:
        dandiset = client.get_dandiset(dandiset_id, "draft")
        prefix = PurePosixPath(inp.rstrip("/") + "/")
        depth = len(prefix.parts)
        paths = sorted({
            asset.path
            for asset in dandiset.get_assets_with_path_prefix(str(prefix))
            if len(PurePosixPath(asset.path).parts) == depth + 1
            and any(token in PurePosixPath(asset.path).name for token in camera_tokens)
        })
    if not paths:
        raise ValueError(
            f"No tile assets found for camera {camera_id} "
            f"(tried tokens {camera_tokens!r}) in DANDI dataset"
        )
    return paths


def _open_raw_channel_volume_and_mask(
    path: str,
    *,
    dandiset_id: Optional[str],
    api_key: Optional[str],
    mip_dir: Optional[str],
    name: str,
    ch: str,
    cam_info,
    channel_affine: Optional[np.ndarray] = None,
    reference_local_crop: Optional[dict] = None,
    pre_flip: Optional[bool] = None,
    mip_pre_split: bool = False,
    x_range=5000
) -> Tuple[da.Array, np.ndarray, float]:
    reader = open_tile_reader(path, dandiset_id=dandiset_id, api_key=api_key)
    vol_channels = crop_volume_channels(reader, cam_info)
    masks, thrs = load_mask_and_thresholds(
        name, mip_dir, cam_info, reader=reader, ch=ch, pre_split=mip_pre_split, x_range=x_range
    )

    vol = vol_channels[ch]
    mask = masks[ch]

    if pre_flip is not None:
        vol = maybe_flip_z_lazy(vol, pre_flip)

    if channel_affine is not None and not np.allclose(channel_affine, np.eye(3)):
        vol = apply_channel_affine_volume(vol, channel_affine)
        mask = apply_channel_affine_mask(mask, channel_affine)

    if reference_local_crop is not None:
        y1, y2 = reference_local_crop["y_start"], reference_local_crop["y_end"]
        z1, z2 = reference_local_crop["z_start"], reference_local_crop["z_end"]
        vol = vol[z1:z2, y1:y2, :]
        mask = mask[y1:y2, :]

    return vol, mask, thrs[ch]


def _corrected_y_chunk(vol, mask, threshold, camera_id, scan_parameters, y0, y1, force_flip=None):
    vol_chunk = vol[:, y0:y1, :]
    mask_chunk = mask[:, y0:y1, :] if mask.ndim == 3 else mask[y0:y1, :]
    return stripe_skew_corr(
        vol_chunk, mask_chunk, threshold, camera_id, scan_parameters, force_flip=force_flip,
    )


def _write_checkpoint(filename, y):
    with open(filename, "w") as f:
        f.write(f"{y}\n")


def _read_checkpoint(filename, default_y):
    try:
        with open(filename, "r") as f:
            content = f.read().strip()
            return int(content)
    except (FileNotFoundError, ValueError):
        return default_y


def load_y_coordinates(coords_yaml_path):
    with open(coords_yaml_path, "r") as f:
        coords = yaml.safe_load(f)
    return [entry[0]["y"] for entry in coords["coordinates"]]


def _checkpoint_path(general_config, ch):
    out = general_config.out.rstrip("/")
    if out.endswith(".ome.zarr"):
        out = out[: -len(".ome.zarr")]
    return f"{out}_{ch}.dat"


@pipeline.default
@autoconfig
def pipeline(
    inp: str,
    yaml_path: str,
    camera_id: int,
    slice_number: int,
    coords_yaml_ch1: str,
    coords_yaml_ch2: str,
    *,
    mip_dir: Optional[str] = None,
    mip_pre_split: bool = False,
    y_chunk_size: int = 256,
    voxel_size: List[float] = [1, 1, 1],
    general_config: Optional[GeneralConfig] = None,
    zarr_config: Optional[ZarrConfig] = None,
    nii_config: Optional[NiftiConfig] = None,
    dandiset_id: Optional[str] = None,
    chunk_min: Optional[int] = None,
    chunk_max: Optional[int] = None,
    channel_affines_path: Optional[str] = None,
    reference_channel: str = "488",
    x_range: int = 5000
) -> None:
    """
    Correct volumetric tile data and stream it directly into a single
    blended OME-Zarr mosaic along the y axis.

    Parameters
    ----------
    mip_dir : str, optional
        Directory containing MIP TIFF files used for mask generation.
        If omitted, the MIP is computed at runtime instead.
    mip_pre_split : bool, default=False
        If True (and `mip_dir` is given), read one MIP file per
        channel, named `{ch}_{tile_name}_proc-mip.tiff`, instead of one
        combined file per camera (`{tile_name}_proc-mip.tiff`). These
        per-channel files must still span the full raw sensor Y range
        (same coordinate system as `cam_info`), not be pre-cropped to
        just that channel's own window -- see
        `load_mask_and_thresholds` for why.
    """
    if camera_id not in (1, 2):
        raise ValueError(f"camera_id must be 1 or 2, got {camera_id}")

    start_timer = time.time()
    voxel_size = list(map(float, reversed(voxel_size)))
    scan_parameters = load_scan_parameters(yaml_path)

    skew_delta_deg = scan_parameters["acquisitionSettings"]["skewCorrection"]["delta_deg"]
    voxel_size[0] = voxel_size[0] / np.cos(np.deg2rad(skew_delta_deg))

    channel_affines = (
        load_channel_affines(channel_affines_path, reference_channel)
        if channel_affines_path else None
    )

    api_key = prompt_dandi_api_key() if dandiset_id else None

    crop_stage = "split" if channel_affines is not None else "stitching"
    cam_info = get_camera_info(
        scan_parameters, camera_id, slice_number, crop_stage=crop_stage)

    if channel_affines is not None:
        reference_camera_id = find_camera_for_channel(
            scan_parameters, reference_channel)
        reference_cam_info_split = get_camera_info(
            scan_parameters, reference_camera_id, slice_number, crop_stage="split"
        )
        reference_tile_paths = discover_tile_paths(
            inp, reference_camera_id, dandiset_id=dandiset_id, api_key=api_key
        )
        reference_reader = open_tile_reader(
            reference_tile_paths[0], dandiset_id=dandiset_id, api_key=api_key
        )
        reference_split_z_depth = crop_volume_channels(
            reference_reader, reference_cam_info_split
        )[reference_channel].shape[0]

        reference_local_crop = get_reference_local_crop(
            scan_parameters, reference_channel, slice_number,
            reference_split_z_depth=reference_split_z_depth,
        )
    else:
        reference_local_crop = None

    if channel_affines is not None:
        pre_flip = bool(
            scan_parameters["channelLayout"][f"Camera{camera_id}"]["verticalFlip"])
        force_flip = False
    else:
        pre_flip = None
        force_flip = None

    tile_paths = discover_tile_paths(
        inp, camera_id, dandiset_id=dandiset_id, api_key=api_key)
    num_tiles = len(tile_paths)

    def tile_name(path):
        return os.path.basename(path.rstrip("/").replace(".ome.zarr", ""))

    channels = get_channel_names(scan_parameters, camera_id)
    if len(channels) != 2:
        raise ValueError(
            f"Expected exactly 2 channels for camera {camera_id}, got {len(channels)}: {channels}"
        )
    coords_yaml_by_channel = dict(
        zip(channels, [coords_yaml_ch1, coords_yaml_ch2]))

    for ch in channels:
        channel_timer = time.time()
        channel_affine = (
            channel_affines.get(ch, np.eye(
                3)) if channel_affines is not None else None
        )

        y_coords = load_y_coordinates(coords_yaml_by_channel[ch])

        sample_path = tile_paths[0]
        sample_raw_vol, sample_mask, sample_thr = _open_raw_channel_volume_and_mask(
            sample_path,
            dandiset_id=dandiset_id,
            api_key=api_key,
            mip_dir=mip_dir,
            name=tile_name(sample_path),
            ch=ch,
            cam_info=cam_info,
            channel_affine=channel_affine,
            reference_local_crop=reference_local_crop,
            pre_flip=pre_flip,
            mip_pre_split=mip_pre_split,
            x_range=x_range
        )
        sample_corrected = stripe_skew_corr(
            sample_raw_vol, sample_mask, sample_thr, camera_id, scan_parameters,
            force_flip=force_flip,
        )
        corrected_sz, corrected_sy, corrected_sx = sample_corrected.shape
        del sample_corrected
        gc.collect()

        full_x = corrected_sx
        full_y = int(round(y_coords[-1])) + corrected_sy
        full_z = corrected_sz
        fullshape = (full_z, full_y, full_x)

        out_dir = f"{general_config.out}/{ch}"
        checkpoint_file = _checkpoint_path(general_config, ch)
        checkpoint = _read_checkpoint(checkpoint_file, -1)

        omz = ZarrPythonGroup.from_config(out_dir, zarr_config)
        try:
            if checkpoint == -1:
                array = omz.create_array(
                    "0", shape=fullshape, zarr_config=zarr_config, dtype=np.uint16)
            else:
                array = omz["0"]
        except Exception:
            logger.info("already exists")
            array = omz["0"]

        logger.info(
            "Writing channel %s, level 0 array with shape %s", ch, fullshape)

        carry = None
        effective_min = max(chunk_min if chunk_min is not None else 0, 0)
        effective_max = min(
            chunk_max if chunk_max is not None else num_tiles - 1, num_tiles - 1)

        if len(tile_paths) > checkpoint + 1:
            for index, path in enumerate(tile_paths):
                gc.collect()
                if index < checkpoint or index < effective_min:
                    continue
                if index > effective_max:
                    break

                name = tile_name(path)
                tile_timer = time.time()

                if index == 0:
                    raw_vol, mask, thr = sample_raw_vol, sample_mask, sample_thr
                else:
                    raw_vol, mask, thr = _open_raw_channel_volume_and_mask(
                        path,
                        dandiset_id=dandiset_id,
                        api_key=api_key,
                        mip_dir=mip_dir,
                        name=name,
                        ch=ch,
                        cam_info=cam_info,
                        channel_affine=channel_affine,
                        reference_local_crop=reference_local_crop,
                        pre_flip=pre_flip,
                        mip_pre_split=mip_pre_split,
                        x_range=x_range
                    )

                is_first = index == 0 or index == checkpoint or (
                    chunk_min is not None and index == chunk_min)
                is_last = index == num_tiles - 1 or index == effective_max

                ystart = int(round(y_coords[index]))

                overlap_with_prev = 0
                if not is_first:
                    overlap_with_prev = corrected_sy - (
                        int(round(y_coords[index])) -
                        int(round(y_coords[index - 1]))
                    )
                overlap_with_next = 0
                if not is_last:
                    overlap_with_next = corrected_sy - (
                        int(round(y_coords[index + 1])) -
                        int(round(y_coords[index]))
                    )
                overlap_with_prev = max(overlap_with_prev, 0)
                overlap_with_next = max(overlap_with_next, 0)
                withhold_from = corrected_sy - overlap_with_next

                if overlap_with_prev > 0:
                    t = np.linspace(0, 1, overlap_with_prev)
                    ramp = (1 - np.cos(np.pi * t)) / 2
                    ramp_inverse = (1 + np.cos(np.pi * t)) / 2
                    ramp = ramp[None, :, None]
                    ramp_inverse = ramp_inverse[None, :, None]

                zstart = 0
                trailing_buffer = None
                y0 = 0
                while y0 < corrected_sy:
                    y1 = min(corrected_sy, y0 + y_chunk_size)
                    lazy_chunk = _corrected_y_chunk(
                        raw_vol, mask, thr, camera_id, scan_parameters, y0, y1, force_flip=force_flip,
                    )
                    with ProgressBar():
                        data = lazy_chunk.compute()

                    if overlap_with_prev > 0 and y0 < overlap_with_prev:
                        blend_len = min(data.shape[1], overlap_with_prev - y0)
                        carry_slice = carry[:, y0:y0 + blend_len, :]
                        ramp_slice = ramp[:, y0:y0 + blend_len, :]
                        ramp_inv_slice = ramp_inverse[:, y0:y0 + blend_len, :]
                        data[:, :blend_len, :] = (
                            carry_slice * ramp_inv_slice +
                            data[:, :blend_len, :] * ramp_slice
                        )

                    if y1 <= withhold_from:
                        to_write, to_withhold = data, None
                    elif y0 >= withhold_from:
                        to_write, to_withhold = None, data
                    else:
                        split = withhold_from - y0
                        to_write, to_withhold = data[:,
                                                     :split, :], data[:, split:, :]

                    out_ystart = ystart + y0
                    if to_write is not None and to_write.shape[1] > 0 and index > checkpoint:
                        array[
                            zstart: zstart + to_write.shape[0],
                            out_ystart: out_ystart + to_write.shape[1],
                            0: to_write.shape[2],
                        ] = to_write

                    if to_withhold is not None:
                        trailing_buffer = (
                            to_withhold if trailing_buffer is None
                            else np.concatenate([trailing_buffer, to_withhold], axis=1)
                        )

                    del data
                    gc.collect()
                    y0 = y1

                carry = trailing_buffer
                _write_checkpoint(checkpoint_file, index)

        gc.collect()
        copy_config = replace(general_config, out=out_dir)
        omz.generate_pyramid_staged(
            levels=zarr_config.levels, copy_config=copy_config, copy_zarr_config=zarr_config,
        )
        omz.write_ome_metadata(axes=["z", "y", "x"], space_scale=voxel_size)

        if nii_config and nii_config.nii:
            header = build_nifti_header(
                zgroup=omz, voxel_size_zyx=tuple(voxel_size), unit="micrometer", nii_config=nii_config,
            )
            omz.write_nifti_header(header)

        logger.info(
            f"Channel {ch} completed in {(time.time() - channel_timer) / 60:.2f} minutes")

    logger.info(
        f"Conversion completed in {(time.time() - start_timer) / 60} minutes.")
