"""Preprocess tiles (background removal, stripe/skew correction) and stream
the corrected volumes directly into a single blended OME-Zarr mosaic along
the y axis, without writing per-tile intermediates to disk.

Chunked along X (not Y): X can be enormous for this dataset, so tiles
are processed in two passes per tile:
  Pass 1 gathers the stripe-correction statistics across X-chunks
  (exact, not approximate -- see StripeStatsAccumulator).
  Pass 2 re-reads each X-chunk (padded, for the skew shear's Z<->X
  coupling), applies flip -> registration -> crop -> stripe correction
  -> skew shear, and writes it out.
Y is never chunked -- it was never the huge axis here, so each
X-chunk's processing handles the tile's full Y range in one shot.
Tile-to-tile Y blending (overlap) is tracked per X-chunk, via a dict
keyed by each X-chunk's starting position, since the same X-chunk
boundaries recur for every tile of a given channel.
"""

import gc
import getpass
import logging
import os
import time
from dataclasses import replace
from glob import glob
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Tuple

import cyclopts
import dask.array as da
import numpy as np
import tifffile
import yaml
from dandi.dandiapi import DandiAPIClient

from linc_convert.modalities.lsm.cli import lsm
from linc_convert.modalities.lsm.preprocessing_utils.corrections import (
    apply_channel_affine_mask,
    apply_channel_affine_volume,
    apply_corr_zy_lazy,
    crop_volume_channels,
    maybe_flip_z_lazy,
    skew_correct_volume_x_chunk,
    skew_shear_amount,
    skew_shear_x_padding,
    StripeStatsAccumulator,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _crop_mask_y(mask: np.ndarray, reference_local_crop: Optional[dict]) -> np.ndarray:
    """Apply the same Y-crop `reference_local_crop` gives the volume
    to the (Y, X) mask -- mask has no Z axis, so only Y is cropped.
    """
    if reference_local_crop is None:
        return mask
    y1, y2 = reference_local_crop["y_start"], reference_local_crop["y_end"]
    return mask[y1:y2, :]


def _load_and_register_mask(
    name, mip_dir, cam_info, ch, mip_pre_split, x_range, channel_affine, reader,
):
    """Load this tile/channel's mask+threshold once (covers the whole
    X range as a 2D array) and, if registration is active, register
    it once -- mask registration is X-identity, so it doesn't need
    chunking, unlike the 3D volume.
    """
    masks, thrs = load_mask_and_thresholds(
        name, mip_dir, cam_info, reader=reader, ch=ch, pre_split=mip_pre_split, x_range=x_range,
    )
    mask = masks[ch]
    thr = thrs[ch]
    if channel_affine is not None and not np.allclose(channel_affine, np.eye(3)):
        mask = apply_channel_affine_mask(mask, channel_affine)
    return mask, thr


def _read_register_crop_x_chunk(
    reader, cam_info, ch, x0, x1, *,
    pre_flip, channel_affine, affine_order, reference_local_crop,
):
    """
    Read, flip, register, and Z/Y-crop one channel's X-chunk [x0, x1)
    from an already-open tile reader. Returns a plain numpy array
    (Z, Y, x1-x0). Shared core step used by both the stats-gathering
    pass and the correction/write pass.
    """
    raw_chunk = crop_volume_channels(
        reader[:, :, x0:x1], cam_info, channels=ch)[ch]

    if pre_flip is not None:
        raw_chunk = maybe_flip_z_lazy(raw_chunk, pre_flip)

    if channel_affine is not None and not np.allclose(channel_affine, np.eye(3)):
        registered = apply_channel_affine_volume(
            raw_chunk, channel_affine, order=affine_order)
        vol_chunk = np.asarray(registered.compute())
    else:
        vol_chunk = np.asarray(raw_chunk.compute())

    if reference_local_crop is not None:
        y1, y2 = reference_local_crop["y_start"], reference_local_crop["y_end"]
        z1, z2 = reference_local_crop["z_start"], reference_local_crop["z_end"]
        vol_chunk = vol_chunk[z1:z2, y1:y2, :]

    return vol_chunk


def _gather_stripe_stats_x_chunked(
    reader, cam_info, ch, mask, threshold, tissue_frac_min, x_total, x_chunk_size,
    *, pre_flip, channel_affine, affine_order, reference_local_crop, log_prefix="",
    stripe_x_stride=8,
):
    """Pass 1: gather stripe-correction stats across X-chunks. `mask`
    must already be registered (see `_load_and_register_mask`) if
    registration is active.
    """

    if x_chunk_size == -1:
        x_chunk_size = x_total
    mask = _crop_mask_y(mask, reference_local_crop)
    acc = StripeStatsAccumulator(
        tissue_frac_min, threshold, x_stride=stripe_x_stride)
    n_chunks = -(-x_total // x_chunk_size)  # ceil division
    x0 = 0
    chunk_idx = 0
    while x0 < x_total:
        x1 = min(x_total, x0 + x_chunk_size)
        t_chunk = time.time()
        vol_chunk = _read_register_crop_x_chunk(
            reader, cam_info, ch, x0, x1,
            pre_flip=pre_flip, channel_affine=channel_affine,
            affine_order=affine_order, reference_local_crop=reference_local_crop,
        )
        mask_chunk = mask[:, x0:x1]
        acc.add_chunk(vol_chunk, mask_chunk, x0)
        chunk_idx += 1
        logger.info(
            f"{log_prefix}pass1 x-chunk {chunk_idx}/{n_chunks} "
            f"[{x0}:{x1}]: {time.time() - t_chunk:.2f}s"
        )
        x0 = x1
    return acc.finalize()


def _iter_corrected_x_chunks(
    reader, cam_info, ch, mask, corr_zy, x_total_input, x_chunk_size, z_final,
    scan_parameters, camera_id,
    *, pre_flip, channel_affine, affine_order, reference_local_crop, force_flip,
    log_prefix="",
):
    """
    Pass 2: for each OUTPUT X-chunk, read a padded INPUT range, apply
    flip -> register -> crop -> stripe correction -> skew shear, trim
    to this chunk's own output range, and yield it. `mask` must
    already be registered (see `_load_and_register_mask`).

    Yields
    ------
    (xo0, xo1, chunk) : (int, int, np.ndarray)
        chunk has shape (Z_final, Y_final, xo1 - xo0).
    """
    mask = _crop_mask_y(mask, reference_local_crop)
    shear = skew_shear_amount(scan_parameters)
    pad = skew_shear_x_padding(scan_parameters, z_final)
    x_out_total = int(np.ceil(shear * z_final)) + x_total_input
    n_chunks = -(-x_out_total // x_chunk_size)  # ceil division

    xo0 = 0
    chunk_idx = 0
    while xo0 < x_out_total:
        xo1 = min(x_out_total, xo0 + x_chunk_size)
        px0 = max(0, xo0 - pad)
        px1 = min(x_total_input, xo1 + pad)
        pad_left_actual = xo0 - px0
        t_chunk = time.time()

        vol_chunk = _read_register_crop_x_chunk(
            reader, cam_info, ch, px0, px1,
            pre_flip=pre_flip, channel_affine=channel_affine,
            affine_order=affine_order, reference_local_crop=reference_local_crop,
        )
        mask_chunk = mask[:, px0:px1]
        mask_b = np.broadcast_to(mask_chunk.astype(bool)[
                                 None], vol_chunk.shape)
        masked_for_corr = np.where(mask_b, vol_chunk, 0)

        corrected = apply_corr_zy_lazy(da.from_array(masked_for_corr), corr_zy)
        corrected = np.asarray(corrected.compute())

        sheared_core = skew_correct_volume_x_chunk(
            corrected, scan_parameters, camera_id,
            pad_left=pad_left_actual, out_width=xo1 - xo0, force_flip=force_flip,
        )
        chunk_idx += 1
        logger.info(
            f"{log_prefix}pass2 x-chunk {chunk_idx}/{n_chunks} "
            f"[{xo0}:{xo1}]: {time.time() - t_chunk:.2f}s"
        )
        yield xo0, xo1, sheared_core
        xo0 = xo1


def _determine_final_shape(
    tile_paths, cam_info, ch, reference_local_crop, scan_parameters,
    *, dandiset_id, api_key,
):
    """
    Determine the final (Z, Y, X) shape for this channel's output, in
    closed form from crop metadata + the shear formula -- no need to
    actually process a "sample" tile's pixel data first. Only reads
    one tile's `.shape` (metadata, no data movement).
    """
    sample_reader = open_tile_reader(
        tile_paths[0], dandiset_id=dandiset_id, api_key=api_key)
    ch_info = next(m for m in cam_info if m["channel"] == ch)

    x_total_input = sample_reader.shape[2]

    if reference_local_crop is not None:
        y_final = reference_local_crop["y_end"] - \
            reference_local_crop["y_start"]
        z_final = reference_local_crop["z_end"] - \
            reference_local_crop["z_start"]
    else:
        y_final = ch_info["y_end"] - ch_info["y_start"]
        if ch_info["z_start"] is not None:
            z_final = ch_info["z_end"] - ch_info["z_start"]
        else:
            z_final = sample_reader.shape[0]

    shear = skew_shear_amount(scan_parameters)
    x_final = int(np.ceil(shear * z_final)) + x_total_input

    return z_final, y_final, x_final, x_total_input


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


def tile_name(path: str) -> str:
    return os.path.basename(path.rstrip("/").replace(".ome.zarr", ""))


def _setup_channel_processing(
    inp, yaml_path, camera_id, slice_number, channel_affines_path, reference_channel,
    dandiset_id, api_key,
):
    """
    Shared setup used by `pipeline()`, `pass1`, `pass2`, and
    `finalize`: loads scan parameters, resolves `cam_info`/crop_stage,
    registration setup (`reference_local_crop`, `pre_flip`/
    `force_flip`), and discovers this camera's tile paths.

    Returns
    -------
    dict
        Keys: scan_parameters, channel_affines, cam_info,
        reference_local_crop, pre_flip, force_flip, tile_paths,
        channels.
    """
    scan_parameters = load_scan_parameters(yaml_path)

    channel_affines = (
        load_channel_affines(channel_affines_path, reference_channel)
        if channel_affines_path else None
    )

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
        pre_flip = bool(
            scan_parameters["channelLayout"][f"Camera{camera_id}"]["verticalFlip"])
        force_flip = False
    else:
        reference_local_crop = None
        pre_flip = None
        force_flip = None

    tile_paths = discover_tile_paths(
        inp, camera_id, dandiset_id=dandiset_id, api_key=api_key)

    channels = get_channel_names(scan_parameters, camera_id)
    if len(channels) != 2:
        raise ValueError(
            f"Expected exactly 2 channels for camera {camera_id}, got {len(channels)}: {channels}"
        )

    return dict(
        scan_parameters=scan_parameters,
        channel_affines=channel_affines,
        cam_info=cam_info,
        reference_local_crop=reference_local_crop,
        pre_flip=pre_flip,
        force_flip=force_flip,
        tile_paths=tile_paths,
        channels=channels,
    )


def _corr_zy_tiff_path(corr_zy_dir: str, ch: str, name: str) -> str:
    """Mirrors the existing MIP naming convention: {ch}_{name}_corr-zy.tiff"""
    return os.path.join(corr_zy_dir, f"{ch}_{name}_corr-zy.tiff")


def _save_corr_zy(corr_zy_dir: str, ch: str, name: str, corr_zy: np.ndarray) -> None:
    os.makedirs(corr_zy_dir, exist_ok=True)
    tifffile.imwrite(_corr_zy_tiff_path(
        corr_zy_dir, ch, name), corr_zy.astype(np.float32))


def _load_corr_zy(corr_zy_dir: str, ch: str, name: str) -> np.ndarray:
    path = _corr_zy_tiff_path(corr_zy_dir, ch, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing corr_zy file: {path}")
    return tifffile.imread(path).astype(np.float32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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
    x_chunk_size: int = 2048,
    voxel_size: List[float] = [1, 1, 1],
    general_config: Optional[GeneralConfig] = None,
    zarr_config: Optional[ZarrConfig] = None,
    nii_config: Optional[NiftiConfig] = None,
    dandiset_id: Optional[str] = None,
    chunk_min: Optional[int] = None,
    chunk_max: Optional[int] = None,
    channel_affines_path: Optional[str] = None,
    reference_channel: str = "488",
    x_range: int = 5000,
    affine_order: int = 1,
    tissue_frac_min: float = 0.02,
    stripe_x_stride: int = 128,
) -> None:
    """
    Correct volumetric tile data and stream it directly into a single
    blended OME-Zarr mosaic along the y axis.

    Chunked along X, not Y (see module docstring) -- X can be
    enormous for this dataset; Y is not, so each X-chunk is processed
    across its tile's full Y range in one shot. Tile-to-tile Y overlap
    is blended per X-chunk.

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
    x_chunk_size : int, default=2048
        Width, in pixels along X, of each chunk processed at a time.
    affine_order : int, default=1
        Spline interpolation order for the registration affine. Lower
        orders (1 = linear, 0 = nearest) are cheaper and don't suffer
        the global cubic-spline prefiltering cost that order=3 does.
    tissue_frac_min : float, default=0.02
        Minimum valid-pixel fraction for the stripe-correction map
        (see `StripeStatsAccumulator`/`compute_corr_zy`).
    stripe_x_stride : int, default=128
        Subsampling stride along X used only for the stripe-correction
        map's median (not the valid-pixel count, which always uses
        every X value). Larger values subsample more aggressively.
    """
    if camera_id not in (1, 2):
        raise ValueError(f"camera_id must be 1 or 2, got {camera_id}")

    start_timer = time.time()
    voxel_size = list(map(float, reversed(voxel_size)))

    api_key = prompt_dandi_api_key() if dandiset_id else None

    setup = _setup_channel_processing(
        inp, yaml_path, camera_id, slice_number, channel_affines_path, reference_channel,
        dandiset_id, api_key,
    )
    scan_parameters = setup["scan_parameters"]
    channel_affines = setup["channel_affines"]
    cam_info = setup["cam_info"]
    reference_local_crop = setup["reference_local_crop"]
    pre_flip = setup["pre_flip"]
    force_flip = setup["force_flip"]
    tile_paths = setup["tile_paths"]
    channels = setup["channels"]
    num_tiles = len(tile_paths)

    skew_delta_deg = scan_parameters["acquisitionSettings"]["skewCorrection"]["delta_deg"]
    voxel_size[0] = voxel_size[0] / np.cos(np.deg2rad(skew_delta_deg))

    coords_yaml_by_channel = dict(
        zip(channels, [coords_yaml_ch1, coords_yaml_ch2]))

    for ch in channels:
        channel_timer = time.time()
        channel_affine = (
            channel_affines.get(ch, np.eye(
                3)) if channel_affines is not None else None
        )

        y_coords = load_y_coordinates(coords_yaml_by_channel[ch])

        z_final, y_final, x_final, x_total_input = _determine_final_shape(
            tile_paths, cam_info, ch, reference_local_crop, scan_parameters,
            dandiset_id=dandiset_id, api_key=api_key,
        )
        logger.info(
            f"[{ch}] final shape (closed-form): Z={z_final}, Y={y_final}, "
            f"X={x_final} (raw input X={x_total_input})"
        )

        full_x = x_final
        full_y = int(round(y_coords[-1])) + y_final
        full_z = z_final
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

        # Trailing Y-overlap withheld from the PREVIOUS tile, awaiting
        # blending with the next tile's leading edge -- keyed by each
        # X-chunk's own starting position, since the same X-chunk
        # boundaries recur for every tile (fixed x_total_input,
        # x_chunk_size per channel).
        carry: Dict[int, Optional[np.ndarray]] = {}

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

                reader = open_tile_reader(
                    path, dandiset_id=dandiset_id, api_key=api_key)

                t_mask = time.time()
                mask, thr = _load_and_register_mask(
                    name, mip_dir, cam_info, ch, mip_pre_split, x_range, channel_affine, reader,
                )
                t_mask = time.time() - t_mask

                t_stats = time.time()
                corr_zy = _gather_stripe_stats_x_chunked(
                    reader, cam_info, ch, mask, thr, tissue_frac_min,
                    x_total_input, x_chunk_size,
                    pre_flip=pre_flip, channel_affine=channel_affine,
                    affine_order=affine_order, reference_local_crop=reference_local_crop,
                    log_prefix=f"[{index}] {name}/{ch} ",
                    stripe_x_stride=stripe_x_stride,
                )
                t_stats = time.time() - t_stats
                logger.info(
                    f"[{index}] {name}/{ch} -- mask/mip: {t_mask:.2f}s, "
                    f"stripe stats (pass 1): {t_stats:.2f}s"
                )

                is_first = index == 0 or index == checkpoint or (
                    chunk_min is not None and index == chunk_min)
                is_last = index == num_tiles - 1 or index == effective_max

                ystart = int(round(y_coords[index]))

                overlap_with_prev = 0
                if not is_first:
                    overlap_with_prev = y_final - (
                        int(round(y_coords[index])) -
                        int(round(y_coords[index - 1]))
                    )
                overlap_with_next = 0
                if not is_last:
                    overlap_with_next = y_final - (
                        int(round(y_coords[index + 1])) -
                        int(round(y_coords[index]))
                    )
                overlap_with_prev = max(overlap_with_prev, 0)
                overlap_with_next = max(overlap_with_next, 0)
                withhold_from = y_final - overlap_with_next

                ramp = ramp_inverse = None
                if overlap_with_prev > 0:
                    t = np.linspace(0, 1, overlap_with_prev)
                    ramp = (1 - np.cos(np.pi * t)) / 2
                    ramp_inverse = (1 + np.cos(np.pi * t)) / 2
                    ramp = ramp[None, :, None]
                    ramp_inverse = ramp_inverse[None, :, None]

                new_carry: Dict[int, Optional[np.ndarray]] = {}

                for xo0, xo1, data in _iter_corrected_x_chunks(
                    reader, cam_info, ch, mask, corr_zy, x_total_input, x_chunk_size,
                    z_final, scan_parameters, camera_id,
                    pre_flip=pre_flip, channel_affine=channel_affine,
                    affine_order=affine_order, reference_local_crop=reference_local_crop,
                    force_flip=force_flip,
                    log_prefix=f"[{index}] {name}/{ch} ",
                ):
                    if overlap_with_prev > 0:
                        prev_carry = carry.get(xo0)
                        if prev_carry is not None:
                            if prev_carry.shape[1] != overlap_with_prev:
                                logger.warning(
                                    f"[{index}] {name}/{ch} x-chunk {xo0}: "
                                    f"previous tile's withheld overlap "
                                    f"({prev_carry.shape[1]} rows) doesn't match "
                                    f"this tile's expected overlap_with_prev "
                                    f"({overlap_with_prev} rows) -- blending only "
                                    f"the smaller of the two; check y_coords for "
                                    f"an irregular gap between these two tiles."
                                )
                            blend_len = min(
                                data.shape[1], overlap_with_prev, prev_carry.shape[1])
                            data[:, :blend_len, :] = (
                                prev_carry[:, :blend_len, :] *
                                ramp_inverse[:, :blend_len, :]
                                + data[:, :blend_len, :] *
                                ramp[:, :blend_len, :]
                            )

                    if withhold_from >= data.shape[1]:
                        to_write, to_withhold = data, None
                    elif withhold_from <= 0:
                        to_write, to_withhold = None, data
                    else:
                        to_write = data[:, :withhold_from, :]
                        to_withhold = data[:, withhold_from:, :]

                    if to_write is not None and to_write.shape[1] > 0 and index > checkpoint:
                        array[
                            0: to_write.shape[0],
                            ystart: ystart + to_write.shape[1],
                            xo0: xo0 + to_write.shape[2],
                        ] = to_write

                    new_carry[xo0] = to_withhold

                carry = new_carry
                _write_checkpoint(checkpoint_file, index)
                logger.info(f"{name} done in {time.time() - tile_timer:.2f}s")

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


# ---------------------------------------------------------------------------
# Parallelized entry points: pass1 (stats) / pass2 (SLURM array, per X-chunk)
# / finalize (pyramid + metadata) -- an alternative to the single-process
# `pipeline()` above for very large X, allowing pass2 to run as an
# embarrassingly-parallel SLURM array (each task handles one disjoint
# X-chunk across all tiles; tasks never need to coordinate with each other).
# ---------------------------------------------------------------------------

pass1 = cyclopts.App(name="pass1", help_format="markdown")
lsm.command(pass1)


@pass1.default
@autoconfig
def pass1_cmd(
    inp: str,
    yaml_path: str,
    camera_id: int,
    slice_number: int,
    coords_yaml_ch1: str,
    coords_yaml_ch2: str,
    corr_zy_dir: str,
    *,
    mip_dir: Optional[str] = None,
    mip_pre_split: bool = False,
    pass1_x_chunk_size: int = -1,
    general_config: Optional[GeneralConfig] = None,
    zarr_config: Optional[ZarrConfig] = None,
    dandiset_id: Optional[str] = None,
    channel_affines_path: Optional[str] = None,
    reference_channel: str = "488",
    x_range: int = 5000,
    affine_order: int = 1,
    tissue_frac_min: float = 0.02,
    stripe_x_stride: int = 128,
) -> None:
    """
    Pass 1 (standalone, single process): for every tile of a channel,
    gather stripe-correction stats (see `_gather_stripe_stats_x_chunked`)
    and save each tile's (Z, Y) corr_zy map to a `.tiff` file, named
    `{ch}_{tile_name}_corr-zy.tiff` in `corr_zy_dir`, for `pass2` to
    load later instead of recomputing.

    Also creates the final output zarr array (with the correct shape),
    since `pass2`'s parallel X-chunk tasks all write into the same
    array and can't safely race to create it themselves.

    Run this once per camera/slice, before launching the `pass2` SLURM
    array.
    """
    if camera_id not in (1, 2):
        raise ValueError(f"camera_id must be 1 or 2, got {camera_id}")

    api_key = prompt_dandi_api_key() if dandiset_id else None

    setup = _setup_channel_processing(
        inp, yaml_path, camera_id, slice_number, channel_affines_path, reference_channel,
        dandiset_id, api_key,
    )
    scan_parameters = setup["scan_parameters"]
    channel_affines = setup["channel_affines"]
    cam_info = setup["cam_info"]
    reference_local_crop = setup["reference_local_crop"]
    pre_flip = setup["pre_flip"]
    tile_paths = setup["tile_paths"]
    channels = setup["channels"]

    coords_yaml_by_channel = dict(
        zip(channels, [coords_yaml_ch1, coords_yaml_ch2]))

    for ch in channels:
        channel_affine = (
            channel_affines.get(ch, np.eye(
                3)) if channel_affines is not None else None
        )
        y_coords = load_y_coordinates(coords_yaml_by_channel[ch])

        z_final, y_final, x_final, x_total_input = _determine_final_shape(
            tile_paths, cam_info, ch, reference_local_crop, scan_parameters,
            dandiset_id=dandiset_id, api_key=api_key,
        )
        logger.info(
            f"[{ch}] final shape (closed-form): Z={z_final}, Y={y_final}, "
            f"X={x_final} (raw input X={x_total_input})"
        )

        full_y = int(round(y_coords[-1])) + y_final
        fullshape = (z_final, full_y, x_final)

        out_dir = f"{general_config.out}/{ch}"
        omz = ZarrPythonGroup.from_config(out_dir, zarr_config)
        try:
            omz.create_array("0", shape=fullshape,
                             zarr_config=zarr_config, dtype=np.uint16)
            logger.info(f"[{ch}] created output array with shape {fullshape}")
        except Exception:
            logger.info(
                f"[{ch}] output array already exists (shape should be {fullshape})")

        for index, path in enumerate(tile_paths):
            name = tile_name(path)
            corr_zy_path = _corr_zy_tiff_path(corr_zy_dir, ch, name)
            if os.path.exists(corr_zy_path):
                logger.info(
                    f"[{index}] {name}/{ch}: corr_zy already exists, skipping")
                continue

            reader = open_tile_reader(
                path, dandiset_id=dandiset_id, api_key=api_key)
            mask, thr = _load_and_register_mask(
                name, mip_dir, cam_info, ch, mip_pre_split, x_range, channel_affine, reader,
            )

            t0 = time.time()
            corr_zy = _gather_stripe_stats_x_chunked(
                reader, cam_info, ch, mask, thr, tissue_frac_min,
                x_total_input, pass1_x_chunk_size,
                pre_flip=pre_flip, channel_affine=channel_affine,
                affine_order=affine_order, reference_local_crop=reference_local_crop,
                log_prefix=f"[{index}] {name}/{ch} ",
                stripe_x_stride=stripe_x_stride,
            )
            _save_corr_zy(corr_zy_dir, ch, name, corr_zy)
            logger.info(
                f"[{index}] {name}/{ch}: corr_zy computed and saved in "
                f"{time.time() - t0:.2f}s"
            )


pass2 = cyclopts.App(name="pass2", help_format="markdown")
lsm.command(pass2)


@pass2.default
@autoconfig
def pass2_cmd(
    inp: str,
    yaml_path: str,
    camera_id: int,
    slice_number: int,
    coords_yaml_ch1: str,
    coords_yaml_ch2: str,
    corr_zy_dir: str,
    x_chunk_index: int,
    *,
    mip_dir: Optional[str] = None,
    mip_pre_split: bool = False,
    x_range: int = 5000,
    x_chunk_size: int = 8192,
    checkpoint_dir: Optional[str] = None,
    general_config: Optional[GeneralConfig] = None,
    zarr_config: Optional[ZarrConfig] = None,
    dandiset_id: Optional[str] = None,
    channel_affines_path: Optional[str] = None,
    reference_channel: str = "488",
    affine_order: int = 1,
    chunk_min: Optional[int] = None,
    chunk_max: Optional[int] = None,
) -> None:
    """
    Pass 2 (SLURM-array-friendly): processes ONE X-chunk (given by
    `x_chunk_index`, width `x_chunk_size`) across every tile, in
    Y-order, for both channels of this camera. Reads each tile's
    corr_zy map from `pass1` rather than recomputing it.

    Different X-chunk tasks write disjoint X-ranges of the output
    array and each handles its own tile-to-tile Y-blending internally,
    so tasks never need to coordinate with each other -- this is
    intended to be launched as a SLURM array, one task per
    `x_chunk_index`.

    `pass1` must have already run (to create the output array and the
    corr_zy files) before any `pass2` task starts.

    Checkpointing is per (channel, x_chunk_index): a preempted task
    resumes on its own, independent of every other task, by
    reprocessing (but not rewriting) its last-completed tile to
    regenerate that tile's withheld Y-overlap before continuing --
    the same pattern `pipeline()`'s single-process loop uses.
    """
    if camera_id not in (1, 2):
        raise ValueError(f"camera_id must be 1 or 2, got {camera_id}")

    api_key = prompt_dandi_api_key() if dandiset_id else None

    setup = _setup_channel_processing(
        inp, yaml_path, camera_id, slice_number, channel_affines_path, reference_channel,
        dandiset_id, api_key,
    )
    scan_parameters = setup["scan_parameters"]
    channel_affines = setup["channel_affines"]
    cam_info = setup["cam_info"]
    reference_local_crop = setup["reference_local_crop"]
    pre_flip = setup["pre_flip"]
    force_flip = setup["force_flip"]
    tile_paths = setup["tile_paths"]
    channels = setup["channels"]
    num_tiles = len(tile_paths)

    coords_yaml_by_channel = dict(
        zip(channels, [coords_yaml_ch1, coords_yaml_ch2]))

    for ch in channels:
        channel_affine = (
            channel_affines.get(ch, np.eye(
                3)) if channel_affines is not None else None
        )
        y_coords = load_y_coordinates(coords_yaml_by_channel[ch])

        z_final, y_final, x_final, x_total_input = _determine_final_shape(
            tile_paths, cam_info, ch, reference_local_crop, scan_parameters,
            dandiset_id=dandiset_id, api_key=api_key,
        )

        xo0 = x_chunk_index * x_chunk_size
        if xo0 >= x_final:
            logger.info(
                f"[{ch}] x_chunk_index {x_chunk_index} starts at {xo0}, beyond "
                f"x_final={x_final} -- nothing to do for this channel."
            )
            continue
        xo1 = min(x_final, xo0 + x_chunk_size)

        out_dir = f"{general_config.out}/{ch}"
        omz = ZarrPythonGroup.from_config(out_dir, zarr_config)
        array = omz["0"]  # must already exist -- created by pass1

        ckpt_base = _checkpoint_path(general_config, ch)
        checkpoint_file = (
            f"{ckpt_base[:-4]}_xchunk{x_chunk_index}.dat"
            if ckpt_base.endswith(".dat") else f"{ckpt_base}_xchunk{x_chunk_index}.dat"
        )
        if checkpoint_dir is not None:
            checkpoint_file = os.path.join(
                checkpoint_dir, os.path.basename(checkpoint_file))
        checkpoint = _read_checkpoint(checkpoint_file, -1)

        effective_min = max(chunk_min if chunk_min is not None else 0, 0)
        effective_max = min(
            chunk_max if chunk_max is not None else num_tiles - 1, num_tiles - 1)

        shear = skew_shear_amount(scan_parameters)
        pad = skew_shear_x_padding(scan_parameters, z_final)
        px0 = max(0, xo0 - pad)
        px1 = min(x_total_input, xo1 + pad)
        pad_left_actual = xo0 - px0

        carry: Optional[np.ndarray] = None

        if num_tiles > checkpoint + 1:
            for index, path in enumerate(tile_paths):
                gc.collect()
                if index < checkpoint or index < effective_min:
                    continue
                if index > effective_max:
                    break

                name = tile_name(path)
                tile_timer = time.time()

                reader = open_tile_reader(
                    path, dandiset_id=dandiset_id, api_key=api_key)
                corr_zy = _load_corr_zy(corr_zy_dir, ch, name)
                mask, _ = _load_and_register_mask(
                    name, mip_dir, cam_info, ch, mip_pre_split, x_range, channel_affine, reader,
                )

                vol_chunk = _read_register_crop_x_chunk(
                    reader, cam_info, ch, px0, px1,
                    pre_flip=pre_flip, channel_affine=channel_affine,
                    affine_order=affine_order, reference_local_crop=reference_local_crop,
                )
                mask_cropped = _crop_mask_y(mask, reference_local_crop)
                mask_chunk = mask_cropped[:, px0:px1]
                mask_b = np.broadcast_to(mask_chunk.astype(bool)[
                                         None], vol_chunk.shape)
                masked_for_corr = np.where(mask_b, vol_chunk, 0)

                corrected = apply_corr_zy_lazy(
                    da.from_array(masked_for_corr), corr_zy)
                corrected = np.asarray(corrected.compute())

                data = skew_correct_volume_x_chunk(
                    corrected, scan_parameters, camera_id,
                    pad_left=pad_left_actual, out_width=xo1 - xo0, force_flip=force_flip,
                )

                is_first = index == 0 or index == checkpoint or (
                    chunk_min is not None and index == chunk_min)
                is_last = index == num_tiles - 1 or index == effective_max

                ystart = int(round(y_coords[index]))

                overlap_with_prev = 0
                if not is_first:
                    overlap_with_prev = y_final - (
                        int(round(y_coords[index])) -
                        int(round(y_coords[index - 1]))
                    )
                overlap_with_next = 0
                if not is_last:
                    overlap_with_next = y_final - (
                        int(round(y_coords[index + 1])) -
                        int(round(y_coords[index]))
                    )
                overlap_with_prev = max(overlap_with_prev, 0)
                overlap_with_next = max(overlap_with_next, 0)
                withhold_from = y_final - overlap_with_next

                if overlap_with_prev > 0 and carry is not None:
                    if carry.shape[1] != overlap_with_prev:
                        logger.warning(
                            f"[{index}] {name}/{ch} x-chunk {x_chunk_index}: "
                            f"previous tile's withheld overlap ({carry.shape[1]} "
                            f"rows) doesn't match this tile's expected "
                            f"overlap_with_prev ({overlap_with_prev} rows) -- "
                            f"blending only the smaller of the two."
                        )
                    blend_len = min(
                        data.shape[1], overlap_with_prev, carry.shape[1])
                    t = np.linspace(0, 1, overlap_with_prev)
                    ramp = ((1 - np.cos(np.pi * t)) / 2)[None, :, None]
                    ramp_inverse = ((1 + np.cos(np.pi * t)) / 2)[None, :, None]
                    data[:, :blend_len, :] = (
                        carry[:, :blend_len, :] *
                        ramp_inverse[:, :blend_len, :]
                        + data[:, :blend_len, :] * ramp[:, :blend_len, :]
                    )

                if withhold_from >= data.shape[1]:
                    to_write, to_withhold = data, None
                elif withhold_from <= 0:
                    to_write, to_withhold = None, data
                else:
                    to_write = data[:, :withhold_from, :]
                    to_withhold = data[:, withhold_from:, :]

                if to_write is not None and to_write.shape[1] > 0 and index > checkpoint:
                    array[
                        0: to_write.shape[0],
                        ystart: ystart + to_write.shape[1],
                        xo0: xo0 + to_write.shape[2],
                    ] = to_write

                carry = to_withhold
                _write_checkpoint(checkpoint_file, index)
                logger.info(
                    f"[{index}] {name}/{ch} x-chunk {x_chunk_index} "
                    f"[{xo0}:{xo1}] done in {time.time() - tile_timer:.2f}s"
                )


finalize = cyclopts.App(name="finalize", help_format="markdown")
lsm.command(finalize)


@finalize.default
@autoconfig
def finalize_cmd(
    inp: str,
    yaml_path: str,
    camera_id: int,
    slice_number: int,
    coords_yaml_ch1: str,
    coords_yaml_ch2: str,
    checkpoint_dir: str,
    *,
    x_chunk_size: int = 8192,
    voxel_size: List[float] = [1, 1, 1],
    general_config: Optional[GeneralConfig] = None,
    zarr_config: Optional[ZarrConfig] = None,
    nii_config: Optional[NiftiConfig] = None,
    dandiset_id: Optional[str] = None,
    channel_affines_path: Optional[str] = None,
    reference_channel: str = "488",
    skip_verification: bool = False,
) -> None:
    """
    Finalize: run once per camera/slice, after ALL `pass2` X-chunk
    tasks have completed for both channels. Generates the pyramid,
    OME metadata, and (optionally) NIfTI header -- these need the
    complete level-0 array, so they can't run per-X-chunk inside
    `pass2`.

    Before doing so, verifies that every expected X-chunk actually
    finished (its checkpoint file exists and shows the last tile index
    completed) -- independent of whatever a SLURM `--dependency`
    already checked. `--dependency=afterok` on the pass2 array job
    should already prevent this command from running at all if any
    task failed or was preempted without recovering, but that
    protection is silent (a cancelled-by-dependency job just
    disappears from the queue) and only catches SLURM-visible
    failures, not e.g. a task that exited cleanly without actually
    processing every tile it was assigned. This check catches both.

    Parameters
    ----------
    skip_verification : bool, default=False
        Skip the completeness check and generate the pyramid
        regardless. Not recommended -- only for manual recovery
        situations where you've already confirmed completeness some
        other way.
    """
    if camera_id not in (1, 2):
        raise ValueError(f"camera_id must be 1 or 2, got {camera_id}")

    api_key = prompt_dandi_api_key() if dandiset_id else None

    setup = _setup_channel_processing(
        inp, yaml_path, camera_id, slice_number, channel_affines_path, reference_channel,
        dandiset_id, api_key,
    )
    scan_parameters = setup["scan_parameters"]
    cam_info = setup["cam_info"]
    reference_local_crop = setup["reference_local_crop"]
    tile_paths = setup["tile_paths"]
    channels = setup["channels"]
    num_tiles = len(tile_paths)
    last_tile_index = num_tiles - 1

    coords_yaml_by_channel = dict(
        zip(channels, [coords_yaml_ch1, coords_yaml_ch2]))

    skew_delta_deg = scan_parameters["acquisitionSettings"]["skewCorrection"]["delta_deg"]
    voxel_size = list(map(float, reversed(voxel_size)))
    voxel_size[0] = voxel_size[0] / np.cos(np.deg2rad(skew_delta_deg))

    if not skip_verification:
        problems = []
        for ch in channels:
            y_coords = load_y_coordinates(coords_yaml_by_channel[ch])
            _, _, x_final, _ = _determine_final_shape(
                tile_paths, cam_info, ch, reference_local_crop, scan_parameters,
                dandiset_id=dandiset_id, api_key=api_key,
            )
            n_x_chunks = -(-x_final // x_chunk_size)  # ceil division

            for x_chunk_index in range(n_x_chunks):
                ckpt_base = _checkpoint_path(general_config, ch)
                checkpoint_file = (
                    f"{ckpt_base[:-4]}_xchunk{x_chunk_index}.dat"
                    if ckpt_base.endswith(".dat") else f"{ckpt_base}_xchunk{x_chunk_index}.dat"
                )
                if checkpoint_dir is not None:
                    checkpoint_file = os.path.join(
                        checkpoint_dir, os.path.basename(checkpoint_file))

                completed = _read_checkpoint(checkpoint_file, default_y=None)
                if completed is None:
                    problems.append(
                        f"[{ch}] x_chunk_index={x_chunk_index}: no checkpoint file "
                        f"found at {checkpoint_file} (pass2 task never started or "
                        f"never completed a single tile)"
                    )
                elif completed < last_tile_index:
                    problems.append(
                        f"[{ch}] x_chunk_index={x_chunk_index}: checkpoint shows "
                        f"tile {completed} as last completed, expected "
                        f"{last_tile_index} (last tile) -- this X-chunk did not "
                        f"finish all tiles"
                    )

        if problems:
            raise RuntimeError(
                "finalize: pass2 does not appear to be fully complete for this "
                "camera/slice -- refusing to generate the pyramid from "
                "incomplete data. Problems found:\n  " + "\n  ".join(problems) +
                "\n\nRe-run (or requeue) the missing/incomplete pass2 X-chunk "
                "tasks, then retry finalize. If you're certain this is a false "
                "positive, pass --skip_verification to bypass this check."
            )
        logger.info(
            f"Verified all X-chunks completed for channels {channels} "
            f"(last tile index {last_tile_index})."
        )

    channels = get_channel_names(scan_parameters, camera_id)

    for ch in channels:
        out_dir = f"{general_config.out}/{ch}"
        omz = ZarrPythonGroup.from_config(out_dir, zarr_config)
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

        logger.info(f"[{ch}] finalized (pyramid + OME metadata written)")
