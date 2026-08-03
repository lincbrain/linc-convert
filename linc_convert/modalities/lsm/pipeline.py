"""Preprocess tiles (background removal, stripe/skew correction) and stream
the corrected volumes directly into a single blended OME-Zarr mosaic along
the y axis, without writing per-tile intermediates to disk.

This merges the per-tile correction pipeline (formerly ``preprocess.py``)
with the tile-mosaicking/blending pipeline (formerly
``convert_spool_or_zarr.py``).

Assumptions baked into this version:
- Tiles are stacked along a single axis (y) and are read in the order
  they're discovered on disk; there's no per-tile (y, z) identity to parse,
  so no filename regex is needed.
- Every tile has the same shape, so we don't need to pre-scan all of them;
  we only correct one tile up front to learn the corrected mosaic geometry.
- Each tile's reader (and per-channel mask) is opened *once*. The tile is
  then walked in fixed-size Y-chunks (`y_chunk_size` rows): each chunk is
  sliced from that single already-open lazy volume, corrected, optionally
  blended against the previous/next tile's overlap, written to the
  destination array, and discarded before moving to the next chunk. Since
  none of the four correction steps mix Y with Z or X (the skew shear only
  couples Z and X), each chunk's correction is exact for whatever rows
  it's given -- there's no need to process a tile's full Y extent at once,
  and no redundant re-reading of the same bytes across chunks, since each
  chunk's bytes are read exactly once from the single open reader.
- Inter-tile overlap can span more than one Y-chunk; rows that fall in an
  overlap region are held in a small carry-over buffer (full tile width,
  but only as tall as the overlap itself) rather than written immediately,
  until they've been blended with the matching rows of the neighboring
  tile.
- Cross-camera affine registration (``get_all_affines``) is removed; it
  isn't currently working, so this pipeline only does background removal
  and stripe/skew correction.
- Per-tile y placement comes from a coordinates YAML file (one per
  channel) rather than a single constant overlap value.
- Scan/crop parameters come from a YAML file whose crop definitions are
  organized into slice-range "configEpochs" (see
  ``preprocessing_utils.io``). A single pipeline run processes tiles
  belonging to one physical slice, given via `slice_number`, and the
  matching configEpoch's crop definitions are used for the whole run.
"""

import gc
import getpass
import logging
import os
import tifffile
import time
import warnings
from dataclasses import replace
from glob import glob
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Tuple

import cyclopts
import dask.array as da
import numpy as np
import yaml
from dandi.dandiapi import DandiAPIClient
from dask.diagnostics import ProgressBar

from linc_convert.modalities.lsm.cli import lsm
from linc_convert.modalities.lsm.preprocessing_utils.corrections import (
    apply_affine_split,
    apply_corr_zy_lazy,
    compute_alt_zy_calibration_for_tile,
    compute_corr_zy,
    crop_volume_channels,
    embed_zy_affine_for_volume,
    generate_skew_affine,
    get_crop_values,
    stripe_skew_corr,
)
from linc_convert.modalities.lsm.preprocessing_utils.io import (
    find_camera_for_channel,
    get_camera_info,
    get_channel_names,
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
    """Check for dandi api key and prompt user if key not found."""
    key = os.environ.get("DANDI_API_KEY")
    return key if key else getpass.getpass("Enter your DANDI API key: ")


def open_tile_reader(
    path: str,
    *,
    dandiset_id: Optional[str] = None,
    api_key: Optional[str] = None,
    chunks: Optional[Tuple[int, ...]] = None,
    zarr_level: int = 0,
) -> da.Array:
    """Lazily open a tile (ome.zarr or spool set) as a dask array.

    No data is read from disk until the returned array is sliced and
    computed.

    Parameters
    ----------
    zarr_level : int, default=0
        Which OME-Zarr pyramid level (array key, e.g. "0", "1", ...) to
        read. Only meaningful for .ome.zarr tiles -- ignored for spool
        sets, which have no pyramid. Caller is responsible for scaling
        everything resolution-dependent (y_coords, crop bounds, affine
        translations) to match this level -- see `pipeline`'s
        `zarr_level` parameter.
    """
    if path.endswith(".ome.zarr"):
        if dandiset_id is None:
            return da.asarray(
                ZarrPythonGroup.open(path)[str(zarr_level)],
                chunks=chunks if chunks is not None else (128, 128, 128),
            )
        return da.asarray(
            ZarrPythonGroup.open_dandi(
                dandiset_id=dandiset_id,
                asset_path=path,
                api_key=api_key,
            )[str(zarr_level)],
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
    mip_dir: str,
    name: str,
    ch: str,
    cam_info,
    background_length: int = 5000,
    mip_pre_split: bool = False,
    reference_ch=None,
    x_min=None,
    x_max=None,
    zarr_level: int = 0,
    mip_cam_info=None,
) -> Tuple[da.Array, np.ndarray, float]:
    """Open one tile/channel exactly once: the reader, the channel crop,
    and the mask/threshold lookup all happen here, a single time per
    tile, regardless of how many Y-chunks it's later split into.

    Parameters
    ----------
    cam_info : list of dict
        Crop bounds for the 3D volume, in the SAME resolution level
        being read (`zarr_level`) -- i.e. already scaled down if
        `zarr_level > 0`.
    mip_cam_info : list of dict, optional
        Crop bounds for the MIP/mask, in FULL RESOLUTION (unscaled) --
        MIP files are separate TIFFs, not part of the zarr pyramid, so
        they stay full-res regardless of `zarr_level`. The resulting
        mask is then downsampled by `zarr_level`'s factor to match the
        volume's actual resolution -- `vol` and `mask` must have
        matching Y (and X) extents for the correction steps downstream
        that use them together. Defaults to `cam_info` itself when
        omitted (correct for `zarr_level=0`, where scaled and unscaled
        bounds are identical, and no downsampling is needed).
    zarr_level : int, default=0
        Which OME-Zarr pyramid level to read (see `open_tile_reader`).

    Returns
    -------
    vol : dask.array.Array
        Raw, channel-cropped (but not yet corrected) lazy volume
        (Z, Y, X) for the whole tile, at `zarr_level`'s resolution.
    mask : np.ndarray
        Mask for this channel, shape (Y, X) or (Z, Y, X), downsampled
        to match `vol`'s actual Y (and X) extent at `zarr_level`.
    threshold : float
        Intensity threshold for this channel.
    """
    if reference_ch is None:
        reference_ch = ch
    if mip_cam_info is None:
        mip_cam_info = cam_info
    reader = open_tile_reader(
        path, dandiset_id=dandiset_id, api_key=api_key, zarr_level=zarr_level)
    vol_channels = crop_volume_channels(reader, cam_info)
    masks, thrs = load_mask_and_thresholds(
        name, mip_dir, mip_cam_info, background_length=background_length, pre_split=mip_pre_split, ch=ch)

    if zarr_level > 0:
        # Mask came from a full-resolution MIP crop; downsample it to
        # match the volume's actual (already-downsampled) Y, X extent.
        # Strided subsampling, then trimmed/padded to the volume's
        # exact shape to absorb any off-by-one rounding difference
        # between (y_end-y_start)/factor and the strided mask's own
        # size.
        factor = 2 ** zarr_level
        target_y = vol_channels[reference_ch].shape[1]
        target_x = vol_channels[reference_ch].shape[2]
        for out_ch in list(masks.keys()):
            m = masks[out_ch][::factor, ::factor]
            m = m[:target_y, :target_x]
            if m.shape != (target_y, target_x):
                padded = np.zeros((target_y, target_x), dtype=m.dtype)
                padded[:m.shape[0], :m.shape[1]] = m
                m = padded
            masks[out_ch] = m

    if x_min is None and x_max is None:
        return vol_channels[reference_ch], masks[reference_ch], thrs[reference_ch]
    x_min = x_min if x_min is not None else 0
    x_max = x_max if x_max is not None else vol_channels[reference_ch].shape[2]
    return vol_channels[reference_ch][:, :, x_min:x_max], masks[reference_ch][:, x_min:x_max], thrs[reference_ch]


def _corrected_y_chunk(
    vol: da.Array,
    mask: np.ndarray,
    affine: np.ndarray,
    corr_zy: np.ndarray,
    y0: int,
    y1: int,
) -> da.Array:
    """Build the lazy corrected dask array for one Y-chunk [y0, y1) of an
    already-opened tile/channel volume.

    `vol` and `mask` are sliced to [y0, y1) here, then run through the
    full correction pipeline -- since none of the correction steps mix Y
    with Z or X, this is exact (not an approximation) for whatever rows
    are in this chunk; no absolute-position information is needed by the
    correction itself, only for slicing the right rows out of `vol` and
    `mask` in the first place.

    Nothing is read from `vol` until the caller calls `.compute()`.
    """

    return apply_affine_split(vol, affine, y0, y1, corr_zy, mask)


def _write_checkpoint(filename: str, y: int) -> None:
    with open(filename, "w") as f:
        f.write(f"{y}\n")


def _read_checkpoint(filename: str, default_y: int) -> int:
    try:
        with open(filename, "r") as f:
            content = f.read().strip()
            y_str = content
            return int(y_str)
    except (FileNotFoundError, ValueError):
        return default_y


def load_y_coordinates(coords_yaml_path: str) -> List[float]:
    """Load per-tile absolute y positions from a coordinates YAML file."""
    with open(coords_yaml_path, "r") as f:
        coords = yaml.safe_load(f)

    return [entry[0]["y"] for entry in coords["coordinates"]]


def _checkpoint_path(general_config: GeneralConfig, ch: str) -> str:
    """Build a checkpoint file path for one channel's mosaic."""
    out = general_config.out.rstrip("/")
    if out.endswith(".ome.zarr"):
        out = out[: -len(".ome.zarr")]
    return f"{out}_{ch}.dat"


def _scale_cam_info(cam_info: List[dict], factor: float) -> List[dict]:
    """
    Return a copy of `cam_info` with y_start/y_end/z_start/z_end divided
    by `factor` and rounded to the nearest int -- for use when reading a
    downsampled OME-Zarr pyramid level (e.g. factor=2 for level 1, under
    the standard OME-Zarr convention that each level halves resolution
    relative to the previous one).

    Does NOT touch `vertical_flip`/`channel`/`camera_id` -- only the
    pixel-position fields.
    """
    scaled = []
    for meta in cam_info:
        new_meta = dict(meta)
        new_meta["y_start"] = round(meta["y_start"] / factor)
        new_meta["y_end"] = round(meta["y_end"] / factor)
        if meta.get("z_start") is not None:
            new_meta["z_start"] = round(meta["z_start"] / factor)
        if meta.get("z_end") is not None:
            new_meta["z_end"] = round(meta["z_end"] / factor)
        scaled.append(new_meta)
    return scaled


def _scale_affine_translations(
    affines: Dict[str, np.ndarray], factor: float
) -> Dict[str, np.ndarray]:
    """
    Return a copy of a {channel: 3x3 (Z,Y) affine} dict with only the
    translation column (indices [0,2] and [1,2]) divided by `factor` --
    the scale/rotation block (indices [0:2, 0:2]) is a dimensionless
    ratio between two coordinate systems and is unaffected by resolution,
    but an absolute pixel-unit translation (e.g. "shift by 20px") means
    half as many pixels at half resolution.
    """
    scaled = {}
    for ch, affine in affines.items():
        new_affine = affine.copy()
        new_affine[0, 2] = affine[0, 2] / factor
        new_affine[1, 2] = affine[1, 2] / factor
        scaled[ch] = new_affine
    return scaled


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@pipeline.default
@autoconfig
def pipeline(
    inp: str,
    mip_dir: str,
    yaml_path: str,
    camera_id: int,
    slice_number: int,
    coords_yaml_ch1: str,
    coords_yaml_ch2: str,
    *,
    y_chunk_size: int = 256,
    voxel_size: List[float] = [1, 1, 1],
    general_config: Optional[GeneralConfig] = None,
    zarr_config: Optional[ZarrConfig] = None,
    nii_config: Optional[NiftiConfig] = None,
    dandiset_id: Optional[str] = None,
    chunk_min: Optional[int] = None,
    chunk_max: Optional[int] = None,
    x_min: Optional[int] = None,
    x_max: Optional[int] = None,
    background_length: int = 5000,
    mip_pre_split: bool = False,
    channel_affines_path: Optional[str] = None,
    reference_channel: str = "488",
    reference_cords: Optional[str] = None,
    tissue_frac_min: float = 0.02,
    zarr_level: int = 0,
    use_alt_zy_correction: bool = False,
    alt_zy_reference_tiles: Optional[List[int]] = None,
    only_channel: Optional[str] = None,
    alt_zy_per_tile: bool = False,
) -> None:
    """
    Correct volumetric tile data and stream it directly into a single
    blended OME-Zarr mosaic along the y axis.

    Per-tile y placement (and therefore overlap) comes from a coordinates
    YAML file, one per channel, rather than a single constant overlap
    value -- this allows tile spacing to vary across the mosaic.

    Each tile's reader is opened once, then walked in fixed-size Y-chunks
    (`y_chunk_size` rows): each chunk is corrected, optionally blended
    against overlap with the neighboring tile, written, and discarded
    before the next chunk is read. This bounds peak memory to roughly one
    Y-chunk's worth of data, while reading each tile's bytes exactly once
    overall (no axis-coupling-induced redundant reads, since the skew
    shear never mixes Y with Z or X).

    Parameters
    ----------
    inp : str
        Path (local or DANDI) to this camera's tiles.
    mip_dir : str
        Directory containing YX MIP TIFF files used for mask generation.
    yaml_path : str
        Path to scan parameter YAML file. Crop definitions in this file
        are organized into slice-range "configEpochs"; the epoch that
        covers `slice_number` is used for this entire run.
    camera_id : int
        Camera to process (1 or 2).
    slice_number : int
        Physical slice number being processed by this run. Used to
        select the configEpoch (and therefore the crop definitions and
        channel layout) that applies to this slice.
    coords_yaml_ch1 : str
        Path to the coordinates YAML file for the first channel of the
        chosen camera (see `get_channel_names`), giving each tile's
        absolute y position.
    coords_yaml_ch2 : str
        Same as `coords_yaml_ch1`, for the second channel.
    y_chunk_size : int, default=256
        Height, in pixels along y, of each chunk that is corrected,
        blended, and written at a time.
    voxel_size : list of float
        Voxel size along X, Y and Z, in microns.
    general_config : GeneralConfig, optional
        Output configuration (must define `.out` directory).
    zarr_config : ZarrConfig, optional
        Zarr storage configuration (chunking, pyramid levels, etc.).
    nii_config : NiftiConfig, optional
        NIfTI header configuration.
    dandiset_id : str, optional
        If provided, tiles are loaded from DANDI instead of local disk.
    chunk_min : int, optional
        First tile index to process (inclusive). If omitted, processing
        starts from tile 0. Useful for splitting a large mosaic across
        multiple Slurm jobs by tile range.
    chunk_max : int, optional
        Last tile index to process (inclusive). If omitted, processing
        runs through the final tile. Combined with `chunk_min`, allows
        a specific contiguous range of tiles to be handled in one job.
    zarr_level : int, default=0
        Which OME-Zarr pyramid level (array key) to read from each
        tile, e.g. 1 to read the half-resolution level instead of the
        full-resolution level 0. Assumes the standard OME-Zarr
        convention of each level being 2x downsampled from the
        previous one. Everything resolution-dependent is automatically
        rescaled to match: y_coords, the volume-cropping bounds
        (`cam_info`), and the translation component of any channel
        registration affines. NOT rescaled: MIP/mask cropping bounds
        (MIP files are separate full-resolution TIFFs, not part of the
        zarr pyramid), `voxel_size` (you must pass the value that's
        actually correct for this level yourself), and `x_min`/`x_max`
        (interpreted in whatever level's pixel units you intend).
    use_alt_zy_correction : bool, default=False
        Use an alternate zy stripe correction instead of the default
        per-tile `compute_corr_zy`. Calibrates ONCE per channel from
        `alt_zy_reference_tiles` (a fixed 3 tiles), instead of
        recomputing a correction for every tile:
        1. For each reference tile: estimate the camera's own
           background noise as the median of the last
           `background_length` X columns, subtract it (clipped at 0).
        2. On the noise-subtracted data, compute a per-(Z, Y) tissue
           scaler (same style as the default correction), normalized
           by that tile's OWN median scaler (not a fixed constant),
           then inverted to a reciprocal multiplier.
        3. Average the noise scalar and the reciprocal maps across the
           3 reference tiles.
        4. Apply the SAME averaged (noise, reciprocal map) pair to
           every tile: `(vol - noise).clip(0) * reciprocal_map`.
        The final averaged noise map and reciprocal (scaler) map are
        written out as `{out_dir}/{ch}_alt_zy_noise.tiff` and
        `{out_dir}/{ch}_alt_zy_scaler.tiff` for inspection.
    alt_zy_reference_tiles : list of int, optional
        Exactly 3 tile indices to calibrate the alternate correction
        from. Required when `use_alt_zy_correction=True`.
    only_channel : str, optional
        If given, process only this one channel instead of both of
        this camera's channels. Must be one of the two channel names
        for `camera_id` (see `get_channel_names`).
    alt_zy_per_tile : bool, default=False
        Only meaningful when `use_alt_zy_correction=True`. Instead of
        calibrating once from `alt_zy_reference_tiles` (a fixed 3
        tiles) and reusing that same correction for every tile,
        recalibrate fresh for EACH tile from its own data alone (no
        averaging across reference tiles, and `alt_zy_reference_tiles`
        is unused in this mode). Each tile's own noise/scaler maps are
        written out as `{out_dir}/{ch}_{tile_name}_alt_zy_noise.tiff`
        and `{out_dir}/{ch}_{tile_name}_alt_zy_scaler.tiff`.

    Raises
    ------
    FileNotFoundError
        If a required MIP file is missing.
    ValueError
        If camera_id is not 1 or 2, or a coordinates file doesn't have
        an entry for every discovered tile.
    """
    if camera_id not in (1, 2):
        raise ValueError(f"camera_id must be 1 or 2, got {camera_id}")

    start_timer = time.time()

    voxel_size = list(map(float, reversed(voxel_size)))

    scan_parameters = load_scan_parameters(yaml_path)
    reference_camera_id = find_camera_for_channel(
        scan_parameters, reference_channel)
    cam_info = get_camera_info(scan_parameters, camera_id, slice_number,
                               crop_stage="stitching" if channel_affines_path is None else "split")
    reference_cam_info = get_camera_info(scan_parameters, reference_camera_id,
                                         slice_number, "stitching") if channel_affines_path is not None else cam_info
    cam_info_stitching = get_camera_info(
        scan_parameters, reference_camera_id, slice_number) if channel_affines_path is not None else cam_info
    # get_crop_values needs the REFERENCE channel's OWN split-crop info,
    # on the reference channel's OWN camera -- which may differ from
    # `camera_id` (the camera THIS run is processing) when the
    # reference channel lives on the other camera. Reusing `cam_info`
    # here would be wrong: `cam_info` only has entries for whichever
    # channels live on `camera_id`, so a lookup for `reference_channel`
    # would raise KeyError whenever it's on the other camera.
    reference_cam_info_split = get_camera_info(
        scan_parameters, reference_camera_id, slice_number, "split",
    ) if channel_affines_path is not None else cam_info

    downsample_factor = 2 ** zarr_level
    # Keep full-resolution cam_info variants for MIP/mask cropping --
    # MIP files are separate full-resolution TIFFs, unaffected by
    # zarr_level -- then overwrite the "working" variable names below
    # with SCALED versions, used everywhere else (3D volume cropping,
    # get_crop_values), so existing code using these names doesn't
    # need further changes.
    cam_info_full_res = cam_info
    reference_cam_info_full_res = reference_cam_info
    cam_info_stitching_full_res = cam_info_stitching
    reference_cam_info_split_full_res = reference_cam_info_split

    if downsample_factor != 1:
        cam_info = _scale_cam_info(cam_info, downsample_factor)
        reference_cam_info = _scale_cam_info(
            reference_cam_info, downsample_factor)
        cam_info_stitching = _scale_cam_info(
            cam_info_stitching, downsample_factor)
        reference_cam_info_split = _scale_cam_info(
            reference_cam_info_split, downsample_factor)

    api_key = prompt_dandi_api_key() if dandiset_id else None

    tile_paths = discover_tile_paths(
        inp, camera_id, dandiset_id=dandiset_id, api_key=api_key
    )
    # Sample tiles used to estimate the reference channel's shape/depth
    # must come from the reference channel's OWN camera -- MIP files
    # are named per-tile, and a camera-1 tile's name has no
    # correspondence to camera-2's MIP files (or vice versa) whenever
    # the reference channel lives on the other camera from `camera_id`.
    reference_tile_paths = (
        tile_paths if reference_camera_id == camera_id
        else discover_tile_paths(inp, reference_camera_id, dandiset_id=dandiset_id, api_key=api_key)
    )

    num_tiles = len(tile_paths)

    def tile_name(path: str) -> str:
        return os.path.basename(path.rstrip("/").replace(".ome.zarr", ""))

    channels = get_channel_names(scan_parameters, camera_id)
    if len(channels) != 2:
        raise ValueError(
            f"Expected exactly 2 channels for camera {camera_id}, "
            f"got {len(channels)}: {channels}"
        )
    coords_yaml_by_channel = dict(
        zip(channels, [coords_yaml_ch1, coords_yaml_ch2]))

    if only_channel is not None:
        if only_channel not in channels:
            raise ValueError(
                f"only_channel '{only_channel}' is not one of this camera's "
                f"channels: {channels}"
            )
        channels_to_process = [only_channel]
    else:
        channels_to_process = channels

    for ch in channels_to_process:

        channel_timer = time.time()

        if channel_affines_path is None:
            reference_channel = ch

        if channel_affines_path is not None:
            y_coords = load_y_coordinates(reference_cords)
        else:
            y_coords = load_y_coordinates(
                coords_yaml_by_channel[ch])
        # Coordinates YAML files give absolute tile positions in
        # full-resolution pixel units -- rescale to match zarr_level.
        if downsample_factor != 1:
            y_coords = [y / downsample_factor for y in y_coords]
        # if len(y_coords) != num_tiles:
        #    raise ValueError(
        #        f"Coordinates file for channel {ch} has {len(y_coords)} "
        #        f"tile entries, but {num_tiles} tiles were discovered."
        #    )

        # --- Estimate the corrected mosaic shape from a single sample tile,
        # taken from the reference channel's OWN camera (see
        # reference_tile_paths above -- not necessarily the same camera
        # as `camera_id`/`tile_paths`).
        sample_path = reference_tile_paths[0]
        sample_raw_vol, sample_mask, sample_thr = (
            _open_raw_channel_volume_and_mask(
                sample_path,
                dandiset_id=dandiset_id,
                api_key=api_key,
                mip_dir=mip_dir,
                name=tile_name(sample_path),
                ch=reference_channel,
                cam_info=reference_cam_info,
                mip_cam_info=reference_cam_info_full_res,
                background_length=background_length,
                mip_pre_split=mip_pre_split,
                x_min=x_min,
                x_max=x_max,
                zarr_level=zarr_level,
            )
        )
        sample_corrected = stripe_skew_corr(
            sample_raw_vol, sample_mask, sample_thr, camera_id, scan_parameters
        )
        corrected_sz, corrected_sy, corrected_sx = sample_corrected.shape
        del sample_corrected
        gc.collect()

        # get_crop_values needs the reference channel's own SPLIT-cropped
        # Z depth (D), not `corrected_sz` above -- that's the STITCHING-
        # cropped sample's depth (after stripe_skew_corr, which doesn't
        # change Z, but stitching-crop and split-crop can still differ in
        # Z extent). Only a shape is needed here, so this is a cheap
        # metadata-only read (dask `.shape`, no data movement), and every
        # tile is assumed to share the same shape elsewhere in this
        # codebase, so measuring it once from the sample tile is valid.
        if channel_affines_path is not None:
            reference_split_sample_vol, _, _ = _open_raw_channel_volume_and_mask(
                sample_path,
                dandiset_id=dandiset_id,
                api_key=api_key,
                mip_dir=mip_dir,
                name=tile_name(sample_path),
                ch=reference_channel,
                cam_info=reference_cam_info_split,
                mip_cam_info=reference_cam_info_split_full_res,
                background_length=background_length,
                mip_pre_split=mip_pre_split,
                x_min=x_min,
                x_max=x_max,
                zarr_level=zarr_level,
            )
            reference_split_z_depth = reference_split_sample_vol.shape[0]
        else:
            reference_split_z_depth = corrected_sz

        full_x = corrected_sx
        full_y = int(round(y_coords[-1])) + corrected_sy
        full_z = corrected_sz
        fullshape = (full_z, full_y, full_x)

        # Alternate zy correction: calibrate ONCE per channel from a
        # fixed set of user-chosen reference tiles, then apply the
        # SAME (averaged) correction to every tile -- instead of the
        # default per-tile compute_corr_zy. Skipped entirely when
        # alt_zy_per_tile=True, since that mode recalibrates fresh for
        # every tile instead (see the main per-tile loop below) and
        # doesn't need this fixed, reference-tile-based calibration at
        # all.
        alt_zy_noise = None
        alt_zy_reciprocal_map = None
        if use_alt_zy_correction and not alt_zy_per_tile:
            if alt_zy_reference_tiles is None or len(alt_zy_reference_tiles) != 3:
                raise ValueError(
                    "use_alt_zy_correction requires exactly 3 tile indices "
                    "in alt_zy_reference_tiles (unless alt_zy_per_tile=True)"
                )
            calib_layout = scan_parameters.get("channelLayout", {})
            calib_vertical_flip = {
                1: bool(calib_layout["Camera1"]["verticalFlip"]),
                2: bool(calib_layout["Camera2"]["verticalFlip"]),
            }
            noises = []
            reciprocal_maps = []
            for ref_index in alt_zy_reference_tiles:
                ref_path = tile_paths[ref_index]
                ref_name = tile_name(ref_path)
                ref_raw_vol, ref_mask, ref_thr = _open_raw_channel_volume_and_mask(
                    ref_path,
                    dandiset_id=dandiset_id,
                    api_key=api_key,
                    mip_dir=mip_dir,
                    name=ref_name,
                    ch=ch,
                    cam_info=cam_info,
                    mip_cam_info=cam_info_full_res,
                    background_length=background_length,
                    mip_pre_split=mip_pre_split,
                    x_min=x_min,
                    x_max=x_max,
                    zarr_level=zarr_level,
                )
                if calib_vertical_flip[camera_id]:
                    ref_raw_vol = ref_raw_vol[::-1]
                ref_noise_map, ref_reciprocal_map = compute_alt_zy_calibration_for_tile(
                    ref_raw_vol, ref_mask, ref_thr,
                    background_length=background_length,
                )
                noises.append(ref_noise_map)
                reciprocal_maps.append(ref_reciprocal_map)
                logger.info(
                    f"[alt zy calibration] tile {ref_index} ({ref_name}): "
                    f"noise_map mean={ref_noise_map.mean():.2f}"
                )

            # noise is a per-(Z, Y) map here, not a single scalar --
            # fixed pattern noise varies per pixel/row, it isn't drawn
            # from one shared distribution -- so average elementwise
            # across the 3 reference tiles, same as the reciprocal map.
            alt_zy_noise = np.mean(np.stack(noises, axis=0), axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                alt_zy_reciprocal_map = np.nanmean(
                    np.stack(reciprocal_maps, axis=0), axis=0)
            # Positions with no valid tissue data in ANY of the 3
            # reference tiles stay NaN after nanmean -- replace with a
            # tiny reciprocal (i.e. a huge equivalent divisor) so those
            # rows get suppressed rather than propagating NaN into the
            # output, matching compute_corr_zy's own
            # insufficient-data-sentinel spirit.
            alt_zy_reciprocal_map = np.nan_to_num(
                alt_zy_reciprocal_map, nan=1e-9)
            logger.info(
                f"[alt zy calibration] averaged noise_map mean="
                f"{alt_zy_noise.mean():.2f} over tiles {alt_zy_reference_tiles}"
            )

        out_dir = f"{general_config.out}/{ch}"

        if use_alt_zy_correction and not alt_zy_per_tile:
            os.makedirs(out_dir, exist_ok=True)
            noise_tiff_path = os.path.join(out_dir, f"{ch}_alt_zy_noise.tiff")
            scaler_tiff_path = os.path.join(
                out_dir, f"{ch}_alt_zy_scaler.tiff")
            tifffile.imwrite(noise_tiff_path, alt_zy_noise.astype(np.float32))
            tifffile.imwrite(scaler_tiff_path,
                             alt_zy_reciprocal_map.astype(np.float32))
            logger.info(
                f"[alt zy calibration] wrote {noise_tiff_path} and "
                f"{scaler_tiff_path}"
            )

        checkpoint_file = _checkpoint_path(general_config, ch)
        checkpoint = _read_checkpoint(checkpoint_file, -1)

        # `omz`/`array` are opened exactly once per channel here, and
        # reused for every tile/chunk below -- no re-opening per tile or
        # per chunk, and no re-opening before pyramid generation.
        omz = ZarrPythonGroup.from_config(out_dir, zarr_config)

        try:
            if checkpoint == -1:
                array = omz.create_array(
                    "0",
                    shape=fullshape,
                    zarr_config=zarr_config,
                    dtype=np.uint16,
                )
            else:
                array = omz["0"]
        except Exception:
            logger.info("already exists")
            array = omz["0"]

        logger.info(
            "Writing channel %s, level 0 array with shape %s", ch, fullshape
        )

        # Rows carried over from the END of the previous tile, awaiting
        # blending with the START of the next one. Full tile width, but
        # only ever as tall as the relevant overlap -- never the whole
        # tile.
        carry: Optional[np.ndarray] = None

        # Clamp the tile range to [chunk_min, chunk_max] if provided.
        # These compose with the checkpoint: we process tiles that are
        # both within the requested range AND past the checkpoint.
        effective_min = max(chunk_min if chunk_min is not None else 0, 0)
        effective_max = min(
            chunk_max if chunk_max is not None else num_tiles - 1,
            num_tiles - 1,
        )

        if chunk_min is not None or chunk_max is not None:
            logger.info(
                f"Processing tile range [{effective_min}, {effective_max}] "
                f"(chunk_min={chunk_min}, chunk_max={chunk_max})"
            )

        if len(tile_paths) > checkpoint + 1:
            for index, path in enumerate(tile_paths):
                gc.collect()
                if index < checkpoint or index < effective_min:
                    continue
                if index > effective_max:
                    break

                name = tile_name(path)
                logger.info(f"[{index}] Processing {name}")
                tile_timer = time.time()

                open_timer = time.time()

                raw_vol, mask, thr = _open_raw_channel_volume_and_mask(
                    path,
                    dandiset_id=dandiset_id,
                    api_key=api_key,
                    mip_dir=mip_dir,
                    name=name,
                    ch=ch,
                    cam_info=cam_info,
                    mip_cam_info=cam_info_full_res,
                    background_length=background_length,
                    mip_pre_split=mip_pre_split,
                    x_min=x_min,
                    x_max=x_max,
                    zarr_level=zarr_level,
                )
                logger.info(
                    f"[{index}] open reader + mask/threshold: "
                    f"{time.time() - open_timer:.2f}s"
                )

                is_first = index == 0 or index == checkpoint or (
                    chunk_min is not None and index == chunk_min)
                is_last = index == num_tiles - 1 or index == effective_max

                ystart = int(round(y_coords[index]))

                overlap_with_prev = 0
                if not is_first:
                    overlap_with_prev = corrected_sy - (
                        int(round(y_coords[index]))
                        - int(round(y_coords[index - 1]))
                    )
                overlap_with_next = 0
                if not is_last:
                    overlap_with_next = corrected_sy - (
                        int(round(y_coords[index + 1]))
                        - int(round(y_coords[index]))
                    )
                overlap_with_prev = max(overlap_with_prev, 0)
                overlap_with_next = max(overlap_with_next, 0)

                # Absolute (within this tile) row at which the tile's
                # trailing overlap region begins; rows at or past this
                # point must be withheld (not written yet) until the
                # next tile's leading chunk(s) have blended with them.
                withhold_from = corrected_sy - overlap_with_next

                if overlap_with_prev > 0:
                    t = np.linspace(0, 1, overlap_with_prev)
                    ramp = (1 - np.cos(np.pi * t)) / 2
                    ramp_inverse = (1 + np.cos(np.pi * t)) / 2
                    ramp = ramp[None, :, None]
                    ramp_inverse = ramp_inverse[None, :, None]

                zstart = 0
                trailing_buffer: Optional[np.ndarray] = None
                layout = scan_parameters.get("channelLayout", {})
                vertical_flip = {1: bool(layout["Camera1"]["verticalFlip"]), 2: bool(
                    layout["Camera2"]["verticalFlip"])}
                z_start, z_end, y_start, y_end = get_crop_values(
                    reference_split_z_depth, reference_cam_info_split, cam_info_stitching, reference_channel, vertical_flip[find_camera_for_channel(scan_parameters, reference_channel)])

                # y0/y1 (the chunk loop below) range over
                # [y_start, corrected_sy + y_start), NOT [0, corrected_sy)
                # -- withhold_from was computed above in the latter
                # (zero-based) frame, so it needs the same y_start offset
                # to be compared correctly against y0/y1.
                withhold_from = withhold_from + y_start

                y0 = y_start
                delta = scan_parameters["acquisitionSettings"]["skewCorrection"]["delta_deg"]
                umps = scan_parameters["voxelSize_um"]["rawAcquisition"]
                factors = [umps["y"], umps["z"], umps["x"]]
                affine = generate_skew_affine(factors, delta)
                if channel_affines_path is not None:
                    affines = load_channel_affines(
                        channel_affines_path, reference_channel)
                    # Translation components are in full-resolution
                    # pixel units in the affines file -- rescale to
                    # match zarr_level. The scale/rotation block is a
                    # dimensionless ratio between coordinate systems
                    # and doesn't change with resolution.
                    if downsample_factor != 1:
                        affines = _scale_affine_translations(
                            affines, downsample_factor)
                    # Combine as (skew @ registration), not (registration @
                    # skew): for forward-mapping matrices composed via
                    # C = B @ A, C(x) = B(A(x)) means A is applied FIRST.
                    # We want registration first, skew second, so skew must
                    # be on the LEFT.
                    affine = affine @ embed_zy_affine_for_volume(affines[ch])

                if vertical_flip[camera_id]:
                    raw_vol = raw_vol[::-1]

                if use_alt_zy_correction:
                    if alt_zy_per_tile:
                        # Recalibrate fresh for THIS tile alone (no
                        # averaging across reference tiles) -- reuses
                        # the same per-tile calibration function the
                        # fixed-calibration mode calls on its 3
                        # reference tiles, just applied to every tile
                        # here instead of just 3 of them.
                        tile_noise_map, tile_reciprocal_map = compute_alt_zy_calibration_for_tile(
                            raw_vol, mask, thr, background_length=background_length,
                        )
                        # Unlike the fixed-calibration mode (which
                        # absorbs NaN positions via nanmean across 3
                        # reference tiles), there's no averaging step
                        # here -- positions with insufficient tissue
                        # data for THIS tile alone would otherwise
                        # propagate NaN straight into corr_zy and the
                        # final uint16 cast. Same fallback: suppress
                        # via a tiny reciprocal instead.
                        tile_reciprocal_map = np.nan_to_num(
                            tile_reciprocal_map, nan=1e-9)
                        os.makedirs(out_dir, exist_ok=True)
                        noise_tiff_path = os.path.join(
                            out_dir, f"{ch}_{name}_alt_zy_noise.tiff")
                        scaler_tiff_path = os.path.join(
                            out_dir, f"{ch}_{name}_alt_zy_scaler.tiff")
                        tifffile.imwrite(
                            noise_tiff_path, tile_noise_map.astype(np.float32))
                        tifffile.imwrite(
                            scaler_tiff_path, tile_reciprocal_map.astype(np.float32))
                    else:
                        tile_noise_map = alt_zy_noise
                        tile_reciprocal_map = alt_zy_reciprocal_map

                    # (vol - noise_map).clip(0) * reciprocal_map is the
                    # same as (vol - noise_map).clip(0) / (1/reciprocal_map),
                    # so this reuses the existing apply_corr_zy_lazy
                    # (division-based) machinery unchanged: pre-subtract
                    # the (per-Z,Y) noise map here, and feed in the
                    # RECIPROCAL of the reciprocal map as if it were
                    # corr_zy.
                    raw_vol = da.clip(
                        raw_vol.astype(np.float32) - tile_noise_map[:, :, None], 0, None)
                    corr_zy = 1.0 / tile_reciprocal_map
                else:
                    corr_zy = compute_corr_zy(
                        raw_vol,
                        mask,
                        tissue_frac_min,
                        thr,
                    )
                while y0 < corrected_sy+y_start:
                    y1 = min(corrected_sy+y_start, y0 + y_chunk_size)

                    lazy_chunk = _corrected_y_chunk(
                        raw_vol, mask, affine, corr_zy, y0, y1)
                    lazy_chunk = lazy_chunk[z_start:z_end]

                    compute_timer = time.time()
                    with ProgressBar():
                        data = lazy_chunk.compute()  # plain numpy array
                    compute_elapsed = time.time() - compute_timer

                    blend_timer = time.time()
                    # Blend the leading edge of this chunk if it falls
                    # within [0, overlap_with_prev) of THIS tile's own
                    # local (zero-based) frame -- y0 itself starts at
                    # y_start, not 0, so it must be re-based here.
                    if overlap_with_prev > 0 and (y0 - y_start) < overlap_with_prev:
                        local_y0 = y0 - y_start
                        blend_len = min(
                            data.shape[1], overlap_with_prev - local_y0)
                        carry_slice = carry[:,
                                            local_y0:local_y0 + blend_len, :]
                        ramp_slice = ramp[:, local_y0:local_y0 + blend_len, :]
                        ramp_inv_slice = ramp_inverse[:,
                                                      local_y0:local_y0 + blend_len, :]
                        data[:, :blend_len, :] = (
                            carry_slice * ramp_inv_slice
                            + data[:, :blend_len, :] * ramp_slice
                        )
                    blend_elapsed = time.time() - blend_timer

                    # Split this chunk into what's safe to write now vs.
                    # what must be withheld (trailing overlap with the
                    # next tile).
                    if y1 <= withhold_from:
                        to_write, to_withhold = data, None
                    elif y0 >= withhold_from:
                        to_write, to_withhold = None, data
                    else:
                        split = withhold_from - y0
                        to_write, to_withhold = (
                            data[:, :split, :], data[:, split:, :]
                        )

                    out_ystart = ystart + y0
                    write_elapsed = 0.0
                    if (
                        to_write is not None
                        and to_write.shape[1] > 0
                        and index > checkpoint
                    ):
                        write_timer = time.time()
                        array[
                            zstart: zstart + to_write.shape[0],
                            out_ystart: out_ystart + to_write.shape[1],
                            0: to_write.shape[2],
                        ] = to_write
                        write_elapsed = time.time() - write_timer

                    logger.info(
                        f"[{index}] chunk y0:{y0}-{y1} (out y:{out_ystart}) -- "
                        f"compute: {compute_elapsed:.2f}s, "
                        f"blend: {blend_elapsed:.2f}s, "
                        f"write: {write_elapsed:.2f}s"
                    )

                    if to_withhold is not None:
                        trailing_buffer = (
                            to_withhold
                            if trailing_buffer is None
                            else np.concatenate(
                                [trailing_buffer, to_withhold], axis=1
                            )
                        )

                    del data
                    gc.collect()
                    y0 = y1

                carry = trailing_buffer

                logger.info(
                    f"{name} done in {time.time() - tile_timer:.2f}s"
                )
                _write_checkpoint(checkpoint_file, index)

        gc.collect()
        copy_config = replace(general_config, out=out_dir)
        pyramid_timer = time.time()
        omz.generate_pyramid_staged(
            levels=zarr_config.levels,
            copy_config=copy_config,
            copy_zarr_config=zarr_config,
        )
        logger.info(
            f"Pyramid generation for channel {ch}: "
            f"{(time.time() - pyramid_timer) / 60:.2f} minutes"
        )

        omz.write_ome_metadata(axes=["z", "y", "x"], space_scale=voxel_size)

        if nii_config and nii_config.nii:
            header = build_nifti_header(
                zgroup=omz,
                voxel_size_zyx=tuple(voxel_size),
                unit="micrometer",
                nii_config=nii_config,
            )
            omz.write_nifti_header(header)

        logger.info(
            f"Channel {ch} completed in "
            f"{(time.time() - channel_timer) / 60:.2f} minutes"
        )

    end_timer = time.time()
    length = end_timer - start_timer
    logger.info(f"Conversion completed in {length / 60} minutes.")
