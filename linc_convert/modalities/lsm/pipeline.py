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
    crop_volume_channels,
    stripe_skew_corr,
)
from linc_convert.modalities.lsm.preprocessing_utils.io import (
    camera_channel_map,
    get_camera_info,
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
) -> da.Array:
    """Lazily open a tile (ome.zarr or spool set) as a dask array.

    No data is read from disk until the returned array is sliced and
    computed.
    """
    if path.endswith(".ome.zarr"):
        if dandiset_id is None:
            return da.asarray(
                ZarrPythonGroup.open(path)["0"],
                chunks=chunks if chunks is not None else (128, 128, 128),
            )
        return da.asarray(
            ZarrPythonGroup.open_dandi(
                dandiset_id=dandiset_id,
                asset_path=path,
                api_key=api_key,
            )["0"],
            chunks=chunks if chunks is not None else (128, 128, 128),
        )

    return da.asarray(
        SpoolSetInterpreter(path, f"{path}_info.mat").assemble_cropped(),
        chunks=chunks if chunks is not None else (128, 128, 128),
    )


def discover_tile_paths(
    inp: str,
    camera_id: int,
    *,
    dandiset_id: Optional[str],
    api_key: Optional[str],
) -> List[str]:
    """Get all tile paths for one camera from the input location, in
    on-disk order.

    Tiles are assumed to be laid out along a single axis (y) in the order
    they're returned here -- the index of a path in this list *is* its
    y-position, so no filename parsing is needed beyond filtering by
    camera.

    Tile names are expected to contain a camera token of the form
    ``acq-camera-01``, ``acq-camera-02``, etc. (zero-padded to two
    digits), e.g.:

        sub-MF283_sample-slice052_chunk-0019_acq-camera-01.ome.zarr

    Only tiles matching `camera_id` are returned.

    Parameters
    ----------
    inp : str
        Path (local directory, or DANDI asset path prefix) to search.
    camera_id : int
        Camera number to filter tiles by (e.g. 1 -> "acq-camera-01").
    dandiset_id : str, optional
        If provided, tiles are listed from DANDI instead of local disk.
    api_key : str, optional
        DANDI API key, required when `dandiset_id` is provided.

    Returns
    -------
    list of str
        Matching tile paths, sorted.

    Raises
    ------
    ValueError
        If no tiles are found, or none match the requested camera.
    """
    camera_token = f"acq-camera-{camera_id:02d}"

    if dandiset_id is None:
        paths = sorted(glob(os.path.join(inp, f"*{camera_token}*.ome.zarr")))
        if not paths:
            raise ValueError(
                f"No tile folders found for camera {camera_id} "
                f"(token '{camera_token}') in input directory"
            )
        return paths

    with DandiAPIClient(
        api_url="https://api.dandiarchive.org/api",
        token=api_key,
    ) as client:
        dandiset = client.get_dandiset(dandiset_id, "draft")
        prefix = PurePosixPath(inp.rstrip("/") + "/")
        depth = len(prefix.parts)

        paths = sorted(
            asset.path
            for asset in dandiset.get_assets_with_path_prefix(str(prefix))
            if len(PurePosixPath(asset.path).parts) == depth + 1
            and camera_token in PurePosixPath(asset.path).name
        )

    if not paths:
        raise ValueError(
            f"No tile assets found for camera {camera_id} "
            f"(token '{camera_token}') in DANDI dataset"
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
) -> Tuple[da.Array, np.ndarray, float]:
    """Open one tile/channel exactly once: the reader, the channel crop,
    and the mask/threshold lookup all happen here, a single time per
    tile, regardless of how many Y-chunks it's later split into.

    Returns
    -------
    vol : dask.array.Array
        Raw, channel-cropped (but not yet corrected) lazy volume
        (Z, Y, X) for the whole tile.
    mask : np.ndarray
        Mask for this channel, shape (Y, X) or (Z, Y, X), matching
        `vol`'s full Y extent.
    threshold : float
        Intensity threshold for this channel.
    """
    reader = open_tile_reader(path, dandiset_id=dandiset_id, api_key=api_key)
    vol_channels = crop_volume_channels(reader, cam_info)
    masks, thrs = load_mask_and_thresholds(name, mip_dir, cam_info)

    return vol_channels[ch], masks[ch], thrs[ch]


def _corrected_y_chunk(
    vol: da.Array,
    mask: np.ndarray,
    threshold: float,
    camera_id: int,
    scan_parameters: dict,
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
    vol_chunk = vol[:, y0:y1, :]
    mask_chunk = mask[:, y0:y1, :] if mask.ndim == 3 else mask[y0:y1, :]

    return stripe_skew_corr(
        vol_chunk, mask_chunk, threshold, camera_id, scan_parameters
    )


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
        Path to scan parameter YAML file.
    camera_id : int
        Camera to process (1 or 2).
    coords_yaml_ch1 : str
        Path to the coordinates YAML file for the first channel of the
        chosen camera (see `camera_channel_map`), giving each tile's
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
    cam_info = get_camera_info(scan_parameters, camera_id)

    api_key = prompt_dandi_api_key() if dandiset_id else None

    tile_paths = discover_tile_paths(
        inp, camera_id, dandiset_id=dandiset_id, api_key=api_key
    )

    num_tiles = len(tile_paths)

    def tile_name(path: str) -> str:
        return os.path.basename(path.rstrip("/").replace(".ome.zarr", ""))

    channels = camera_channel_map[camera_id]
    if len(channels) != 2:
        raise ValueError(
            f"Expected exactly 2 channels for camera {camera_id}, "
            f"got {len(channels)}: {channels}"
        )
    coords_yaml_by_channel = dict(
        zip(channels, [coords_yaml_ch1, coords_yaml_ch2]))

    for ch in channels:

        channel_timer = time.time()

        y_coords = load_y_coordinates(coords_yaml_by_channel[ch])
        # if len(y_coords) != num_tiles:
        #    raise ValueError(
        #        f"Coordinates file for channel {ch} has {len(y_coords)} "
        #        f"tile entries, but {num_tiles} tiles were discovered."
        #    )

        # --- Estimate the corrected mosaic shape from a single sample tile.
        sample_path = tile_paths[0]
        sample_raw_vol, sample_mask, sample_thr = (
            _open_raw_channel_volume_and_mask(
                sample_path,
                dandiset_id=dandiset_id,
                api_key=api_key,
                mip_dir=mip_dir,
                name=tile_name(sample_path),
                ch=ch,
                cam_info=cam_info,
            )
        )
        sample_corrected = stripe_skew_corr(
            sample_raw_vol, sample_mask, sample_thr, camera_id, scan_parameters
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
                if index == 0:
                    raw_vol, mask, thr = (
                        sample_raw_vol, sample_mask, sample_thr
                    )
                else:
                    raw_vol, mask, thr = _open_raw_channel_volume_and_mask(
                        path,
                        dandiset_id=dandiset_id,
                        api_key=api_key,
                        mip_dir=mip_dir,
                        name=name,
                        ch=ch,
                        cam_info=cam_info,
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
                    # Power-curve blend: both ramps change rapidly at the
                    # start of the overlap and slowly at the end.
                    # ramp (incoming tile weight):   0 -> 1, fast rise early
                    # ramp_inverse (carry weight):   1 -> 0, fast drop early
                    # ramp + ramp_inverse = 1 everywhere by construction.
                    ramp = t ** 0.4
                    ramp_inverse = 1 - t ** 0.4
                    ramp = ramp[None, :, None]
                    ramp_inverse = ramp_inverse[None, :, None]

                zstart = 0
                trailing_buffer: Optional[np.ndarray] = None

                y0 = 0
                while y0 < corrected_sy:
                    y1 = min(corrected_sy, y0 + y_chunk_size)

                    lazy_chunk = _corrected_y_chunk(
                        raw_vol, mask, thr, camera_id, scan_parameters,
                        y0, y1,
                    )

                    compute_timer = time.time()
                    with ProgressBar():
                        data = lazy_chunk.compute()  # plain numpy array
                    compute_elapsed = time.time() - compute_timer

                    blend_timer = time.time()
                    # Blend the leading edge of this chunk if it falls
                    # within [0, overlap_with_prev).
                    if overlap_with_prev > 0 and y0 < overlap_with_prev:
                        blend_len = min(data.shape[1], overlap_with_prev - y0)
                        carry_slice = carry[:, y0:y0 + blend_len, :]
                        ramp_slice = ramp[:, y0:y0 + blend_len, :]
                        ramp_inv_slice = ramp_inverse[:, y0:y0 + blend_len, :]
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
