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
- There is no chunking along x: each tile is read, corrected, blended, and
  written as a single unit, which is also what keeps memory use to one
  "chunk" (one tile) at a time.
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
from glob import glob
from pathlib import PurePosixPath
from typing import Callable, List, Literal, Optional, Tuple

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


def _corrected_volume(
    path: str,
    *,
    dandiset_id: Optional[str],
    api_key: Optional[str],
    mip_dir: str,
    name: str,
    camera_id: int,
    ch: str,
    cam_info,
    scan_parameters,
) -> da.Array:
    """Build the lazy (uncomputed) corrected dask array for one tile/channel.

    Building this graph does not read or correct any data yet; nothing is
    materialized until the caller calls `.compute()` on it (or a slice of
    it), which is what lets us keep only one tile's worth of corrected data
    resident in memory at a time.
    """
    reader = open_tile_reader(path, dandiset_id=dandiset_id, api_key=api_key)

    vol_channels = crop_volume_channels(reader, cam_info)
    masks, thrs = load_mask_and_thresholds(name, mip_dir, cam_info)

    mask = masks[ch]
    thr = thrs[ch]

    vol = vol_channels[ch]
    vol = stripe_skew_corr(vol, mask, thr, camera_id, scan_parameters)

    return vol


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
    """Load per-tile absolute y positions from a coordinates YAML file.

    The file is expected to have the structure produced by the tile
    registration step, in particular a top-level ``coordinates`` key:

        coordinates:
        - - {x: 0.0, y: 0.0}
        - - {x: 0.0, y: 522.0}
        - - {x: 0.0, y: 1044.0}
          ...

    i.e. a list with one single-element sub-list per tile, each holding
    an ``{x, y}`` mapping. The x values are ignored; only the y position
    of each tile (already in corrected, post stripe/skew-correction pixel
    space) is used.

    Parameters
    ----------
    coords_yaml_path : str
        Path to the coordinates YAML file for one channel.

    Returns
    -------
    list of float
        Absolute y position of each tile, in on-disk tile order.
    """
    with open(coords_yaml_path, "r") as f:
        coords = yaml.safe_load(f)

    return [entry[0]["y"] for entry in coords["coordinates"]]


def _checkpoint_path(general_config: GeneralConfig, ch: str) -> str:
    """Build a checkpoint file path for one channel's mosaic.

    Strips a trailing ``.ome.zarr`` suffix from `general_config.out` if
    present (as an actual suffix removal, not `str.rstrip`, which treats
    its argument as a set of characters and would otherwise eat unrelated
    trailing characters from the path).
    """
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
    inp_cm1: str,
    inp_cm2: str,
    mip_dir: str,
    yaml_path: str,
    camera_id: int,
    coords_yaml_ch1: str,
    coords_yaml_ch2: str,
    *,
    voxel_size: List[float] = [1, 1, 1],
    general_config: Optional[GeneralConfig] = None,
    zarr_config: Optional[ZarrConfig] = None,
    nii_config: Optional[NiftiConfig] = None,
    dandiset_id: Optional[str] = None,
) -> None:
    """
    Correct volumetric tile data and stream it directly into a single
    blended OME-Zarr mosaic along the y axis.

    Per-tile y placement (and therefore overlap) comes from a coordinates
    YAML file, one per channel, rather than a single constant overlap
    value -- this allows tile spacing to vary across the mosaic.

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
    coords_yaml_ch1 : str
        Path to the coordinates YAML file for the first channel of the
        chosen camera (see `camera_channel_map`), giving each tile's
        absolute y position.
    coords_yaml_ch2 : str
        Same as `coords_yaml_ch1`, for the second channel.
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

    inp = inp_cm1 if camera_id == 1 else inp_cm2
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

        y_coords = load_y_coordinates(coords_yaml_by_channel[ch])
        if len(y_coords) != num_tiles:
            raise ValueError(
                f"Coordinates file for channel {ch} has {len(y_coords)} "
                f"tile entries, but {num_tiles} tiles were discovered."
            )

        sample_path = tile_paths[0]
        sample_vol = _corrected_volume(
            sample_path,
            dandiset_id=dandiset_id,
            api_key=api_key,
            mip_dir=mip_dir,
            name=tile_name(sample_path),
            camera_id=camera_id,
            ch=ch,
            cam_info=cam_info,
            scan_parameters=scan_parameters,
        )
        corrected_sz, corrected_sy, corrected_sx = sample_vol.shape

        full_x = corrected_sx
        full_y = int(round(y_coords[-1])) + corrected_sy
        full_z = corrected_sz
        fullshape = (full_z, full_y, full_x)

        out_dir = f"{general_config.out}/{ch}"

        checkpoint_file = _checkpoint_path(general_config, ch)
        checkpoint = _read_checkpoint(checkpoint_file, -1)

        try:
            if checkpoint == -1:
                omz = ZarrPythonGroup.from_config(out_dir, zarr_config)
                array = omz.create_array(
                    "0",
                    shape=fullshape,
                    zarr_config=zarr_config,
                    dtype=np.uint16,
                )
        except Exception:
            logger.info("already exists")

        logger.info(
            "Writing channel %s, level 0 array with shape %s", ch, fullshape
        )

        bottom_overlap = None

        for index, path in enumerate(tile_paths):
            gc.collect()
            if index >= checkpoint:
                omz = ZarrPythonGroup.from_config(out_dir, zarr_config)
                array = omz["0"]
                name = tile_name(path)
                logger.info(f"[{index}] Processing {name}")

                lazy_vol = sample_vol if index == 0 else _corrected_volume(
                    path,
                    dandiset_id=dandiset_id,
                    api_key=api_key,
                    mip_dir=mip_dir,
                    name=name,
                    camera_id=camera_id,
                    ch=ch,
                    cam_info=cam_info,
                    scan_parameters=scan_parameters,
                )

                with ProgressBar():
                    data = lazy_vol.compute()  # plain numpy array

                is_first = index == 0 or index == checkpoint
                is_last = index == num_tiles - 1

                ystart = int(round(y_coords[index]))

                overlap_with_prev = None
                if not is_first:
                    overlap_with_prev = corrected_sy - (
                        int(round(y_coords[index]))
                        - int(round(y_coords[index - 1]))
                    )
                overlap_with_next = None
                if not is_last:
                    overlap_with_next = corrected_sy - (
                        int(round(y_coords[index + 1]))
                        - int(round(y_coords[index]))
                    )

                if not is_first and overlap_with_prev and overlap_with_prev > 0:
                    t = np.linspace(0, 1, overlap_with_prev)
                    ramp = (1 - np.cos(np.pi * t)) / 2
                    ramp_inverse = (1 + np.cos(np.pi * t)) / 2
                    ramp = ramp[None, :, None]
                    ramp_inverse = ramp_inverse[None, :, None]

                    top_overlap = data[:, :overlap_with_prev, :]
                    data = data[:, overlap_with_prev:, :]

                    blended = bottom_overlap * ramp_inverse + top_overlap * ramp
                    data = np.concatenate([blended, data], axis=1)

                if not is_last and overlap_with_next and overlap_with_next > 0:
                    bottom_overlap = data[:, -overlap_with_next:, :].copy()
                    data = data[:, :-overlap_with_next, :]
                elif not is_last:
                    # No positive overlap with the next tile (e.g. a gap,
                    # or exactly adjacent tiles) -- nothing to carry
                    # forward for blending.
                    bottom_overlap = None

                zstart = 0

                if index > checkpoint:
                    logger.info(f"Storing tile {index} at y:{ystart}")
                    array[
                        zstart: zstart + data.shape[0],
                        ystart: ystart + data.shape[1],
                        0: data.shape[2],
                    ] = data

                logger.info(f"{name} done")
                _write_checkpoint(checkpoint_file, index)

        gc.collect()
        omz = ZarrPythonGroup.from_config(out_dir, zarr_config)
        array = omz["0"]
        omz.generate_pyramid_staged(
            levels=zarr_config.levels,
            copy_config=general_config,
            copy_zarr_config=zarr_config,
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

    end_timer = time.time()
    length = end_timer - start_timer
    logger.info(f"Conversion completed in {length / 60} minutes.")
