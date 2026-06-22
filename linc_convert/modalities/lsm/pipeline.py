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
"""

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
    *,
    dandiset_id: Optional[str],
    api_key: Optional[str],
) -> List[str]:
    """Get all tile paths from the input location, in on-disk order.

    Tiles are assumed to be laid out along a single axis (y) in the order
    they're returned here -- the index of a path in this list *is* its
    y-position, so no filename parsing is needed.
    """
    if dandiset_id is None:
        paths = sorted(glob(os.path.join(inp, "*.ome.zarr")))
        if not paths:
            raise ValueError("No tile folders found in input directory")
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
        )

    if not paths:
        raise ValueError("No tile assets found in DANDI dataset")

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
    *,
    overlap: int = 192,
    voxel_size: List[float] = [1, 1, 1],
    general_config: Optional[GeneralConfig] = None,
    zarr_config: Optional[ZarrConfig] = None,
    nii_config: Optional[NiftiConfig] = None,
    dandiset_id: Optional[str] = None,
) -> None:
    """
    Correct volumetric tile data and stream it directly into a single
    blended OME-Zarr mosaic along the y axis.

    This pipeline:
    1. Loads scan parameters and discovers tile paths for the chosen camera,
       in on-disk order (tiles are assumed to lie along y in that order)
    2. Corrects one tile to learn the post-correction tile shape, and
       estimates the full mosaic shape from that shape, the tile count,
       and the requested overlap
    3. For each tile in order: reads and corrects it (background mask,
       stripe/skew correction), blends its y-overlap with its neighbor,
       and writes it directly into the destination array

    No per-tile intermediate `.ome.zarr` is ever written to disk, and only
    one tile's worth of corrected data is held in memory at a time (there
    is no chunking along x; each tile is processed as a single unit).

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
    overlap : int
        Number of pixels of y-overlap between consecutive tiles.
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
        If camera_id is not 1 or 2.
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
        inp, dandiset_id=dandiset_id, api_key=api_key
    )

    num_tiles = len(tile_paths)

    def tile_name(path: str) -> str:
        return os.path.basename(path.rstrip("/").replace(".ome.zarr", ""))

    for ch in camera_channel_map[camera_id]:

        # --- Estimate the corrected mosaic shape from a single sample tile.
        #
        # Stripe/skew correction changes a tile's shape relative to its
        # raw, on-disk shape, so we can't use the raw reader shape directly.
        # All tiles are assumed to share the same shape, so we correct just
        # the first tile to learn the corrected (z, y, x) shape, and
        # extrapolate the full mosaic from that one tile's size, the tile
        # count, and the requested overlap -- without loading every tile.
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
        full_y = num_tiles * corrected_sy - (num_tiles - 1) * overlap
        full_z = corrected_sz
        fullshape = (full_z, full_y, full_x)

        out_dir = f"{general_config.out}/{ch}"
        omz = ZarrPythonGroup.from_config(out_dir, zarr_config)
        checkpoint_file = _checkpoint_path(general_config, ch)
        checkpoint = _read_checkpoint(checkpoint_file, -1)

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
            array = omz["0"]

        logger.info(
            "Writing channel %s, level 0 array with shape %s", ch, fullshape
        )

        bottom_overlap = None

        for index, path in enumerate(tile_paths):
            if index >= checkpoint:
                omz = ZarrPythonGroup.from_config(out_dir, zarr_config)
                array = omz["0"]
                name = tile_name(path)
                logger.info(f"[{index}] Processing {name}")

                # Build the lazy corrected volume graph for this tile, then
                # compute it -- one tile is one chunk, so this is the single
                # point where pixel data for this tile becomes resident in
                # memory. Tile 0's graph was already built above (to learn the
                # mosaic geometry), so reuse it instead of building it twice.
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

                if (overlap and (not is_first or not is_last)) and num_tiles > 1:
                    if not is_first:
                        t = np.linspace(0, 1, overlap)
                        ramp = (1 - np.cos(np.pi * t)) / 2
                        ramp_inverse = (1 + np.cos(np.pi * t)) / 2
                        ramp = ramp[None, :, None]
                        ramp_inverse = ramp_inverse[None, :, None]

                        top_overlap = data[:, :overlap, :]
                        data = data[:, overlap:, :]

                        blended = bottom_overlap * ramp_inverse + top_overlap * ramp
                        data = np.concatenate([blended, data], axis=1)

                    if not is_last:
                        bottom_overlap = data[:, -overlap:, :]
                        data = data[:, :-overlap, :]

                ystart = index * corrected_sy - \
                    (index * overlap if index > 0 else 0)
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
