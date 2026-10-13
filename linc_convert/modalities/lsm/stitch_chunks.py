"""Stitch the chunks of one slice into a blended OME-Zarr mosaic along y.

A deliberately small version of ``pipeline``: no channel registration and a
single correction. Needs ``linc_convert`` only for its zarr output and config
(``ZarrConfig``, ``GeneralConfig``, ``ZarrPythonGroup``); the helpers are in the
``stitching_utils`` folder next to this file.

    stitch_chunks(
        input_path="…/strips_raw/sample-slice060", yaml_path="LSM_metadata.yaml",
        camera_id=1, slice_number=60,
        coords_path="…/registration_{channel}.yaml",  # from register_chunks
        corrections_dir="…/60_corrections",       # written by create_corrections
        voxel_size=[1.0, 1.0, 1.0],
        general_config=GeneralConfig(out="…/stitched"), zarr_config=ZarrConfig(...))

For each channel of the camera, and each chunk (level 0 only):

1. Crop with the channel's ``stitchingCrop`` (y, and z if defined) and flip z
   if the camera's ``verticalFlip`` is set.
2. Correction: ``(vol - noise).clip(0) * scaler`` using
   ``{corrections_dir}/{ch}/{ch}_alt_zy_noise.tiff`` and ``..._alt_zy_scaler.tiff``,
   cast to uint16.
3. Skew correction: shear x as a function of z (``x += shear_x * z``) from the
   skew angle and raw voxel sizes in the metadata YAML, and optionally z as a
   function of y (``z += shear_z * y``, from register_chunks' ``dz`` when
   ``dz_as_shear_z`` is set). The chunk grows so nothing is cut off.
4. Blend the overlap with the previous chunk (power ramp times 1/scaler
   confidence) and write everything lazily, chunk by chunk, into one uint16
   array.

Chunk positions come from the register_chunks YAML: chunk ``i`` starts at
``round(i * weighted_average.y_offset)`` (full-resolution pixels, in the
stitching-cropped frame). All chunks start at z = 0. ``weighted_average.dz``
is ignored unless ``dz_as_shear_z`` is set, which applies it as a z shear
(``shear_z = -dz / y_offset``) of every chunk.

Output: ``{general_config.out}/{channel}`` (OME-Zarr with a pyramid, written
with ``space_scale=voxel_size``).

Checkpoints: after each chunk (its blended overlap and its own rows) is fully
written, ``{general_config.out}/{channel}.stitch_checkpoint.json`` is updated.
Running again with the same inputs reopens the output, skips the finished
chunks (the last finished one is only rebuilt lazily, not rewritten, so the next
blend has its data) and carries on; a finished channel is skipped. Changed
inputs, or ``restart=True``, start over.
"""

import json
import os
import shutil
from dataclasses import replace
from typing import List, Optional

import cyclopts
import dask.array as da
import numpy as np
import yaml

from linc_convert.modalities.lsm.stitching_utils.blending import (
    blend_overlap,
    sheared_weights,
)
from linc_convert.modalities.lsm.stitching_utils.corrections import (
    apply_corrections,
    load_corrections,
    shear_volume,
    sheared_shape,
)
from linc_convert.modalities.lsm.stitching_utils.io import (
    crop_and_flip,
    atomic_write,
    discover_tile_paths,
    open_tile_array,
    start_dask,
)
from linc_convert.modalities.lsm.stitching_utils.registration import (
    load_tile_positions,
)
from linc_convert.modalities.lsm.stitching_utils.scan_metadata import (
    Crop,
    get_channel_names,
    get_crops,
    get_skew_shear,
    get_vertical_flip,
)
from linc_convert.utils.io.zarr.drivers.zarr_python import ZarrPythonGroup
from linc_convert.utils.zarr_config import GeneralConfig, ZarrConfig, autoconfig

# --------------------------------------------------------------------------
# One channel
# --------------------------------------------------------------------------


def _resolve_tile_paths(input_path: str, camera_id: int, coords_path: str) -> List[str]:
    with open(coords_path) as f:
        listed = (yaml.safe_load(f) or {}).get("chunks")
    if not listed:
        return discover_tile_paths(input_path, camera_id)
    tile_paths = [os.path.join(input_path, os.path.basename(str(c).rstrip("/")))
                  for c in listed]
    missing = [p for p in tile_paths if not os.path.isdir(p)]
    if missing:
        raise FileNotFoundError(f"Chunk(s) listed in {coords_path} not found in "
                                f"{input_path}: {missing[:3]}")
    return tile_paths


def _checkpoint_path(out_dir: str) -> str:
    """Checkpoint file of a channel, next to its output folder."""
    return out_dir.rstrip("/") + ".stitch_checkpoint.json"


def _read_checkpoint(path: str, signature: dict) -> Optional[dict]:
    """Return the checkpoint if it exists and belongs to this exact stitch."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            state = json.load(f)
    except (OSError, ValueError):
        return None
    if state.get("signature") != signature:
        print("  checkpoint was made with different inputs; starting over")
        return None
    return state


def _write_checkpoint(path: str, signature: dict, last_chunk: int,
                      done: bool = False) -> None:
    """Record that chunks ``0..last_chunk`` are completely written."""
    text = json.dumps({"signature": signature, "last_chunk": last_chunk,
                       "done": done})
    atomic_write(path, lambda tmp: tmp.write_text(text))


def corrected_tile(vol: da.Array, noise: np.ndarray, scaler: np.ndarray,
                   shear_x: float, shear_z: float) -> da.Array:
    """Lazily correct and shear a whole chunk.

    Parameters
    ----------
    vol : dask.array.Array
        Cropped, flipped raw chunk, shape (Z, Y, X).
    noise : numpy.ndarray
        Background map, shape (Z, Y).
    scaler : numpy.ndarray
        Multiplier map, shape (Z, Y).
    shear_x : float
        x shift per z step (>= 0).
    shear_z : float
        z shift per y step; negative values shift z towards smaller values with
        increasing y, and the chunk is padded so that z starts at 0.

    Returns
    -------
    dask.array.Array
        uint16 array of shape ``sheared_shape(vol.shape, ...)``; computing it
        runs ``(vol - noise).clip(0) * scaler``, then the shear (nearest
        neighbour). Nothing is read until it is computed.
    """
    corrected = apply_corrections(vol, noise, scaler)
    if shear_z >= 0:
        return shear_volume(corrected, shear_x, shear_z)
    # negative z shear: shear x first (it needs the original z), then flip z,
    # shear by |shear_z| and flip back
    corrected = shear_volume(corrected, shear_x, 0.0)
    return shear_volume(corrected[::-1], 0.0, -shear_z)[::-1]


def _weights(scaler: np.ndarray, shear_z: float, z_out: int) -> np.ndarray:
    """``sheared_weights`` that also accepts a negative ``shear_z``."""
    if shear_z >= 0:
        return sheared_weights(scaler, shear_z, z_out)
    return sheared_weights(scaler[::-1], -shear_z, z_out)[::-1]


def stitch_channel(
    tile_paths: List[str], channel: str, crop: Crop,
    vertical_flip: bool, shear_x: float, shear_z: float, positions_y: List[int],
    positions_z: List[int], corrections_dir: str, out_dir: str,
    voxel_size: List[float], general_config: GeneralConfig, zarr_config: ZarrConfig,
) -> None:
    """Stitch all chunks of one channel into an OME-Zarr mosaic, lazily.

    Parameters
    ----------
    tile_paths : list of str
        Chunk ``.ome.zarr`` folders in acquisition order.
    channel : str
        Channel name.
    crop : tuple
        ``(z_start, z_end, y_start, y_end)`` stitching crop.
    vertical_flip : bool
        Flip each chunk along z.
    shear_x : float
        x shift per z step (>= 0).
    shear_z : float
        z shift per y step (may be negative).
    positions_y : list of int
        y start of every chunk in the mosaic, increasing.
    positions_z : list of int
        z origin of every *sheared* chunk in the mosaic (their minimum is
        moved to 0). This is the final placement, shear included.
    corrections_dir : str
        Folder with the ``create_corrections`` output.
    out_dir : str
        Output ``.ome.zarr`` folder of this channel.
    voxel_size : list of float
        Voxel size in z, y, x order (already reversed).
    general_config : GeneralConfig
        Output configuration (its ``out`` is replaced for the pyramid copy).
    zarr_config : ZarrConfig
        Zarr storage configuration.

    Notes
    -----
    Progress is checkpointed after every chunk in
    ``{out_dir}.stitch_checkpoint.json`` and resumed from there; see the module
    docstring.

    Raises
    ------
    ValueError
        If the position lists do not match the tiles, ``shear_x`` is negative,
        chunk positions do not increase, chunk shapes differ, or the overlap
        from the previous and next chunk exceed the chunk height together.
    """
    n = len(tile_paths)
    if not (len(positions_y) == len(positions_z) == n):
        raise ValueError(f"{n} tiles but {len(positions_y)} y positions and "
                         f"{len(positions_z)} z positions")
    if shear_x < 0:
        raise ValueError("shear_x must be >= 0")
    steps = np.diff(positions_y)
    if n > 1 and steps.min() <= 0:
        raise ValueError("chunk positions must increase")

    def open_tile(path: str) -> da.Array:
        return crop_and_flip(open_tile_array(path), crop, vertical_flip)

    raw_shape = open_tile(tile_paths[0]).shape
    sz, sy, sx = raw_shape
    z_out, _, x_out = sheared_shape(raw_shape, shear_x, abs(shear_z))
    noise, scaler = load_corrections(corrections_dir, channel)
    weights = _weights(scaler, shear_z, z_out)
    if n > 1 and steps.max() > sy:
        print(f"  WARNING: gap between chunks ({steps.max()} > chunk height {sy}); "
              "the gap is left empty")

    z_origin = [int(z) - int(min(positions_z)) for z in positions_z]
    fullshape = (z_out + max(z_origin), positions_y[-1] + sy, x_out)
    print(
        f"channel {channel}: {n} chunks, chunk (Z,Y,X)={raw_shape}, mosaic {fullshape}")

    signature = {"tiles": [os.path.basename(p) for p in tile_paths],
                 "positions_y": [int(y) for y in positions_y],
                 "z_origin": z_origin, "shape": list(fullshape),
                 "shear": [float(shear_x), float(shear_z)]}
    ckpt = _checkpoint_path(out_dir)
    state = _read_checkpoint(ckpt, signature)
    if state is not None and not os.path.isdir(out_dir):
        state = None
    if state is not None and state["done"]:
        print(
            f"channel {channel}: already stitched (checkpoint {ckpt}); skipping")
        return

    if state is None:
        # fresh start: discard any partial output of an unfinished earlier run
        shutil.rmtree(out_dir, ignore_errors=True)
        if os.path.exists(ckpt):
            os.remove(ckpt)
        group = ZarrPythonGroup.from_config(out_dir, zarr_config)
        array = group.create_array("0", shape=fullshape, zarr_config=zarr_config,
                                   dtype=np.uint16)
        start = 0
    else:
        # resume: reopen the output and rebuild the last finished chunk lazily
        # (nothing is written for it) so the next blend can use it
        group = ZarrPythonGroup.open(out_dir, mode="r+")
        array = group["0"]
        start = int(state["last_chunk"]) + 1
        print(f"channel {channel}: resuming after chunk {start - 1} "
              f"({start}/{n} written)")

    prev: Optional[da.Array] = None
    if start > 0:
        raw = open_tile(tile_paths[start - 1])
        if raw.shape != raw_shape:
            raise ValueError(f"{os.path.basename(tile_paths[start - 1])} has shape "
                             f"{raw.shape}, expected {raw_shape}")
        prev = corrected_tile(raw, noise, scaler, shear_x, shear_z)

    for index in range(start, n):
        path = tile_paths[index]
        raw = open_tile(path)
        if raw.shape != raw_shape:
            raise ValueError(f"{os.path.basename(path)} has shape {raw.shape}, "
                             f"expected {raw_shape}")
        ystart, zstart = positions_y[index], z_origin[index]
        overlap_prev = max(
            sy - (ystart - positions_y[index - 1]), 0) if index > 0 else 0
        overlap_next = (max(sy - (positions_y[index + 1] - ystart), 0)
                        if index < n - 1 else 0)
        if overlap_prev + overlap_next > sy:
            raise ValueError(f"chunk {index}: overlaps ({overlap_prev} + "
                             f"{overlap_next}) exceed the chunk height {sy}")
        print(f"[{index}] {os.path.basename(path)}  y={ystart} z={zstart}  "
              f"overlap prev/next = {overlap_prev}/{overlap_next}")

        cur = corrected_tile(raw, noise, scaler, shear_x, shear_z)
        if overlap_prev > 0:
            assert prev is not None
            blended = blend_overlap(prev, cur, weights, weights, z_origin[index - 1],
                                    zstart, fullshape[0], overlap_prev)
            da.store(blended, array, regions=(slice(0, fullshape[0]),
                                              slice(ystart, ystart +
                                                    overlap_prev),
                                              slice(0, x_out)), lock=True)
        first_row, last_row = overlap_prev, sy - overlap_next
        if last_row > first_row:
            da.store(cur[:, first_row:last_row],
                     array,
                     regions=(slice(zstart, zstart + z_out),
                              slice(ystart + first_row, ystart + last_row),
                              slice(0, x_out)), lock=True)
        prev = cur
        _write_checkpoint(ckpt, signature, index)

    group.generate_pyramid_staged(levels=zarr_config.levels,
                                  copy_config=replace(
                                      general_config, out=out_dir),
                                  copy_zarr_config=zarr_config)
    group.write_ome_metadata(axes=["z", "y", "x"], space_scale=voxel_size)
    _write_checkpoint(ckpt, signature, n - 1, done=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

app = cyclopts.App(name="stitch_chunks", help_format="markdown")


@app.default
@autoconfig
def stitch_chunks(
    input_path: str,
    yaml_path: str,
    camera_id: int,
    slice_number: int,
    coords_path: str,
    corrections_dir: str,
    *,
    voxel_size: List[float] = [1, 1, 1],
    memory_gb: int = 64,
    dz_as_shear_z: bool = False,
    restart: bool = False,
    general_config: Optional[GeneralConfig] = None,
    zarr_config: Optional[ZarrConfig] = None,
) -> None:
    """Stitch the chunks of one slice and camera into blended OME-Zarr mosaics.

    Parameters
    ----------
    input_path : str
        Folder with the chunk ``*.ome.zarr`` folders. If the register_chunks
        YAML lists ``chunks``, only those (matched by file name) are stitched,
        in that order; otherwise all chunks of the camera, sorted.
    yaml_path : str
        Scan metadata YAML (crop definitions, flip, skew).
    camera_id : int
        Camera to stitch (1 or 2). Every channel of the camera is written.
    slice_number : int
        Physical slice number (selects the config epoch).
    coords_path : str
        YAML written by ``register_chunks``. May contain ``{channel}``, which is
        replaced by each channel's name, so that every channel uses its own
        registration (e.g. ``registration/slice060_{channel}.yaml``).
    corrections_dir : str
        Folder written by ``create_corrections``.
    voxel_size : list of float
        Voxel size in x, y, z (microns) for the output metadata.
    memory_gb : int, default=64
        Total memory, in GB, that the dask cluster may use.
    dz_as_shear_z : bool, default=False
        If False, ``weighted_average.dz`` from register_chunks is not applied.
        If True, it is applied as a z shear of every chunk instead of as
        offsets between chunks: ``shear_z = -dz / y_offset`` (z shift per y
        step), with all chunk z offsets left at 0. A chunk's rows then move
        in z by ``dz`` over one chunk step, which removes the z drift measured
        between neighbouring chunks.
    restart : bool, default=False
        Ignore any checkpoints and rewrite every channel from scratch.
    general_config : GeneralConfig
        Output configuration; ``out`` is the output folder.
    zarr_config : ZarrConfig
        Zarr storage configuration (chunking, pyramid levels, ...).

    Raises
    ------
    ValueError
        If ``general_config.out`` is not set, no chunks are found, or the
        chunks and the register_chunks result disagree.
    FileNotFoundError
        If a listed chunk or a correction file is missing.
    """
    if general_config is None or not getattr(general_config, "out", None):
        raise ValueError("general_config.out (the output folder) is required")
    voxel_size = list(map(float, reversed(voxel_size)))

    with open(yaml_path) as f:
        scan_parameters = yaml.safe_load(f)
    crops = get_crops(scan_parameters, camera_id, slice_number)
    flip = get_vertical_flip(scan_parameters, camera_id)
    shear_x = get_skew_shear(scan_parameters)
    channels = get_channel_names(scan_parameters, camera_id)
    for channel in channels:
        if channel not in crops:
            raise KeyError(f"No stitchingCrop for channel '{channel}'")
        if not os.path.exists(coords_path.format(channel=channel)):
            raise FileNotFoundError(
                f"No register_chunks result for channel '{channel}': "
                f"{coords_path.format(channel=channel)}")

    client = start_dask(memory_gb)
    try:
        for channel in channels:
            if restart:
                ckpt = _checkpoint_path(
                    os.path.join(general_config.out, channel))
                if os.path.exists(ckpt):
                    os.remove(ckpt)
            coords = coords_path.format(channel=channel)
            tile_paths = _resolve_tile_paths(input_path, camera_id, coords)
            positions_y, dz = load_tile_positions(coords, len(tile_paths))
            positions_z = [0] * len(tile_paths)
            shear_z = 0.0
            if dz_as_shear_z:
                with open(coords) as f:
                    y_offset = float(
                        yaml.safe_load(f)["weighted_average"]["y_offset"])
                shear_z = -dz / y_offset
                print(f"channel {channel}: dz = {dz:.3f} px over y_offset "
                      f"{y_offset:.2f} px -> shear_z = {shear_z:.6f}")
            elif abs(dz) > 0.5:
                print(f"channel {channel}: NOTE dz = {dz:.2f} px is not applied "
                      "(dz_as_shear_z=False)")
            stitch_channel(tile_paths, channel, crops[channel], flip, shear_x,
                           shear_z, positions_y, positions_z, corrections_dir,
                           os.path.join(general_config.out, channel),
                           voxel_size, general_config, zarr_config)
    finally:
        client.close()


if __name__ == "__main__":
    app()
