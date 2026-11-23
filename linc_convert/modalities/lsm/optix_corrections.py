"""Noise / scaler maps from an optix shading estimator (CIDRE, IlluminationFunction, GammaMixture).

An alternative to ``create_corrections``: the maps have the same names and layout
(``{output_dir}/{channel}/{channel}_alt_zy_{noise,scaler}.tiff``), so registration and
``stitch_chunks`` use them unchanged (``--corrections_dir``). The estimator is fitted on frames
(the (z, y) planes at evenly spread x) taken from the given tiles, at ``fit_layer`` (a coarse
level keeps it fast); the smooth maps are then resized to the layer the stitch runs at.
"""
import os
from typing import List, Optional

import cyclopts
import numpy as np
import yaml

from linc_convert.modalities.lsm.cli import lsm
from linc_convert.modalities.lsm.stitching_utils.corrections import (
    compute_noise_map,
    correction_paths,
)
from linc_convert.modalities.lsm.stitching_utils.io import (
    crop_and_flip,
    discover_tile_paths,
    open_tile_array,
    write_tiff,
)
from linc_convert.modalities.lsm.stitching_utils.optix_shading import (
    METHODS,
    fit_maps,
    frame_columns,
    upsample_maps,
)
from linc_convert.modalities.lsm.stitching_utils.scan_metadata import (
    get_channel_names,
    get_crops,
    get_vertical_flip,
)
from linc_convert.modalities.lsm.stitching_utils.tile_selection import available_layer

optix_corrections = cyclopts.App(name="optix_corrections", help_format="markdown")
lsm.command(optix_corrections)


def collect_frames(tile_paths: List[str], crop, flip: bool, layer: int, background_length: int,
                   multiplier: float, frames_per_tile: int, min_signal: float):
    """Frames (n, z, y) and the median noise map (z, y) of a channel over several tiles."""
    scale = 2 ** layer
    bg = max(1, background_length // scale)
    vols = []
    for path in tile_paths:
        tile = open_tile_array(path, layer, chunks=(128, 128, 128))
        vols.append(crop_and_flip(tile, crop, flip, layer))
    noise = np.median(np.stack([compute_noise_map(v, bg) for v in vols]), axis=0)
    frames = []
    for path, vol in zip(tile_paths, vols):
        cols = frame_columns(vol, noise, multiplier, frames_per_tile, min_signal, bg)
        print(f"    {os.path.basename(path)}: {len(cols)} tissue frames", flush=True)
        for x in cols:
            frames.append(np.asarray(vol[:, :, int(x)].compute(), dtype=np.float32))
    if len(frames) < 4:
        raise ValueError("fewer than 4 frames with tissue were found: lower "
                         "min_frame_signal or pick other tiles")
    return np.stack(frames), noise.astype(np.float32)


@optix_corrections.default
def optix_corrections_cmd(
    input_path: str, yaml_path: str, camera_id: int, slice_number: int, *, method: str,
    output_dir: str, tiles: Optional[List[int]] = None, tiles_file: Optional[str] = None,
    layer: int = 0, fit_layer: int = 2, frames_per_tile: int = 60,
    min_frame_signal: float = 0.3, background_length: int = 5000,
    zy_threshold_multiplier: float = 1.1, crop_kind: str = "stitchingCrop",
    gain: Optional[float] = None, max_iterations: Optional[int] = None,
    dark_source: str = "estimate", only_channel: Optional[str] = None, force: bool = False,
) -> None:
    """Fit CIDRE, IlluminationFunction or GammaMixture and write noise / scaler maps.

    Parameters
    ----------
    input_path : str
        Folder with the slice's ``*.ome.zarr`` chunks.
    yaml_path : str
        Scan metadata YAML (channels, crops, camera flip).
    camera_id : int
        1 or 2.
    slice_number : int
        Slice number (for the crops).
    method : {"cidre", "illumination", "gamma_mixture"}
        The optix estimator. ``illumination`` is given the background (noise) map as its
        dark, because its method assumes dark-subtracted frames.
    output_dir : str
        Folder of the maps (same layout as ``create_corrections``).
    tiles : list of int, optional
        1-based tile numbers the estimator is fitted on.
    tiles_file : str, optional
        YAML with a ``tiles`` list (``pick_tissue_chunks``); used when ``tiles`` is not given.
    layer : int, default=0
        Pyramid level the maps are made for (the one the stitch runs at). 0 = full resolution.
    fit_layer : int, default=2
        Pyramid level the estimator is fitted at (never finer than ``layer``). The maps are
        resized to ``layer`` afterwards.
    frames_per_tile : int, default=60
        Frames with tissue taken from every tile.
    min_frame_signal : float, default=0.3
        A frame is used if at least this share of its pixels is above the noise map times
        ``zy_threshold_multiplier``.
    background_length : int, default=5000
        Trailing x columns (full-resolution pixels) the noise map is the median of.
    zy_threshold_multiplier : float, default=1.1
        Signal threshold multiplier.
    crop_kind : {"stitchingCrop", "splitCrop"}, default="stitchingCrop"
        Region the maps cover (``splitCrop`` when the stitch registers the channels).
    gain : float, optional
        Nominal gain (ADU per electron) of ``gamma_mixture``.
    max_iterations : int, optional
        Iteration limit of ``cidre`` / ``gamma_mixture``.
    dark_source : {"estimate", "noise_map"}, default="estimate"
        ``noise`` map of ``cidre`` / ``gamma_mixture``: the estimator's own dark, or the
        background map measured from the last x columns (as ``create_corrections`` does).
        Use ``noise_map`` if the estimated dark looks wrong (it needs background pixels).
    only_channel : str, optional
        Only this channel of the camera.
    force : bool, default=False
        Recompute maps that exist.

    Raises
    ------
    ValueError
        If ``method``, ``camera_id`` or ``crop_kind`` is unknown or no tiles are given.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if camera_id not in (1, 2):
        raise ValueError(f"camera_id must be 1 or 2, got {camera_id}")
    if crop_kind not in ("stitchingCrop", "splitCrop"):
        raise ValueError(f"crop_kind must be 'stitchingCrop' or 'splitCrop', got {crop_kind!r}")
    if not tiles and tiles_file:
        with open(tiles_file) as f:
            tiles = list(yaml.safe_load(f)["tiles"])
    if not tiles:
        raise ValueError("give tiles or tiles_file")
    with open(yaml_path) as f:
        params = yaml.safe_load(f)
    channels = get_channel_names(params, camera_id)
    if only_channel is not None:
        if only_channel not in channels:
            raise ValueError(f"only_channel {only_channel!r} is not in {channels}")
        channels = [only_channel]
    all_paths = discover_tile_paths(input_path, camera_id)
    bad = [t for t in tiles if not 1 <= t <= len(all_paths)]
    if bad:
        raise ValueError(f"tiles {bad} out of range (1..{len(all_paths)})")
    paths = [all_paths[t - 1] for t in tiles]
    crops = get_crops(params, camera_id, slice_number, kind=crop_kind, include_z=True)
    flip = get_vertical_flip(params, camera_id)
    fit_layer = available_layer(paths[0], max(fit_layer, layer))
    for ch in channels:
        out = correction_paths(output_dir, ch)
        if not force and all(os.path.exists(p) for p in out.values()):
            print(f"channel {ch}: maps exist, skipping (--force to recompute)")
            continue
        print(f"channel {ch}: {method} on tiles {tiles} at level {fit_layer}", flush=True)
        frames, noise = collect_frames(paths, crops[ch], flip, fit_layer, background_length,
                                       zy_threshold_multiplier, frames_per_tile,
                                       min_frame_signal)
        noise_map, scaler = fit_maps(method, frames, dark=noise, gain=gain,
                                     max_iterations=max_iterations,
                                     dark_source=dark_source)
        target = crop_and_flip(open_tile_array(paths[0], layer, chunks=(128, 128, 128)),
                               crops[ch], flip, layer).shape[:2]
        noise_map, scaler = upsample_maps(noise_map, scaler, target)
        write_tiff(out["noise"], noise_map)
        write_tiff(out["scaler"], scaler)
        print(f"  noise mean {noise_map.mean():.2f}, scaler {scaler.min():.3f}.."
              f"{scaler.max():.3f}, maps {target}\n  wrote {out['noise']}\n  wrote {out['scaler']}")