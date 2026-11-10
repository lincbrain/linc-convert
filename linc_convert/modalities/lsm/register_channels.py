"""Register each channel to a reference channel (488) with multiview-stitcher.

Writes one YAML per channel in the format that ``stitch_chunks
--channel_registration`` reads: ``affine``, a 4 x 4 matrix in (z, y, x) order,
channel -> reference, in full-resolution voxels. The frames are the ones the
stitching uses: every chunk is cut to its channel's ``splitCrop`` (y range, all
z) and flipped in z if its camera flips.

The same chunk number of the channel's camera and of the reference's camera is
compared (the two cameras record the same place at the same time). Several chunks
can be used; the translations are averaged, weighted by the registration quality.
Registration is a 3-D translation (phase correlation) and is best done on a very
small pyramid layer (4 or 5): the file holds full-resolution values.

Usage::

    linc-convert lsm register_channels <slice dir> --yaml_path meta.yaml \
        --slice_number 37 --channels 594 --channels 660 --channels 561 \
        --chunk_numbers 6 --chunk_numbers 7 --layer 4 --z_trim_end 10 \
        --out "registration/channel_{channel}.yaml"
"""

import re
from pathlib import Path
from typing import Any, Optional

import cyclopts
import numpy as np
import yaml

from linc_convert.modalities.lsm.cli import lsm
from linc_convert.modalities.lsm.register_chunks import (
    registration_corrections,
    z_trim_planes,
)
from linc_convert.modalities.lsm.stitching_utils.io import (
    crop_and_flip,
    discover_tile_paths,
    open_tile_array,
    write_yaml,
)
from linc_convert.modalities.lsm.stitching_utils.registration import (
    read_volume_rows,
    register_volumes_3d,
)
from linc_convert.modalities.lsm.stitching_utils.scan_metadata import (
    get_channel_names,
    get_crop,
    get_vertical_flip,
)


def _camera_of(params: dict, channel: str) -> int:
    """Camera (1 or 2) that records ``channel``."""
    for camera in (1, 2):
        if channel in [str(c) for c in get_channel_names(params, camera)]:
            return camera
    raise ValueError(f"channel {channel} is not in the channel layout of either camera")


def _chunk_number(path: str) -> int:
    found = re.search(r"chunk-?(\d+)", Path(path).name)
    if found is None:
        raise ValueError(f"cannot read the chunk number from {Path(path).name}")
    return int(found.group(1))


def _chunks_by_number(input_path: str, camera: int) -> dict[int, str]:
    return {_chunk_number(p): p for p in discover_tile_paths(input_path, camera)}


def load_registration_volume(path: str, slice_number: int, camera: int, channel: str,
                             layer: int, params: dict[str, Any], z_trim_start: float,
                             z_trim_end: float, smooth_sigma: float,
                             subtract_background: bool, background_length: int,
                             corrections_dir: Optional[str],
                             corrections_layer: int,
                             corrections_crop: str = "stitchingCrop") -> np.ndarray:
    """One chunk of one channel: split crop, flip, z trim, background, smoothing.

    Returns
    -------
    numpy.ndarray
        float32 (Z, Y, X) at ``layer``.
    """
    arr = open_tile_array(path, layer)
    crop = get_crop(params, camera, slice_number, channel,
                    kind="splitCrop", include_z=False)
    vol = crop_and_flip(arr, crop, get_vertical_flip(params, camera), layer)
    depth = vol.shape[0]
    start = z_trim_planes(z_trim_start, layer, depth, "z_trim_start") if z_trim_start > 0 else 0
    end = z_trim_planes(z_trim_end, layer, depth) if z_trim_end > 0 else 0
    if start + end >= depth:
        raise ValueError(f"z_trim_start {z_trim_start:g} + z_trim_end {z_trim_end:g} "
                         f"full-res px remove every z plane at layer {layer}")
    noise = scaler = None
    if corrections_dir is not None:
        noise, scaler = registration_corrections(
            path, slice_number, camera, channel, layer, params, corrections_dir,
            False, corrections_layer, corrections_crop)
        noise, scaler = noise[start:depth - end], scaler[start:depth - end]
    vol = vol[start:depth - end]
    out = read_volume_rows(vol, 0, vol.shape[1], 0, vol.shape[2],
                           max(1, background_length // 2 ** layer),
                           subtract_background, scaler=scaler, noise=noise)
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        out = gaussian_filter(out, smooth_sigma)
    print(f"  {Path(path).name} ({channel}, camera {camera}): {out.shape} at layer "
          f"{layer}, z trim first {start} / last {end} planes")
    return out


register_channels = cyclopts.App(name="register_channels", help_format="markdown")
lsm.command(register_channels)


@register_channels.default
def register_channels(input_path: str, *, yaml_path: str, slice_number: int,
                      channels: list[str], out: str, reference_channel: str = "488",
                      chunk_numbers: list[int], layer: int = 4,
                      z_trim_end: float = 10, z_trim_start: float = 0,
                      smooth_sigma: float = 1.0, subtract_background: bool = True,
                      background_length: int = 400,
                      corrections_dir: Optional[str] = None,
                      corrections_layer: int = 0,
                      corrections_crop: str = "splitCrop",
                      force: bool = False) -> None:
    """Register each channel to the reference channel and write one YAML per channel.

    Parameters
    ----------
    input_path : str
        Slice folder with the raw ``*.ome.zarr`` chunks of both cameras.
    yaml_path : str
        Scan metadata YAML.
    slice_number : int
        Physical slice number.
    channels : list of str
        Channels to register (the reference may be listed; it gets the identity).
    out : str
        Output path containing ``{channel}``, e.g. ``reg/slice037_{channel}.yaml``.
    reference_channel : str, default="488"
        Channel everything is registered to.
    chunk_numbers : list of int
        Chunk numbers (1 = chunk-0001) compared; the same number is read from the
        channel's camera and the reference's camera. Results are averaged,
        weighted by quality.
    layer : int, default=4
        Pyramid level; use a very small one (4 or 5).
    z_trim_end : float, default=10
        z planes (full-res px) removed from the end of every chunk (after crop and
        flip) before registering. z_trim_start does the same at the beginning.
    smooth_sigma : float, default=1.0
        Gaussian sigma (voxels of the layer) applied before registering; 0 = off.
    subtract_background, background_length
        Per-row background subtraction (see register_chunks).
    corrections_dir : str, optional
        ``create_corrections`` folder (stitching-crop maps made at
        ``corrections_layer``); used like register_chunks does.
    corrections_crop : {"splitCrop", "stitchingCrop"}, default="splitCrop"
        Crop the maps were made for (``create_corrections --crop_kind``); the
        channel registration of ``stitch_chunks`` needs ``splitCrop`` maps.
    force : bool, default=False
        Recompute files that already exist.

    Raises
    ------
    ValueError
        If ``out`` has no ``{channel}``, no chunks are given, or a channel/chunk is
        missing.
    """
    if "{channel}" not in out:
        raise ValueError("out must contain {channel}")
    if not chunk_numbers:
        raise ValueError("give at least one chunk number")
    reference_channel = str(reference_channel)
    channels = [str(c) for c in channels]
    scale = 2 ** layer
    with open(yaml_path) as f:
        params = yaml.safe_load(f)
    ref_cam = _camera_of(params, reference_channel)
    ref_chunks = _chunks_by_number(input_path, ref_cam)
    load_kw = dict(slice_number=slice_number, layer=layer, params=params,
                   z_trim_start=z_trim_start, z_trim_end=z_trim_end,
                   smooth_sigma=smooth_sigma, subtract_background=subtract_background,
                   background_length=background_length, corrections_dir=corrections_dir,
                   corrections_layer=corrections_layer,
                   corrections_crop=corrections_crop)

    for channel in channels:
        target = Path(out.format(channel=channel))
        if target.exists() and not force:
            print(f"{target} already exists, skipping (force=True to recompute)")
            continue
        print(f"=== channel {channel} -> {reference_channel} ===")
        if channel == reference_channel:
            affine, results = np.eye(4), []
        else:
            cam = _camera_of(params, channel)
            chunks = _chunks_by_number(input_path, cam)
            results = []
            for n in chunk_numbers:
                if n not in chunks or n not in ref_chunks:
                    raise ValueError(f"chunk {n} is missing for camera {cam} or "
                                     f"camera {ref_cam} in {input_path}")
                fixed = load_registration_volume(
                    ref_chunks[n], camera=ref_cam, channel=reference_channel, **load_kw)
                moving = load_registration_volume(
                    chunks[n], camera=cam, channel=channel, **load_kw)
                aff, quality = register_volumes_3d(fixed, moving)
                t = aff[:3, 3] * scale
                print(f"  chunk {n}: translation (z,y,x) = "
                      f"({t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}) full-res px, "
                      f"quality {quality:.3f}")
                results.append((n, t, quality))
            q = np.array([r[2] for r in results], dtype=float)
            w = np.where(np.isfinite(q) & (q > 0), q, 0.0)
            if w.sum() == 0:
                print("  WARNING: no chunk has a positive quality; plain average")
                w = np.ones_like(q)
            t_avg = sum(wi * r[1] for wi, r in zip(w, results)) / w.sum()
            affine = np.eye(4)
            affine[:3, 3] = t_avg
            print(f"  weighted translation (z,y,x) = ({t_avg[0]:+.2f}, "
                  f"{t_avg[1]:+.2f}, {t_avg[2]:+.2f}) full-res px")
        write_yaml(target, {
            "slice": int(slice_number), "channel": channel,
            "reference_channel": reference_channel,
            "units": "full-resolution pixels, (z, y, x), channel -> reference",
            "affine": [[float(v) for v in row] for row in affine],
            "layer": int(layer),
            "chunks": [{"chunk": int(n), "translation_zyx": [float(v) for v in t],
                        "quality": None if not np.isfinite(qq) else float(qq)}
                       for n, t, qq in results],
            "settings": {"z_trim_end": float(z_trim_end),
                         "z_trim_start": float(z_trim_start),
                         "smooth_sigma": float(smooth_sigma), "crop": "splitCrop",
                         "model": "translation (phase correlation)"}})