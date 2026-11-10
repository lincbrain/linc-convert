"""Register each channel to a reference channel (488) with multiview-stitcher.

Writes one YAML per channel in the format that ``stitch_chunks
--channel_registration`` reads: ``affine``, a 4 x 4 matrix in (z, y, x) order,
channel -> reference, in full-resolution voxels. The frames are the ones the
stitching uses: every chunk is cut to its channel's ``splitCrop`` (y range, all
z) and flipped in z if its camera flips.

The same chunk number of the channel's camera and of the reference's camera is
compared (the two cameras record the same place at the same time). Several chunks
can be used; the translations are averaged, weighted by the registration quality.
Registration is by default a translation (phase correlation; z and y only), or with
``--model rigid|similarity|affine`` also rotation / scale / shear (ANTsPy) and is best done on a very
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
from linc_convert.modalities.lsm.stitching_utils.channel_registration import (
    reference_origin,
)
from linc_convert.modalities.lsm.stitching_utils.io import (
    crop_and_flip,
    level_range,
    discover_tile_paths,
    open_tile_array,
    write_yaml,
)
from linc_convert.modalities.lsm.stitching_utils.registration import (
    read_volume_rows,
    register_volumes_model,
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


def dark_weighted(vol: np.ndarray, gamma: float) -> np.ndarray:
    """Give dim structure more weight than bright structure.

    The volume is scaled by its 99.5th percentile (so channels of different
    brightness become comparable) and raised to ``gamma``: with ``gamma < 1`` the
    bright values are compressed and the dim ones relatively stretched, so bright
    spots no longer decide the correlation. ``gamma = 1`` only scales.
    """
    if gamma <= 0:
        raise ValueError(f"dark_gamma must be positive, got {gamma}")
    top = float(np.percentile(vol, 99.5))
    if top <= 0:
        return vol.astype(np.float32)
    return (np.clip(vol / top, 0, 1) ** gamma).astype(np.float32)


def load_registration_volume(path: str, slice_number: int, camera: int, channel: str,
                             layer: int, params: dict[str, Any], z_trim_start: float,
                             z_trim_end: float, smooth_sigma: float,
                             subtract_background: bool, background_length: int,
                             corrections_dir: Optional[str],
                             corrections_layer: int,
                             corrections_crop: str = "stitchingCrop",
                             dark_gamma: float = 0.5,
                             crop_kind: str = "stitchingCrop",
                             ) -> tuple[np.ndarray, np.ndarray]:
    """One chunk of one channel: crop, flip, z trim, background, smoothing.

    With ``crop_kind="stitchingCrop"`` the volume is the region the stitch uses
    (the stitching crop in y and z), cut out of the split, flipped chunk; with
    ``"splitCrop"`` it is the whole split, flipped chunk.

    Returns
    -------
    volume : numpy.ndarray
        float32 (Z, Y, X) at ``layer``.
    origin : numpy.ndarray
        (z, y, x) position, in voxels of ``layer``, of the volume's first voxel in
        the channel's split, flipped chunk frame (the frame the registration file
        is in).
    """
    if crop_kind not in ("stitchingCrop", "splitCrop"):
        raise ValueError(f"crop_kind must be 'stitchingCrop' or 'splitCrop', got "
                         f"{crop_kind!r}")
    arr = open_tile_array(path, layer)
    crop = get_crop(params, camera, slice_number, channel,
                    kind="splitCrop", include_z=False)
    flip = get_vertical_flip(params, camera)
    vol = crop_and_flip(arr, crop, flip, layer)
    full_vol = vol
    origin = np.zeros(3)
    noise = scaler = None
    if corrections_dir is not None:
        noise, scaler = registration_corrections(
            path, slice_number, camera, channel, layer, params, corrections_dir,
            False, corrections_layer, corrections_crop)
    if crop_kind == "stitchingCrop":
        z0, z1, y0, y1 = get_crop(params, camera, slice_number, channel,
                                  kind="stitchingCrop", include_z=True)
        raw_z, raw_y, _ = arr.shape
        sy = level_range((y0, y1), raw_y, layer)
        split_y = level_range((crop[2], crop[3]), raw_y, layer)
        sz = (None, None) if z0 is None or z1 is None else level_range(
            (z0, z1), raw_z, layer)
        oz, oy, _ = reference_origin(split_y, sy, sz, raw_z, flip, layer)
        hz = raw_z if sz[0] is None else sz[1] - sz[0]
        oz, oy, hz, hy = int(oz), int(oy), int(hz), int(sy[1] - sy[0])
        vol = vol[oz:oz + hz, oy:oy + hy]
        if noise is not None:
            noise, scaler = noise[oz:oz + hz, oy:oy + hy], scaler[oz:oz + hz, oy:oy + hy]
        origin = np.array([oz, oy, 0], dtype=float)
    depth = vol.shape[0]
    start = z_trim_planes(z_trim_start, layer, depth, "z_trim_start") if z_trim_start > 0 else 0
    end = z_trim_planes(z_trim_end, layer, depth) if z_trim_end > 0 else 0
    if start + end >= depth:
        raise ValueError(f"z_trim_start {z_trim_start:g} + z_trim_end {z_trim_end:g} "
                         f"full-res px remove every z plane at layer {layer}")
    if noise is not None:
        noise, scaler = noise[start:depth - end], scaler[start:depth - end]
    vol = vol[start:depth - end]
    origin[0] += start
    out = read_volume_rows(vol, 0, vol.shape[1], 0, vol.shape[2],
                           max(1, background_length // 2 ** layer),
                           subtract_background, scaler=scaler, noise=noise)
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        out = gaussian_filter(out, smooth_sigma)
    out = dark_weighted(out, dark_gamma)
    print(f"  {Path(path).name} ({channel}, camera {camera}): {out.shape} at layer "
          f"{layer} ({crop_kind}, starts at (z,y)=({origin[0]:g},{origin[1]:g}) of the "
          f"split chunk), z trim first {start} / last {end} planes")
    return out, origin


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
                      crop_kind: str = "stitchingCrop",
                      dark_gamma: float = 0.5, fit_x: bool = False,
                      model: str = "translation",
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
    crop_kind : {"stitchingCrop", "splitCrop"}, default="stitchingCrop"
        Region that is registered: ``stitchingCrop`` = the stitching crop (y and z),
        i.e. the data that ends up in the stitched image; ``splitCrop`` = the whole
        split chunk. Either way the YAML is in the split, flipped chunk frames
        (what ``stitch_chunks --channel_registration`` reads); the stitching-crop
        corner is folded into the translation.
    dark_gamma : float, default=0.5
        Each volume is scaled by its 99.5th percentile and raised to this power
        before registering: below 1 the dim tissue counts more than bright spots,
        and channels of different brightness are made comparable. 1 = only scaled.
    fit_x : bool, default=False
        Also register x. By default only z and y are registered: the transform acts in
        the (z, y) plane and x is left alone.
    model : {"translation", "rigid", "similarity", "affine"}, default="translation"
        Transform family. ``translation`` is phase correlation; the others use ANTsPy
        through multiview-stitcher (needs ``antspyx``): rigid = rotation + translation,
        similarity = + uniform scale, affine = + anisotropic scale and shear. With
        several chunks the matrices and translations are averaged, weighted by quality.
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
                   corrections_crop=corrections_crop, dark_gamma=dark_gamma,
                   crop_kind=crop_kind)

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
                fixed, o_fix = load_registration_volume(
                    ref_chunks[n], camera=ref_cam, channel=reference_channel, **load_kw)
                moving, o_mov = load_registration_volume(
                    chunks[n], camera=cam, channel=channel, **load_kw)
                aff, quality = register_volumes_model(fixed, moving, model, fit_x=fit_x)
                # the volumes start at o (stitching crop and/or z trim) inside their
                # split chunks, and the registration is found about their own corners:
                # q - o_fix = A (p - o_mov) + t   =>   q = A p + t + o_fix - A o_mov,
                # the registration in the split frames (what stitch_chunks reads)
                aff = aff.copy()
                aff[:3, 3] += o_fix - aff[:3, :3] @ o_mov
                full = aff.copy()
                full[:3, 3] = aff[:3, 3] * scale     # the matrix part is unitless
                t = full[:3, 3]
                print(f"  chunk {n}: translation (z,y,x) = "
                      f"({t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}) full-res px, "
                      f"quality {quality:.3f}; matrix diag "
                      f"{np.round(np.diag(aff)[:3], 4).tolist()}")
                results.append((n, t, quality, full))
            q = np.array([r[2] for r in results], dtype=float)
            w = np.where(np.isfinite(q) & (q > 0), q, 0.0)
            if w.sum() == 0:
                print("  WARNING: no chunk has a positive quality; plain average")
                w = np.ones_like(q)
            affine = sum(wi * r[3] for wi, r in zip(w, results)) / w.sum()
            t_avg = affine[:3, 3]
            print(f"  weighted translation (z,y,x) = ({t_avg[0]:+.2f}, "
                  f"{t_avg[1]:+.2f}, {t_avg[2]:+.2f}) full-res px; matrix\n"
                  f"{np.array2string(affine[:3, :3], precision=4, suppress_small=True)}")
        write_yaml(target, {
            "slice": int(slice_number), "channel": channel,
            "reference_channel": reference_channel,
            "units": "full-resolution pixels, (z, y, x), channel -> reference",
            "affine": [[float(v) for v in row] for row in affine],
            "layer": int(layer),
            "chunks": [{"chunk": int(n), "affine": [[float(v) for v in row] for row in full],
                        "quality": None if not np.isfinite(qq) else float(qq)}
                       for n, t, qq, full in results],
            "settings": {"z_trim_end": float(z_trim_end),
                         "z_trim_start": float(z_trim_start),
                         "smooth_sigma": float(smooth_sigma), "crop": crop_kind, "frame": "split, flipped chunk",
                         "model": model, "axes": "z,y,x" if fit_x else "z,y", "dark_gamma": float(dark_gamma)}})