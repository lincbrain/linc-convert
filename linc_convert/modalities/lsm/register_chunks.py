"""Register consecutive raw LSM chunks with multiview-stitcher (z and y only).

Reads existing local files only (never contacts DANDI).

    from register_chunks import register_chunks
    register_chunks(
        chunks=["…/sub-MF283_sample-slice060_chunk-0023_acq-camera-01.ome.zarr",
                "…/sub-MF283_sample-slice060_chunk-0024_acq-camera-01.ome.zarr"],
        channel="594", slice_numb=60, camera_numb=1,
        yaml_path="LSM_metadata.yaml", overlap=150, layer=1, out="registration.yaml")

* Each chunk is cropped to the channel's ``splitCrop`` yRange (from the metadata
  YAML; all z) and then flipped in z per the camera's ``verticalFlip``.
* Starting offset: ``overlap`` is your estimate of how many y rows (full-res
  pixels, after the splitCrop) the end of chunk A shares with the start of
  chunk B. The starting y offset is then (cropped chunk height - overlap); z = 0.
* The per-row background (median of the last ``background_length`` x columns) is
  subtracted, then the whole overlap volume (all x) is registered with
  phase correlation in 3-D. x is never translated: its fitted component is
  printed (unconstrained x shift) and forced to 0.

``layer`` is the pyramid level (0 = full resolution). Memory is about
overlap rows x Z x X x 4 bytes x a few; use layer >= 1 for full-size chunks.

Output (``out``): a YAML file with the quality-weighted average z shift and y
offset (full-resolution pixels), the mean quality, and the per-pair results.

Needs: multiview-stitcher, dask, numpy, scipy, zarr, pyyaml, cyclopts, and the
``stitching_utils`` folder next to this file.
"""

import time
from pathlib import Path
from typing import Any, Union

import cyclopts
import dask.array as da
import numpy as np
import yaml

from linc_convert.modalities.lsm.cli import lsm
from linc_convert.modalities.lsm.stitching_utils.io import (
    crop_and_flip,
    open_tile_array,
    write_yaml,
)
from linc_convert.modalities.lsm.stitching_utils.registration import (
    Pair,
    average_pairs,
    read_volume_rows,
    register_volume_zy,
)
from linc_convert.modalities.lsm.stitching_utils.scan_metadata import (
    get_crop,
    get_vertical_flip,
)
from linc_convert.utils.zarr_config import autoconfig

PathLike = Union[str, Path]

# --------------------------------------------------------------------------
# Registering chunks
# --------------------------------------------------------------------------


def load_chunk(path: PathLike, slice_numb: int, camera_numb: int, channel: str,
               layer: int, params: dict[str, Any]) -> da.Array:
    """Open one raw chunk lazily, cropped and flipped.

    The chunk is cropped to the channel's ``splitCrop`` yRange (all z), then
    flipped in z per the camera's ``verticalFlip``.

    Parameters
    ----------
    path : str or pathlib.Path
        Chunk ``.ome.zarr`` folder.
    slice_numb : int
        Physical slice number (selects the crop definitions).
    camera_numb : int
        Camera number (1 or 2).
    channel : str
        Channel name, e.g. ``"594"``.
    layer : int
        Pyramid level (0 = full resolution).
    params : dict
        Parsed scan metadata YAML.

    Returns
    -------
    dask.array.Array
        Lazy volume, shape (Z, Y, X).

    Raises
    ------
    FileNotFoundError
        If the chunk folder does not exist.
    """
    arr = open_tile_array(path, layer)
    crop = get_crop(params, camera_numb, slice_numb, channel,
                    kind="splitCrop", include_z=False)
    vol = crop_and_flip(arr, crop, get_vertical_flip(
        params, camera_numb), layer)
    print(f"  {Path(path).name}: splitCrop y {crop[2]}:{crop[3]} "
          f"(layer {layer}, array {arr.shape})")
    return vol


def register_two_chunks(chunk_a: PathLike, chunk_b: PathLike, channel: str,
                        slice_numb: int, camera_numb: int, yaml_path: PathLike,
                        overlap: float, layer: int = 1, background_length: int = 400,
                        subtract_background: bool = True,
                        smooth_sigma: float = 1.0) -> Pair:
    """Register chunk B against chunk A (translation in z and y only).

    Parameters
    ----------
    chunk_a, chunk_b : str or pathlib.Path
        Raw chunk ``.ome.zarr`` folders; B follows A.
    channel : str
        Channel name, e.g. ``"594"``.
    slice_numb : int
        Physical slice number (selects the crop definitions).
    camera_numb : int
        Camera that recorded the channel (1 or 2).
    yaml_path : str or pathlib.Path
        Scan metadata YAML (crop definitions, channel layout).
    overlap : float
        Estimated y overlap of A's end and B's start, in full-resolution
        pixels (after the splitCrop).
    layer : int, default=1
        Pyramid level (0 = full resolution).
    background_length : int, default=400
        Trailing x columns for the per-row background (full-resolution pixels).
    subtract_background : bool, default=True
        Subtract that background before registering.
    smooth_sigma : float, default=1.0
        Gaussian sigma (in this level's pixels) applied before registering;
        0 disables it.

    Returns
    -------
    dz : float
        z shift of B relative to A, in full-resolution pixels.
    y_offset : float
        y distance between the origins of A and B, in full-resolution pixels.
    quality : float
        Registration quality.

    Raises
    ------
    FileNotFoundError
        If a chunk folder does not exist.
    ValueError
        If ``overlap`` is not between 0 and the cropped chunk height.
    """
    channel = str(channel)
    scale = 2 ** layer

    with open(yaml_path) as f:
        params = yaml.safe_load(f)

    print(f"slice {slice_numb} channel {channel} camera {camera_numb} chunks "
          f"{Path(chunk_a).name}->{Path(chunk_b).name} layer {layer}")

    vol_a = load_chunk(chunk_a, slice_numb, camera_numb,
                       channel, layer, params)
    vol_b = load_chunk(chunk_b, slice_numb, camera_numb,
                       channel, layer, params)
    Za, sy, sx = vol_a.shape
    if vol_b.shape != vol_a.shape:
        print(f"WARNING: chunk shapes differ: {vol_a.shape} vs {vol_b.shape}")
    # starting y offset = cropped chunk height - estimated overlap
    init_y = sy - overlap / scale
    init_y_full = init_y * scale
    iy = int(round(init_y))
    print(f"estimated overlap {overlap:g} px full-res -> starting y offset "
          f"{init_y_full:g} px full-res ({init_y:g} at this layer; chunk is "
          f"{sy} rows tall at this layer), z = 0")
    if not 0 < iy < sy:
        raise ValueError(f"overlap {overlap:g} (full-res px) is outside the chunk "
                         f"(height {sy * scale} full-res px); it must be between "
                         "0 and the height")
    # nominal overlap: A's rows [iy, sy) and B's rows [0, sy - iy)
    a_y0, b_rows = iy, min(vol_b.shape[1], sy - iy)
    bg_len = max(1, background_length // scale)
    nbytes = ((sy - a_y0) + b_rows) * Za * sx * 4
    print(f"A rows {a_y0}:{sy} (of {sy}), B rows 0:{b_rows}; "
          f"reading {nbytes / 2**30:.2f} GiB of float32 (+ similar temporaries)"
          + ("; consider a higher layer" if nbytes > 2**33 else ""))

    t0 = time.time()
    sub_a = read_volume_rows(vol_a, a_y0, sy, 0, sx,
                             bg_len, subtract_background)
    sub_b = read_volume_rows(vol_b, 0, b_rows, 0, sx,
                             bg_len, subtract_background)
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        sub_a = gaussian_filter(sub_a, smooth_sigma)
        sub_b = gaussian_filter(sub_b, smooth_sigma)
    print(f"read in {time.time() - t0:.1f}s: A {sub_a.shape}, B {sub_b.shape}")
    residual, quality, info = register_volume_zy(sub_a, sub_b, a_y0, (0.0, iy))
    del sub_a, sub_b
    print("unconstrained x shift (ignored): "
          f"{info['unconstrained_x_shift_px']:+.3f} px")

    dz, dy = residual
    tot_y = iy + dy

    return dz * scale, tot_y * scale, quality


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

register_chunks = cyclopts.App(
    name="register_chunks", help_format="markdown")
lsm.command(register_chunks)


@register_chunks.default
@autoconfig
def register_chunks(chunks: list[str],
                    *, channel: str, slice_numb: int, camera_numb: int,
                    yaml_path: str, overlap: float, out: str, layer: int = 1,
                    background_length: int = 400, subtract_background: bool = True,
                    smooth_sigma: float = 1.0, force: bool = False) -> None:
    """Register every pair of consecutive chunks and average by registration quality.

    Each pair ``(chunks[i], chunks[i+1])`` is run through
    ``register_two_chunks``, giving ``(dz, y_offset, quality)`` in
    full-resolution pixels: the z shift of the next chunk relative to the
    previous one, and the y distance between their origins. The YAML holds the
    quality-weighted averages

        dz = sum(q_i * dz_i) / sum(q_i),   y = sum(q_i * y_i) / sum(q_i)

    the mean quality, and every pair's values and weight. Pairs whose quality is
    NaN or <= 0 get no weight (and are reported); if no pair has positive
    quality the plain average is used instead, with a warning.

    Parameters
    ----------
    chunks : list of str
        Raw chunk ``.ome.zarr`` paths in acquisition order (chunk i+1 follows
        chunk i); at least two.
    channel : str
        Channel name, e.g. ``"594"``.
    slice_numb : int
        Physical slice number (selects the crop definitions).
    camera_numb : int
        Camera that recorded the channel (1 or 2).
    yaml_path : str
        Scan metadata YAML (crop definitions, channel layout).
    overlap : float
        Estimated y overlap of consecutive chunks, in full-resolution pixels
        (after the splitCrop). The same estimate is used for every pair.
    out : str
        Path of the YAML file to write.
    layer : int, default=1
        Pyramid level (0 = full resolution).
    background_length : int, default=400
        Trailing x columns for the per-row background (full-resolution pixels).
    subtract_background : bool, default=True
        Subtract that background before registering.
    smooth_sigma : float, default=1.0
        Gaussian sigma (in this level's pixels) applied before registering;
        0 disables it.
    force : bool, default=False
        Recompute even if ``out`` already exists.

    Raises
    ------
    ValueError
        If fewer than two chunks are given, or ``overlap`` is outside the chunk.
    FileNotFoundError
        If a chunk folder does not exist.
    """
    out_path = Path(out)
    if out_path.exists() and not force:
        print(f"{out_path} already exists, skipping (force=True to recompute)")
        return
    chunks = list(chunks)
    if len(chunks) < 2:
        raise ValueError("need at least two chunk paths")

    pairs: list[Pair] = []
    for i, (a, b) in enumerate(zip(chunks[:-1], chunks[1:])):
        print(f"=== pair {i + 1}/{len(chunks) - 1}: "
              f"{Path(a).name} -> {Path(b).name} ===")
        res = register_two_chunks(a, b, channel, slice_numb, camera_numb, yaml_path,
                                  overlap, layer=layer,
                                  background_length=background_length,
                                  subtract_background=subtract_background,
                                  smooth_sigma=smooth_sigma)
        pairs.append((float(res[0]), float(res[1]), float(res[2])))

    avg = average_pairs(pairs)
    w = avg["weights"]
    mean_dz, mean_y, mean_q = avg["dz"], avg["y_offset"], avg["mean_quality"]
    for i, (d, yy, qq) in enumerate(pairs):
        print(f"  pair {i + 1}: dz={d:+.3f}  y={yy:.3f}  quality={qq:.3f}  "
              f"weight={w[i]:.2f}")
    print(f"weighted average: dz={mean_dz:+.3f}, y offset={mean_y:.3f} px (full-res); "
          f"mean quality {mean_q:.3f}")

    result: dict[str, Any] = {
        "slice": int(slice_numb),
        "channel": str(channel),
        "camera": int(camera_numb),
        "layer": int(layer),
        "estimated_overlap_full_res": float(overlap),
        "units": "full-resolution pixels",
        "weighted_average": {
            "dz": mean_dz,
            "y_offset": mean_y,
            "mean_quality": mean_q,
        },
        "pairs": [
            {"chunk_a": Path(a).name, "chunk_b": Path(b).name,
             "dz": float(d), "y_offset": float(yy),
             "quality": None if not np.isfinite(qq) else float(qq),
             "weight": float(w[i])}
            for i, ((a, b), (d, yy, qq)) in enumerate(
                zip(zip(chunks[:-1], chunks[1:]), pairs))
        ],
        "settings": {"background_length": int(background_length),
                     "subtract_background": bool(subtract_background),
                     "smooth_sigma": float(smooth_sigma), "crop": "splitCrop"},
        "chunks": [str(c) for c in chunks],
    }
    write_yaml(out_path, result)
    print(f"wrote {out_path}")
