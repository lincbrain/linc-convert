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
  chunk B. ``overlap_factor`` (default 1.5) scales it: the rows registered are
  overlap x overlap_factor, so an overlap larger than estimated is still inside the
  window. The starting y offset is then (cropped chunk height - exposed overlap);
  z = 0. The returned y offset is absolute, so it does not depend on the factor.
* ``z_trim_start`` is the same for the BEGINNING of z (full-res pixels, default 0).
* ``z_trim_end`` (full-res pixels, default 0) removes that many z planes from the END
  of every cropped, flipped chunk before registering, e.g. an empty region that
  would mislead the correlation. Only the registration volumes are cut: the
  correction maps are cut the same way for this step, and neither the maps nor the
  stitching crops change. z positions start at the first plane, so the registered
  ``dz`` keeps its meaning.
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

import re
import time
import warnings
from pathlib import Path
from typing import Any, Optional, Union

import cyclopts
import dask.array as da
import numpy as np
import yaml

from linc_convert.modalities.lsm.cli import lsm
from linc_convert.modalities.lsm.stitching_utils.corrections import (
    load_corrections,
)
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


def z_trim_planes(z_trim_end: float, layer: int, depth: int,
                  name: str = "z_trim_end") -> int:
    """z planes at ``layer`` that ``z_trim_end`` (full-resolution pixels) removes.

    Rounded up so a requested empty region is fully removed.

    Parameters
    ----------
    z_trim_end : float
        Planes to remove from the end of z, in full-resolution pixels (>= 0).
    layer : int
        Pyramid level of the volume.
    depth : int
        Number of z planes of the volume at ``layer``.

    Returns
    -------
    int
        Planes to cut at ``layer``.

    Raises
    ------
    ValueError
        If ``z_trim_end`` is negative or would remove every plane.
    """
    if z_trim_end < 0:
        raise ValueError(f"{name} must be >= 0")
    planes = int(-(-float(z_trim_end) // (2 ** layer)))
    if planes >= depth:
        raise ValueError(f"{name} {z_trim_end:g} full-res px removes {planes} of "
                         f"the {depth} z planes at layer {layer}")
    return planes


def _block_mean(frame: np.ndarray, factor: int) -> np.ndarray:
    """Mean over ``factor`` x ``factor`` blocks, ignoring NaN (all-NaN stays NaN)."""
    if factor == 1:
        return frame
    nz, ny = frame.shape
    pz, py = -nz % factor, -ny % factor
    padded = np.pad(frame, ((0, pz), (0, py)), constant_values=np.nan)
    blocks = padded.reshape(padded.shape[0] // factor, factor,
                            padded.shape[1] // factor, factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(blocks, axis=(1, 3)).astype(np.float32)


def registration_corrections(
        path: PathLike, slice_numb: int, camera_numb: int, channel: str,
        layer: int, params: dict[str, Any], corrections_dir: PathLike,
        per_chunk_corrections: bool = False, corrections_layer: int = 0,
        corrections_crop: str = "stitchingCrop",
) -> tuple[np.ndarray, np.ndarray]:
    """Load the noise and scaler maps of a chunk, laid out on the registration volume.

    ``create_corrections`` makes its maps for the ``stitchingCrop`` (z range
    and y range, z flipped if the camera is), at the pyramid level
    ``corrections_layer``. The chunks are registered at ``layer`` (>=
    ``corrections_layer``) on the ``splitCrop`` y range with all z. The maps
    are put back into the raw chunk's frame, averaged down to ``layer`` and cut
    to the registration volume. Rows the maps do not cover get scaler 1 and
    noise NaN (meaning: measure the background from the chunk).

    Parameters
    ----------
    path : str or pathlib.Path
        Chunk ``.ome.zarr`` folder.
    slice_numb, camera_numb : int
        Slice and camera (select the crop definitions).
    channel : str
        Channel name.
    layer : int
        Pyramid level the chunks are registered at.
    params : dict
        Parsed scan metadata YAML.
    corrections_dir : str or pathlib.Path
        Folder written by ``create_corrections``.
    per_chunk_corrections : bool, default=False
        Use the chunk's own maps; the chunk number is read from the folder
        name (``..._chunk-0023_...`` is chunk 23).
    corrections_layer : int, default=0
        Pyramid level ``create_corrections`` was run at (``--layer`` there).
    corrections_crop : {"stitchingCrop", "splitCrop"}, default="stitchingCrop"
        Crop the maps were made for (``create_corrections --crop_kind``).

    Returns
    -------
    noise, scaler : numpy.ndarray
        float32 background and multiplier, shape (Z, Y) of the registration
        volume (``load_chunk``); noise is NaN where there is no map.

    Raises
    ------
    ValueError
        If ``corrections_layer`` is above ``layer``, the map shape does not
        match the stitching crop at ``corrections_layer``, or the chunk number
        cannot be read from the name.
    """
    if corrections_layer > layer:
        raise ValueError(f"corrections_layer {corrections_layer} is coarser than "
                         f"the registration layer {layer}")
    chunk = None
    if per_chunk_corrections:
        found = re.search(r"chunk-?(\d+)", Path(path).name)
        if found is None:
            raise ValueError(
                f"cannot read the chunk number from {Path(path).name}")
        chunk = int(found.group(1))
    noise, scaler = load_corrections(str(corrections_dir), channel, chunk)

    # 1. maps -> the raw chunk's frame at corrections_layer (unflipped, uncropped)
    c_scale = 2 ** corrections_layer
    raw_zc, raw_yc, _ = open_tile_array(path, corrections_layer).shape
    if corrections_crop not in ("stitchingCrop", "splitCrop"):
        raise ValueError("corrections_crop must be 'stitchingCrop' or 'splitCrop', "
                         f"got {corrections_crop!r}")
    z0, z1, t0, t1 = get_crop(params, camera_numb, slice_numb, channel,
                              kind=corrections_crop,
                              include_z=corrections_crop == "stitchingCrop")
    t0, t1 = t0 // c_scale, min(t1 // c_scale, raw_yc)
    zc0, zc1 = (0, raw_zc) if z0 is None or z1 is None else (
        max(z0 // c_scale, 0), min(z1 // c_scale, raw_zc))
    if scaler.shape != (zc1 - zc0, t1 - t0):
        raise ValueError(
            f"corrections for channel {channel} have shape {scaler.shape} but "
            f"the {corrections_crop} at layer {corrections_layer} is (Z, Y)="
            f"({zc1 - zc0}, {t1 - t0}); pass the layer create_corrections was "
            "run at as corrections_layer")
    flip = get_vertical_flip(params, camera_numb)
    frames = []
    for m in (noise, scaler):
        frame = np.full((raw_zc, raw_yc), np.nan, dtype=np.float32)
        # crop is flipped after cutting
        frame[zc0:zc1, t0:t1] = m[::-1] if flip else m
        frames.append(_block_mean(frame, 2 ** (layer - corrections_layer)))

    # 2. cut to the registration volume: splitCrop rows, all z, flipped
    raw_z, raw_y, _ = open_tile_array(path, layer).shape
    scale = 2 ** layer
    _, _, s0, s1 = get_crop(params, camera_numb, slice_numb, channel,
                            kind="splitCrop", include_z=False)
    s0, s1 = s0 // scale, min(s1 // scale, raw_y)
    out = []
    for frame in frames:
        full = np.full((raw_z, raw_y), np.nan, dtype=np.float32)
        nz_, ny_ = min(raw_z, frame.shape[0]), min(raw_y, frame.shape[1])
        full[:nz_, :ny_] = frame[:nz_, :ny_]
        vol = full[:, s0:s1]
        out.append(vol[::-1] if flip else vol)
    noise_reg, scaler_reg = out
    covered = float(np.isfinite(scaler_reg).mean())
    scaler_reg = np.where(np.isfinite(scaler_reg),
                          scaler_reg, 1.0).astype(np.float32)
    print(f"  {Path(path).name}: correction maps cover {covered:.0%} of the "
          "registration volume (other rows: own background, no scaling)")
    return noise_reg, scaler_reg


def register_two_chunks(chunk_a: PathLike, chunk_b: PathLike, channel: str,
                        slice_numb: int, camera_numb: int, yaml_path: PathLike,
                        overlap: float, layer: int = 1, background_length: int = 400,
                        subtract_background: bool = True,
                        smooth_sigma: float = 1.0,
                        corrections_dir: Optional[PathLike] = None,
                        per_chunk_corrections: bool = False,
                        corrections_layer: int = 0,
                        z_trim_end: float = 0, z_trim_start: float = 0,
                        overlap_factor: float = 1.5,
                        corrections_crop: str = "stitchingCrop") -> Pair:
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
    corrections_dir : str or pathlib.Path, optional
        If given, the noise and zy scaler maps of ``create_corrections`` (made
        with the same ``layer``) are applied to both chunks, as
        ``(vol - noise).clip(0) * scaler``, exactly like the stitching. The
        background is then not measured again from each chunk (rows the maps
        do not cover still are).
    per_chunk_corrections : bool, default=False
        Use each chunk's own maps (``create_corrections`` with ``per_chunk``).
    corrections_layer : int, default=0
        Pyramid level ``create_corrections`` was run at; its maps are averaged
        down to ``layer``.
    z_trim_end : float, default=0
        z planes (full-resolution pixels) removed from the end of each cropped,
        flipped chunk before registering; the correction maps are cut the same way.
    z_trim_start : float, default=0
        Same, from the beginning of z.
    overlap_factor : float, default=1.5
        The overlap that is registered is ``overlap_factor`` times the estimated
        ``overlap``: the last ``overlap * overlap_factor`` rows of A and the first
        ``overlap * overlap_factor`` rows of B are read, and B starts at the cropped
        chunk height minus that. Use a factor above 1 when the true overlap may be
        larger than the estimate; 1 registers exactly the estimated overlap. The
        result is an absolute y offset, so it does not depend on the factor.
    corrections_crop : {"stitchingCrop", "splitCrop"}, default="stitchingCrop"
        Crop the correction maps were made for (``create_corrections --crop_kind``).

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
    trim_a = trim_b = 0                    # planes removed from the end
    start_a = start_b = 0                  # planes removed from the beginning
    if z_trim_start > 0:
        start_a = z_trim_planes(z_trim_start, layer, vol_a.shape[0], "z_trim_start")
        start_b = z_trim_planes(z_trim_start, layer, vol_b.shape[0], "z_trim_start")
    if z_trim_end > 0:
        trim_a = z_trim_planes(z_trim_end, layer, vol_a.shape[0])
        trim_b = z_trim_planes(z_trim_end, layer, vol_b.shape[0])
    if z_trim_start > 0 or z_trim_end > 0:
        if start_a + trim_a >= vol_a.shape[0] or start_b + trim_b >= vol_b.shape[0]:
            raise ValueError(f"z_trim_start {z_trim_start:g} + z_trim_end "
                             f"{z_trim_end:g} full-res px remove every z plane at "
                             f"layer {layer}")
        vol_a = vol_a[start_a:vol_a.shape[0] - trim_a]
        vol_b = vol_b[start_b:vol_b.shape[0] - trim_b]
        print(f"  z trim (full-res px): first {z_trim_start:g} / last {z_trim_end:g}: "
              f"first {start_a} and last {trim_a} z planes of each chunk are not used "
              f"for registration (layer {layer})")
    Za, sy, sx = vol_a.shape
    if vol_b.shape != vol_a.shape:
        print(f"WARNING: chunk shapes differ: {vol_a.shape} vs {vol_b.shape}")
    # starting y offset = cropped chunk height - exposed overlap, the exposed
    # overlap being the estimate times overlap_factor
    if overlap_factor <= 0:
        raise ValueError(f"overlap_factor must be positive, got {overlap_factor}")
    if not 0 < overlap / scale < sy:
        raise ValueError(f"overlap {overlap:g} (full-res px) is outside the chunk "
                         f"(height {sy * scale} full-res px); it must be between "
                         "0 and the height")
    exposed = overlap * overlap_factor
    init_y = sy - exposed / scale
    iy = int(round(init_y))
    if iy < 1:
        iy = 1
        print(f"  overlap x {overlap_factor:g} = {exposed:g} px is more than the "
              f"chunk; using all but one row")
    init_y_full = iy * scale
    print(f"estimated overlap {overlap:g} px full-res x {overlap_factor:g} = "
          f"{exposed:g} px exposed -> starting y offset "
          f"{init_y_full:g} px full-res ({iy} at this layer; chunk is "
          f"{sy} rows tall at this layer), z = 0")
    # nominal overlap: A's rows [iy, sy) and B's rows [0, sy - iy)
    a_y0, b_rows = iy, min(vol_b.shape[1], sy - iy)
    bg_len = max(1, background_length // scale)
    nbytes = ((sy - a_y0) + b_rows) * Za * sx * 4
    print(f"A rows {a_y0}:{sy} (of {sy}), B rows 0:{b_rows}; "
          f"reading {nbytes / 2**30:.2f} GiB of float32 (+ similar temporaries)"
          + ("; consider a higher layer" if nbytes > 2**33 else ""))

    noise_a = scaler_a = noise_b = scaler_b = None
    if corrections_dir is not None:
        noise_a, scaler_a = registration_corrections(
            chunk_a, slice_numb, camera_numb, channel, layer, params,
            corrections_dir, per_chunk_corrections, corrections_layer,
            corrections_crop)
        noise_b, scaler_b = registration_corrections(
            chunk_b, slice_numb, camera_numb, channel, layer, params,
            corrections_dir, per_chunk_corrections, corrections_layer,
            corrections_crop)
        # the maps are laid out on the full (cropped, flipped) volume: cut them too
        za = slice(start_a, start_a + vol_a.shape[0])
        zb = slice(start_b, start_b + vol_b.shape[0])
        noise_a, scaler_a = noise_a[za], scaler_a[za]
        noise_b, scaler_b = noise_b[zb], scaler_b[zb]

    t0 = time.time()
    sub_a = read_volume_rows(vol_a, a_y0, sy, 0, sx,
                             bg_len, subtract_background, scaler=scaler_a,
                             noise=noise_a)
    sub_b = read_volume_rows(vol_b, 0, b_rows, 0, sx,
                             bg_len, subtract_background, scaler=scaler_b,
                             noise=noise_b)
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
                    smooth_sigma: float = 1.0, force: bool = False,
                    corrections_dir: Optional[str] = None,
                    per_chunk_corrections: bool = False,
                    corrections_layer: int = 0, z_trim_end: float = 0,
                    z_trim_start: float = 0, overlap_factor: float = 1.5,
                    corrections_crop: str = "stitchingCrop") -> None:
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
    corrections_dir : str, optional
        Folder written by ``create_corrections`` (with the same ``layer``). If
        given, the chunks are corrected with its noise and scaler maps,
        ``(vol - noise).clip(0) * scaler``, before registering (instead of
        measuring the background again), so dark or unevenly lit rows no
        longer dominate the correlation.
    per_chunk_corrections : bool, default=False
        Use each chunk's own scaler map (``create_corrections`` with
        ``per_chunk``; the chunk number is read from the folder name).
    corrections_layer : int, default=0
        Pyramid level that ``create_corrections`` was run at (its ``--layer``,
        0 = full resolution). The maps are averaged down to ``layer``, so
        registering at a lower resolution than the corrections is fine; it
        cannot be above ``layer``.
    z_trim_end : float, default=0
        Remove this many z planes (full-resolution pixels) from the END of every
        chunk, after the crop and the vertical flip, before registering (for example
        200 for an empty region of that depth). The correction maps are cut the same
        way for registration only; the maps, the stitching crops and the stitched
        output are not changed, and the registered ``dz`` keeps its meaning.
    z_trim_start : float, default=0
        The same from the BEGINNING of z (full-resolution pixels), for an empty
        region at the first planes. It can be combined with ``z_trim_end``. Both
        chunks lose the same planes, so the registered ``dz`` (B relative to A) keeps
        its meaning.
    overlap_factor : float, default=1.5
        The overlap that is registered is ``overlap_factor`` times the estimated
        ``overlap``: the last ``overlap * overlap_factor`` rows of A and the first
        ``overlap * overlap_factor`` rows of B are read, and B starts at the cropped
        chunk height minus that. Use a factor above 1 when the true overlap may be
        larger than the estimate; 1 registers exactly the estimated overlap. The
        result is an absolute y offset, so it does not depend on the factor.
    corrections_crop : {"stitchingCrop", "splitCrop"}, default="stitchingCrop"
        Crop the correction maps were made for (``create_corrections --crop_kind``;
        ``splitCrop`` when the stitch registers the channels to 488).

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
                                  smooth_sigma=smooth_sigma,
                                  corrections_dir=corrections_dir,
                                  per_chunk_corrections=per_chunk_corrections,
                                  corrections_layer=corrections_layer,
                                  z_trim_end=z_trim_end,
                                  z_trim_start=z_trim_start,
                                  overlap_factor=overlap_factor,
                                  corrections_crop=corrections_crop)
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
        "overlap_factor": float(overlap_factor),
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
                     "smooth_sigma": float(smooth_sigma), "crop": "splitCrop",
                     "scaler_correction": corrections_dir is not None,
                     "corrections_dir": (None if corrections_dir is None
                                         else str(corrections_dir)),
                     "per_chunk_corrections": bool(per_chunk_corrections),
                     "corrections_layer": int(corrections_layer),
                     "z_trim_end": float(z_trim_end),
                     "z_trim_start": float(z_trim_start)},
        "chunks": [str(c) for c in chunks],
    }
    write_yaml(out_path, result)
    print(f"wrote {out_path}")