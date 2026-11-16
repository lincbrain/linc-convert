"""Contrast projections along z of a stitched mosaic, written as OME-Zarr images.

Used by ``stitch_chunks`` to write quick-look images next to every stitched channel,
one ``slice{NNN}_{channel}_{kind}.ome.zarr`` (a 2-D float32 image with a small pyramid)
per kind. All kinds work on the 3-D high-pass filtered volume and on every (y, x) pixel
separately, over its z column:

``max``
    mean of the brightest ``top_fraction`` (default 10%) of the voxels
    minus the median of the column.
``min``
    median of the column minus the mean of the dimmest ``top_fraction`` of its voxels
    (so dim structures come out positive, i.e. bright).
``minrel``
    like ``min``, but on the *relative* deviation ``(v - background) / background``
    (the background being the smooth Gaussian estimate below), so a dim structure counts
    the same in a bright and in a dark region.

The mean of the extreme tenth averages noise out (a plain maximum picks the noisiest
voxel) and the median over depth is the homogeneous background that is present at all
depths. Before that the slowly varying background is removed with a 3-D high-pass filter
(``v - G(v)``, a Gaussian with one sigma per axis, given in micrometres and converted to
voxels, normalised by the Gaussian of the validity mask so that the empty shear padding
does not drag the background down at the tissue border). The Gaussian is applied with an
FFT, so its cost does not grow with sigma. ``highpass_sigma = 0`` skips the filter. Zero
voxels are the padding of the shear and are never counted. The first and the last z planes
can be left out (``z_start`` / ``z_end``).

Memory is kept low on purpose, since the mosaic can be tens of thousands of pixels on a
side: the stored array is read in blocks (aligned to the stored chunks, with all of z and
only 2-D results are kept, and the images are written
tile by tile.
"""

import os
import shutil
from pathlib import Path
from typing import Sequence, Tuple, Union

import dask.array as da
import numpy as np
import zarr
from dask.distributed import progress
from scipy import fft as sfft

from linc_convert.modalities.lsm.stitching_utils.io import start_dask

TILE = 1024                      # chunk edge of the projection arrays
STRIP_ROWS = 256                 # rows handled at a time when building the pyramid
KINDS = ("max", "min", "minrel")
BLOCK_BYTES = 48 * 2 ** 20       # target size of one dask block (uint16 bytes)
BYTES_PER_VOXEL = 64.0           # working memory per voxel of a block (float32 / complex64 copies)
FFT_WORKERS = 2                  # threads of one FFT


def block_shape(shape: Tuple[int, int, int], chunks: Tuple[int, ...],
                block_bytes: int = BLOCK_BYTES) -> Tuple[int, int, int]:
    """Dask block (z, y, x) of about ``block_bytes`` made of whole stored chunks.

    All of z is read per block (the projections reduce over z); y and x are
    multiples of the stored chunk, as square as possible.
    """
    nz, ny, nx = shape
    cy, cx = max(1, int(chunks[1])), max(1, int(chunks[2]))
    k = 1
    while True:
        by, bx = cy * (k + 1), cx * (k + 1)
        if nz * by * bx * 2 > block_bytes or (by >= ny and bx >= nx):
            break
        k += 1
    return nz, min(cy * k, ny), min(cx * k, nx)


def sigma_halo(sigma: Sequence[float]) -> Tuple[int, int, int]:
    """Halo (z, y, x) in voxels that a Gaussian of ``sigma`` voxels needs (3 sigma)."""
    return tuple(int(np.ceil(3.0 * s)) if s > 0 else 0 for s in sigma)  # type: ignore[return-value]


def gaussian_fft(a: np.ndarray, sigma: Sequence[float], workers: int = FFT_WORKERS) -> np.ndarray:
    """Gaussian filter of ``a`` (3-D) with one sigma (voxels) per axis, done with an FFT.

    The array is mirrored by 3 sigma on every filtered axis first, so the borders behave
    like ``mode="reflect"`` and the periodic wrap of the FFT never reaches the data. An
    axis with sigma 0 is left alone. The cost does not depend on sigma.
    """
    a = np.asarray(a, dtype=np.float32)
    halo = sigma_halo(sigma)
    padded = np.pad(a, [(h, h) for h in halo], mode="symmetric") if any(halo) else a
    shape = tuple(sfft.next_fast_len(n, real=True) for n in padded.shape)
    spec = sfft.rfftn(padded, s=shape, workers=workers)
    del padded
    for axis, (n, sg) in enumerate(zip(shape, sigma)):
        if sg <= 0:
            continue
        f = sfft.rfftfreq(n) if axis == len(shape) - 1 else sfft.fftfreq(n)
        gain = np.exp(-2.0 * (np.pi * sg) ** 2 * f ** 2).astype(np.float32)
        spec *= gain.reshape([-1 if i == axis else 1 for i in range(len(shape))])
    out: np.ndarray = np.asarray(sfft.irfftn(spec, s=shape, workers=workers))
    del spec
    (hz, hy, hx), (nz, ny, nx) = halo, a.shape
    return out[hz:hz + nz, hy:hy + ny, hx:hx + nx].astype(np.float32)


def highpass(vol: np.ndarray, sigma: Sequence[float]):
    """High-pass of ``vol`` (z, y, x): ``v - G(v)`` and the background ``G(v)``.

    ``G`` is a Gaussian (``sigma`` voxels per axis) normalised by the same Gaussian of
    the validity mask (``vol != 0``), so the padding is not averaged in. Padding voxels
    are returned as 0 in both arrays (callers use the mask of ``vol`` to ignore them).

    Returns
    -------
    residual, background : numpy.ndarray
        float32 arrays of the shape of ``vol``.
    """
    v = vol.astype(np.float32)
    valid = (vol != 0).astype(np.float32)
    num = gaussian_fft(v * valid, sigma)
    den = gaussian_fft(valid, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        background = np.where(den > 1e-3, num / den, 0.0).astype(np.float32)
    return ((v - background) * valid).astype(np.float32), (background * valid).astype(np.float32)


def _column_stats(v: np.ndarray, valid: np.ndarray, n: np.ndarray, fraction: float,
                  kinds: Sequence[str]) -> list:
    """Extreme-mean minus median (``max``) / median minus extreme-mean (``min``) of every
    z column of ``v`` over its ``n`` valid voxels. ``v`` is overwritten."""
    nz = v.shape[0]
    np.copyto(v, -np.inf, where=~valid)
    v.sort(axis=0)                                   # ascending; the padding comes first
    first = nz - n                                   # index of the first valid value
    k = np.minimum(np.maximum(1, np.ceil(fraction * n).astype(np.int64)), np.maximum(n, 1))
    lo = np.clip(first + (n - 1) // 2, 0, nz - 1)
    hi = np.clip(first + n // 2, 0, nz - 1)
    median = (np.take_along_axis(v, lo[None], axis=0)[0]
              + np.take_along_axis(v, hi[None], axis=0)[0]) / 2
    np.copyto(v, 0.0, where=~np.isfinite(v))
    csum = np.cumsum(v, axis=0, dtype=np.float32)    # running sums of the sorted values
    out = []
    for kind in kinds:
        if kind == "max":
            below = np.where(nz - k - 1 >= 0,
                             np.take_along_axis(csum, np.clip(nz - k - 1, 0, nz - 1)[None],
                                                axis=0)[0], 0.0)
            out.append((csum[-1] - below) / k - median)          # mean of the k largest
        else:
            out.append(median - np.take_along_axis(
                csum, np.clip(first + k - 1, 0, nz - 1)[None], axis=0)[0] / k)
    return out


def contrast_block(vol: np.ndarray, kinds: Sequence[str], fraction: float = 0.1,
                   sigma: Union[float, Sequence[float]] = 0.0) -> np.ndarray:
    """``max`` / ``min`` / ``minrel`` contrast of one block ``vol`` (z, y, x), uint16.

    Parameters
    ----------
    vol : numpy.ndarray
        Block of the stitched array; zeros are padding and are ignored.
    kinds : sequence of str
        Which of ``"max"`` / ``"min"`` / ``"minrel"`` to return, in this order.
    fraction : float
        Share of the (non-zero) voxels of a column used for the extreme mean: the
        ``ceil(fraction * n)`` (at least one) largest / smallest of the ``n`` voxels.
    sigma : float or sequence of 3 floats
        High-pass Gaussian sigma in voxels, (z, y, x) (a single number is used for all
        three axes; 0 = no filter). ``minrel`` needs a filter.

    Returns
    -------
    numpy.ndarray
        float32 (len(kinds), y, x); 0 where a column has no data.
    """
    sig = (float(sigma),) * 3 if np.isscalar(sigma) else tuple(float(x) for x in sigma)
    filtered = any(x > 0 for x in sig)
    if "minrel" in kinds and not filtered:
        raise ValueError("the 'minrel' projection needs a high-pass sigma > 0")
    valid = vol != 0
    n = valid.sum(axis=0)
    if filtered:
        residual, background = highpass(vol, sig)
    else:
        residual, background = vol.astype(np.float32), None
    plain = [k for k in kinds if k != "minrel"]
    got = dict(zip(plain, _column_stats(residual.copy() if "minrel" in kinds else residual,
                                        valid, n, fraction, plain))) if plain else {}
    if "minrel" in kinds:
        positive = background[valid & (background > 0)]
        floor = max(2.0, 0.03 * float(np.median(positive))) if positive.size else 2.0
        relative = residual / np.maximum(background, floor)
        got["minrel"] = _column_stats(relative, valid, n, fraction, ["min"])[0]
    return np.where(n[None] > 0, np.stack([got[k] for k in kinds]), 0.0).astype(np.float32)


def project(zarr_dir: str, kinds: Sequence[str], group: str = "0", memory_gb: int = 32,
            z_start: int = 0, z_end: int = 0, fraction: float = 0.1,
            highpass_sigma: Union[float, Sequence[float]] = 0.0) -> dict:
    """Contrast projections over z of one stitched array: ``{kind: float32 (y, x)}``.

    Starts (and closes) its own dask cluster, limited to ``memory_gb``.
    ``highpass_sigma`` is the 3-D high-pass sigma in voxels of the stored array, one
    number or (z, y, x) (0 = no filter).
    """
    sig = (float(highpass_sigma),) * 3 if np.isscalar(highpass_sigma) \
        else tuple(float(x) for x in highpass_sigma)
    kinds = parse_kinds(kinds)
    stored = zarr.open_group(str(zarr_dir), mode="r")[group]
    nz, ny, nx = stored.shape
    if z_start < 0 or z_end < 0 or z_start + z_end >= nz:
        raise ValueError(f"projection z crop (first {z_start}, last {z_end}) "
                         f"leaves no plane of the {nz} z planes")
    bz, by, bx = block_shape(stored.shape, stored.chunks)
    arr = da.from_array(stored, chunks=(bz, by, bx))[z_start:nz - z_end]
    halo = sigma_halo(sig)
    depth = {0: 0, 1: halo[1], 2: halo[2]}
    if halo[1] or halo[2]:
        arr = da.overlap.overlap(arr, depth=depth, boundary="reflect")
    stacked = arr.map_blocks(contrast_block, kinds, fraction, sig,
                             chunks=((len(kinds),),) + arr.chunks[1:],
                             dtype=np.float32, meta=np.empty((0, 0, 0), np.float32))
    if halo[1] or halo[2]:
        stacked = da.overlap.trim_internal(stacked, depth, boundary="reflect")
    keep_gb = 4.0 * len(kinds) * ny * nx / 2 ** 30 * 1.2 + 1.0
    task_gb = BYTES_PER_VOXEL * bz * (by + 2 * halo[1]) * (bx + 2 * halo[2]) / 2 ** 30
    client = start_dask(memory_gb, task_gb=task_gb, reserve_gb=keep_gb)
    try:
        future = client.compute(stacked)
        progress(future, notebook=False)
        print(flush=True)
        result = np.asarray(client.gather(future), dtype=np.float32)
    finally:
        client.close()
    return {kind: result[i] for i, kind in enumerate(kinds)}


def save_projection(shape: Tuple[int, int], piece, path: Path,
                    pixel_size: Tuple[float, float] = (1.0, 1.0),
                    min_side: int = 2048) -> None:
    """Write a 2-D float32 image as an OME-Zarr (v0.5, zarr v3) with a pyramid, atomically.

    The image is written tile by tile; level ``i + 1`` is the 2 x 2 mean of level ``i``
    and levels are added until the image is at most ``min_side`` pixels on a side.

    Parameters
    ----------
    shape : tuple of int
        Image size (y, x).
    piece : callable
        ``piece(rows, cols)`` returns that float32 part of the image.
    path : pathlib.Path
        Output ``.ome.zarr`` folder (replaced if it exists).
    pixel_size : tuple of float
        Pixel size (y, x) in micrometres of level 0.
    min_side : int
        Stop adding levels once both sides are at most this many pixels.
    """
    ny, nx = shape
    tmp = path.with_name(path.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    root = zarr.open_group(str(tmp), mode="w")

    def create(level: int, h: int, w: int):
        return root.create_array(str(level), shape=(h, w),
                                 chunks=(min(TILE, h), min(TILE, w)),
                                 dtype=np.float32, fill_value=0.0)

    level0 = create(0, ny, nx)
    for r in range(0, ny, TILE):
        for c in range(0, nx, TILE):
            level0[r:r + TILE, c:c + TILE] = piece(slice(r, min(r + TILE, ny)),
                                                   slice(c, min(c + TILE, nx)))
    sizes = [(ny, nx)]
    prev = level0
    while max(sizes[-1]) > min_side:
        h, w = sizes[-1]
        nh, nw = max(1, h // 2), max(1, w // 2)
        arr = create(len(sizes), nh, nw)
        step = max(2, (STRIP_ROWS // 2) * 2)
        for r in range(0, nh, step // 2):
            rows = min(step // 2, nh - r)
            block = np.asarray(prev[2 * r:2 * (r + rows), :2 * nw], dtype=np.float32)
            if block.shape[0] < 2 * rows:                        # odd edge row
                block = np.pad(block, ((0, 2 * rows - block.shape[0]), (0, 0)), mode="edge")
            arr[r:r + rows, :] = block.reshape(rows, 2, nw, 2).mean(axis=(1, 3))
        sizes.append((nh, nw))
        prev = arr
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [{
            "name": path.name.replace(".ome.zarr", ""),
            "axes": [{"name": "y", "type": "space", "unit": "micrometer"},
                     {"name": "x", "type": "space", "unit": "micrometer"}],
            "datasets": [{"path": str(i), "coordinateTransformations": [
                {"type": "scale", "scale": [float(pixel_size[0]) * ny / h,
                                            float(pixel_size[1]) * nx / w]}]}
                         for i, (h, w) in enumerate(sizes)]}]}
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)


def parse_kinds(kinds: Union[str, Sequence[str]]) -> Tuple[str, ...]:
    """Validate a projection kind list given as ``"max,min"`` or a sequence."""
    if isinstance(kinds, str):
        kinds = [k.strip() for k in kinds.split(",") if k.strip()]
    kinds = tuple(dict.fromkeys(kinds))
    bad = [k for k in kinds if k not in KINDS]
    if bad or not kinds:
        raise ValueError(f"projection kinds must be a non-empty subset of {KINDS}, "
                         f"got {kinds!r}")
    return kinds


def write_projections(zarr_dir: str, out_dir: str, slice_number: int,
                      channel: str, memory_gb: int = 32,
                      z_start: int = 0, z_end: int = 0,
                      kinds: Union[str, Sequence[str]] = ("max", "min"),
                      fraction: float = 0.1,
                      highpass_sigma: Union[float, Sequence[float]] = 0.0,
                      pixel_size: Tuple[float, float] = (1.0, 1.0)) -> None:
    """Write ``slice{NNN}_{channel}_{kind}.ome.zarr`` into ``out_dir`` for every kind.

    A projection that already exists is not recomputed, so a rerun (or a resumed job)
    only fills in what is missing. ``memory_gb`` bounds this process and its dask workers
    together. ``z_start`` / ``z_end`` (z planes of the stored array) are left out.
    ``pixel_size`` is the (y, x) size in micrometres of the stored array's pixels and
    ``highpass_sigma`` is the 3-D high-pass sigma, (z, y, x) in voxels of the stored array.
    """
    kinds = parse_kinds(kinds)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {k: out / f"slice{slice_number:03d}_{channel}_{k}.ome.zarr" for k in kinds}
    todo = [k for k in kinds if not (files[k] / "zarr.json").exists()]
    if not todo:
        print(f"channel {channel}: projections already exist in {out}; skipping", flush=True)
        return
    print(f"channel {channel}: {', '.join(todo)} projection(s) -> {out}", flush=True)
    images = project(zarr_dir, todo, memory_gb=memory_gb, z_start=z_start, z_end=z_end,
                     fraction=fraction, highpass_sigma=highpass_sigma)
    for kind, img in images.items():
        save_projection(img.shape, lambda r, c, img=img: img[r, c], files[kind], pixel_size)
        print(f"  {files[kind].name}: done", flush=True)