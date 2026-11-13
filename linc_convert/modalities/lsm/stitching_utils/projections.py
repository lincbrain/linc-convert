"""Contrast projections along z of a stitched mosaic, written as OME-Zarr images.

Used by ``stitch_chunks`` to write quick-look images next to every stitched channel,
one ``slice{NNN}_{channel}_{kind}.ome.zarr`` (a 2-D float32 image with a small pyramid)
per kind. Both kinds work on every (y, x) pixel separately, over its z column:

``max``
    mean of the brightest ``top_fraction`` (default 10%) of the voxels
    minus the median of the column.
``min``
    median of the column minus the mean of the dimmest ``top_fraction`` of its voxels
    (so dim structures come out positive, i.e. bright).

The mean of the extreme tenth averages noise out (a plain maximum picks the noisiest
voxel) and the median over depth is the homogeneous background that is present at all
depths. The slowly varying background of the *image* is then removed with a 2-D high-pass
filter of the projection (``img - G(img)``, a Gaussian of ``highpass_sigma`` pixels
normalised so that empty pixels do not drag the background down at the tissue border).
This is much cheaper than filtering the 3-D volume first. ``highpass_sigma=0`` skips the
filter. Zero voxels are the padding of the shear and are never counted. The first and the
last z planes can be left out (``z_start`` / ``z_end``).

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
from scipy import ndimage

from linc_convert.modalities.lsm.stitching_utils.io import start_dask

TILE = 1024                      # chunk edge of the projection arrays
STRIP_ROWS = 256                 # rows handled at a time when building the pyramid
KINDS = ("max", "min")
BLOCK_BYTES = 48 * 2 ** 20       # target size of one dask block (uint16 bytes)
BYTES_PER_VOXEL = 44.0           # working memory per voxel of a block (float32 copies)


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


def highpass(vol: np.ndarray, sigma: float) -> np.ndarray:
    """High-pass filtered float32 copy of ``vol`` (z, y, x): ``v - G(v)``.

    ``G`` is a Gaussian (``sigma`` voxels, all axes) normalised by the same Gaussian of
    the validity mask (``vol != 0``), so the padding is not averaged in. Voxels of the
    padding are returned as 0 (callers use the mask of ``vol`` to ignore them).
    """
    v = vol.astype(np.float32)
    valid = (vol != 0).astype(np.float32)
    num = ndimage.gaussian_filter(v * valid, sigma, mode="nearest")
    den = ndimage.gaussian_filter(valid, sigma, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        background = np.where(den > 1e-3, num / den, 0.0)
    return ((v - background) * valid).astype(np.float32)


def highpass2d(img: np.ndarray, sigma: float, band: int = 2048) -> np.ndarray:
    """2-D high-pass ``img - G(img)`` of a projection, normalised by the mask ``img != 0``.

    Works in bands of ``band`` rows (plus a halo of ``4 sigma`` rows) so the memory
    stays small for a very large mosaic. Pixels outside the mask stay 0.
    """
    ny = img.shape[0]
    halo = int(np.ceil(4 * sigma)) + 1
    out = np.zeros(img.shape, np.float32)
    for r0 in range(0, ny, band):
        r1 = min(r0 + band, ny)
        a, b = max(0, r0 - halo), min(ny, r1 + halo)
        v = np.asarray(img[a:b], dtype=np.float32)
        valid = (v != 0).astype(np.float32)
        num = ndimage.gaussian_filter(v * valid, sigma, mode="nearest")
        den = ndimage.gaussian_filter(valid, sigma, mode="nearest")
        with np.errstate(divide="ignore", invalid="ignore"):
            bg = np.where(den > 1e-3, num / den, 0.0)
        out[r0:r1] = ((v - bg) * valid)[r0 - a:r1 - a]
    return out


def contrast_block(vol: np.ndarray, kinds: Sequence[str], fraction: float = 0.1,
                   sigma: float = 0.0) -> np.ndarray:
    """``max`` and/or ``min`` contrast of one block ``vol`` (z, y, x), uint16.

    Parameters
    ----------
    vol : numpy.ndarray
        Block of the stitched array; zeros are padding and are ignored.
    kinds : sequence of str
        Which of ``"max"`` / ``"min"`` to return, in this order.
    fraction : float
        Share of the (non-zero) voxels of a column used for the extreme mean: the
        ``ceil(fraction * n)`` (at least one) largest / smallest of the ``n`` voxels.
    sigma : float
        High-pass Gaussian sigma in voxels (0 = no filter).

    Returns
    -------
    numpy.ndarray
        float32 (len(kinds), y, x); 0 where a column has no data.
    """
    nz = vol.shape[0]
    valid = vol != 0
    n = valid.sum(axis=0)
    v = highpass(vol, sigma) if sigma > 0 else vol.astype(np.float32)
    v = np.where(valid, v, -np.inf).astype(np.float32)
    v.sort(axis=0)                                   # ascending; the padding comes first
    first = nz - n                                   # index of the first valid value
    k = np.minimum(np.maximum(1, np.ceil(fraction * n).astype(np.int64)), np.maximum(n, 1))
    # median of every column
    lo = np.clip(first + (n - 1) // 2, 0, nz - 1)
    hi = np.clip(first + n // 2, 0, nz - 1)
    median = (np.take_along_axis(v, lo[None], axis=0)[0]
              + np.take_along_axis(v, hi[None], axis=0)[0]) / 2
    np.copyto(v, 0.0, where=~np.isfinite(v))
    csum = np.cumsum(v, axis=0, dtype=np.float32)    # running sums of the sorted values
    del v
    out = []
    for kind in kinds:
        if kind == "max":
            below = np.where(nz - k - 1 >= 0,
                             np.take_along_axis(csum, np.clip(nz - k - 1, 0, nz - 1)[None],
                                                axis=0)[0], 0.0)
            out.append(csum[-1] - below)             # sum of the k largest
            out[-1] = out[-1] / k - median
        else:
            out.append(median - np.take_along_axis(
                csum, np.clip(first + k - 1, 0, nz - 1)[None], axis=0)[0] / k)
    return np.where(n[None] > 0, np.stack(out), 0.0).astype(np.float32)


def project(zarr_dir: str, kinds: Sequence[str], group: str = "0", memory_gb: int = 32,
            z_start: int = 0, z_end: int = 0, fraction: float = 0.1,
            highpass_sigma: float = 0.0) -> dict:
    """Contrast projections over z of one stitched array: ``{kind: float32 (y, x)}``.

    Starts (and closes) its own dask cluster, limited to ``memory_gb``.
    ``highpass_sigma`` (pixels, 0 = no filter) is the 2-D high-pass applied to every
    finished projection.
    """
    kinds = parse_kinds(kinds)
    stored = zarr.open_group(str(zarr_dir), mode="r")[group]
    nz, ny, nx = stored.shape
    if z_start < 0 or z_end < 0 or z_start + z_end >= nz:
        raise ValueError(f"projection z crop (first {z_start}, last {z_end}) "
                         f"leaves no plane of the {nz} z planes")
    bz, by, bx = block_shape(stored.shape, stored.chunks)
    arr = da.from_array(stored, chunks=(bz, by, bx))[z_start:nz - z_end]
    stacked = arr.map_blocks(contrast_block, kinds, fraction, 0.0,
                             chunks=((len(kinds),),) + arr.chunks[1:],
                             dtype=np.float32, meta=np.empty((0, 0, 0), np.float32))
    keep_gb = 4.0 * len(kinds) * ny * nx / 2 ** 30 * 1.2 + 1.0
    task_gb = BYTES_PER_VOXEL * bz * by * bx / 2 ** 30
    client = start_dask(memory_gb, task_gb=task_gb, reserve_gb=keep_gb)
    try:
        future = client.compute(stacked)
        progress(future, notebook=False)
        print(flush=True)
        result = np.asarray(client.gather(future), dtype=np.float32)
    finally:
        client.close()
    images = {kind: result[i] for i, kind in enumerate(kinds)}
    if highpass_sigma > 0:
        images = {kind: highpass2d(img, highpass_sigma) for kind, img in images.items()}
    return images


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
                      fraction: float = 0.1, highpass_sigma: float = 0.0,
                      pixel_size: Tuple[float, float] = (1.0, 1.0)) -> None:
    """Write ``slice{NNN}_{channel}_{kind}.ome.zarr`` into ``out_dir`` for every kind.

    A projection that already exists is not recomputed, so a rerun (or a resumed job)
    only fills in what is missing. ``memory_gb`` bounds this process and its dask workers
    together. ``z_start`` / ``z_end`` (z planes of the stored array) are left out.
    ``pixel_size`` is the (y, x) size in micrometres of the stored array's pixels and
    ``highpass_sigma`` is the 2-D high-pass sigma in pixels of the stored array.
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