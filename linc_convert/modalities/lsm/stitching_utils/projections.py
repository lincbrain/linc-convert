"""Maximum and minimum intensity projections (along z) of a stitched mosaic.

Used by ``stitch_chunks`` to write a quick-look TIFF pair next to every stitched
channel. The definitions follow the standalone export script: the maximum
projection is the plain max over z; the minimum projection ignores zeros (the
empty padding of the shear) and is then inverted against the 99th percentile of
the whole image, so that dim structures show up bright.

Memory is kept low on purpose, since the mosaic can be tens of thousands of pixels
on a side: the stored array is read in blocks of a few hundred MB (aligned to the
stored chunks, so a block never reads more of a shard than it needs), everything is
kept as uint16, the percentile comes from a histogram, and the TIFFs are written
tile by tile.
"""

import os
from pathlib import Path
from typing import Iterator, Optional, Tuple

import dask.array as da
import numpy as np
import tifffile
import zarr
from dask.distributed import progress

from linc_convert.modalities.lsm.stitching_utils.io import atomic_write, start_dask

BLOCK_BYTES = 256 * 2 ** 20      # target size of one dask block (uint16)
TILE = 1024                      # TIFF tile edge
STRIP_ROWS = 256                 # rows handled at a time outside dask


def block_shape(shape: Tuple[int, int, int], chunks: Tuple[int, ...]
                ) -> Tuple[int, int, int]:
    """Dask block (z, y, x) of about ``BLOCK_BYTES`` made of whole stored chunks.

    All of z is read per block (the projections reduce over z); y and x are
    multiples of the stored chunk, as square as possible.
    """
    nz, ny, nx = shape
    cy, cx = max(1, int(chunks[1])), max(1, int(chunks[2]))
    k = 1
    while True:
        by, bx = cy * (k + 1), cx * (k + 1)
        if nz * by * bx * 2 > BLOCK_BYTES or (by >= ny and bx >= nx):
            break
        k += 1
    return nz, min(cy * k, ny), min(cx * k, nx)


def percentile_from_counts(counts: np.ndarray, q: float) -> float:
    """``np.percentile(values, q)`` (linear interpolation) from a histogram of
    non-negative integers, ``counts[v]`` = how often ``v`` occurs."""
    total = int(counts.sum())
    if total == 0:
        raise ValueError("no values")
    cum = np.cumsum(counts)
    pos = q / 100.0 * (total - 1)
    lo, hi = int(np.floor(pos)), int(np.ceil(pos))
    v_lo = int(np.searchsorted(cum, lo, side="right"))
    v_hi = int(np.searchsorted(cum, hi, side="right"))
    return v_lo + (pos - lo) * (v_hi - v_lo)


def project_zarr(zarr_dir: str, want_max: bool = True, want_min: bool = True,
                 group: str = "0", memory_gb: int = 32
                 ) -> Tuple[Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray,
                                                                 float]]]:
    """Max and/or min projection over z of one stitched array.

    Starts (and closes) its own dask cluster, limited to ``memory_gb`` in total
    together with the arrays this process keeps.

    Returns
    -------
    max_img : numpy.ndarray or None
        uint16 (y, x).
    min_parts : tuple or None
        ``(raw_min, has_data, ref)``: uint16 per-column minimum over the non-zero
        voxels, bool mask of columns with data, and the 99th percentile of
        ``raw_min`` over those columns. The min image is
        ``where(has_data, ref - raw_min, 0)`` (see ``min_tile``).
    """
    stored = zarr.open_group(str(zarr_dir), mode="r")[group]
    nz, ny, nx = stored.shape
    bz, by, bx = block_shape(stored.shape, stored.chunks)
    arr = da.from_array(stored, chunks=(bz, by, bx))

    # what this process keeps: max (2 B), raw_min (2 B), has_data (1 B) per pixel,
    # plus working copies of a strip; the rest goes to the workers
    keep_gb = 5.0 * ny * nx / 2 ** 30 * 1.2 + 1.0
    # a task holds the block, a bool mask and a where() copy, plus slack
    task_gb = 4.5 * bz * by * bx * 2 / 2 ** 30
    client = start_dask(memory_gb, task_gb=task_gb, reserve_gb=keep_gb)
    try:
        todo = {}
        if want_max:
            todo["max"] = arr.max(axis=0)
        if want_min:
            valid = arr != 0
            todo["has_data"] = valid.any(axis=0)
            todo["raw_min"] = da.where(valid, arr, np.uint16(65535)).min(axis=0)
        futures = client.compute(list(todo.values()))
        progress(futures, notebook=False)
        print(flush=True)
        res = dict(zip(todo, client.gather(futures)))
    finally:
        client.close()

    max_img = res["max"].astype(np.uint16) if want_max else None
    min_parts = None
    if want_min:
        has_data = res["has_data"]
        raw_min = res["raw_min"].astype(np.uint16)
        if has_data.any():
            counts = np.zeros(65536, dtype=np.int64)
            for r in range(0, ny, STRIP_ROWS):
                vals = raw_min[r:r + STRIP_ROWS][has_data[r:r + STRIP_ROWS]]
                counts += np.bincount(vals, minlength=65536)
            ref = np.float32(percentile_from_counts(counts, 99.0))
        else:
            ref = np.float32(0)
        min_parts = (raw_min, has_data, ref)
    return max_img, min_parts


def min_tile(parts: Tuple[np.ndarray, np.ndarray, float], rows: slice,
             cols: slice) -> np.ndarray:
    """A float32 piece of the min image from ``project_zarr``'s ``min_parts``."""
    raw_min, has_data, ref = parts
    return np.where(has_data[rows, cols],
                    ref - raw_min[rows, cols].astype(np.float32),
                    np.float32(0)).astype(np.float32)


def save_projection(shape: Tuple[int, int], piece, path: Path) -> None:
    """Write a 2D float32 image as a tiled TIFF, one tile at a time, atomically.

    Parameters
    ----------
    shape : tuple of int
        Image size (y, x).
    piece : callable
        ``piece(rows, cols)`` returns that float32 part of the image.
    path : pathlib.Path
        Output file (BigTIFF if the image is over 4 GB).
    """
    ny, nx = shape

    def tiles() -> Iterator[np.ndarray]:
        for r in range(0, ny, TILE):
            for c in range(0, nx, TILE):
                part = piece(slice(r, min(r + TILE, ny)), slice(c, min(c + TILE, nx)))
                if part.shape != (TILE, TILE):          # edge tiles are padded
                    padded = np.zeros((TILE, TILE), dtype=np.float32)
                    padded[:part.shape[0], :part.shape[1]] = part
                    part = padded
                yield part

    atomic_write(path, lambda tmp: tifffile.imwrite(
        str(tmp), tiles(), shape=(ny, nx), dtype=np.float32,
        bigtiff=ny * nx * 4 > 2 ** 32 - 2 ** 25, photometric="minisblack",
        tile=(TILE, TILE), compression="zlib", resolution=(1.0, 1.0)))


def write_projections(zarr_dir: str, out_dir: str, slice_number: int,
                      channel: str, memory_gb: int = 32) -> None:
    """Write ``slice{NNN}_{channel}_max.tiff`` and ``_min.tiff`` into ``out_dir``.

    A projection that already exists is not recomputed, so a rerun (or a resumed
    job) only fills in what is missing. ``memory_gb`` bounds this process and its
    dask workers together.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f_max = out / f"slice{slice_number:03d}_{channel}_max.tiff"
    f_min = out / f"slice{slice_number:03d}_{channel}_min.tiff"
    want_max, want_min = not f_max.exists(), not f_min.exists()
    if not (want_max or want_min):
        print(f"channel {channel}: projections already exist in {out}; skipping",
              flush=True)
        return
    print(f"channel {channel}: max/min projections -> {out}", flush=True)
    mx, mn = project_zarr(zarr_dir, want_max, want_min, memory_gb=memory_gb)
    if want_max:
        save_projection(mx.shape, lambda r, c: mx[r, c].astype(np.float32), f_max)
        print(f"  {f_max.name}: done", flush=True)
    if want_min:
        save_projection(mn[1].shape, lambda r, c: min_tile(mn, r, c), f_min)
        print(f"  {f_min.name}: done", flush=True)