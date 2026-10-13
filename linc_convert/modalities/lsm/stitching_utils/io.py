"""Reading and writing to files for stitching."""

import math
import os
from glob import glob
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, cast

import dask.array as da
import numpy as np
import tifffile
import yaml
import zarr
from dask.distributed import Client, LocalCluster

from linc_convert.modalities.lsm.stitching_utils.scan_metadata import Crop

PathLike = Union[str, Path]


def discover_tile_paths(input_path: str, camera_id: int) -> List[str]:
    """List the chunk folders of one camera.

    Both ``acq-camera-01`` and ``acq-camera01`` naming are accepted.

    Parameters
    ----------
    input_path : str
        Folder containing the ``*.ome.zarr`` chunk folders.
    camera_id : int
        Camera number (1 or 2).

    Returns
    -------
    list of str
        Chunk folder paths, sorted.

    Raises
    ------
    ValueError
        If no chunk folder is found for the camera.
    """
    tokens = [f"acq-camera-{camera_id:02d}", f"acq-camera{camera_id:02d}"]
    paths = sorted({p for t in tokens
                    for p in glob(os.path.join(input_path, f"*{t}*.ome.zarr"))})
    if not paths:
        raise ValueError(f"No tile folders found for camera {camera_id} "
                         f"(tried tokens {tokens!r}) in {input_path}")
    return paths


def open_tile_array(path: PathLike, layer: int = 0,
                    chunks: Optional[Tuple[int, ...]] = None) -> da.Array:
    """Open one pyramid level of a chunk as a lazy dask array.

    Parameters
    ----------
    path : str or pathlib.Path
        Chunk ``.ome.zarr`` folder.
    layer : int, default=0
        Pyramid level (0 = full resolution).
    chunks : tuple of int, optional
        Dask chunks; default is the stored zarr chunks.

    Returns
    -------
    dask.array.Array
        Lazy volume, shape (Z, Y, X).

    Raises
    ------
    FileNotFoundError
        If the chunk folder does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"raw chunk not found: {path} (this code only reads existing files)")
    arr = cast(zarr.Array, zarr.open_group(str(path), mode="r")[str(layer)])
    return da.asarray(arr, chunks=chunks or arr.chunks)


def level_range(rng: Sequence[int], size: int, layer: int) -> Tuple[int, int]:
    """Convert a full-resolution crop range to a pyramid level.

    Parameters
    ----------
    rng : sequence of int
        ``(lo, hi)`` in full-resolution pixels.
    size : int
        Size of the array axis at this level.
    layer : int
        Pyramid level (0 = full resolution).

    Returns
    -------
    tuple of int
        ``[lo, hi)`` at this level, clamped to the array (a warning is printed
        if it had to be clamped).
    """
    lo, hi = int(rng[0]) // (2 ** layer), int(rng[1]) // (2 ** layer)
    if hi > size or lo < 0:
        print(f"  WARNING: crop range {list(rng)} (level {layer}: {lo}:{hi}) exceeds "
              f"the array size {size}; clamped")
    return max(lo, 0), min(hi, size)


def crop_and_flip(tile: da.Array, crop: Crop, vertical_flip: bool,
                  layer: int = 0) -> da.Array:
    """Crop a chunk to a channel's crop and flip z if needed.

    The y crop is applied first, then the z crop (only if both z entries are
    given), then the flip.

    Parameters
    ----------
    tile : dask.array.Array
        Chunk volume at pyramid level ``layer``, shape (Z, Y, X).
    crop : tuple
        ``(z_start, z_end, y_start, y_end)`` in full-resolution pixels; z
        entries may be None to keep all z.
    vertical_flip : bool
        Flip the cropped volume along z if True.
    layer : int, default=0
        Pyramid level of ``tile``; the crop is scaled to it.

    Returns
    -------
    dask.array.Array
        Lazy cropped (and possibly flipped) volume, shape (Z', Y', X).
    """
    z0, z1, y0, y1 = crop
    y_lo, y_hi = level_range((y0, y1), tile.shape[1], layer)
    vol = tile[:, y_lo:y_hi, :]
    if z0 is not None and z1 is not None:
        z_lo, z_hi = level_range((z0, z1), tile.shape[0], layer)
        vol = vol[z_lo:z_hi, :, :]
    return vol[::-1] if vertical_flip else vol


def atomic_write(path: Union[str, Path], write: Callable[[Path], Any]) -> None:
    """Write a file through a temporary name, then move it into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.partial{path.suffix}")
    write(tmp)
    tmp.replace(path)


def write_yaml(path: Union[str, Path], data: dict) -> None:
    """Write a YAML file atomically, keeping the key order."""
    text = yaml.safe_dump(data, sort_keys=False)
    atomic_write(path, lambda tmp: tmp.write_text(text))


def write_tiff(path: Union[str, Path], array: np.ndarray) -> None:
    """Write a float32 TIFF atomically."""
    atomic_write(path, lambda tmp: tifffile.imwrite(
        tmp, array.astype(np.float32)))


def start_dask(memory_gb: int, max_threads_per_worker: int = 8) -> Client:
    """Start a local dask cluster that is limited to a total amount of memory.

    The CPU count comes from ``SLURM_CPUS_PER_TASK``, else the CPUs the process
    may use. Workers and threads are chosen from it automatically.

    Parameters
    ----------
    memory_gb : int
        Total memory, in GB, that dask may use across all workers.
    max_threads_per_worker : int, default=8
        Upper bound on threads in one worker; more CPUs are split over more
        workers.

    Returns
    -------
    distributed.Client
        Client connected to the new cluster (``client.close()`` when done).

    Raises
    ------
    ValueError
        If ``memory_gb`` is not a positive integer.
    """
    if int(memory_gb) != memory_gb or memory_gb < 1:
        raise ValueError(
            f"memory_gb must be a positive integer, got {memory_gb!r}")
    if "SLURM_CPUS_PER_TASK" in os.environ:
        n_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    elif hasattr(os, "sched_getaffinity"):
        n_cpus = len(os.sched_getaffinity(0))
    else:
        n_cpus = os.cpu_count() or 1
    n_workers = max(
        1, min(math.ceil(n_cpus / max_threads_per_worker), int(memory_gb)))
    threads = max(1, n_cpus // n_workers)
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads,
                           memory_limit=f"{memory_gb / n_workers:.3f}GiB",
                           dashboard_address=None)
    print(f"dask: {n_workers} worker(s) x {threads} thread(s), "
          f"{memory_gb / n_workers:.1f} GiB each ({memory_gb} GiB total)")
    return Client(cluster)
