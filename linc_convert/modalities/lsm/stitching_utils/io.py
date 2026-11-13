"""Reading and writing to files for stitching."""

import math
import os
from glob import glob
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, cast

import dask
import dask.array as da
import numpy as np
import tifffile
import yaml
import zarr
from dask.distributed import Client, LocalCluster

from linc_convert.modalities.lsm.stitching_utils.scan_metadata import Crop

PathLike = Union[str, Path]


STORED_AXES_ENV = "LINC_LSM_STORED_AXES"


def stored_axes_permutation() -> Tuple[int, int, int]:
    """Permutation that turns the stored raw chunk axes into the code's (z, y, x).

    The code works on chunks whose axes are z = depth, y = lateral axis (the one that
    holds the two channels of a camera and is stitched) and x = scanning direction
    (the M2 layout). A dataset that stores its chunks in another order says so with the
    environment variable ``LINC_LSM_STORED_AXES``: three letters, one per *stored* axis
    (first to last), naming the code axis it is. The default ``zyx`` is the M2 layout;
    ``xzy`` means the stored axes are (x, z, y), as in sub-MF283 slice 122 (stored z is
    the M2 x, stored y the M2 z, stored x the M2 y).

    Returns
    -------
    tuple of int
        ``perm`` such that ``stored.transpose(perm)`` has the axes (z, y, x).
    """
    order = os.environ.get(STORED_AXES_ENV, "zyx").strip().lower()
    if sorted(order) != ["x", "y", "z"]:
        raise ValueError(f"{STORED_AXES_ENV} must be a permutation of 'zyx', got {order!r}")
    return tuple(order.index(a) for a in "zyx")  # type: ignore[return-value]


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


def coarse_chunks(stored: Sequence[int], shape: Sequence[int],
                  target: Sequence[int]) -> Tuple[int, ...]:
    """Dask block shape: the largest multiple of the stored chunks up to ``target``.

    Whole multiples of the stored chunks keep every read aligned; blocks never
    exceed the array. Fewer, larger blocks mean far fewer dask tasks, which keeps
    the task graph (and the scheduler) manageable at full resolution.
    """
    out = []
    for c, n, t in zip(stored, shape, target):
        c = max(int(c), 1)
        out.append(int(min(n, max(c, (int(t) // c) * c))))
    return tuple(out)


def open_tile_array(path: PathLike, layer: int = 0,
                    chunks: Optional[Tuple[int, ...]] = None,
                    block: Optional[Sequence[int]] = None) -> da.Array:
    """Open one pyramid level of a chunk as a lazy dask array.

    Parameters
    ----------
    path : str or pathlib.Path
        Chunk ``.ome.zarr`` folder.
    layer : int, default=0
        Pyramid level (0 = full resolution).
    chunks : tuple of int, optional
        Dask chunks; default is the stored zarr chunks.
    block : sequence of int, optional
        Target dask block shape (z, y, x); used instead of the stored chunks,
        rounded down to a multiple of them (see ``coarse_chunks``). Ignored if
        ``chunks`` is given.

    Returns
    -------
    dask.array.Array
        Lazy volume, shape (Z, Y, X) (axes reordered if ``LINC_LSM_STORED_AXES`` is set,
        see ``stored_axes_permutation``; ``chunks`` and ``block`` are then in (Z, Y, X)).

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
    perm = stored_axes_permutation()
    shape = tuple(arr.shape[p] for p in perm)            # (Z, Y, X) of the code
    stored_chunks = tuple(arr.chunks[p] for p in perm)
    if chunks is None and block is not None:
        chunks = coarse_chunks(stored_chunks, shape, block)
    chunks = tuple(chunks) if chunks else stored_chunks
    if perm == (0, 1, 2):
        return da.asarray(arr, chunks=chunks)
    # dask chunks of the stored array, then reorder the axes
    chunks_stored = [0, 0, 0]
    for code_axis, stored_axis in enumerate(perm):
        chunks_stored[stored_axis] = chunks[code_axis]
    return da.asarray(arr, chunks=tuple(chunks_stored)).transpose(perm)


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


def start_dask(memory_gb: int, max_threads_per_worker: int = 8,
               task_gb: Optional[float] = None, reserve_gb: float = 0.0) -> Client:
    """Start a local dask cluster that is limited to a total amount of memory.

    The CPU count comes from ``SLURM_CPUS_PER_TASK``, else the CPUs the process
    may use. Workers and threads are chosen from it automatically.

    Memory is kept under ``memory_gb`` in three ways. The workers share
    ``memory_gb - reserve_gb`` (the rest is for this process, which holds the
    results it gathers). Every worker pauses new tasks at 75% of its limit and
    spills finished results to disk before that. And with ``task_gb`` the number of
    threads is capped so that tasks running at once cannot need more than 70% of
    the workers' memory together (dask cannot split a running task, so this is the
    only way to bound it).

    Parameters
    ----------
    memory_gb : int
        Total memory, in GB, that this process and dask together may use.
    max_threads_per_worker : int, default=8
        Upper bound on threads in one worker; more CPUs are split over more
        workers.
    task_gb : float, optional
        Estimated peak memory of the largest task, in GB. Fewer threads are used
        if all CPUs running such tasks would not fit.
    reserve_gb : float, default=0
        Memory kept for the calling process (not given to the workers).

    Returns
    -------
    distributed.Client
        Client connected to the new cluster (``client.close()`` when done).

    Raises
    ------
    ValueError
        If ``memory_gb`` is not a positive integer, or ``reserve_gb`` leaves less
        than 1 GB for the workers.
    """
    if int(memory_gb) != memory_gb or memory_gb < 1:
        raise ValueError(
            f"memory_gb must be a positive integer, got {memory_gb!r}")
    usable = float(memory_gb) - float(reserve_gb)
    if usable < 1:
        raise ValueError(f"memory_gb {memory_gb} minus reserve_gb {reserve_gb:.1f} "
                         "leaves less than 1 GB for the dask workers")
    if "SLURM_CPUS_PER_TASK" in os.environ:
        n_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    elif hasattr(os, "sched_getaffinity"):
        n_cpus = len(os.sched_getaffinity(0))
    else:
        n_cpus = os.cpu_count() or 1
    n_threads = n_cpus
    if task_gb:
        n_threads = max(1, min(n_cpus, int(0.7 * usable // float(task_gb))))
    n_workers = max(
        1, min(math.ceil(n_threads / max_threads_per_worker), int(usable)))
    threads = max(1, n_threads // n_workers)
    # spill early and pause new tasks before a worker gets near its limit;
    # terminate stays as a last resort
    # the scheduler runs inside this process: while it builds or ships a large graph
    # it can stall for minutes, and the default 30 s tcp timeout would then drop the
    # client and cancel every running write
    dask.config.set({"distributed.comm.timeouts.tcp": "900s",
                     "distributed.comm.timeouts.connect": "900s",
                     "distributed.worker.memory.target": 0.55,
                     "distributed.worker.memory.spill": 0.65,
                     "distributed.worker.memory.pause": 0.75,
                     "distributed.worker.memory.terminate": 0.95})
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads,
                           memory_limit=f"{usable / n_workers:.3f}GiB",
                           dashboard_address=None)
    note = ""
    if n_threads < n_cpus:
        note = (f"; {n_cpus - n_threads} of {n_cpus} CPUs left idle because a task "
                f"may need up to {task_gb:.1f} GiB")
    print(f"dask: {n_workers} worker(s) x {threads} thread(s), "
          f"{usable / n_workers:.1f} GiB each ({usable:.0f} GiB of the {memory_gb} "
          f"GiB total; {reserve_gb:.1f} GiB kept for the main process){note}",
          flush=True)
    return Client(cluster)