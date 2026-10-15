"""Single-pass, bounded-memory writer for a 3D array and its pyramid.

Frames are pushed along the first (z) axis in order. Each pyramid level keeps
a buffer of one write unit (shard, or chunk if unsharded) along z, so peak
memory is roughly ``3 * unit_z * Y * X * itemsize`` for level 0 plus a
geometric tail for the coarser levels, independent of the volume size.

The output is identical to writing level 0 and then calling
`ZarrGroup.generate_pyramid` with ``mode="mean"``: 2x windows, excess voxels
trimmed, factor 1 along singleton or `no_pyramid_axis` dimensions, mean cast
back to the input dtype (truncation).

Level 0 dominates the cost (compression). zarr-python serializes most of the
shard encoding within one process, so with the zarr-python driver level-0
tiles are encoded by a pool of worker *processes* reading the slab from
shared memory, while the main process downsamples and writes the coarser
levels. With other drivers (tensorstore), worker threads are used.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
from typing import Callable, Iterator, Literal, Optional, Sequence

import numpy as np
from numpy.typing import DTypeLike

from linc_convert.utils.io.zarr.abc import ZarrArray, ZarrGroup
from linc_convert.utils.io.zarr.generate_pyramid import default_levels, next_level_shape
from linc_convert.utils.zarr_config import ZarrConfig

logger = logging.getLogger(__name__)


def _downsample(block: np.ndarray, factors: Sequence[int]) -> np.ndarray:
    """Mean over non-overlapping windows, trimming excess voxels.

    Matches ``da.coarsen(da.nanmean, ..., trim_excess=True).astype(dtype)``
    for NaN-free input: integer data uses an exact floor division, which
    equals truncating the float64 mean of at most 8 values.
    """
    dtype = block.dtype
    out_shape = [s // f for s, f in zip(block.shape, factors)]
    block = block[tuple(slice(0, o * f) for o, f in zip(out_shape, factors))]
    integer = np.issubdtype(dtype, np.integer)
    if integer and np.issubdtype(dtype, np.unsignedinteger) and dtype.itemsize <= 2:
        acc_dtype = np.uint32  # 8 * 65535 fits
    else:
        acc_dtype = np.int64 if integer else np.float64
    acc = None
    count = 0
    for dz in range(factors[0]):
        for dy in range(factors[1]):
            for dx in range(factors[2]):
                part = block[dz:: factors[0], dy:: factors[1], dx:: factors[2]]
                if acc is None:
                    acc = part.astype(acc_dtype)
                else:
                    acc += part
                count += 1
    if integer:
        if np.issubdtype(dtype, np.signedinteger):
            # truncate toward zero, like float -> int casting
            out = np.sign(acc) * (np.abs(acc) // count)
        else:
            out = acc // count
    else:
        out = acc / count
    return out.astype(dtype)


def _downsample_parallel(
    block: np.ndarray,
    factors: Sequence[int],
    pool: Optional[ThreadPoolExecutor],
    band_bytes: int = 2**24,
) -> np.ndarray:
    """`_downsample` in y-bands, so temporaries stay ~`band_bytes` each.

    Bands are processed by `pool` if given (numpy releases the GIL).
    """
    out_shape = [s // f for s, f in zip(block.shape, factors)]
    ny_out = out_shape[1]
    # bytes of input touched per output row
    row_bytes = block.nbytes // max(block.shape[1], 1) * factors[1]
    nbands = max(1, min(ny_out, int(np.ceil(row_bytes * ny_out / band_bytes))))
    if pool is not None:
        nbands = max(nbands, min(pool._max_workers, ny_out))
    if nbands <= 1:
        return _downsample(block, factors)
    out = np.empty(out_shape, dtype=block.dtype)
    edges = np.linspace(0, ny_out, nbands + 1).astype(int)

    def band(i: int) -> None:
        o0, o1 = edges[i], edges[i + 1]
        out[:, o0:o1] = _downsample(
            block[:, o0 * factors[1]: o1 * factors[1]], factors
        )

    if pool is None:
        for i in range(nbands):
            band(i)
    else:
        list(pool.map(band, range(nbands)))
    return out


# --------------------------------------------------------------------------
# process workers (zarr-python driver, local store)
# --------------------------------------------------------------------------
_W_BUFS: list[np.ndarray] = []
_W_SHMS: list[shared_memory.SharedMemory] = []
_W_ARRS: list = []


def _attach_shm(name: str) -> shared_memory.SharedMemory:
    """Attach to an existing segment without registering it for cleanup."""
    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:  # Python < 3.13
        from multiprocessing import resource_tracker

        register = resource_tracker.register
        resource_tracker.register = lambda *a, **k: None
        try:
            return shared_memory.SharedMemory(name=name)
        finally:
            resource_tracker.register = register


def _worker_init(
    buffers: list[tuple[str, tuple]], dtype: str, locations: list[tuple[str, str]]
) -> None:
    import zarr

    # one shard at a time per process; avoid per-thread malloc arenas
    zarr.config.set({"threading.max_workers": 1, "async.concurrency": 4})

    _W_SHMS[:] = [_attach_shm(name) for name, _ in buffers]
    _W_BUFS[:] = [
        np.ndarray(shape, dtype, buffer=s.buf)
        for s, (_, shape) in zip(_W_SHMS, buffers)
    ]
    _W_ARRS[:] = [
        zarr.open_array(zarr.storage.LocalStore(root), path=path, mode="r+")
        for root, path in locations
    ]


def _worker_put(task: tuple) -> None:
    b, lvl, z0, n, y0, y1, x0, x1 = task
    _W_ARRS[lvl][z0: z0 + n, y0:y1, x0:x1] = _W_BUFS[b][:n, y0:y1, x0:x1]


def _spawn_pool(processes: int, initializer: Callable, initargs: tuple):  # noqa: ANN202
    """Spawn a worker pool, limiting glibc malloc arenas in the children."""
    import os

    env = os.environ.get("MALLOC_ARENA_MAX")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    try:
        return mp.get_context("spawn").Pool(processes, initializer, initargs)
    finally:
        if env is None:
            os.environ.pop("MALLOC_ARENA_MAX", None)


def _local_zarr_location(arr: ZarrArray) -> Optional[tuple[str, str]]:
    """(root, path) of a zarr-python array in a LocalStore, else None."""
    inner = getattr(arr, "_array", None)
    store = getattr(inner, "store", None)
    root = getattr(store, "root", None)
    if inner is None or root is None or type(store).__name__ != "LocalStore":
        return None
    return str(root), str(inner.path)


# per-worker overhead (interpreter + numpy/zarr + one shard being encoded)
WORKER_OVERHEAD_BYTES = 160 * 2**20
# main process heap besides the buffers
MAIN_OVERHEAD_BYTES = 300 * 2**20


def estimate_memory(
    shape: Sequence[int],
    itemsize: int,
    unit: Sequence[int],
    workers: int,
    levels: int,
) -> int:
    """Rough peak memory (bytes) of `StreamingPyramidWriter`.

    `unit` is the (z, y, x) write unit (shard, or chunk if unsharded).
    Counts two level-0 slabs (read/write ping-pong), one slab per coarser
    level, and a fixed cost per worker.
    """
    slab = unit[0] * shape[1] * shape[2] * itemsize
    pyramid = sum(slab / 4**k for k in range(1, levels + 1))
    shard = int(np.prod(unit)) * itemsize
    per_worker = WORKER_OVERHEAD_BYTES + 2 * shard
    return int(2 * slab + pyramid + MAIN_OVERHEAD_BYTES + workers * per_worker)


def workers_for_memory(
    max_bytes: int, shape: Sequence[int], itemsize: int, unit: Sequence[int],
    levels: int, max_workers: int,
) -> int:
    """Largest worker count (<= max_workers, >= 1) whose estimate fits."""
    for w in range(max_workers, 0, -1):
        if estimate_memory(shape, itemsize, unit, w, levels) <= max_bytes:
            return w
    need = estimate_memory(shape, itemsize, unit, 1, levels)
    raise MemoryError(
        f"Need ~{need / 2**30:.1f} GiB even with one worker; use a smaller "
        f"shard/chunk along z (current {unit[0]}) or allow more memory."
    )


class StreamingPyramidWriter:
    """Write a z-stream of frames into level 0 and all pyramid levels at once."""

    def __init__(
        self,
        zgroup: ZarrGroup,
        shape: Sequence[int],
        dtype: DTypeLike,
        zarr_config: ZarrConfig,
        levels: int = -1,
        no_pyramid_axis: Optional[int] = None,
        max_workers: Optional[int] = None,
        worker_type: Optional[Literal["process", "thread"]] = None,
    ) -> None:
        self.dtype = np.dtype(dtype)
        base = zgroup.create_array(
            "0", shape=tuple(shape), zarr_config=zarr_config, dtype=self.dtype
        )
        if levels == -1:
            levels = default_levels(tuple(shape), base.chunks[-3:], no_pyramid_axis)
        self.shapes = [list(shape)]
        self.arrays: list[ZarrArray] = [base]
        for lvl in range(1, levels + 1):
            self.shapes.append(next_level_shape(self.shapes[-1], no_pyramid_axis))
            self.arrays.append(
                zgroup.create_array_from_base(str(lvl), tuple(self.shapes[-1]))
            )
        # downsampling factors to go from level l-1 to level l
        self.factors = [None] + [
            [1 if (i == no_pyramid_axis or prev[i] == 1) else 2 for i in range(3)]
            for prev in self.shapes[:-1]
        ]
        # z write unit per level: one shard (or chunk) thick, so each write
        # touches whole chunks/shards and never needs read-modify-write
        self.unit = [int((a.shards or a.chunks)[0]) for a in self.arrays]
        self.fill = [0] * self.nlevels
        self.cursor = [0] * self.nlevels
        # frames of level l-1 waiting to be paired along z for level l
        self.carry: list[Optional[np.ndarray]] = [None] * self.nlevels

        workers = max(1, int(max_workers or 1))
        locations = [_local_zarr_location(a) for a in self.arrays]
        local = all(loc is not None for loc in locations)
        if worker_type is None:
            worker_type = "process" if (workers > 1 and local) else "thread"
        if worker_type == "process" and not local:
            logger.warning("Process workers need a local zarr-python store; "
                           "falling back to threads.")
            worker_type = "thread"
        self.worker_type = worker_type
        self._ppool = None
        self._shms: list[shared_memory.SharedMemory] = []
        use_procs = worker_type == "process" and workers > 1
        # threads: write tiles (thread mode) and downsample y-bands (both)
        self._tpool = ThreadPoolExecutor(workers) if workers > 1 else None
        # buffers: level 0 gets two (reader/writer ping-pong) + one staging;
        # each coarser level gets one. Index i of `self._bufs` -> level.
        shapes = [(self.unit[0], *self.shapes[0][1:])] * 3 + [
            (u, *s[1:]) for u, s in zip(self.unit[1:], self.shapes[1:])
        ]
        self._buf_level = [0, 0, 0] + list(range(1, self.nlevels))
        if use_procs:
            for shp in shapes:
                nbytes = int(np.prod(shp)) * self.dtype.itemsize
                self._shms.append(
                    shared_memory.SharedMemory(create=True, size=max(nbytes, 1)))
            self._bufs = [
                np.ndarray(shp, self.dtype, buffer=m.buf)
                for shp, m in zip(shapes, self._shms)
            ]
            self._ppool = _spawn_pool(
                workers,
                _worker_init,
                ([(m.name, shp) for m, shp in zip(self._shms, shapes)],
                 self.dtype.str, locations),
            )
        else:
            self._bufs = [np.empty(shp, self.dtype) for shp in shapes]
        self._base_bufs = self._bufs[:3]
        self.buf = [self._bufs[2]] + self._bufs[3:]
        logger.info(
            "Streaming %d levels, z write units %s, %d %s worker(s), "
            "buffers %.1f MB",
            self.nlevels, self.unit, workers, worker_type,
            sum(b.nbytes for b in self._bufs) / 2**20,
        )

    @property
    def nlevels(self) -> int:
        """Number of levels including the base level."""
        return len(self.arrays)

    def base_buffers(self) -> list[np.ndarray]:
        """Two reusable (unit_z, Y, X) buffers for feeding `write_frames`."""
        return self._base_bufs[:2]

    # ------------------------------------------------------------------ write
    def _tiles(self, lvl: int, ny: int, nx: int) -> list[tuple[int, int, int, int]]:
        arr = self.arrays[lvl]
        _, uy, ux = (int(u) for u in (arr.shards or arr.chunks))
        return [
            (y0, min(y0 + uy, ny), x0, min(x0 + ux, nx))
            for y0 in range(0, ny, uy)
            for x0 in range(0, nx, ux)
        ]

    def _advance(self, lvl: int, n: int) -> int:
        z0 = self.cursor[lvl]
        if z0 + n > self.shapes[lvl][0]:
            raise RuntimeError(
                f"Level {lvl} overflow: {z0 + n} > {self.shapes[lvl][0]}")
        self.cursor[lvl] = z0 + n
        return z0

    def _write(self, lvl: int, data: np.ndarray):  # noqa: ANN202
        """Write `data` at the level cursor. May return an async handle."""
        if self._ppool is not None:
            idx = next(
                (i for i, b in enumerate(self._bufs)
                 if self._buf_level[i] == lvl
                 and b.ctypes.data == data.ctypes.data), None)
            if idx is None:
                staging = 2 if lvl == 0 else self._bufs.index(self.buf[lvl])
                self._bufs[staging][: data.shape[0]] = data
                idx = staging
            z0 = self._advance(lvl, data.shape[0])
            tasks = [
                (idx, lvl, z0, data.shape[0], y0, y1, x0, x1)
                for y0, y1, x0, x1 in self._tiles(lvl, *data.shape[1:])
            ]
            return self._ppool.map_async(_worker_put, tasks, chunksize=1)
        arr = self.arrays[lvl]
        z0 = self._advance(lvl, data.shape[0])
        z1 = z0 + data.shape[0]
        if self._tpool is None:
            arr[z0:z1] = data
            return None

        def put(tile: tuple[int, int, int, int]) -> None:
            y0, y1, x0, x1 = tile
            arr[z0:z1, y0:y1, x0:x1] = data[:, y0:y1, x0:x1]

        list(self._tpool.map(put, self._tiles(lvl, *data.shape[1:])))
        return None

    def _push(self, lvl: int, frames: np.ndarray) -> None:
        """Append frames (already at the resolution of `lvl`) to its buffer."""
        i = 0
        n = frames.shape[0]
        while i < n:
            k = min(self.unit[lvl] - self.fill[lvl], n - i)
            self.buf[lvl][self.fill[lvl]: self.fill[lvl] + k] = frames[i: i + k]
            self.fill[lvl] += k
            i += k
            if self.fill[lvl] == self.unit[lvl]:
                self._flush(lvl)

    def _flush(self, lvl: int) -> None:
        n = self.fill[lvl]
        if n == 0:
            return
        self._write_and_feed(lvl, self.buf[lvl][:n])
        self.fill[lvl] = 0

    def _write_and_feed(self, lvl: int, data: np.ndarray) -> None:
        pending = self._write(lvl, data)
        if lvl + 1 < self.nlevels:
            # overlaps with the asynchronous level-0 encode, if any
            self._feed_next(lvl + 1, data)
        if pending is not None:
            pending.get()

    def _feed_next(self, lvl: int, frames: np.ndarray) -> None:
        """Downsample frames of level `lvl-1` and push them into `lvl`."""
        fz = self.factors[lvl][0]
        if self.carry[lvl] is not None:
            frames = np.concatenate([self.carry[lvl], frames], axis=0)
            self.carry[lvl] = None
        usable = (frames.shape[0] // fz) * fz
        if usable < frames.shape[0]:
            self.carry[lvl] = frames[usable:].copy()
        if usable:
            self._push(
                lvl,
                _downsample_parallel(frames[:usable], self.factors[lvl], self._tpool),
            )

    # ------------------------------------------------------------------ API
    def write_frames(self, frames: np.ndarray) -> None:
        """Push the next frames (n, Y, X) of the base level, in z order.

        Returns once `frames` may be reused by the caller.
        """
        if tuple(frames.shape[1:]) != tuple(self.shapes[0][1:]):
            raise ValueError(f"Bad frame shape {frames.shape[1:]}")
        last = self.cursor[0] + frames.shape[0] == self.shapes[0][0]
        if self.fill[0] == 0 and (frames.shape[0] == self.unit[0] or last):
            # aligned full (or final) unit: write straight from the caller's
            # buffer; the staging buffer is then never touched
            self._write_and_feed(0, frames)
        else:
            self._push(0, frames)

    def close(self) -> None:
        """Flush partial buffers; trailing odd frames are trimmed."""
        try:
            for lvl in range(self.nlevels):
                self._flush(lvl)
            for lvl, (cur, shp) in enumerate(zip(self.cursor, self.shapes)):
                if cur != shp[0]:
                    raise RuntimeError(
                        f"Level {lvl}: wrote {cur} of {shp[0]} slices")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Release worker pools and shared memory."""
        if self._tpool is not None:
            self._tpool.shutdown()
            self._tpool = None
        if self._ppool is not None:
            self._ppool.close()
            self._ppool.join()
            self._ppool = None
        self._bufs = []
        self._base_bufs = []
        self.buf = []
        for s in self._shms:
            s.close()
            s.unlink()
        self._shms = []


def stream_volume(
    writer: StreamingPyramidWriter,
    read: Callable[[int, int, np.ndarray], np.ndarray],
    progress: Optional[Callable[[int], None]] = None,
) -> None:
    """Drive `writer` from a `read(start, stop, out)` frame source.

    Reads one base write unit at a time, prefetching the next unit in a
    background thread while the current one is compressed and written.
    """
    nz = writer.shapes[0][0]
    unit = writer.unit[0]
    bufs = writer.base_buffers()

    def ranges() -> Iterator[tuple[int, int]]:
        for z0 in range(0, nz, unit):
            yield z0, min(z0 + unit, nz)

    try:
        with ThreadPoolExecutor(1) as io:
            it = ranges()
            nxt = next(it, None)
            fut = io.submit(read, *nxt, bufs[0]) if nxt else None
            k = 0
            while fut is not None:
                data = fut.result()
                nxt = next(it, None)
                fut = io.submit(read, *nxt, bufs[(k + 1) % 2]) if nxt else None
                writer.write_frames(data)
                if progress:
                    progress(data.shape[0])
                k += 1
        writer.close()
    finally:
        writer.shutdown()
