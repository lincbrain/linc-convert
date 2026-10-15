"""Single-pass, bounded-memory writer for a 3D array and its pyramid.

Frames are pushed along the first (z) axis in order. Each pyramid level keeps
a buffer of one write unit along z, so peak memory is roughly
``2 * unit_z * Y * X * itemsize`` for level 0 plus a geometric tail for the
coarser levels, independent of the volume size.

The output is identical to writing level 0 and then calling
`ZarrGroup.generate_pyramid` with ``mode="mean"``: 2x windows, excess voxels
trimmed, factor 1 along singleton or `no_pyramid_axis` dimensions, mean cast
back to the input dtype (truncation).

Two sinks write a level:

* `ShardAppendSink` (sharded zarr v3, zarr-python driver, local store).
  Inner chunks are compressed by a thread pool and appended to their shard
  file as soon as a chunk-thick z slab is available; the shard index is
  written when the shard is complete. The write unit is one *inner chunk*
  along z, so shards can be arbitrarily large (e.g. one shard per volume)
  without read-modify-write. Shard files are written as ``<key>.partial`` and
  renamed when complete.
* `ZarrApiSink` (anything else). Whole shards (or chunks) are written through
  the array API by a thread pool; the write unit is one shard along z.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
from numpy.typing import DTypeLike

from linc_convert.utils.io.zarr.abc import ZarrArray, ZarrGroup
from linc_convert.utils.io.zarr.generate_pyramid import default_levels, next_level_shape
from linc_convert.utils.zarr_config import ZarrConfig

logger = logging.getLogger(__name__)

MAX_UINT_64 = 2**64 - 1


# --------------------------------------------------------------------------
# downsampling
# --------------------------------------------------------------------------
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
# sinks
# --------------------------------------------------------------------------
class ZarrApiSink:
    """Write whole shards (or chunks) of one level through the array API."""

    def __init__(self, arr: ZarrArray) -> None:
        self.arr = arr
        self.unit = tuple(int(u) for u in (arr.shards or arr.chunks))
        self.unit_z = self.unit[0]

    def write(
        self, z0: int, data: np.ndarray, pool: Optional[ThreadPoolExecutor]
    ) -> None:
        """Write `data` at z offset `z0`, one shard/chunk tile per task."""
        z1 = z0 + data.shape[0]
        if pool is None:
            self.arr[z0:z1] = data
            return
        _, uy, ux = self.unit
        ny, nx = data.shape[1:]
        tiles = [(y0, min(y0 + uy, ny), x0, min(x0 + ux, nx))
                 for y0 in range(0, ny, uy) for x0 in range(0, nx, ux)]

        def put(tile: tuple[int, int, int, int]) -> None:
            y0, y1, x0, x1 = tile
            self.arr[z0:z1, y0:y1, x0:x1] = data[:, y0:y1, x0:x1]

        list(pool.map(put, tiles))

    def close(self) -> None:
        """Nothing to flush."""


def _inner_encoder(arr: ZarrArray) -> Optional[Callable[[np.ndarray], bytes]]:
    """Return a numpy->bytes encoder matching the shard's inner codecs.

    Supports ``[bytes, blosc|zstd|gzip]`` inner codecs with a
    ``[bytes, crc32c]`` index at the end of the shard; None otherwise.
    """
    import zarr.codecs as zc

    meta = getattr(getattr(arr, "_array", None), "metadata", None)
    codecs = getattr(meta, "codecs", None)
    if not codecs or len(codecs) != 1 or not isinstance(codecs[0], zc.ShardingCodec):
        return None
    shard_codec = codecs[0]
    location = getattr(shard_codec.index_location, "value", shard_codec.index_location)
    if location != "end":
        return None
    idx = list(shard_codec.index_codecs)
    if not (len(idx) == 2 and isinstance(idx[0], zc.BytesCodec)
            and isinstance(idx[1], zc.Crc32cCodec)):
        return None
    inner = list(shard_codec.codecs)
    if not inner or not isinstance(inner[0], zc.BytesCodec):
        return None
    endian = getattr(inner[0].endian, "value", inner[0].endian)
    steps: list[Callable[[bytes], bytes]] = []
    for c in inner[1:]:
        if isinstance(c, zc.BloscCodec):
            steps.append(c._blosc_codec.encode)
        elif isinstance(c, zc.ZstdCodec):
            import numcodecs

            steps.append(numcodecs.Zstd(level=c.level, checksum=c.checksum).encode)
        elif isinstance(c, zc.GzipCodec):
            import numcodecs

            steps.append(numcodecs.GZip(level=c.level).encode)
        else:
            return None
    target = np.dtype(arr.dtype).newbyteorder(
        "<" if endian in (None, "little") else ">")

    def encode(chunk: np.ndarray) -> bytes:
        buf = np.ascontiguousarray(chunk, dtype=target).tobytes()
        for step in steps:
            buf = bytes(step(buf))
        return buf

    return encode


class ShardAppendSink:
    """Stream inner chunks of one level into append-only shard files."""

    def __init__(self, arr: ZarrArray, encode: Callable[[np.ndarray], bytes]) -> None:
        inner = arr._array
        self.encode = encode
        self.shape = tuple(int(s) for s in arr.shape)
        self.chunk = tuple(int(c) for c in arr.chunks)
        self.shard = tuple(int(s) for s in arr.shards)
        self.unit_z = self.chunk[0]
        self.cps = tuple(s // c for s, c in zip(self.shard, self.chunk))
        self.fill = inner.metadata.fill_value
        self.dtype = np.dtype(arr.dtype)
        self.root = os.path.join(str(inner.store.root), inner.path)
        self.key = inner.metadata.chunk_key_encoding.encode_chunk_key
        # shards open in the current shard row along z: (sy, sx) -> state
        self._open: dict[tuple[int, int], dict] = {}
        self._shard_z: Optional[int] = None

    def _shard_state(self, sz: int, sy: int, sx: int) -> dict:
        st = self._open.get((sy, sx))
        if st is None:
            path = os.path.join(self.root, self.key((sz, sy, sx)))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            st = dict(
                path=path,
                fh=open(path + ".partial", "wb"),
                offset=0,
                index=np.full(self.cps + (2,), MAX_UINT_64, dtype="<u8"),
            )
            self._open[(sy, sx)] = st
        return st

    def write(
        self, z0: int, data: np.ndarray, pool: Optional[ThreadPoolExecutor]
    ) -> None:
        """Compress the inner chunks of slab `data` (at `z0`) and append them."""
        cz, cy, cx = self.chunk
        if z0 % cz:
            raise RuntimeError("Appends must be aligned on inner chunks")
        n = data.shape[0]
        sz = z0 // self.shard[0]
        if self._shard_z is not None and sz != self._shard_z:
            self._finalize()
        self._shard_z = sz
        ny, nx = self.shape[1:]
        tiles = [(y0, x0) for y0 in range(0, ny, cy) for x0 in range(0, nx, cx)]
        full = (cz, cy, cx)

        def enc(tile: tuple[int, int]) -> Optional[bytes]:
            y0, x0 = tile
            block = data[:, y0:y0 + cy, x0:x0 + cx]
            if block.shape != full:  # edge chunk: pad with the fill value
                padded = np.full(full, self.fill, dtype=self.dtype)
                padded[: block.shape[0], : block.shape[1], : block.shape[2]] = block
                block = padded
            if np.all(block == self.fill):
                return None  # like zarr-python's write_empty_chunks=False
            return self.encode(block)

        encoded = pool.map(enc, tiles) if pool is not None else map(enc, tiles)
        iz = (z0 % self.shard[0]) // cz
        for (y0, x0), buf in zip(tiles, encoded):
            if buf is None:
                continue
            sy, sx = y0 // self.shard[1], x0 // self.shard[2]
            st = self._shard_state(sz, sy, sx)
            st["fh"].write(buf)
            iy = (y0 % self.shard[1]) // cy
            ix = (x0 % self.shard[2]) // cx
            st["index"][iz, iy, ix] = (st["offset"], len(buf))
            st["offset"] += len(buf)
        # shard row complete?
        if z0 + n >= min((sz + 1) * self.shard[0], self.shape[0]):
            self._finalize()

    def _finalize(self) -> None:
        import google_crc32c

        for st in self._open.values():
            index = st["index"].tobytes()
            st["fh"].write(index)
            st["fh"].write(np.uint32(google_crc32c.value(index)).tobytes())
            st["fh"].close()
            os.replace(st["path"] + ".partial", st["path"])
        self._open = {}
        self._shard_z = None

    def close(self) -> None:
        """Finalize any shard still open."""
        self._finalize()


def make_sink(arr: ZarrArray, append: Optional[bool] = None):  # noqa: ANN201
    """Pick `ShardAppendSink` when possible (or forced), else `ZarrApiSink`."""
    encode = None
    if append is not False and getattr(arr, "shards", None):
        inner = getattr(arr, "_array", None)
        store = getattr(inner, "store", None)
        if type(store).__name__ == "LocalStore":
            encode = _inner_encoder(arr)
    if encode is None:
        if append:
            raise ValueError("Append-only shard writing is not supported for "
                             "this array (needs sharded zarr v3, local store).")
        return ZarrApiSink(arr)
    return ShardAppendSink(arr, encode)


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------
# main-process heap besides the buffers, and per-thread working memory
MAIN_OVERHEAD_BYTES = 300 * 2**20
THREAD_OVERHEAD_BYTES = 16 * 2**20


def estimate_memory(
    shape: Sequence[int],
    itemsize: int,
    unit_z: int,
    tile: Sequence[int],
    workers: int,
    levels: int,
) -> int:
    """Rough peak memory (bytes) of `StreamingPyramidWriter`.

    `unit_z` is the z write unit and `tile` the block encoded by one thread
    (inner chunk for the append sink, shard otherwise).
    """
    slab = unit_z * shape[1] * shape[2] * itemsize
    pyramid = sum(slab / 4**k for k in range(1, levels + 1))
    per_thread = THREAD_OVERHEAD_BYTES + 3 * int(np.prod(tile)) * itemsize
    return int(3 * slab + pyramid + MAIN_OVERHEAD_BYTES + workers * per_thread)


def workers_for_memory(
    max_bytes: int, shape: Sequence[int], itemsize: int, unit_z: int,
    tile: Sequence[int], levels: int, max_workers: int,
) -> int:
    """Largest worker count (<= max_workers, >= 1) whose estimate fits."""
    for w in range(max_workers, 0, -1):
        if estimate_memory(shape, itemsize, unit_z, tile, w, levels) <= max_bytes:
            return w
    need = estimate_memory(shape, itemsize, unit_z, tile, 1, levels)
    raise MemoryError(
        f"Need ~{need / 2**30:.1f} GiB even with one worker; use a smaller "
        f"write unit along z (current {unit_z}) or allow more memory."
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
        append_shards: Optional[bool] = None,
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
        self.sinks = [make_sink(a, append_shards) for a in self.arrays]
        # downsampling factors to go from level l-1 to level l
        self.factors = [None] + [
            [1 if (i == no_pyramid_axis or prev[i] == 1) else 2 for i in range(3)]
            for prev in self.shapes[:-1]
        ]
        self.unit = [s.unit_z for s in self.sinks]
        self.fill = [0] * self.nlevels
        self.cursor = [0] * self.nlevels
        # frames of level l-1 waiting to be paired along z for level l
        self.carry: list[Optional[np.ndarray]] = [None] * self.nlevels
        workers = max(1, int(max_workers or 1))
        self._pool = ThreadPoolExecutor(workers) if workers > 1 else None
        # two buffers for the reader/writer ping-pong; one staging buffer per
        # level for partial units
        self._base_bufs = [
            np.empty((self.unit[0], *self.shapes[0][1:]), self.dtype)
            for _ in range(2)
        ]
        self.buf = [
            np.empty((u, *s[1:]), dtype=self.dtype)
            for u, s in zip(self.unit, self.shapes)
        ]
        logger.info(
            "Streaming %d levels with %s, z write units %s, %d thread(s), "
            "buffers %.1f MB",
            self.nlevels, type(self.sinks[0]).__name__, self.unit, workers,
            (sum(b.nbytes for b in self._base_bufs)
             + sum(b.nbytes for b in self.buf)) / 2**20,
        )

    @property
    def nlevels(self) -> int:
        """Number of levels including the base level."""
        return len(self.arrays)

    def base_buffers(self) -> list[np.ndarray]:
        """Two reusable (unit_z, Y, X) buffers for feeding `write_frames`."""
        return self._base_bufs

    # ------------------------------------------------------------------ write
    def _write(self, lvl: int, data: np.ndarray) -> None:
        z0 = self.cursor[lvl]
        if z0 + data.shape[0] > self.shapes[lvl][0]:
            raise RuntimeError(
                f"Level {lvl} overflow: {z0 + data.shape[0]} > {self.shapes[lvl][0]}")
        self.sinks[lvl].write(z0, data, self._pool)
        self.cursor[lvl] = z0 + data.shape[0]

    def _write_and_feed(self, lvl: int, data: np.ndarray) -> None:
        self._write(lvl, data)
        if lvl + 1 < self.nlevels:
            self._feed_next(lvl + 1, data)

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
                _downsample_parallel(frames[:usable], self.factors[lvl], self._pool),
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
            # aligned full (or final) unit: write straight from the caller
            self._write_and_feed(0, frames)
        else:
            self._push(0, frames)

    def close(self) -> None:
        """Flush partial buffers; trailing odd frames are trimmed."""
        try:
            for lvl in range(self.nlevels):
                self._flush(lvl)
            for sink in self.sinks:
                sink.close()
            for lvl, (cur, shp) in enumerate(zip(self.cursor, self.shapes)):
                if cur != shp[0]:
                    raise RuntimeError(
                        f"Level {lvl}: wrote {cur} of {shp[0]} slices")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Release the thread pool."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None


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
