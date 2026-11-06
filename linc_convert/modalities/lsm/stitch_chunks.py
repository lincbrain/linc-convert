"""Stitch the chunks of one slice into a blended OME-Zarr mosaic along y.

A deliberately small version of ``pipeline``: no channel registration and a
single correction. Needs ``linc_convert`` only for its zarr output and config
(``ZarrConfig``, ``GeneralConfig``, ``ZarrPythonGroup``); the helpers are in the
``stitching_utils`` folder next to this file.

    stitch_chunks(
        input_path="…/strips_raw/sample-slice060", yaml_path="LSM_metadata.yaml",
        camera_id=1, slice_number=60,
        coords_path="…/registration_{channel}.yaml",  # from register_chunks
        corrections_dir="…/60_corrections",       # written by create_corrections
        voxel_size=[1.0, 1.0, 1.0],
        general_config=GeneralConfig(out="…/stitched"), zarr_config=ZarrConfig(...))

For each channel of the camera, and each chunk (level 0 only):

1. Crop with the channel's ``stitchingCrop`` (y, and z if defined) and flip z
   if the camera's ``verticalFlip`` is set.
2. Correction: ``(vol - noise).clip(0) * scaler`` using
   ``{corrections_dir}/{ch}/{ch}_alt_zy_noise.tiff`` and ``..._alt_zy_scaler.tiff``,
   cast to uint16.
3. Skew correction: shear x as a function of z (``x += shear_x * z``) from the
   skew angle and raw voxel sizes in the metadata YAML, and optionally z as a
   function of y (``z += shear_z * y``, from register_chunks' ``dz`` when
   ``dz_as_shear_z`` is set). The chunk grows so nothing is cut off.
4. Blend the overlap with the previous chunk (power ramp times 1/scaler
   confidence) and write everything lazily, chunk by chunk, into one uint16
   array.

Chunk positions come from the register_chunks YAML: chunk ``i`` starts at
``round(i * weighted_average.y_offset)`` (full-resolution pixels, in the
stitching-cropped frame). All chunks start at z = 0. ``weighted_average.dz``
is ignored unless ``dz_as_shear_z`` is set, which applies it as a z shear
(``shear_z = -dz / y_offset``) of every chunk.

Output: ``{general_config.out}/{channel}`` (OME-Zarr with a pyramid, written
with ``space_scale=voxel_size``).

Per-chunk values (optional): ``per_chunk_registration`` uses each pair's own
``y_offset`` and ``dz`` from the register_chunks YAML instead of the average,
and ``per_chunk_corrections`` uses each chunk's own noise and scaler maps (made
by ``create_corrections`` with ``per_chunk``).

Lower resolution (optional): ``layer`` stitches a lower pyramid level of the
raw chunks (each level halves every axis) for fast test runs. Use the same
``layer`` for ``create_corrections`` (smaller correction maps) and
``register_chunks``; the registration YAML stays in full-resolution pixels and
is scaled here.

Region (optional): ``min_y``/``max_y`` and ``min_x``/``max_x`` restrict the output
to a window of the mosaic (full-resolution pixels of the mosaic frame, half-open
``[min, max)``, scaled to ``layer``). Only that window is written, so the output
array is smaller, chunks outside the y range are not read, and the result equals
the same slice of the full mosaic. x is the x of the sheared mosaic. z is never
cropped.

Bands (default): the output is written in bands, one band being one row of shards
(the shard height in y, e.g. 2048 rows). Each band is the lazy join of the mosaic
rows it covers (every chunk's blended overlap and own rows), cut along the shard grid
and stored in one go, so every shard is written exactly once; chunks are re-read
by the bands they cross. Bands cover different shards, so ``parallel_bands`` of them
run at the same time. Checkpoints count finished bands (``last_band``) in order.
``bands=False`` gives the older chunk-by-chunk writing, where a shard is rewritten
by every chunk that crosses it and chunks run together only if they share no shards.

Projections (default on): when a channel is stitched, its maximum and minimum
intensity projections over z are written as float32 TIFFs
``{out}/projections/slice{NNN}_{channel}_max.tiff`` and ``..._min.tiff`` (the min
projection ignores zeros and is inverted against its 99th percentile). Existing
files are not recomputed. They stay in ``out``, next to the channel's zarr folder.

Checkpoints: after each chunk (its blended overlap and its own rows) is fully
written, ``{general_config.out}/{channel}.stitch_checkpoint.json`` is updated.
Running again with the same inputs reopens the output, skips the finished
chunks (the last finished one is only rebuilt lazily, not rewritten, so the next
blend has its data) and carries on; a finished channel is skipped. Changed
inputs, or ``restart=True``, start over.
"""

import json
import os
import shutil
from typing import List, Optional, Tuple

import cyclopts
import dask.array as da
import numpy as np
import yaml
from dask.distributed import progress

from linc_convert.modalities.lsm.cli import lsm
from linc_convert.modalities.lsm.stitching_utils.blending import (
    blend_overlap,
    sheared_weights,
)
from linc_convert.modalities.lsm.stitching_utils.corrections import (
    apply_corrections,
    correction_paths,
    load_corrections,
    shear_volume,
    sheared_shape,
)
from linc_convert.modalities.lsm.stitching_utils.io import (
    atomic_write,
    crop_and_flip,
    discover_tile_paths,
    open_tile_array,
    start_dask,
)
from linc_convert.modalities.lsm.stitching_utils.projections import write_projections
from linc_convert.modalities.lsm.stitching_utils.registration import (
    load_pair_registration,
    load_tile_positions,
)
from linc_convert.modalities.lsm.stitching_utils.scan_metadata import (
    Crop,
    get_channel_names,
    get_crops,
    get_skew_shear,
    get_vertical_flip,
)
from linc_convert.utils.io.zarr.drivers.zarr_python import ZarrPythonGroup
from linc_convert.utils.zarr_config import GeneralConfig, ZarrConfig, autoconfig

# --------------------------------------------------------------------------
# One channel
# --------------------------------------------------------------------------


def _resolve_tile_paths(input_path: str, camera_id: int, coords_path: str,
                        all_chunks: bool = False) -> List[str]:
    with open(coords_path) as f:
        listed = (yaml.safe_load(f) or {}).get("chunks")
    if all_chunks or not listed:
        return discover_tile_paths(input_path, camera_id)
    tile_paths = [os.path.join(input_path, os.path.basename(str(c).rstrip("/")))
                  for c in listed]
    missing = [p for p in tile_paths if not os.path.isdir(p)]
    if missing:
        raise FileNotFoundError(f"Chunk(s) listed in {coords_path} not found in "
                                f"{input_path}: {missing[:3]}")
    return tile_paths


# target dask block (z, y, x) of the raw chunks, rounded down to a multiple of the
# stored chunks: a few large blocks instead of very many small ones keeps the task
# graph small at full resolution (a graph of ~200 MiB stalled the scheduler).
# y is never split: with a z shear the nearest-neighbour shear of dask-image is off
# by one voxel in z at some rows next to every y block edge (0.4% of the voxels;
# splitting z or x is exact), so a chunk is kept whole along y.
DASK_BLOCK = (128, 1 << 30, 512)


def _checkpoint_path(out_dir: str) -> str:
    """Checkpoint file of a channel, next to its output folder."""
    return out_dir.rstrip("/") + ".stitch_checkpoint.json"


def _read_checkpoint(path: str, signature: dict) -> Optional[dict]:
    """Return the checkpoint if it exists and belongs to this exact stitch."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            state = json.load(f)
    except (OSError, ValueError):
        return None
    if state.get("signature") != signature:
        print("  checkpoint was made with different inputs; starting over")
        return None
    return state


def _write_checkpoint(path: str, signature: dict, last: int, done: bool = False,
                      key: str = "last_chunk", **extra: int) -> None:
    """Record that units ``0..last`` (chunks, or bands with ``key="last_band"``)
    are completely written. ``extra`` values are stored next to them."""
    text = json.dumps({"signature": signature, key: last, "done": done, **extra})
    atomic_write(path, lambda tmp: tmp.write_text(text))


def _aligned_rechunk(arr: da.Array, start: Tuple[int, ...],
                     grid: Tuple[int, ...]) -> da.Array:
    """Rechunk so block edges fall on the output array's chunk/shard grid.

    ``arr`` is written to the output starting at index ``start`` (per axis).
    Blocks are cut at the multiples of ``grid`` in *output* coordinates, so
    each block is a whole shard (or chunk) except for the pieces at the two
    ends of the region. Writing a block that is not aligned would make zarr
    read, modify and rewrite the shards it straddles.

    Parameters
    ----------
    arr : dask.array.Array
        Data to be stored.
    start : tuple of int
        Index in the output array of ``arr``'s first element, per axis.
    grid : tuple of int
        Shard shape (or chunk shape if there are no shards) of the output.

    Returns
    -------
    dask.array.Array
        ``arr`` rechunked along the grid.
    """
    chunks = {}
    for axis, (first, step) in enumerate(zip(start, grid)):
        pos, end, sizes = first, first + arr.shape[axis], []
        while pos < end:
            nxt = min((pos // step + 1) * step, end)
            sizes.append(nxt - pos)
            pos = nxt
        chunks[axis] = tuple(sizes)
    return arr.rechunk(chunks)


def _clip_range(bounds: Tuple[Optional[int], Optional[int]], size: int,
                axis: str) -> Tuple[int, int]:
    """Clip optional ``(lo, hi)`` bounds to ``[0, size)``.

    Raises
    ------
    ValueError
        If nothing of the axis is left.
    """
    lo = 0 if bounds[0] is None else int(bounds[0])
    hi = size if bounds[1] is None else int(bounds[1])
    lo, hi = max(lo, 0), min(hi, size)
    if hi <= lo:
        raise ValueError(f"the {axis} range {tuple(bounds)} leaves nothing of the "
                         f"mosaic ({axis} size {size} at this layer)")
    return lo, hi


def _layout(coords: str, tile_paths: List[str], channel: str, dz_as_shear_z: bool,
            per_chunk_registration: bool,
            layer: int = 0) -> Tuple[List[int], List[int], float]:
    """Y positions, z origins and z shear of the chunks of one channel.

    Shared registration: chunk ``i`` starts at ``round(i * y_offset)``, all z
    origins are 0, and ``dz`` becomes ``shear_z = -dz / y_offset`` if
    ``dz_as_shear_z`` is set.

    Per-chunk registration: every pair uses its own ``y_offset`` and ``dz``
    (see ``load_pair_registration``). The y positions are the running sum of the
    offsets. ``shear_z`` is the same for all chunks (``-dz_avg / y_avg`` with
    ``dz_as_shear_z``, else 0), so the chunks keep one shape. Each pair's own
    ``dz`` is applied as a z offset of the next chunk,
    ``dz_i + shear_z * y_offset_i``, which lines the two chunks up exactly
    (derivation: a feature at row ``r`` of chunk A is at row ``r - y_offset`` of
    chunk B and ``dz`` lower in z, and the shear moves it by ``shear_z * r``).

    The registration YAML is in full-resolution pixels. On a lower pyramid
    ``layer`` the positions are divided by ``2 ** layer``; ``shear_z`` is a
    ratio and does not change.

    Returns
    -------
    positions_y, positions_z : list of int
        In pixels of ``layer``.
    shear_z : float
    """
    n = len(tile_paths)
    scale = 2 ** layer
    if not per_chunk_registration:
        positions_y, dz = load_tile_positions(coords, n)
        shear_z = 0.0
        if dz_as_shear_z:
            with open(coords) as f:
                y_offset = float(yaml.safe_load(
                    f)["weighted_average"]["y_offset"])
            shear_z = -dz / y_offset
            print(f"channel {channel}: dz = {dz:.3f} px over y_offset "
                  f"{y_offset:.2f} px -> shear_z = {shear_z:.6f}")
        elif abs(dz) > 0.5:
            print(f"channel {channel}: NOTE dz = {dz:.2f} px is not applied "
                  "(dz_as_shear_z=False)")
        return [int(round(y / scale)) for y in positions_y], [0] * n, shear_z

    names = [os.path.basename(p.rstrip("/")) for p in tile_paths]
    y, dz_pairs, y_avg, dz_avg = load_pair_registration(coords, names)
    shear_z = -dz_avg / y_avg if dz_as_shear_z else 0.0
    positions_y = [int(round(v / scale))
                   for v in np.concatenate([[0.0], np.cumsum(y)])]
    z_offsets = np.concatenate([[0.0], np.cumsum(dz_pairs + shear_z * y)])
    positions_z = [int(round(v / scale)) for v in z_offsets]
    print(f"channel {channel}: per-chunk registration, shear_z = {shear_z:.6f}; "
          f"y offsets {y.min():.1f}..{y.max():.1f}, dz {dz_pairs.min():+.2f}.."
          f"{dz_pairs.max():+.2f} px")
    return positions_y, positions_z, shear_z


def corrected_tile(vol: da.Array, noise: np.ndarray, scaler: np.ndarray,
                   shear_x: float, shear_z: float) -> da.Array:
    """Lazily correct and shear a whole chunk.

    Parameters
    ----------
    vol : dask.array.Array
        Cropped, flipped raw chunk, shape (Z, Y, X).
    noise : numpy.ndarray
        Background map, shape (Z, Y).
    scaler : numpy.ndarray
        Multiplier map, shape (Z, Y).
    shear_x : float
        x shift per z step (>= 0).
    shear_z : float
        z shift per y step; negative values shift z towards smaller values with
        increasing y, and the chunk is padded so that z starts at 0.

    Returns
    -------
    dask.array.Array
        uint16 array of shape ``sheared_shape(vol.shape, ...)``; computing it
        runs ``(vol - noise).clip(0) * scaler``, then the shear (nearest
        neighbour). Nothing is read until it is computed.
    """
    corrected = apply_corrections(vol, noise, scaler)
    if shear_z >= 0:
        return shear_volume(corrected, shear_x, shear_z)
    # negative z shear: shear x first (it needs the original z), then flip z,
    # shear by |shear_z| and flip back
    corrected = shear_volume(corrected, shear_x, 0.0)
    return shear_volume(corrected[::-1], 0.0, -shear_z)[::-1]


def _weights(scaler: np.ndarray, shear_z: float, z_out: int) -> np.ndarray:
    """``sheared_weights`` that also accepts a negative ``shear_z``."""
    if shear_z >= 0:
        return sheared_weights(scaler, shear_z, z_out)
    return sheared_weights(scaler[::-1], -shear_z, z_out)[::-1]


def _stitch_bands(first_band: int, n: int, sy: int, z_out: int,
                  mosaic: Tuple[int, int, int], window: Tuple[int, int, int, int],
                  grid: Tuple[int, ...], target: object, client: object,
                  positions_y: List[int], z_origin: List[int], tile_paths: List[str],
                  open_tile, raw_shape: Tuple[int, int, int], chunk_maps,
                  shear_x: float, shear_z: float, ckpt: str, signature: dict,
                  parallel_bands: int) -> None:
    """Write the mosaic band by band, a band being one row of shards.

    Every band is the lazy join (along y) of the mosaic rows it covers: for each
    chunk its blended overlap with the previous chunk and its own rows (a gap
    between chunks is zeros). It is cut along the shard grid and stored in one
    ``da.store``, so every shard is written exactly once and never read back.
    Bands cover different shard rows, so up to ``parallel_bands`` of them can run at
    the same time. A checkpoint records the last band of the finished prefix.
    """
    y0, y1, x0, x1 = window
    gy = int(grid[1])
    n_bands = -(-(y1 - y0) // gy)

    # which rows come from where: (first row, end row, kind, chunk, overlap_prev)
    segments = []
    for k in range(n):
        ystart = positions_y[k]
        overlap_prev = max(sy - (ystart - positions_y[k - 1]), 0) if k > 0 else 0
        overlap_next = (max(sy - (positions_y[k + 1] - ystart), 0)
                        if k < n - 1 else 0)
        if overlap_prev + overlap_next > sy:
            raise ValueError(f"chunk {k}: overlaps ({overlap_prev} + "
                             f"{overlap_next}) exceed the chunk height {sy}")
        for kind, a, b in (("blend", ystart, ystart + overlap_prev),
                           ("own", ystart + overlap_prev, ystart + sy - overlap_next)):
            a, b = max(a, y0), min(b, y1)
            if b > a:
                segments.append((a, b, kind, k, overlap_prev))

    tiles: dict = {}

    def tile(k: int) -> Tuple[da.Array, np.ndarray]:
        """Corrected, sheared chunk ``k`` (lazy) and its blend weights."""
        if k not in tiles:
            raw = open_tile(tile_paths[k])
            if raw.shape != raw_shape:
                raise ValueError(f"{os.path.basename(tile_paths[k])} has shape "
                                 f"{raw.shape}, expected {raw_shape}")
            noise, scaler, weights = chunk_maps(k)
            tiles[k] = (corrected_tile(raw, noise, scaler, shear_x, shear_z),
                        weights)
        return tiles[k]

    def zeros(rows: int) -> da.Array:
        return da.zeros((mosaic[0], rows, x1 - x0), dtype=np.uint16,
                        chunks=(mosaic[0], rows, x1 - x0))

    def band_array(band: int) -> Tuple[da.Array, str]:
        lo, hi = y0 + band * gy, min(y0 + (band + 1) * gy, y1)
        pieces, row, used = [], lo, []
        for a, b, kind, k, overlap_prev in segments:
            a, b = max(a, lo), min(b, hi)
            if b <= a:
                continue
            if a > row:
                pieces.append(zeros(a - row))
            ystart, zstart = positions_y[k], z_origin[k]
            cur, cur_w = tile(k)
            if kind == "blend":
                prev, prev_w = tile(k - 1)
                blended = blend_overlap(prev, cur, prev_w, cur_w, z_origin[k - 1],
                                        zstart, mosaic[0], overlap_prev)
                pieces.append(blended[:, a - ystart:b - ystart, x0:x1])
            else:
                own = cur[:, a - ystart:b - ystart, x0:x1]
                if zstart > 0 or zstart + z_out < mosaic[0]:
                    # same z extent as the other pieces; nothing else lies in
                    # these rows, so the padding is zeros, as in the empty output
                    own = da.pad(own, ((zstart, mosaic[0] - zstart - z_out),
                                       (0, 0), (0, 0)))
                pieces.append(own)
            row = b
            if k not in used:
                used.append(k)
        if row < hi:
            pieces.append(zeros(hi - row))
        data = pieces[0] if len(pieces) == 1 else da.concatenate(pieces, axis=1)
        label = f"chunks {min(used) + 1}..{max(used) + 1}" if used else "no chunk"
        return data, label

    pending: List[Tuple[int, list]] = []

    def finish_oldest() -> None:
        band, futures = pending.pop(0)
        print(f"  waiting for band {band + 1}/{n_bands}", flush=True)
        progress(futures, notebook=False)
        for fut in futures:
            fut.result()
        print(flush=True)
        _write_checkpoint(ckpt, signature, band, key="last_band", band_rows=gy)

    for band in range(first_band, n_bands):
        data, label = band_array(band)
        start = band * gy
        data = _aligned_rechunk(data, (0, start, 0), grid)
        region = (slice(0, mosaic[0]), slice(start, start + data.shape[1]),
                  slice(0, x1 - x0))
        stores = da.store([data], [target], regions=[region], lock=True,
                          compute=False)
        futures = client.compute(
            list(stores) if isinstance(stores, (list, tuple)) else [stores])
        print(f"band {band + 1}/{n_bands}: rows {start}:{start + data.shape[1]} from "
              f"{label} ({len(pending)} other band(s) in flight)", flush=True)
        pending.append((band, futures))
        while len(pending) >= max(1, int(parallel_bands)):
            finish_oldest()
    while pending:
        finish_oldest()


def stitch_channel(
    tile_paths: List[str], channel: str, crop: Crop,
    vertical_flip: bool, shear_x: float, shear_z: float, positions_y: List[int],
    positions_z: List[int], corrections_dir: str, out_dir: str,
    voxel_size: List[float], zarr_config: ZarrConfig,
    per_chunk_corrections: bool = False, layer: int = 0,
    y_range: Tuple[Optional[int], Optional[int]] = (None, None),
    x_range: Tuple[Optional[int], Optional[int]] = (None, None),
    memory_gb: int = 128, parallel_chunks: int = 2, bands: bool = True,
    parallel_bands: int = 2, x_tolerance: int = 64,
) -> None:
    """Stitch all chunks of one channel into an OME-Zarr mosaic, lazily.

    Parameters
    ----------
    tile_paths : list of str
        Chunk ``.ome.zarr`` folders in acquisition order.
    channel : str
        Channel name.
    crop : tuple
        ``(z_start, z_end, y_start, y_end)`` stitching crop.
    vertical_flip : bool
        Flip each chunk along z.
    shear_x : float
        x shift per z step (>= 0).
    shear_z : float
        z shift per y step (may be negative).
    positions_y : list of int
        y start of every chunk in the mosaic, increasing.
    positions_z : list of int
        z origin of every *sheared* chunk in the mosaic (their minimum is
        moved to 0). This is the final placement, shear included.
    corrections_dir : str
        Folder with the ``create_corrections`` output.
    out_dir : str
        Output ``.ome.zarr`` folder of this channel.
    voxel_size : list of float
        Voxel size in z, y, x order (already reversed).
    zarr_config : ZarrConfig
        Zarr storage configuration.
    per_chunk_corrections : bool, default=False
        Use each chunk's own maps (``create_corrections`` with ``per_chunk``,
        chunk ``i`` uses chunk number ``i + 1``) instead of the shared ones.
        Each chunk's own blend confidence follows from its own scaler.
    layer : int, default=0
        Pyramid level of the raw chunks to stitch (0 = full resolution). The
        crop is scaled to it; ``positions_y``/``positions_z``, the corrections
        and ``voxel_size`` must already be for this level.
    y_range, x_range : tuple of int or None
        ``(lo, hi)`` window of the mosaic to write, in pixels of ``layer``
        (``hi`` exclusive; None leaves that side open). Values are clipped to
        the mosaic. Only the window is written; the output array has its size.
    memory_gb : int, default=128
        Total memory this process and its dask workers may use together. The
        number of worker threads is reduced if tasks writing one output block
        (a shard) could not all fit; see ``start_dask``.
    parallel_chunks : int, default=2
        Chunk mode only (``bands=False``): how many chunks are computed and
        written at the same time.
    bands : bool, default=True
        Write the mosaic in bands of whole shard rows instead of chunk by chunk
        (see the module docstring).
    parallel_bands : int, default=2
        Band mode only: how many bands are computed and written at the same time.
    x_tolerance : int, default=64
        A chunk whose x length differs from the first chunk's by at most this many
        full-resolution pixels (scaled to ``layer``, at least 1) and whose z and y
        lengths agree is made as long as the first chunk: zeros are appended at the
        end of x if it is shorter, the extra columns are cut if it is longer, and a
        warning is printed. 0 requires identical shapes.

    Notes
    -----
    Progress is checkpointed after every chunk in
    ``{out_dir}.stitch_checkpoint.json`` and resumed from there; see the module
    docstring.

    Raises
    ------
    ValueError
        If the position lists do not match the tiles, ``shear_x`` is negative,
        chunk positions do not increase, chunk shapes differ, the overlap
        from the previous and next chunk exceed the chunk height together, or a
        requested y or x range leaves nothing of the mosaic.
    """
    n = len(tile_paths)
    if not (len(positions_y) == len(positions_z) == n):
        raise ValueError(f"{n} tiles but {len(positions_y)} y positions and "
                         f"{len(positions_z)} z positions")
    if shear_x < 0:
        raise ValueError("shear_x must be >= 0")
    steps = np.diff(positions_y)
    if n > 1 and steps.min() <= 0:
        raise ValueError("chunk positions must increase")

    def open_tile_exact(path: str) -> da.Array:
        return crop_and_flip(open_tile_array(path, layer, block=DASK_BLOCK), crop,
                             vertical_flip, layer)

    raw_shape = open_tile_exact(tile_paths[0]).shape
    x_tol = (max(1, int(x_tolerance) >> layer) if x_tolerance > 0 else 0)
    warned_x: set = set()

    def open_tile(path: str) -> da.Array:
        """Chunk as stitched; its x length is matched to the first chunk's if the
        difference is within ``x_tol`` (z and y must agree)."""
        arr = open_tile_exact(path)
        dx = arr.shape[2] - raw_shape[2]
        if dx != 0 and arr.shape[:2] == raw_shape[:2] and abs(dx) <= x_tol:
            if path not in warned_x:
                warned_x.add(path)
                print(f"  WARNING: {os.path.basename(path)} has x length "
                      f"{arr.shape[2]}, the first chunk {raw_shape[2]}: "
                      + ("zeros added at the end of x" if dx < 0
                         else "the last x columns are cut"), flush=True)
            if dx < 0:
                arr = da.pad(arr, ((0, 0), (0, 0), (0, -dx)), mode="constant",
                             constant_values=0)
            else:
                arr = arr[:, :, :raw_shape[2]]
        return arr
    sz, sy, sx = raw_shape
    z_out, _, x_out = sheared_shape(raw_shape, shear_x, abs(shear_z))
    if per_chunk_corrections:
        missing = [p for i in range(n)
                   for p in correction_paths(corrections_dir, channel, i + 1).values()
                   if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"Missing per-chunk correction file(s), e.g. {missing[0]} "
                f"({len(missing)} missing)")

    def chunk_maps(index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Noise map, scaler map and blend confidence of one chunk."""
        noise, scaler = load_corrections(
            corrections_dir, channel, index + 1 if per_chunk_corrections else None)
        if noise.shape != (sz, sy) or scaler.shape != (sz, sy):
            raise ValueError(
                f"corrections for channel {channel} have shape {noise.shape}, "
                f"but chunks at layer {layer} are (Z, Y)=({sz}, {sy}); run "
                f"create_corrections with layer={layer}")
        return noise, scaler, _weights(scaler, shear_z, z_out)

    if n > 1 and steps.max() > sy:
        print(f"  WARNING: gap between chunks ({steps.max()} > chunk height {sy}); "
              "the gap is left empty")

    z_origin = [int(z) - int(min(positions_z)) for z in positions_z]
    mosaic = (z_out + max(z_origin), positions_y[-1] + sy, x_out)
    y0, y1 = _clip_range(y_range, mosaic[1], "y")
    x0, x1 = _clip_range(x_range, mosaic[2], "x")
    fullshape = (mosaic[0], y1 - y0, x1 - x0)
    windowed = fullshape != mosaic
    print(f"channel {channel}: {n} chunks, chunk (Z,Y,X)={raw_shape}, mosaic {mosaic}"
          + (f"; writing y {y0}:{y1}, x {x0}:{x1} -> {fullshape}" if windowed else ""))

    signature = {"tiles": [os.path.basename(p) for p in tile_paths],
                 "positions_y": [int(y) for y in positions_y],
                 "z_origin": z_origin, "shape": list(fullshape),
                 "shear": [float(shear_x), float(shear_z)],
                 "per_chunk_corrections": bool(per_chunk_corrections)}
    if windowed:
        signature["window"] = [y0, y1, x0, x1]
    if bands:
        signature["mode"] = "bands"
    ckpt = _checkpoint_path(out_dir)
    state = _read_checkpoint(ckpt, signature)
    if state is not None and not os.path.isdir(out_dir):
        state = None
    if state is not None and state["done"]:
        print(
            f"channel {channel}: already stitched (checkpoint {ckpt}); skipping")
        return

    group = array = None
    if state is not None:
        group = ZarrPythonGroup.open(out_dir, mode="r+")
        array = group["0"]
        if bands and state.get("band_rows") != int((array.shards or array.chunks)[1]):
            print("  the shard height changed since the checkpoint; starting over")
            state, group, array = None, None, None

    if state is None:
        # fresh start: discard any partial output of an unfinished earlier run
        shutil.rmtree(out_dir, ignore_errors=True)
        if os.path.exists(ckpt):
            os.remove(ckpt)
        group = ZarrPythonGroup.from_config(out_dir, zarr_config)
        array = group.create_array("0", shape=fullshape, zarr_config=zarr_config,
                                   dtype=np.uint16)
        start = 0
    elif bands:
        start = int(state["last_band"]) + 1
        print(f"channel {channel}: resuming after band {start - 1} "
              f"({start} band(s) written)")
    else:
        # resume: the last finished chunk is rebuilt lazily below (nothing is
        # written for it) so the next blend can use it
        start = int(state["last_chunk"]) + 1
        print(f"channel {channel}: resuming after chunk {start - 1} "
              f"({start}/{n} written)")

    # dask pickles the store target to its workers; hand them the plain zarr
    # array (a wrapper with a recursive __getattr__ cannot be unpickled)
    target = getattr(array, "_array", array)

    grid = tuple(int(g) for g in (array.shards or array.chunks))
    block = [min(g, n) for g, n in zip(grid, fullshape)]
    block_gib = float(np.prod(block)) * 2 / 2 ** 30                  # uint16
    # a store task holds the block and the pieces it is merged from, plus the
    # float32 working copy of the raw chunk pieces it reads; a band block also
    # holds the float32 blend of the overlap rows
    task_gb = (4.0 if bands else 2.5) * block_gib + 0.5
    client = start_dask(memory_gb, task_gb=task_gb,
                        reserve_gb=max(2.0, 0.1 * memory_gb))
    try:
        if bands:
            _stitch_bands(
                start, n, sy, z_out, mosaic, (y0, y1, x0, x1), grid, target, client,
                positions_y, z_origin, tile_paths, open_tile, raw_shape, chunk_maps,
                shear_x, shear_z, ckpt, signature, parallel_bands)
            group.generate_pyramid(levels=zarr_config.levels)
            group.write_ome_metadata(axes=["z", "y", "x"], space_scale=voxel_size)
            _write_checkpoint(ckpt, signature, -(-(y1 - y0) // grid[1]) - 1,
                              done=True, key="last_band", band_rows=grid[1])
            return
        prev: Optional[da.Array] = None
        prev_weights: Optional[np.ndarray] = None
        if start > 0:
            raw = open_tile(tile_paths[start - 1])
            if raw.shape != raw_shape:
                raise ValueError(f"{os.path.basename(tile_paths[start - 1])} has "
                                 f"shape {raw.shape}, expected {raw_shape}")
            noise, scaler, prev_weights = chunk_maps(start - 1)
            prev = corrected_tile(raw, noise, scaler, shear_x, shear_z)

        # chunks are independent (every blend is built lazily from the corrected
        # chunks, nothing reads the output), so up to ``parallel_chunks`` of them are
        # computed at the same time, with one hard rule: two chunks are in flight
        # together only if they write to different rows of shards. Writing part of a
        # shard reads, changes and rewrites the whole shard, so two writers of one
        # shard would overwrite each other's rows. (With shards taller than the
        # chunk step, as at layer 0, neighbouring chunks share shards and are
        # therefore written one after another.) Checkpoints go in chunk order:
        # chunk k is recorded only after chunks 0..k are written.
        pending: List[Tuple[int, list, Tuple[int, int]]] = []

        def finish_oldest() -> None:
            idx, futures, _ = pending.pop(0)
            if futures:
                print(f"  waiting for chunk {idx + 1}/{n}", flush=True)
                progress(futures, notebook=False)
                for fut in futures:
                    fut.result()
                print(flush=True)
            _write_checkpoint(ckpt, signature, idx)

        for index in range(start, n):
            path = tile_paths[index]
            raw = open_tile(path)
            if raw.shape != raw_shape:
                raise ValueError(f"{os.path.basename(path)} has shape {raw.shape}, "
                                 f"expected {raw_shape}")
            ystart, zstart = positions_y[index], z_origin[index]
            overlap_prev = max(
                sy - (ystart - positions_y[index - 1]), 0) if index > 0 else 0
            overlap_next = (max(sy - (positions_y[index + 1] - ystart), 0)
                            if index < n - 1 else 0)
            if overlap_prev + overlap_next > sy:
                raise ValueError(f"chunk {index}: overlaps ({overlap_prev} + "
                                 f"{overlap_next}) exceed the chunk height {sy}")
            print(f"[{index + 1}] {os.path.basename(path)}  y={ystart} z={zstart}  "
                  f"overlap prev/next = {overlap_prev}/{overlap_next}")

            noise, scaler, cur_weights = chunk_maps(index)
            cur = corrected_tile(raw, noise, scaler, shear_x, shear_z)
            # rows of the mosaic this chunk writes: its blended overlap with the
            # previous chunk, then its own rows. They are contiguous, so they are
            # joined and written as one array: every shard they touch is written by
            # exactly one task.
            parts, rows_written, full_z = [], None, False
            if overlap_prev > 0:
                r0, r1 = max(ystart, y0), min(ystart + overlap_prev, y1)
                if r1 > r0:
                    assert prev is not None and prev_weights is not None
                    blended = blend_overlap(prev, cur, prev_weights, cur_weights,
                                            z_origin[index - 1],
                                            zstart, mosaic[0], overlap_prev)
                    parts.append(blended[:, r0 - ystart:r1 - ystart, x0:x1])
                    rows_written, full_z = (r0, r1), True
            first_row, last_row = overlap_prev, sy - overlap_next
            r0, r1 = max(ystart + first_row, y0), min(ystart + last_row, y1)
            if r1 > r0:
                own = cur[:, r0 - ystart:r1 - ystart, x0:x1]
                if parts:
                    # same z extent as the blended rows (nothing else is in these
                    # rows, so the padding is zeros, as in the empty output)
                    own = da.pad(own, ((zstart, mosaic[0] - zstart - z_out),
                                       (0, 0), (0, 0)))
                parts.append(own)
                rows_written = (rows_written[0] if rows_written else r0, r1)
            elif windowed:
                print("  outside the requested y range; nothing to write", flush=True)
            futures: list = []
            shard_rows = (0, -1)
            if parts:
                r0, r1 = rows_written
                data = parts[0] if len(parts) == 1 else da.concatenate(parts, axis=1)
                region_z = (slice(0, mosaic[0]) if full_z
                            else slice(zstart, zstart + z_out))
                data = _aligned_rechunk(data, (region_z.start, r0 - y0, 0), grid)
                shard_rows = ((r0 - y0) // grid[1], (r1 - y0 - 1) // grid[1])
                # wait for every chunk that writes to one of the same shard rows
                while any(not (shard_rows[1] < p[2][0] or shard_rows[0] > p[2][1])
                          for p in pending):
                    finish_oldest()
                futures = client.compute(da.store(
                    [data], [target], regions=[(region_z, slice(r0 - y0, r1 - y0),
                                                slice(0, x1 - x0))],
                    lock=True, compute=False))
                futures = list(futures) if isinstance(futures, (list, tuple)) \
                    else [futures]
                print(f"  chunk {index + 1}/{n}: submitted rows {r0 - y0}:{r1 - y0} "
                      f"(shard rows {shard_rows[0]}..{shard_rows[1]}; "
                      f"{len(pending)} other chunk(s) in flight)", flush=True)
            pending.append((index, futures, shard_rows))
            prev, prev_weights = cur, cur_weights
            while len(pending) >= max(1, int(parallel_chunks)):
                finish_oldest()
        while pending:
            finish_oldest()

        group.generate_pyramid(levels=zarr_config.levels)
        group.write_ome_metadata(axes=["z", "y", "x"], space_scale=voxel_size)
        _write_checkpoint(ckpt, signature, n - 1, done=True)
    finally:
        client.close()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

stitch_chunks = cyclopts.App(
    name="stitch_chunks", help_format="markdown")
lsm.command(stitch_chunks)


@stitch_chunks.default
@autoconfig
def stitch_chunks(
    input_path: str,
    yaml_path: str,
    camera_id: int,
    slice_number: int,
    coords_path: str,
    corrections_dir: str,
    *,
    voxel_size: List[float] = [1, 1, 1],
    memory_gb: int = 128,
    all_chunks: bool = False,
    dz_as_shear_z: bool = False,
    per_chunk_registration: bool = False,
    per_chunk_corrections: bool = False,
    restart: bool = False,
    parallel_chunks: int = 2,
    bands: bool = True,
    parallel_bands: int = 2,
    x_tolerance: int = 64,
    projections: bool = True,
    projections_dir: Optional[str] = None,
    layer: int = 0,
    min_x: Optional[int] = None,
    max_x: Optional[int] = None,
    min_y: Optional[int] = None,
    max_y: Optional[int] = None,
    general_config: Optional[GeneralConfig] = None,
    zarr_config: Optional[ZarrConfig] = None,
) -> None:
    """Stitch the chunks of one slice and camera into blended OME-Zarr mosaics.

    Parameters
    ----------
    input_path : str
        Folder with the chunk ``*.ome.zarr`` folders. If the register_chunks
        YAML lists ``chunks`` (and ``all_chunks`` is False), only those
        (matched by file name) are stitched, in that order; otherwise all
        chunks of the camera, sorted.
    yaml_path : str
        Scan metadata YAML (crop definitions, flip, skew).
    camera_id : int
        Camera to stitch (1 or 2). Every channel of the camera is written.
    slice_number : int
        Physical slice number (selects the config epoch).
    coords_path : str
        YAML written by ``register_chunks``. May contain ``{channel}``, which is
        replaced by each channel's name, so that every channel uses its own
        registration (e.g. ``registration/slice060_{channel}.yaml``).
    corrections_dir : str
        Folder written by ``create_corrections``.
    voxel_size : list of float
        Voxel size in x, y, z (microns) for the output metadata.
    memory_gb : int, default=128
        Total memory, in GB, that this process and its dask workers may use
        together (a tenth, at least 2 GB, is kept for this process). Worker
        threads are reduced if tasks writing one shard could not all fit, so
        a smaller shard size allows more threads.
    bands : bool, default=True
        Write the mosaic in bands of whole shard rows (every shard is written once)
        instead of chunk by chunk. See the module docstring.
    parallel_bands : int, default=2
        With ``bands``: how many bands are computed and written at the same time.
    x_tolerance : int, default=64
        A chunk whose x length differs from the first chunk's by at most this many
        full-resolution pixels (and whose z and y lengths agree) is padded with zeros
        at the end of x, or cut, to the first chunk's length, with a warning
        (e.g. 45198 instead of 45204 columns). 0 requires identical shapes.
    parallel_chunks : int, default=2
        Without ``bands``: how many chunks are computed and written at the same
        time, only when they write to different shard rows (1 = one chunk after
        another). Checkpoints still follow the chunk order.
    all_chunks : bool, default=False
        Stitch every chunk of the camera (sorted), ignoring the ``chunks`` list
        in the register_chunks YAML. Use it when the registration was run on a
        subset of the chunks; positions are still ``round(i * y_offset)``.
    dz_as_shear_z : bool, default=False
        If False, ``weighted_average.dz`` from register_chunks is not applied.
        If True, it is applied as a z shear of every chunk instead of as
        offsets between chunks: ``shear_z = -dz / y_offset`` (z shift per y
        step), with all chunk z offsets left at 0. A chunk's rows then move
        in z by ``dz`` over one chunk step, which removes the z drift measured
        between neighbouring chunks.
    per_chunk_registration : bool, default=False
        Give every chunk its own registration: each neighbouring pair uses its
        own ``y_offset`` and ``dz`` from the ``pairs`` in the register_chunks
        YAML (chunk positions are the running sum of the offsets), instead of
        the weighted average. Pairs that were not registered, or whose quality
        is missing or <= 0, use the average. A pair's ``dz`` becomes a z
        offset of the next chunk (on top of the common ``dz_as_shear_z`` shear,
        if set).
    per_chunk_corrections : bool, default=False
        Give every chunk its own noise and scaler maps (create_corrections
        with ``per_chunk``), also used for each chunk's blend confidence.
    restart : bool, default=False
        Ignore any checkpoints and rewrite every channel from scratch.
    projections : bool, default=True
        After a channel is stitched, write its maximum and minimum intensity
        projection over z as TIFFs (see the module docstring). Projections that
        already exist are skipped, also when the stitch itself was already done.
    projections_dir : str, optional
        Where the projection TIFFs go; default ``{out}/projections``.
    layer : int, default=0
        Pyramid level of the raw chunks to stitch (0 = full resolution), for
        quick low-resolution test runs. Everything is scaled to the level: the
        registration positions (``/ 2**layer``), the output voxel size
        (``* 2**layer``) and the crops. ``corrections_dir`` must come from
        ``create_corrections`` with the same ``layer``, and ``out`` should be
        a different folder from the full-resolution output.
    min_x, max_x, min_y, max_y : int, optional
        Write only this window of the mosaic: pixels ``min <= index < max`` in
        full-resolution units of the mosaic frame (the same units as the
        register_chunks offsets; divided by ``2**layer`` for a lower ``layer``).
        Leave a bound out to keep that side open. y is the stitching axis; x is
        the x of the sheared mosaic. The output array is only as large as the
        window, chunks that lie outside the y range are not read, and the data
        equal the same slice of the full mosaic. The output's origin is not
        recorded in the metadata, and a different window starts a new stitch
        (checkpoints include the window). z is not cropped.
    general_config : GeneralConfig
        Output configuration; ``out`` is the output folder.
    zarr_config : ZarrConfig
        Zarr storage configuration (chunking, pyramid levels, ...).

    Raises
    ------
    ValueError
        If ``general_config.out`` is not set, no chunks are found, or the
        chunks and the register_chunks result disagree, a bound is negative,
        ``min`` is not below ``max``, or the window leaves nothing of the mosaic.
    FileNotFoundError
        If a listed chunk or a correction file is missing.
    """
    if general_config is None or not getattr(general_config, "out", None):
        raise ValueError("general_config.out (the output folder) is required")
    voxel_size = [float(v) * 2 ** layer for v in reversed(voxel_size)]
    for name, lo, hi in (("x", min_x, max_x), ("y", min_y, max_y)):
        if (lo is not None and lo < 0) or (hi is not None and hi < 0):
            raise ValueError(f"min_{name} and max_{name} must be >= 0")
        if lo is not None and hi is not None and hi <= lo:
            raise ValueError(f"max_{name} ({hi}) must be greater than min_{name} "
                             f"({lo})")

    def at_layer(value: Optional[int]) -> Optional[int]:
        return None if value is None else int(round(value / 2 ** layer))

    y_range = (at_layer(min_y), at_layer(max_y))
    x_range = (at_layer(min_x), at_layer(max_x))

    with open(yaml_path) as f:
        scan_parameters = yaml.safe_load(f)
    crops = get_crops(scan_parameters, camera_id, slice_number)
    flip = get_vertical_flip(scan_parameters, camera_id)
    shear_x = get_skew_shear(scan_parameters)
    channels = get_channel_names(scan_parameters, camera_id)
    for channel in channels:
        if channel not in crops:
            raise KeyError(f"No stitchingCrop for channel '{channel}'")
        if not os.path.exists(coords_path.format(channel=channel)):
            raise FileNotFoundError(
                f"No register_chunks result for channel '{channel}': "
                f"{coords_path.format(channel=channel)}")

    for channel in channels:
        if restart:
            ckpt = _checkpoint_path(os.path.join(general_config.out, channel))
            if os.path.exists(ckpt):
                os.remove(ckpt)
        coords = coords_path.format(channel=channel)
        tile_paths = _resolve_tile_paths(input_path, camera_id, coords,
                                         all_chunks)
        positions_y, positions_z, shear_z = _layout(
            coords, tile_paths, channel, dz_as_shear_z,
            per_chunk_registration, layer)
        stitch_channel(tile_paths, channel, crops[channel], flip, shear_x,
                       shear_z, positions_y, positions_z, corrections_dir,
                       os.path.join(general_config.out, channel),
                       voxel_size, zarr_config, per_chunk_corrections, layer,
                       y_range, x_range, memory_gb, parallel_chunks, bands,
                       parallel_bands, x_tolerance)
        if projections:
            write_projections(
                os.path.join(general_config.out, channel),
                projections_dir or os.path.join(general_config.out, "projections"),
                slice_number, channel, memory_gb)