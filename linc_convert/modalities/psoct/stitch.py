from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
from numpy.typing import ArrayLike


class TileInfo(dataclass):
    x: int
    y: int
    image: da.Array


class MosaicInfo(dataclass):
    tiles: List[TileInfo]
    full_shape: Tuple[int, int, int]
    blend_ramp: da.Array
    chunk_size: Tuple[int, int]
    circular_mean: bool


def _to_dask(arr: ArrayLike, chunks="auto") -> da.Array:
    """Convert array-like to dask array."""
    if isinstance(arr, da.Array):
        return arr
    return da.from_array(np.asarray(arr), chunks=chunks)


def _normalize_overlap(
    overlap: Union[int, Tuple[int, int]],
) -> Tuple[int, int]:
    """Normalize overlap parameter to (row_overlap, col_overlap) tuple."""
    if isinstance(overlap, tuple):
        if len(overlap) != 2:
            raise ValueError("overlap tuple must be (row_overlap, col_overlap).")
        row_overlap, col_overlap = overlap
    else:
        row_overlap = col_overlap = int(overlap)

    if row_overlap < 0 or col_overlap < 0:
        raise ValueError("Overlap must be non-negative.")
    return row_overlap, col_overlap


def _blending_ramp(
    shape: tuple(int),
    overlap: tuple(int),
) -> np.ndarray:
    """
    Create a separable 2D blending ramp similar to build_slice().
    """
    wx = np.ones(shape[0], dtype=np.float32)
    wy = np.ones(shape[1], dtype=np.float32)
    if overlap[0] > 0:
        wx[:overlap[0]] = np.linspace(0, 1, overlap[0], dtype=np.float32)
        wx[-overlap[0]:] = np.linspace(1, 0, overlap[0], dtype=np.float32)
    if overlap[1] > 0:
        wy[:overlap[1]] = np.linspace(0, 1, overlap[1], dtype=np.float32)
        wy[-overlap[1]:] = np.linspace(1, 0, overlap[1], dtype=np.float32)
    return np.outer(wx, wy)


def _combine_block(
    _, block_tiles, block_weights, circular_mean, *args, block_info=None, **kwargs
    ):
    chunk_id = tuple(block_info[None]['chunk-location'][:2])
    paints = block_tiles[chunk_id]
    weights = block_weights[chunk_id]
    shape = block_info[None]['chunk-shape']

    if not paints:
        return np.broadcast_to(np.zeros((), dtype=np.float32), shape)

    total_paint = da.sum(da.stack(paints, axis=0), axis=0)
    total_weight = da.sum(da.stack(weights, axis=0), axis=0)
    normalized = total_paint / np.broadcast_to(total_weight, total_paint.shape)
    if not circular_mean:
        return normalized
    return da.rad2deg(da.arctan2(normalized[..., 1], normalized[..., 0])) / 2


def stitch_tiles(
    tile_infos,
    full_shape,
    blend_ramp,
    chunk_size=None,
    circular_mean=False,
    **_
    ):
    """
    Chunk‐aligned padding + per‐block summation (build_slice_chunked_padding).
    """
    if not chunk_size:
        chunk_size = tile_infos[0].image.shape[:2]
    pw, ph = chunk_size
    no_chunk_dim = full_shape[2:]
    canvas = da.zeros(full_shape,
                      chunks=(pw, ph, *no_chunk_dim),
                      dtype=np.float32)
    weight = da.zeros(full_shape[:2],
                      chunks=(pw, ph),
                      dtype=np.float32)

    # collect per‐chunk pieces
    block_tiles = defaultdict(list)
    block_weights = defaultdict(list)

    for tile_info in tile_infos:
        x0, y0, t = tile_info.x, tile_info.y, tile_info.image
        x0c = x0 // pw
        y0c = y0 // ph
        x1c = (x0 + pw - 1) // pw
        y1c = (y0 + ph - 1) // ph

        # pad region covering those chunks
        x_start = x0c * pw
        y_start = y0c * ph
        x_end = (x1c + 1) * pw
        y_end = (y1c + 1) * ph

        if not circular_mean:
            block_canvas = da.zeros((x_end - x_start, y_end - y_start, *no_chunk_dim),
                                    chunks=(pw, ph, *no_chunk_dim),
                                    dtype=np.float32)
        else:
            block_canvas = da.zeros(
                (x_end - x_start, y_end - y_start, *no_chunk_dim, 2),
                chunks=(pw, ph, *no_chunk_dim, 2),
                dtype=np.float32)

        block_weight = da.zeros((x_end - x_start, y_end - y_start),
                                chunks=(pw, ph),
                                dtype=np.float32)

        # place tile into that big block
        xs = slice(x0 - x_start, x0 - x_start + pw)
        ys = slice(y0 - y_start, y0 - y_start + ph)
        if not circular_mean:
            block_canvas[xs, ys, ...] = t * blend_ramp
        else:
            rad = da.deg2rad(t) * 2
            block_canvas[xs, ys, ..., 0] = da.cos(rad)
            block_canvas[xs, ys, ..., 1] = da.sin(rad)

        block_weight[xs, ys] = blend_ramp

        # chop into per‐chunk pieces
        for cx in range(x0c, x1c + 1):
            for cy in range(y0c, y1c + 1):
                bid = (cx, cy)
                sub_x = slice((cx - x0c) * pw, (cx - x0c + 1) * pw)
                sub_y = slice((cy - y0c) * ph, (cy - y0c + 1) * ph)
                block_tiles[bid].append(block_canvas[sub_x, sub_y, ...])
                block_weights[bid].append(block_weight[sub_x, sub_y])

    canvas = da.map_blocks(_combine_block, canvas, block_tiles, block_weights,
                           circular_mean,
                           dtype=canvas.dtype,
                           chunks=(pw, ph, *no_chunk_dim))
    # crop canvas to get rid of excessive padded pixels
    canvas = canvas[:full_shape[0], :full_shape[1], ...]
    return canvas
