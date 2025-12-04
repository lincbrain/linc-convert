from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np
from numpy.typing import ArrayLike


@dataclass
class TileInfo:
    x: int
    y: int
    image: da.Array


@dataclass
class MosaicInfo:
    """Container for mosaic information with integrated blending and stitching."""
    
    tiles: List[TileInfo]
    full_shape: Tuple[int, ...]  # Can be 2D (width, height) or 3D (width, height, depth)
    blend_ramp: da.Array
    chunk_size: Tuple[int, int]
    circular_mean: bool = False
    
    @classmethod
    def from_tiles(
        cls,
        tiles: List[TileInfo],
        depth: Optional[int] = None,
        chunk_size: Optional[Tuple[int, int]] = None,
        circular_mean: bool = False,
        tile_overlap: Union[float, int, Tuple[float, float], Tuple[int, int], Literal["auto"]] = "auto",
    ) -> "MosaicInfo":
        """
        Create MosaicInfo from tiles, extracting dimensions and coordinates automatically.
        
        Parameters
        ----------
        tiles : List[TileInfo]
            List of tile information with coordinates and images.
        depth : Optional[int]
            Depth dimension for 3D mosaics. If None, creates 2D mosaic.
        chunk_size : Optional[Tuple[int, int]]
            Chunk size for dask arrays. If None, uses tile dimensions.
        circular_mean : bool
            Whether to use circular mean for blending.
        tile_overlap : Union[float, int, Tuple[float, float], Tuple[int, int], Literal["auto"]]
            Overlap specification:
            - float in (0, 1): percentile of tile size (e.g., 0.2 = 20% overlap on each side)
            - int: number of pixels overlap on each side
            - Tuple[float, float]: percentiles for (x, y) dimensions
            - Tuple[int, int]: pixel counts for (x, y) dimensions
            - "auto": compute maximum overlap from coordinates (default)
        
        Returns
        -------
        MosaicInfo
            Configured MosaicInfo instance.
        """
        if not tiles:
            raise ValueError("No tiles provided")
        
        # Extract tile dimensions from first tile
        first_tile = tiles[0]
        tile_height, tile_width = first_tile.image.shape[:2]
        
        # Extract depth from first tile if 3D
        if depth is None and len(first_tile.image.shape) == 3:
            depth = first_tile.image.shape[2]
        
        # Extract coordinates from tiles
        x_coords = np.array([tile.x for tile in tiles])
        y_coords = np.array([tile.y for tile in tiles])
        
        # Compute full mosaic dimensions
        full_width = int(np.nanmax(x_coords) + tile_width)
        full_height = int(np.nanmax(y_coords) + tile_height)
        
        if depth is not None:
            full_shape = (full_width, full_height, depth)
        else:
            full_shape = (full_width, full_height)
        
        # Normalize tile_overlap to pixels
        x_overlap, y_overlap = _normalize_tile_overlap(
            tile_overlap, tile_width, tile_height, x_coords, y_coords
        )
        
        # Compute blending ramp with explicit overlap
        blend_ramp = cls._compute_blending_ramp(
            tile_width, tile_height, x_overlap, y_overlap
        )
        
        if chunk_size is None:
            chunk_size = (tile_width, tile_height)
        
        return cls(
            tiles=tiles,
            full_shape=full_shape,
            blend_ramp=blend_ramp,
            chunk_size=chunk_size,
            circular_mean=circular_mean,
        )
    
    @staticmethod
    def _compute_blending_ramp(
        tile_width: int,
        tile_height: int,
        x_overlap: int,
        y_overlap: int,
    ) -> da.Array:
        """
        Compute blending ramp for tile stitching with explicit overlap values.
        
        Parameters
        ----------
        tile_width : int
            Width of each tile.
        tile_height : int
            Height of each tile.
        x_overlap : int
            Number of overlapping pixels in x dimension (on each side).
        y_overlap : int
            Number of overlapping pixels in y dimension (on each side).
        
        Returns
        -------
        da.Array
            Blending ramp as a dask array.
        """
        # Create blending ramp
        wx = np.ones(tile_height, dtype=np.float32)
        wy = np.ones(tile_width, dtype=np.float32)
        
        if x_overlap > 0:
            wx[:x_overlap] = np.linspace(0, 1, x_overlap, dtype=np.float32)
            wx[-x_overlap:] = np.linspace(1, 0, x_overlap, dtype=np.float32)
        
        if y_overlap > 0:
            wy[:y_overlap] = np.linspace(0, 1, y_overlap, dtype=np.float32)
            wy[-y_overlap:] = np.linspace(1, 0, y_overlap, dtype=np.float32)
        
        ramp = np.outer(wx, wy)
        return da.from_array(ramp, chunks="auto")
    
    def stitch(self) -> da.Array:
        """
        Stitch tiles into a mosaic using lazy dask operations.
        
        Returns
        -------
        da.Array
            Stitched mosaic as a dask array. Shape is (width, height, ...) for 3D
            or (width, height) for 2D.
        """
        if not self.tiles:
            raise ValueError("No tiles to stitch")
        
        pw, ph = self.chunk_size
        no_chunk_dim = self.full_shape[2:]  # Empty for 2D, (depth,) for 3D
        
        # Create canvas with appropriate shape
        canvas = da.zeros(
            self.full_shape,
            chunks=(pw, ph, *no_chunk_dim),
            dtype=np.float32
        )
        weight = da.zeros(
            self.full_shape[:2],
            chunks=(pw, ph),
            dtype=np.float32
        )
        
        # Collect per-chunk pieces
        block_tiles = defaultdict(list)
        block_weights = defaultdict(list)
        
        for tile_info in self.tiles:
            x0, y0, t = tile_info.x, tile_info.y, tile_info.image
            blend_ramp = self.blend_ramp
            
            # Determine which chunks this tile falls into
            x0c = x0 // pw
            y0c = y0 // ph
            x1c = (x0 + pw - 1) // pw
            y1c = (y0 + ph - 1) // ph
            
            # Pad region covering those chunks
            x_start = x0c * pw
            y_start = y0c * ph
            x_end = (x1c + 1) * pw
            y_end = (y1c + 1) * ph
            
            # Create block canvas with appropriate shape
            if not self.circular_mean:
                block_canvas = da.zeros(
                    (x_end - x_start, y_end - y_start, *no_chunk_dim),
                    chunks=(pw, ph, *no_chunk_dim),
                    dtype=np.float32
                )
            else:
                block_canvas = da.zeros(
                    (x_end - x_start, y_end - y_start, *no_chunk_dim, 2),
                    chunks=(pw, ph, *no_chunk_dim, 2),
                    dtype=np.float32
                )
            
            block_weight = da.zeros(
                (x_end - x_start, y_end - y_start),
                chunks=(pw, ph),
                dtype=np.float32
            )
            
            # Place tile into that big block
            xs = slice(x0 - x_start, x0 - x_start + pw)
            ys = slice(y0 - y_start, y0 - y_start + ph)
            
            # Apply blending ramp - handle both 2D and 3D
            if not self.circular_mean:
                # For 2D: t * blend_ramp
                # For 3D: t * blend_ramp[..., None]
                if len(no_chunk_dim) == 0:  # 2D
                    weighted_tile = t * blend_ramp
                else:  # 3D
                    weighted_tile = t * blend_ramp[:, :, None]
                block_canvas[xs, ys, ...] = weighted_tile
            else:
                # Circular mean: convert to sin/cos representation
                rad = da.deg2rad(t) * 2
                block_canvas[xs, ys, ..., 0] = da.cos(rad)
                block_canvas[xs, ys, ..., 1] = da.sin(rad)
            
            block_weight[xs, ys] = blend_ramp
            
            # Chop into per-chunk pieces
            for cx in range(x0c, x1c + 1):
                for cy in range(y0c, y1c + 1):
                    bid = (cx, cy)
                    sub_x = slice((cx - x0c) * pw, (cx - x0c + 1) * pw)
                    sub_y = slice((cy - y0c) * ph, (cy - y0c + 1) * ph)
                    block_tiles[bid].append(block_canvas[sub_x, sub_y, ...])
                    block_weights[bid].append(block_weight[sub_x, sub_y])
        
        # Combine blocks using map_blocks
        canvas = da.map_blocks(
            _combine_block,
            canvas,
            block_tiles,
            block_weights,
            self.circular_mean,
            dtype=canvas.dtype,
            chunks=(pw, ph, *no_chunk_dim)
        )
        
        # Crop canvas to get rid of excessive padded pixels
        canvas = canvas[:self.full_shape[0], :self.full_shape[1], ...]
        return canvas


def _combine_block(
    _, block_tiles, block_weights, circular_mean, *args, block_info=None, **kwargs
):
    """Combine overlapping tile blocks with weighted averaging."""
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


def _to_dask(arr: ArrayLike, chunks="auto") -> da.Array:
    """Convert array-like to dask array."""
    if isinstance(arr, da.Array):
        return arr
    return da.from_array(np.asarray(arr), chunks=chunks)


def _normalize_tile_overlap(
    tile_overlap: Union[float, int, Tuple[float, float], Tuple[int, int], Literal["auto"]],
    tile_width: int,
    tile_height: int,
    x_coords: Optional[np.ndarray] = None,
    y_coords: Optional[np.ndarray] = None,
) -> Tuple[int, int]:
    """
    Normalize tile_overlap parameter to (x_overlap, y_overlap) in pixels.
    
    Parameters
    ----------
    tile_overlap : Union[float, int, Tuple[float, float], Tuple[int, int], Literal["auto"]]
        Overlap specification:
        - float in (0, 1): percentile of tile size (e.g., 0.2 = 20% overlap on each side)
        - int: number of pixels overlap on each side
        - Tuple[float, float]: percentiles for (x, y) dimensions
        - Tuple[int, int]: pixel counts for (x, y) dimensions
        - "auto": compute maximum overlap from coordinates
    tile_width : int
        Width of each tile.
    tile_height : int
        Height of each tile.
    x_coords : Optional[np.ndarray]
        X coordinates for auto computation.
    y_coords : Optional[np.ndarray]
        Y coordinates for auto computation.
    
    Returns
    -------
    Tuple[int, int]
        (x_overlap, y_overlap) in pixels.
    """
    if tile_overlap == "auto":
        if x_coords is None or y_coords is None:
            raise ValueError("x_coords and y_coords required for auto tile_overlap")
        return _compute_auto_overlap(tile_width, tile_height, x_coords, y_coords)
    
    # Handle tuple case
    if isinstance(tile_overlap, tuple):
        if len(tile_overlap) != 2:
            raise ValueError("tile_overlap tuple must have 2 elements (x_overlap, y_overlap)")
        x_overlap_val, y_overlap_val = tile_overlap
    else:
        x_overlap_val = y_overlap_val = tile_overlap
    
    # Convert to pixels if float (percentile)
    if isinstance(x_overlap_val, float):
        if not (0 < x_overlap_val < 1):
            raise ValueError(f"Float tile_overlap must be in range (0, 1), got {x_overlap_val}")
        x_overlap = int(tile_width * x_overlap_val)
    else:
        x_overlap = int(x_overlap_val)
    
    if isinstance(y_overlap_val, float):
        if not (0 < y_overlap_val < 1):
            raise ValueError(f"Float tile_overlap must be in range (0, 1), got {y_overlap_val}")
        y_overlap = int(tile_height * y_overlap_val)
    else:
        y_overlap = int(y_overlap_val)
    
    if x_overlap < 0 or y_overlap < 0:
        raise ValueError("Overlap must be non-negative.")
    
    return x_overlap, y_overlap


def _compute_auto_overlap(
    tile_width: int, tile_height: int, x_coords: np.ndarray, y_coords: np.ndarray
) -> Tuple[int, int]:
    """
    Compute maximum overlap from tile coordinates.
    
    Returns the maximum number of overlapping pixels in each dimension.
    """
    if len(x_coords) == 0 or len(y_coords) == 0:
        return 0, 0
    
    x_coords_flat = x_coords[~np.isnan(x_coords)]
    y_coords_flat = y_coords[~np.isnan(y_coords)]
    
    x_overlap = 0
    y_overlap = 0
    
    if len(x_coords_flat) > 1:
        x_spacing = np.min(np.diff(np.sort(np.unique(x_coords_flat))))
        x_overlap = max(0, tile_width - int(x_spacing)) if x_spacing < tile_width else 0
    
    if len(y_coords_flat) > 1:
        y_spacing = np.min(np.diff(np.sort(np.unique(y_coords_flat))))
        y_overlap = max(0, tile_height - int(y_spacing)) if y_spacing < tile_height else 0
    
    return x_overlap, y_overlap


# Backward compatibility: keep stitch_tiles as a function that uses MosaicInfo
def stitch_tiles(
    tile_infos: List[TileInfo],
    full_shape: Tuple[int, ...],
    blend_ramp: Union[np.ndarray, da.Array],
    chunk_size: Optional[Tuple[int, int]] = None,
    circular_mean: bool = False,
    **_
) -> da.Array:
    """
    Stitch tiles into a mosaic (backward compatibility wrapper).
    
    This function is kept for backward compatibility. New code should use
    MosaicInfo.stitch() directly.
    """
    if not tile_infos:
        raise ValueError("No tiles provided")
    
    if chunk_size is None:
        chunk_size = tile_infos[0].image.shape[:2]
    
    mosaic_info = MosaicInfo(
        tiles=tile_infos,
        full_shape=full_shape,
        blend_ramp=blend_ramp,
        chunk_size=chunk_size,
        circular_mean=circular_mean,
    )
    
    return mosaic_info.stitch()
