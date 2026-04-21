"""
Unit tests for linc_convert.utils.stitch module.
"""

import dask.array as da
import numpy as np
import pytest

from linc_convert.utils.stitch import (
    MosaicInfo,
    TileInfo,
    _normalize_tile_overlap,
    stitch_tiles,
)


class TestNormalizeTileOverlap:
    """Test _normalize_tile_overlap function."""

    def test_single_float(self):
        """Test single float input (percentile)."""
        tile_width, tile_height = 100, 200
        overlap = 0.2
        x_overlap, y_overlap = _normalize_tile_overlap(overlap, tile_width, tile_height)
        assert x_overlap == 20  # 20% of 100
        assert y_overlap == 40  # 20% of 200

    def test_single_int(self):
        """Test single int input (pixels)."""
        tile_width, tile_height = 100, 200
        overlap = 10
        x_overlap, y_overlap = _normalize_tile_overlap(overlap, tile_width, tile_height)
        assert x_overlap == 10
        assert y_overlap == 10

    def test_tuple_float(self):
        """Test tuple of floats (percentiles)."""
        tile_width, tile_height = 100, 200
        overlap = (0.1, 0.2)
        x_overlap, y_overlap = _normalize_tile_overlap(overlap, tile_width, tile_height)
        assert x_overlap == 10  # 10% of 100
        assert y_overlap == 40  # 20% of 200

    def test_tuple_int(self):
        """Test tuple of ints (pixels)."""
        tile_width, tile_height = 100, 200
        overlap = (5, 10)
        x_overlap, y_overlap = _normalize_tile_overlap(overlap, tile_width, tile_height)
        assert x_overlap == 5
        assert y_overlap == 10

    def test_invalid_float_range(self):
        """Test invalid float range raises ValueError."""
        tile_width, tile_height = 100, 200
        with pytest.raises(ValueError, match="Float tile_overlap must be in range"):
            _normalize_tile_overlap(1.5, tile_width, tile_height)
        with pytest.raises(ValueError, match="Float tile_overlap must be in range"):
            _normalize_tile_overlap(0.0, tile_width, tile_height)
        with pytest.raises(ValueError, match="Float tile_overlap must be in range"):
            _normalize_tile_overlap(-0.1, tile_width, tile_height)

    def test_invalid_tuple_length(self):
        """Test invalid tuple length raises ValueError."""
        tile_width, tile_height = 100, 200
        with pytest.raises(ValueError, match="tile_overlap tuple must have 2 elements"):
            _normalize_tile_overlap((1, 2, 3), tile_width, tile_height)

    def test_negative_overlap(self):
        """Test negative overlap raises ValueError."""
        tile_width, tile_height = 100, 200
        with pytest.raises(ValueError, match="Overlap must be non-negative"):
            _normalize_tile_overlap(-5, tile_width, tile_height)
        with pytest.raises(ValueError, match="Overlap must be non-negative"):
            _normalize_tile_overlap((-5, 10), tile_width, tile_height)


class TestMosaicInfo:
    """Test MosaicInfo class."""

    def test_from_tiles_2d(self):
        """Test creating 2D MosaicInfo from tiles."""
        # Create two small tiles
        tile1 = TileInfo(
            x=0, y=0, image=da.ones((10, 10), chunks=(5, 5), dtype=np.float32)
        )
        tile2 = TileInfo(
            x=5, y=5, image=da.ones((10, 10), chunks=(5, 5), dtype=np.float32) * 2
        )

        mosaic = MosaicInfo.from_tiles(
            tiles=[tile1, tile2],
            depth=None,  # 2D mosaic
            tile_overlap=0.2,
        )

        # Check full_shape: max x + tile_width, max y + tile_height
        assert mosaic.full_shape == (15, 15)  # (15, 15) for 2D
        assert mosaic.chunk_size == (10, 10)
        assert mosaic.circular_mean is False

    def test_from_tiles_3d(self):
        """Test creating 3D MosaicInfo from tiles."""
        # Create two small 3D tiles
        tile1 = TileInfo(
            x=0, y=0, image=da.ones((10, 10, 5), chunks=(5, 5, 5), dtype=np.float32)
        )
        tile2 = TileInfo(
            x=5, y=5, image=da.ones((10, 10, 5), chunks=(5, 5, 5), dtype=np.float32) * 2
        )

        mosaic = MosaicInfo.from_tiles(
            tiles=[tile1, tile2],
            depth=5,
            tile_overlap=0.2,
        )

        # Check full_shape: (width, height, depth)
        assert mosaic.full_shape == (15, 15, 5)
        assert mosaic.chunk_size == (10, 10)

    def test_normalize_tile_coordinates(self):
        """Test normalize_tile_coordinates method."""
        # Create tiles with non-zero minimum coordinates
        tile1 = TileInfo(
            x=10, y=20, image=da.ones((10, 10), chunks=(5, 5), dtype=np.float32)
        )
        tile2 = TileInfo(
            x=15, y=25, image=da.ones((10, 10), chunks=(5, 5), dtype=np.float32)
        )

        mosaic = MosaicInfo.from_tiles(
            tiles=[tile1, tile2],
            depth=None,
            tile_overlap=0.2,
        )

        # Before normalization
        assert tile1.x == 10
        assert tile1.y == 20
        assert tile2.x == 15
        assert tile2.y == 25
        assert mosaic.full_shape == (25, 35)  # max(15+10, 10+10), max(25+10, 20+10)

        # Normalize
        mosaic.normalize_tile_coordinates()

        # After normalization
        assert tile1.x == 0
        assert tile1.y == 0
        assert tile2.x == 5
        assert tile2.y == 5
        # full_shape should be adjusted
        assert mosaic.full_shape == (15, 15)  # (25-10, 35-20)

    def test_stitch_2d_simple(self):
        """Test stitching 2D tiles with simple overlap."""
        # Create two overlapping tiles
        tile1 = TileInfo(
            x=0, y=0, image=da.ones((10, 10), chunks=(5, 5), dtype=np.float32) * 10
        )
        tile2 = TileInfo(
            x=5,
            y=5,  # 5 pixel overlap in both dimensions
            image=da.ones((10, 10), chunks=(5, 5), dtype=np.float32) * 20,
        )

        mosaic = MosaicInfo.from_tiles(
            tiles=[tile1, tile2],
            depth=None,
            tile_overlap=5,  # 5 pixels overlap
        )

        result = mosaic.stitch()
        result_computed = result.compute()

        # Check shape
        assert result_computed.shape == (15, 15)

        # Check non-overlapping regions
        # Top-left corner should be from tile1
        print(np.array(result_computed[0:5, 0:5]))
        assert np.allclose(result_computed[0:5, 0:5], 10.0)
        # Bottom-right corner should be from tile2
        assert np.allclose(result_computed[10:15, 10:15], 20.0)

        # Overlapping region should be blended (weighted average)
        # The overlap region should have values between 10 and 20
        overlap_region = result_computed[5:10, 5:10]
        assert np.all(overlap_region >= 10.0)
        assert np.all(overlap_region <= 20.0)

    def test_stitch_2d_no_overlap(self):
        """Test stitching 2D tiles with no overlap."""
        tile1 = TileInfo(
            x=0, y=0, image=da.ones((10, 10), chunks=(5, 5), dtype=np.float32) * 10
        )
        tile2 = TileInfo(
            x=10,
            y=10,  # No overlap
            image=da.ones((10, 10), chunks=(5, 5), dtype=np.float32) * 20,
        )

        mosaic = MosaicInfo.from_tiles(
            tiles=[tile1, tile2],
            depth=None,
            tile_overlap=0,
        )

        result = mosaic.stitch()
        result_computed = result.compute()

        # Check shape
        assert result_computed.shape == (20, 20)

        # Check non-overlapping regions
        assert np.allclose(result_computed[0:10, 0:10], 10.0)
        assert np.allclose(result_computed[10:20, 10:20], 20.0)

    def test_stitch_3d(self):
        """Test stitching 3D tiles."""
        tile1 = TileInfo(
            x=0,
            y=0,
            image=da.ones((10, 10, 5), chunks=(5, 5, 5), dtype=np.float32) * 10,
        )
        tile2 = TileInfo(
            x=5,
            y=5,
            image=da.ones((10, 10, 5), chunks=(5, 5, 5), dtype=np.float32) * 20,
        )

        mosaic = MosaicInfo.from_tiles(
            tiles=[tile1, tile2],
            depth=5,
            tile_overlap=5,
        )

        result = mosaic.stitch()
        result_computed = result.compute()

        # Check shape: (width, height, depth)
        assert result_computed.shape == (15, 15, 5)

        # Check that depth dimension is preserved
        assert result_computed.shape[2] == 5

    def test_stitch_circular_mean(self):
        """Test stitching with circular_mean=True."""
        # Create tiles with angle values (in degrees)
        # Use simple angles for easy verification
        tile1 = TileInfo(
            x=0,
            y=0,
            image=da.full((10, 10), chunks=(5, 5), dtype=np.float32, fill_value=0.0),
            # 0 degrees
        )
        tile2 = TileInfo(
            x=5,
            y=5,
            image=da.full((10, 10), chunks=(5, 5), dtype=np.float32, fill_value=180.0),
            # 180 degrees
        )

        mosaic = MosaicInfo.from_tiles(
            tiles=[tile1, tile2],
            depth=None,
            tile_overlap=5,
            circular_mean=True,
        )

        result = mosaic.stitch()
        result_computed = result.compute()

        # Check shape
        assert result_computed.shape == (15, 15)

        # In the overlap region, circular mean of 0° and 180° should be around 90° or
        # 270°
        # (depending on implementation, but should be consistent)
        overlap_region = result_computed[5:10, 5:10]
        # The circular mean should produce a value that makes sense
        # For 0° and 180°, the mean could be 90° or 270° (both are valid)
        # We just check it's not NaN and is in a reasonable range
        assert not np.any(np.isnan(overlap_region))
        assert np.all(overlap_region >= 0) and np.all(overlap_region < 360)


class TestStitchTiles:
    """Test backward compatibility stitch_tiles function."""

    def test_stitch_tiles_backward_compat(self):
        """Test that stitch_tiles produces same result as MosaicInfo.stitch()."""
        tile1 = TileInfo(
            x=0, y=0, image=da.ones((10, 10), chunks=(5, 5), dtype=np.float32) * 10
        )
        tile2 = TileInfo(
            x=5, y=5, image=da.ones((10, 10), chunks=(5, 5), dtype=np.float32) * 20
        )

        # Create MosaicInfo and stitch
        mosaic = MosaicInfo.from_tiles(
            tiles=[tile1, tile2],
            depth=None,
            tile_overlap=5,
        )
        result1 = mosaic.stitch()

        # Use stitch_tiles function
        result2 = stitch_tiles(
            tile_infos=[tile1, tile2],
            full_shape=mosaic.full_shape,
            blend_ramp=mosaic.blend_ramp,
            chunk_size=mosaic.chunk_size,
            circular_mean=False,
        )

        # Results should be the same
        np.testing.assert_array_almost_equal(
            result1.compute(), result2.compute(), decimal=5
        )

    def test_stitch_tiles_empty(self):
        """Test stitch_tiles with empty tile list raises ValueError."""
        with pytest.raises(ValueError, match="No tiles provided"):
            stitch_tiles(
                tile_infos=[],
                full_shape=(10, 10),
                blend_ramp=da.ones((10, 10), dtype=np.float32),
                chunk_size=(5, 5),
            )
