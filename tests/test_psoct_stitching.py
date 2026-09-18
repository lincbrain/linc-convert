"""Numerical regression tests for PS-OCT tile placement and blending."""

import dask.array as da
import numpy as np
import pytest

from linc_convert.utils.stitch import MosaicInfo, TileInfo, _normalize_tile_overlap


@pytest.mark.parametrize("depth", [None, 3])
def test_overlap_preserves_coverage_and_blends_values(depth):
    shape = (10, 8) if depth is None else (10, 8, depth)
    tiles = [
        TileInfo(x=x, y=20, image=da.full(shape, value, chunks=shape))
        for x, value in [(100, 20.0), (108, 40.0)]
    ]
    result = np.asarray(
        MosaicInfo.from_tiles(tiles, tile_overlap=0.2).stitch().compute()
    )
    assert result.shape == ((18, 8) if depth is None else (18, 8, depth))
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[:8], 20)
    np.testing.assert_allclose(result[8], 80 / 3, rtol=1e-6)
    np.testing.assert_allclose(result[9], 100 / 3, rtol=1e-6)
    np.testing.assert_allclose(result[10:], 40)


def test_circular_mean_respects_orientation_periodicity():
    tiles = [
        TileInfo(x=0, y=0, image=da.full((4, 4), angle, chunks=(4, 4)))
        for angle in [89.0, -89.0]
    ]
    result = np.asarray(
        MosaicInfo.from_tiles(tiles, tile_overlap=0, circular_mean=True)
        .stitch()
        .compute()
    )
    np.testing.assert_allclose(np.abs(result), 90, atol=1e-5)


@pytest.mark.parametrize(
    "overlap, expected", [(0.2, (20, 40)), (10, (10, 10)), ((0.1, 0.2), (10, 40))]
)
def test_overlap_units(overlap, expected):
    assert _normalize_tile_overlap(overlap, 100, 200) == expected


def test_empty_tiles_rejected():
    with pytest.raises(ValueError, match="No tiles provided"):
        MosaicInfo.from_tiles([], tile_overlap=0.1)
