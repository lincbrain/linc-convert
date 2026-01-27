import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
import zarr.storage
from scipy.io import savemat
from utils.compare_file import assert_zarr_equal

from linc_convert.modalities.psoct import mosaic_complex


@pytest.fixture
def mosaic_complex_tiles(tmp_path):
    """
    Create synthetic complex 3D tiles for mosaic_complex testing.
    
    Each tile has shape (4 * raw_tile_width, tile_height, tile_width) where
    the first dimension is split into 4 parts: j1r, j1i, j2r, j2i.
    """
    base_dir = tmp_path / "tiles"
    base_dir.mkdir()

    # Use small dimensions for fast tests
    raw_tile_width = 16
    tile_height = 32
    tile_width = 32
    depth = 20

    # Create 4 tiles in a 2x2 grid
    tiles = []
    for y_idx in range(2):
        for x_idx in range(2):
            x = x_idx * tile_width
            y = y_idx * tile_height

            # Create synthetic complex data
            # Shape: (4 * raw_tile_width, tile_height, tile_width)
            # Split into: j1r, j1i, j2r, j2i
            tile_data = np.zeros((4 * raw_tile_width, tile_height, depth),
                                 dtype=np.float32)

            # Fill with synthetic data that varies by position
            for i in range(4 * raw_tile_width):
                for j in range(tile_height):
                    for k in range(depth):
                        # Create some variation based on position
                        val = (i + j + k) % 100 + 1.0
                        tile_data[i, j, k] = val

            # Save as .mat file
            tile_path = base_dir / f"tile_y{y_idx}_x{x_idx}.mat"
            savemat(str(tile_path), {"complex3d": tile_data})

            tiles.append({
                "x": x,
                "y": y,
                "filepath": f"tile_y{y_idx}_x{x_idx}.mat"
            })

    # Create YAML file
    yaml_path = tmp_path / "tile_info.yaml"
    tile_info = {
        "metadata": {
            "base_dir": str(base_dir),
            "tile_width": tile_width,
            "tile_height": tile_height,
            "depth": depth,
            "raw_tile_width": raw_tile_width,
            "scan_resolution": [1.0, 1.0, 1.0],
            "unit": "millimeter",
            "tile_overlap": 0.2,
            "flip_z": False,
            "clip_x": 0,
            "clip_y": 0,
            "file_key": "complex3d"
        },
        "tiles": tiles
    }

    with open(yaml_path, "w") as f:
        yaml.dump(tile_info, f)

    return yaml_path


def test_psoct_mosaic_complex_convert(
    tmp_path, mosaic_complex_tiles, zarr_version, driver
    ):
    """Test psoct.mosaic_complex conversion."""
    dbi_output = tmp_path / "dbi.nii.zarr"
    o3d_output = tmp_path / "o3d.nii.zarr"
    r3d_output = tmp_path / "r3d.nii.zarr"

    mosaic_complex.mosaic_complex(
        tile_info_file=str(mosaic_complex_tiles),
        dbi_output=str(dbi_output),
        o3d_output=str(o3d_output),
        r3d_output=str(r3d_output),
        zarr_version=zarr_version,
        overwrite=True,
        driver=driver,
    )

    # Compare dBI output as primary regression target
    expected_zarr = f"data/psoct_mosaic_complex_dbi_zarr{zarr_version}.nii.zarr.zip"
    assert_zarr_equal(
        str(dbi_output),
        zarr.storage.ZipStore(expected_zarr, mode="r"),
    )


@pytest.mark.golden
def test_psoct_mosaic_complex_regen_golden(
    tmp_path, mosaic_complex_tiles, zarr_version
    ):
    """Regenerate golden archives for psoct.mosaic_complex."""
    dbi_output = tmp_path / "dbi_output.nii.zarr"
    o3d_output = tmp_path / "o3d_output.nii.zarr"
    r3d_output = tmp_path / "r3d_output.nii.zarr"

    mosaic_complex.mosaic_complex(
        tile_info_file=str(mosaic_complex_tiles),
        dbi_output=str(dbi_output),
        o3d_output=str(o3d_output),
        r3d_output=str(r3d_output),
        zarr_version=zarr_version,
        overwrite=True,
        driver="zarr-python",
    )

    # Regenerate all three golden archives
    for output, name in [
        (dbi_output, "psoct_mosaic_complex_dbi"),
        (o3d_output, "psoct_mosaic_complex_o3d"),
        (r3d_output, "psoct_mosaic_complex_r3d"),
    ]:
        expected_zarr = f"data/{name}_zarr{zarr_version}.nii.zarr.zip"
        base = Path(expected_zarr).with_suffix("")
        shutil.make_archive(str(base), "zip", str(output))
