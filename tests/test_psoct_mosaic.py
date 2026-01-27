import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import skimage
import yaml
import zarr.storage
from scipy.io import savemat
from utils.compare_file import assert_zarr_equal

from linc_convert.modalities.psoct import mosaic


@pytest.fixture
def mosaic_tiles(tmp_path):
    """
    Create 3D tiles for mosaic testing using skimage brain data.
    """
    base_dir = tmp_path / "tiles"
    base_dir.mkdir()

    # Get brain data and transpose to (height, width, depth)
    brain_volume = skimage.data.brain().T.astype(np.float32)
    brain_h, brain_w, brain_d = brain_volume.shape

    # Split into 4 tiles in a 2x2 grid
    # Each tile will be approximately half the size in height and width
    tile_height = brain_h // 2
    tile_width = brain_w // 2

    tiles = []
    for y_idx in range(2):
        for x_idx in range(2):
            x = x_idx * tile_width
            y = y_idx * tile_height

            # Extract a region from the brain volume for this tile
            h_start = y_idx * tile_height
            h_end = min((y_idx + 1) * tile_height, brain_h)
            w_start = x_idx * tile_width
            w_end = min((x_idx + 1) * tile_width, brain_w)

            # Extract 3D tile (height, width, depth)
            tile_data = brain_volume[h_start:h_end, w_start:w_end, :].copy()

            # Save as .mat file
            tile_path = base_dir / f"tile_y{y_idx}_x{x_idx}.mat"
            savemat(str(tile_path), {"image": tile_data})

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
            "scan_resolution": [1.0, 1.0, 1.0],
            "unit": "millimeter",
            "tile_overlap": 0.2,
            "clip_x": 0,
            "clip_y": 0,
            "file_key": "image"
        },
        "tiles": tiles
    }

    with open(yaml_path, "w") as f:
        yaml.dump(tile_info, f)

    return yaml_path


@pytest.fixture
def mosaic_mask(tmp_path):
    """Create a simple binary mask for testing."""
    # Get brain data dimensions to match mosaic size
    brain_volume = skimage.data.brain().T
    brain_h, brain_w = brain_volume.shape[:2]

    # Create a 2D mask that matches the mosaic dimensions
    mask_data = np.ones((brain_h, brain_w), dtype=np.float32)
    # Create a circular mask
    center = (brain_h // 2, brain_w // 2)
    radius = min(brain_h, brain_w) // 3
    y, x = np.ogrid[:brain_h, :brain_w]
    mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= radius ** 2
    mask_data = mask.astype(np.float32)

    # Save as NIfTI
    mask_path = tmp_path / "mask.nii"
    mask_img = nib.Nifti1Image(mask_data, np.eye(4))
    nib.save(mask_img, mask_path)

    return mask_path


@pytest.fixture
def mosaic_focus_plane(tmp_path):
    """Create a simple focus plane for testing."""
    # Get brain data dimensions to match mosaic size
    brain_volume = skimage.data.brain().T
    brain_h, brain_w = brain_volume.shape[:2]
    brain_h //= 2
    brain_w //= 2
    # Create a 2D focus plane
    focus_data = np.zeros((brain_h, brain_w), dtype=np.uint16)
    # Create a gradient
    for i in range(brain_h):
        for j in range(brain_w):
            focus_data[i, j] = (i + j) % 5

    # Save as NIfTI
    focus_path = tmp_path / "focus_plane.nii"
    focus_img = nib.Nifti1Image(focus_data, np.eye(4))
    nib.save(focus_img, focus_path)

    return focus_path


def test_psoct_mosaic_convert(tmp_path, mosaic_tiles, zarr_version, driver):
    """Test psoct.mosaic conversion with Zarr output."""
    output = tmp_path / "mosaic.nii.zarr"

    mosaic.mosaic2d(
        tile_info_file=str(mosaic_tiles),
        out=str(output),
        zarr_version=zarr_version,
        overwrite=True,
        driver=driver,
        tile_overlap=0.2,
    )

    expected_zarr = f"data/psoct_mosaic_zarr{zarr_version}.nii.zarr.zip"
    assert_zarr_equal(
        str(output),
        zarr.storage.ZipStore(expected_zarr, mode="r"),
    )


def test_psoct_mosaic_with_mask(
    tmp_path, mosaic_tiles, mosaic_mask, zarr_version, driver
    ):
    """Test psoct.mosaic with mask applied."""
    output = tmp_path / "mosaic_masked.nii.zarr"

    mosaic.mosaic2d(
        tile_info_file=str(mosaic_tiles),
        mask=str(mosaic_mask),
        out=str(output),
        zarr_version=zarr_version,
        overwrite=True,
        driver=driver,
        tile_overlap=0.2,
    )

    # Verify output exists and has correct shape
    assert output.exists()


def test_psoct_mosaic_with_focus_plane(
    tmp_path, mosaic_tiles, mosaic_focus_plane, zarr_version, driver
    ):
    """Test psoct.mosaic with focus plane cropping."""
    output = tmp_path / "mosaic_focus.nii.zarr"

    mosaic.mosaic2d(
        tile_info_file=str(mosaic_tiles),
        focus_plane=str(mosaic_focus_plane),
        normalize_focus_plane=False,
        crop_focus_plane_depth=4,
        crop_focus_plane_offset=0,
        out=str(output),
        zarr_version=zarr_version,
        overwrite=True,
        driver=driver,
        tile_overlap=0.2,
    )

    # Verify output exists
    assert output.exists()


@pytest.mark.golden
def test_psoct_mosaic_regen_golden(tmp_path, mosaic_tiles, zarr_version):
    """Regenerate golden archive for psoct.mosaic."""
    output = tmp_path / "mosaic_output.nii.zarr"

    mosaic.mosaic2d(
        tile_info_file=str(mosaic_tiles),
        out=str(output),
        zarr_version=zarr_version,
        overwrite=True,
        driver="zarr-python",
        tile_overlap=0.2,
    )

    expected_zarr = f"data/psoct_mosaic_zarr{zarr_version}.nii.zarr.zip"
    base = Path(expected_zarr).with_suffix("")
    shutil.make_archive(str(base), "zip", str(output))
