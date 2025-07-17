import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

from linc_convert.modalities.lsm import mosaic
from utils.compare_file import assert_zarr_equal
from utils.sample_data import generate_sample_data_variation


@pytest.fixture
def mosaic_tiff(tmp_path):
    root = Path(tmp_path)
    brain = np.array(generate_sample_data_variation(3))
    brain = brain.reshape(3, 2, 5, 4, 64, 256)

    for z in range(1, 6):
        for y in range(1, 5):
            folder = root / f"test_z{z}_y{y}"
            folder.mkdir(parents=True, exist_ok=True)
            for plane_index in range(1, 3):
                for c in range(1, 4):
                    out_path = folder / f"test_z{z}_y{y}_plane{plane_index}_c{c}.tiff"
                    image = brain[c - 1, plane_index - 1, z - 1, y - 1, :]
                    tifffile.imwrite(out_path, image)
    return root


def test_lsm_mosaic(tmp_path, mosaic_tiff, zarr_version, driver):
    """
    Convert multiple JP2 slices into a Zarr store and compare against golden.
    """
    expected_zarr = f"data/lsm_mosaic_zarr{zarr_version}.nii.zarr.zip"
    output = tmp_path / "mosaic.nii.zarr"
    mosaic.convert(
        str(mosaic_tiff),
        out=str(output),
        zarr_version=zarr_version,
        driver=driver
    )
    assert_zarr_equal(
        str(output),
        zarr.storage.ZipStore(expected_zarr, mode="r")
    )


@pytest.mark.golden
def test_lsm_mosaic_regen_golden(tmp_path, mosaic_tiff, zarr_version):
    expected_zarr = f"data/lsm_mosaic_zarr{zarr_version}.nii.zarr.zip"
    output = tmp_path / "mosaic.nii.zarr"
    mosaic.convert(
        str(mosaic_tiff),
        out=str(output),
        zarr_version=zarr_version,
    )
    base = Path(expected_zarr).with_suffix("")
    shutil.make_archive(str(base), "zip", str(output))
