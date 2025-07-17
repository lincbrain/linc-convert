import shutil
from pathlib import Path

import numpy as np
import pytest
import skimage
import tifffile
import zarr

from linc_convert.modalities.lsm import multi_slice
from utils.compare_file import assert_zarr_equal
from utils.sample_data import generate_sample_data_variation


@pytest.fixture
def multi_slice_tiff(tmp_path):
    root = Path(tmp_path)
    brain = skimage.data.brain()
    brain = brain.reshape(10, 4, 64, 256)
    for z in range(1, 11):
        for y in range(1, 4):
            out_path = root / f"test_y{y:02d}_z{z:02d}.tiff"
            image = brain[z - y, y - 1][None, ...]
            tifffile.imwrite(out_path, image)
    return root


@pytest.mark.parametrize(
    "zarr_version, expected_zarr",
    [
        (2, "data/lsm_multi_slice_zarr2.nii.zarr.zip"),
        (3, "data/lsm_multi_slice_zarr3.nii.zarr.zip"),
    ],
)
def test_lsm_multi_slice(tmp_path, multi_slice_tiff, zarr_version, expected_zarr,
                         driver):
    """
    Convert multiple JP2 slices into a Zarr store and compare against golden.
    """
    output = tmp_path / "multi_slice.zarr"
    multi_slice.convert(
        multi_slice_tiff,
        overlap=0,
        out=str(output),
        zarr_version=zarr_version,
        driver=driver
    )
    assert_zarr_equal(
        str(output),
        zarr.storage.ZipStore(expected_zarr, mode="r")
    )


@pytest.mark.golden
@pytest.mark.parametrize(
    ("zarr_version", "expected_zarr"),
    [
        (2, "data/lsm_multi_slice_zarr2.nii.zarr.zip"),
        (3, "data/lsm_multi_slice_zarr3.nii.zarr.zip"),
    ],
)
def test_lsm_multi_slice_regen_golden(tmp_path, multi_slice_tiff, zarr_version,
                                      expected_zarr):
    output = tmp_path / "single_slice.zarr"
    multi_slice.convert(
        str(multi_slice_tiff),
        overlap=0,
        out=str(output),
        zarr_version=zarr_version,
    )
    base = Path(expected_zarr).with_suffix("")
    shutil.make_archive(str(base), "zip", str(output))
