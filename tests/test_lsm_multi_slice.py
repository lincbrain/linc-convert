import shutil
from pathlib import Path

import pytest
import skimage
import tifffile
import zarr
from utils.compare_file import assert_zarr_equal

from linc_convert.modalities.lsm import multi_slice


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


def test_lsm_multi_slice(tmp_path, multi_slice_tiff, zarr_version, driver):
    expected_zarr = f"data/lsm_multi_slice_zarr{zarr_version}.nii.zarr.zip"
    output = tmp_path / "multi_slice.nii.zarr"
    multi_slice.convert(
            str(multi_slice_tiff),
            overlap=0,
            out=str(output),
            zarr_version=zarr_version,
            driver=driver,
    )
    assert_zarr_equal(str(output), zarr.storage.ZipStore(expected_zarr, mode="r"))


@pytest.mark.golden
def test_lsm_multi_slice_regen_golden(tmp_path, multi_slice_tiff, zarr_version):
    expected_zarr = f"data/lsm_multi_slice_zarr{zarr_version}.nii.zarr.zip"
    output = tmp_path / "multi_slice.nii.zarr"
    multi_slice.convert(
            str(multi_slice_tiff),
            overlap=0,
            out=str(output),
            zarr_version=zarr_version,
    )
    base = Path(expected_zarr).with_suffix("")
    shutil.make_archive(str(base), "zip", str(output))
