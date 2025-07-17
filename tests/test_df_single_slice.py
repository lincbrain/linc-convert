import shutil
from pathlib import Path

import glymur
import pytest
import skimage
import zarr

from linc_convert.modalities.df import single_slice
from utils.compare_file import assert_zarr_equal


@pytest.fixture
def single_slice_jp2(tmp_path):
    path = tmp_path / "slice.jp2"
    image = skimage.data.brain().T[..., :3]
    glymur.Jp2k(
        path,
        data=image,
    )
    return path


def test_single_slice_df(tmp_path, single_slice_jp2, zarr_version, driver):
    expected_zarr = f"data/df_single_slice_zarr{zarr_version}.nii.zarr.zip"
    output = tmp_path / "single_slice.zarr"
    single_slice.convert(
        str(single_slice_jp2),
        out=str(output),
        zarr_version=zarr_version,
        driver=driver
    )
    assert_zarr_equal(
        str(output),
        zarr.storage.ZipStore(expected_zarr, mode="r")
    )


@pytest.mark.golden
def test_single_slice_df_regen_golden(tmp_path, single_slice_jp2, zarr_version):
    expected_zarr = f"data/df_single_slice_zarr{zarr_version}.nii.zarr.zip"
    output = tmp_path / "single_slice.zarr"
    single_slice.convert(
        str(single_slice_jp2),
        out=str(output),
        zarr_version=zarr_version,
    )
    base = Path(expected_zarr).with_suffix("")
    shutil.make_archive(str(base), "zip", str(output))
