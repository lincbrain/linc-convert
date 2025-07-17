import glob
import os
import shutil
from pathlib import Path

import glymur
import numpy as np
import pytest
import zarr

from linc_convert.modalities.df import multi_slice
from utils.compare_file import assert_zarr_equal
from utils.sample_data import generate_sample_data_variation


@pytest.fixture
def multi_slice_jp2(tmp_path):
    image = generate_sample_data_variation(3)
    image = np.array(image).transpose(2, 3, 0, 1)
    for i in range(4):
        glymur.Jp2k(
            f"{tmp_path}/slice{i:03d}.jp2",
            data=image[..., i],
        )
    files = glob.glob(str(os.path.join(tmp_path, "*.jp2")))
    files.sort()
    return files


def test_multi_slice_df(tmp_path, multi_slice_jp2, zarr_version, driver):
    expected_zarr = f"data/df_multi_slice_zarr{zarr_version}.nii.zarr.zip"
    output = tmp_path / "multi_slice.zarr"
    multi_slice.convert(
        multi_slice_jp2,
        out=str(output),
        zarr_version=zarr_version,
        driver=driver
    )
    assert_zarr_equal(
        str(output),
        zarr.storage.ZipStore(expected_zarr, mode="r")
    )


@pytest.mark.golden
def test_multi_slice_df_regen_golden(tmp_path, multi_slice_jp2, zarr_version):
    expected_zarr = f"data/df_multi_slice_zarr{zarr_version}.nii.zarr.zip"
    output = tmp_path / "multi_slice.zarr"
    multi_slice.convert(
        multi_slice_jp2,
        out=str(output),
        zarr_version=zarr_version,
    )
    base = Path(expected_zarr).with_suffix("")
    shutil.make_archive(str(base), "zip", str(output))
