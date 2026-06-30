import shutil
from pathlib import Path

import numpy as np
import pytest
import skimage
import tifffile
import zarr.storage
from utils.compare_file import assert_zarr_equal

from linc_convert.modalities.lsm import single_volume


@pytest.fixture
def single_volume_tiff(tmp_path):
    """Create a temporary multi-page TIFF file containing a small volume."""
    path = tmp_path / "single_volume.tif"
    slice2d = skimage.data.brain().astype(np.uint16)
    volume = np.stack([slice2d] * 4, axis=0)
    tifffile.imwrite(str(path), volume)
    return path


@pytest.mark.parametrize(
    "zarr_version, expected_zarr",
    [
        (2, "data/lsm_single_volume_zarr2.nii.zarr.zip"),
        (3, "data/lsm_single_volume_zarr3.nii.zarr.zip"),
    ],
)
def test_lsm_single_volume(
    tmp_path, single_volume_tiff, zarr_version, expected_zarr, driver
):
    output = tmp_path / "single_volume.nii.zarr"

    single_volume.convert(
        str(single_volume_tiff),
        out=str(output),
        zarr_version=zarr_version,
        overwrite=True,
        chunk=(64,),
        driver=driver,
    )

    assert_zarr_equal(
        str(output),
        zarr.storage.ZipStore(expected_zarr, mode="r"),
    )


@pytest.mark.golden
@pytest.mark.parametrize(
    "zarr_version, expected_zarr",
    [
        (2, "data/lsm_single_volume_zarr2.nii.zarr.zip"),
        (3, "data/lsm_single_volume_zarr3.nii.zarr.zip"),
    ],
)
def test_lsm_single_volume_regen_golden(
    tmp_path, single_volume_tiff, zarr_version, expected_zarr
):
    """
    Rebuild lsm single-volume golden archives. Only run with --regenerate-golden.
    """
    output = tmp_path / "single_volume.nii.zarr"
    single_volume.convert(
        str(single_volume_tiff),
        out=str(output),
        zarr_version=zarr_version,
        overwrite=True,
        chunk=(64,),
        driver="zarr-python",
    )
    base = Path(expected_zarr).with_suffix("")
    shutil.make_archive(str(base), "zip", str(output))


