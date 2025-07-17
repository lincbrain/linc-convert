import shutil
from pathlib import Path

import pytest
import skimage
import zarr.storage
from scipy.io import savemat

from linc_convert.modalities.psoct import single_volume
from utils.compare_file import assert_zarr_equal


@pytest.fixture
def single_volume_mat(tmp_path):
    """
    Create a temporary .mat file containing a single brain volume.
    """
    path = tmp_path / "single_volume.mat"
    volume = skimage.data.brain().T
    savemat(str(path), {"volume": volume})
    return path


@pytest.mark.parametrize(
        "zarr_version, expected_zarr",
        [
            (2, "data/oct_single_volume_zarr2.nii.zarr.zip"),
            (3, "data/oct_single_volume_zarr3.nii.zarr.zip"),
            ],
        )
def test_oct_single_volume(tmp_path, single_volume_mat, zarr_version, expected_zarr,
                           driver):
    output = tmp_path / "single_volume.nii.zarr"

    single_volume.convert(
            str(single_volume_mat),
            out=str(output),
            key="volume",
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
            (2, "data/oct_single_volume_zarr2.nii.zarr.zip"),
            (3, "data/oct_single_volume_zarr3.nii.zarr.zip"),
            ],
        )
def test_oct_single_volume_regen_golden(tmp_path, single_volume_mat, zarr_version,
                                        expected_zarr):
    """
    Rebuild single-volume golden archives. Only run with --regenerate-golden.
    """
    output = tmp_path / "single_volume.nii.zarr"
    single_volume.convert(
            str(single_volume_mat),
            out=str(output),
            key="volume",
            zarr_version=zarr_version,
            overwrite=True,
            chunk=(64,),
            driver="zarr-python",
            )
    base = Path(expected_zarr).with_suffix("")
    shutil.make_archive(str(base), "zip", str(output))
