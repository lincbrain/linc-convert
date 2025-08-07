import glob
import shutil
from pathlib import Path

import pytest
import skimage
import zarr.storage
from scipy.io import savemat

from linc_convert.modalities.psoct import multi_slice
from utils.compare_file import assert_zarr_equal


@pytest.fixture
def multi_slice_mats(tmp_path):
    """
    Create a set of temporary .mat files representing slices of a volume.
    """
    dir_path = tmp_path / "slices"
    dir_path.mkdir(parents=True, exist_ok=True)

    volume = skimage.data.brain().T
    files = []
    for i in range(5):
        file_path = dir_path / f"slice_{i:03d}.mat"
        savemat(str(file_path), {"Psi_ObsLSQ": volume[..., i]})
        files.append(str(file_path))
    return sorted(files)


def test_oct_multi_slice(tmp_path, multi_slice_mats, zarr_version, driver):
    expected_zarr = f"data/oct_multi_slice_zarr{zarr_version}.nii.zarr.zip"
    output = tmp_path / "multi_slice.nii.zarr"
    multi_slice.convert(
        multi_slice_mats,
        out=str(output),
        key="Psi_ObsLSQ",
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
def test_oct_multi_slice_regen_golden(tmp_path, multi_slice_mats, zarr_version):
    expected_zarr = f"data/oct_multi_slice_zarr{zarr_version}.nii.zarr.zip"
    output = tmp_path / "multi_slice.nii.zarr"
    multi_slice.convert(
        multi_slice_mats,
        out=str(output),
        key="Psi_ObsLSQ",
        zarr_version=zarr_version,
        overwrite=True,
        chunk=(64,),
        driver="zarr-python",
    )
    base = Path(expected_zarr).with_suffix("")
    shutil.make_archive(str(base), "zip", str(output))


@pytest.mark.heavy
def test_oct_multi_slice_heavy(test_data_heavy_dir, tmp_path, zarr_version, driver):
    multi_slice_heavy_data_dir = test_data_heavy_dir / "sub-test_oct_multi_slice"
    expected_zarr = multi_slice_heavy_data_dir / f"zarr{zarr_version}.nii.zarr.zip"
    files = glob.glob(str(multi_slice_heavy_data_dir / "*.mat"))
    files.sort()
    output_zarr = tmp_path / "output.zarr"
    multi_slice.convert(
        files,
        out=str(output_zarr),
        key="Psi_ObsLSQ",
        zarr_version=2,
        overwrite=True,
        driver=driver,
    )
    assert_zarr_equal(
        str(output_zarr),
        zarr.storage.ZipStore(expected_zarr, mode="r"),
    )
    base = Path(expected_zarr).with_suffix("")
    shutil.make_archive(str(base), "zip", str(output_zarr))


@pytest.mark.golden
@pytest.mark.heavy
def test_oct_multi_slice_heavy_regen_golden(
    test_data_heavy_dir, tmp_path, zarr_version
):
    multi_slice_heavy_data_dir = test_data_heavy_dir / "sub-test_oct_multi_slice"
    expected_zarr = multi_slice_heavy_data_dir / f"zarr{zarr_version}.nii.zarr.zip"
    files = glob.glob(str(multi_slice_heavy_data_dir / "*.mat"))
    files.sort()
    output_zarr = tmp_path / "output.zarr"
    multi_slice.convert(
        files,
        out=str(output_zarr),
        key="Psi_ObsLSQ",
        zarr_version=zarr_version,
        overwrite=True,
    )
    base = expected_zarr.with_suffix("")
    shutil.make_archive(str(base), "zip", str(output_zarr))
