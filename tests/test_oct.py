import glob
import shutil
import tempfile
from os import PathLike
from pathlib import Path
from typing import Union

import skimage
import zarr.storage
from scipy.io import savemat
import pytest

from linc_convert.modalities.psoct import single_volume, multi_slice
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


@ pytest.mark.oct
@ pytest.mark.parametrize(
    "zarr_version, expected_zarr",
    [
        (2, "data/oct_single_volume_zarr2.nii.zarr.zip"),
        (3, "data/oct_single_volume_zarr3.nii.zarr.zip"),
    ],
)
def test_single_volume(tmp_path, single_volume_mat, zarr_version, expected_zarr):
    output = tmp_path / "single_volume.nii.zarr"

    single_volume.convert(
        str(single_volume_mat),
        out=str(output),
        key="volume",
        zarr_version=zarr_version,
        overwrite=True,
        chunk=(64,),
        driver="tensorstore",
    )

    assert_zarr_equal(
        str(output),
        zarr.storage.ZipStore(expected_zarr, mode="r"),
    )


@ pytest.mark.oct
@ pytest.mark.parametrize(
    "zarr_version, expected_zarr",
    [
        (2, "data/oct_multi_slice_zarr2.nii.zarr.zip"),
        (3, "data/oct_multi_slice_zarr3.nii.zarr.zip"),
    ],
)

def test_multi_slice(tmp_path, multi_slice_mats, zarr_version, expected_zarr):
    output = tmp_path / "multi_slice.nii.zarr"

    multi_slice.convert(
        multi_slice_mats,
        out=str(output),
        key="Psi_ObsLSQ",
        zarr_version=zarr_version,
        overwrite=True,
        chunk=(64,),
        driver="tensorstore",
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
def test_single_volume_regen_golden(tmp_path, single_volume_mat, zarr_version, expected_zarr):
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


@pytest.mark.golden
@pytest.mark.parametrize(
    "zarr_version, expected_zarr",
    [
        (2, "data/oct_multi_slice_zarr2.nii.zarr.zip"),
        (3, "data/oct_multi_slice_zarr3.nii.zarr.zip"),
    ],
)
def test_multi_slice_regen_golden(tmp_path, multi_slice_mats, zarr_version, expected_zarr):
    """
    Rebuild multi-slice golden archives. Only run with --regenerate-golden.
    """
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

@ pytest.mark.heavy
@ pytest.mark.parametrize(
    "zarr_version, expected_zarr",
    [
        (2, "zarr2.nii.zarr.zip"),
        (3, "zarr3.nii.zarr.zip"),
    ],
)
def test_multi_slice_heavy(test_data_heavy_dir, tmp_path, zarr_version, expected_zarr):
    files = glob.glob(test_data_heavy_dir / "sub-test_oct_multi_slice" / "*.mat")
    files.sort()
    output_zarr = tmp_path / "output.zarr"
    multi_slice.convert(
        files, out=str(output_zarr), key="Psi_ObsLSQ", zarr_version = 2, overwrite=True
    )
    assert_zarr_equal(
        str(output_zarr),
        zarr.storage.ZipStore(expected_zarr, mode="r"),
    )
    base = Path(expected_zarr).with_suffix("")
    shutil.make_archive(str(base), "zip", str(output_zarr))

@ pytest.mark.golden
@ pytest.mark.heavy
@ pytest.mark.parametrize(
    "zarr_version, expected_zarr",
    [
        (2, "zarr2.nii.zarr.zip"),
        (3, "zarr3.nii.zarr.zip"),
    ],
)
def test_multi_slice_heavy_regen_golden(test_data_heavy_dir, tmp_path, zarr_version, expected_zarr):
    multi_slice_heavy_data_dir = test_data_heavy_dir / "sub-test_oct_multi_slice"
    files = glob.glob(str(multi_slice_heavy_data_dir / "*.mat"))
    files.sort()
    output_zarr = tmp_path / "output.zarr"
    multi_slice.convert(
        files, out=str(output_zarr), key="Psi_ObsLSQ", zarr_version = zarr_version, overwrite=True
    )
    base = Path(multi_slice_heavy_data_dir / expected_zarr).with_suffix("")
    shutil.make_archive(str(base), "zip", str(output_zarr))

