import glob
import shutil
import tempfile
from os import PathLike
from pathlib import Path
from typing import Union

import skimage
import zarr.storage
from scipy.io import savemat

from linc_convert.modalities.psoct import single_volume, multi_slice
from utils.compare_file import _cmp_zarr_archives


# def test_oct(tmp_path):
#     files = glob.glob("data/test_oct/*")
#     files.sort()
#     output_zarr = tmp_path / "output.zarr"
#     multi_slice.convert(
#         files, out=str(output_zarr), key="Psi_ObsLSQ", zarr_version = 2, overwrite=True
#     )
#     assert _cmp_zarr_archives(str(output_zarr), "data/oct.zarr")

def generate_single_volume_data(path: Union[str, PathLike]) -> None:
    """
    Generate a single-brain-volume test dataset and save it as a .mat file.

    Parameters
    ----------
    path : Union[str, PathLike]
        Path to save the .mat file. Can be:
         - a directory (e.g. "tests/data/") → will write "tests/data/single_volume.mat"
         - a filename (e.g. "out/myvol.mat" or "out/myvol") → will write to that file (appending .mat if needed)
    """
    p = Path(path)
    if p.is_dir() or str(p).endswith(('/', '\\')):
        p = p / "single_volume.mat"
    if p.suffix.lower() != ".mat":
        p = p.with_suffix(".mat")
    p.parent.mkdir(parents=True, exist_ok=True)

    volume = skimage.data.brain().T
    savemat(str(p), {"volume": volume})


def test_oct_single_volume(tmp_path):
    input_path = tmp_path / "single_volume.mat"
    output_path = tmp_path / "single_volume.nii.zarr"
    generate_single_volume_data(input_path)
    single_volume.convert(input_path, out=str(output_path), key="volume",
                          zarr_version=2, overwrite=True, chunk=(64,))

    assert _cmp_zarr_archives(str(output_path), zarr.storage.ZipStore(
        "data/oct_single_volume_zarr2.nii.zarr.zip", mode="r"))


def test_oct_single_volume_zarr3(tmp_path):
    input_path = tmp_path / "single_volume.mat"
    output_path = tmp_path / "single_volume.nii.zarr"
    generate_single_volume_data(input_path)
    single_volume.convert(input_path, out=str(output_path), key="volume",
                          zarr_version=3, overwrite=True, chunk=(64,))

    assert _cmp_zarr_archives(str(output_path), zarr.storage.ZipStore(
        "data/oct_single_volume_zarr3.nii.zarr.zip", mode="r"))


def generate_single_volume_test_result():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        try:
            test_oct_single_volume(tmpdir)
        except Exception as e:
            pass
        shutil.make_archive("data/oct_single_volume_zarr2.nii.zarr", "zip",
                            str(tmpdir / "single_volume.nii.zarr"), )
        try:
            test_oct_single_volume_zarr3(tmpdir)
        except Exception as e:
            pass
        shutil.make_archive("data/oct_single_volume_zarr3.nii.zarr", "zip",
                            str(tmpdir / "single_volume.nii.zarr"), )


def generate_multi_slice_data(path: Union[str, PathLike]) -> None:
    """
    Generate a single-brain-volume test dataset and save it as a .mat file.

    Parameters
    ----------
    path : Union[str, PathLike]
        Path to save the .mat file.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    volume = skimage.data.brain().T
    for i in range(5):
        savemat(str(p / f"slice_{i:03d}.mat"), {"Psi_ObsLSQ": volume[..., i]})


def test_oct_multi_slices(tmp_path):
    input_path = tmp_path
    output_path = tmp_path / "multi_slice.nii.zarr"
    generate_multi_slice_data(input_path)
    input_path = glob.glob(str(input_path / "*.mat"))
    multi_slice.convert(input_path, out=str(output_path), zarr_version=2,
                        overwrite=True, chunk=(64,))

    assert _cmp_zarr_archives(str(output_path), zarr.storage.ZipStore(
        "data/oct_multi_slice_zarr2.nii.zarr.zip", mode="r"))


def test_oct_multi_slices_zarr3(tmp_path):
    input_path = tmp_path
    output_path = tmp_path / "multi_slice.nii.zarr"
    generate_multi_slice_data(input_path)
    input_path = glob.glob(str(input_path / "*.mat"))
    multi_slice.convert(input_path, out=str(output_path), zarr_version=3,
                        overwrite=True, chunk=(64,))

    assert _cmp_zarr_archives(str(output_path), zarr.storage.ZipStore(
        "data/oct_multi_slice_zarr3.nii.zarr.zip", mode="r"))


def generate_multi_slice_test_result():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        try:
            test_oct_multi_slices(tmpdir)
        except Exception as e:
            pass
        shutil.make_archive("data/oct_multi_slice_zarr2.nii.zarr", "zip",
                            str(tmpdir / "multi_slice.nii.zarr"), )
        try:
            test_oct_multi_slices_zarr3(tmpdir)
        except Exception as e:
            pass
        shutil.make_archive("data/oct_multi_slice_zarr3.nii.zarr", "zip",
                            str(tmpdir / "multi_slice.nii.zarr"), )
