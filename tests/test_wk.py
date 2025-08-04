import shutil
from pathlib import Path

import niizarr
import numpy as np
import pytest
import wkw
import zarr
from nibabel import Nifti1Image

from linc_convert.modalities.wk import webknossos_annotation
from linc_convert.modalities.wk.webknossos_annotation import get_mask_name
from utils.compare_file import assert_zarr_equal
from utils.sample_data import generate_sample_data_variation


@pytest.fixture
def wk_annotation(tmp_path):
    image, annotation = generate_sample_data_variation(2, output_dtype=np.uint8)
    image, annotation = (
        Nifti1Image(image, np.eye(4)),
        Nifti1Image(annotation, np.eye(4)),
    )

    niizarr.nii2zarr(image, tmp_path / "image.nii.zarr", chunk=64)
    niizarr.nii2zarr(annotation, tmp_path / "annotation.nii.zarr", chunk=64)
    wkw_dir = tmp_path / "wkw"
    annotation_data = zarr.open(tmp_path / "annotation.nii.zarr", mode="r")
    for level in range(3):
        wkw_filepath = str(wkw_dir / get_mask_name(level))

        with wkw.Dataset.create(wkw_filepath, wkw.Header(np.uint8)) as dataset:
            dataset.write((0, 0, 0), np.array(annotation_data[str(level)]).T)
    return tmp_path


def test_wkw(tmp_path, wk_annotation, zarr_version, driver):
    expected_zarr = f"data/wkw_zarr{zarr_version}.nii.zarr.zip"
    output = tmp_path / "wkw.nii.zarr"
    webknossos_annotation.convert(
        str(wk_annotation / "wkw"),
        str(wk_annotation / "image.nii.zarr"),
        out=output,
        dic="{}",
    )
    assert_zarr_equal(str(output), zarr.storage.ZipStore(expected_zarr, mode="r"))


@pytest.mark.golden
def test_wkw_regen_golden(tmp_path, wk_annotation, zarr_version):
    expected_zarr = f"data/wkw_zarr{zarr_version}.nii.zarr.zip"
    output = tmp_path / "wkw.nii.zarr"
    webknossos_annotation.convert(
        str(wk_annotation / "wkw"),
        str(wk_annotation / "image.nii.zarr"),
        out=output,
        dic="{}",
    )
    base = Path(expected_zarr).with_suffix("")
    shutil.make_archive(str(base), "zip", str(output))
