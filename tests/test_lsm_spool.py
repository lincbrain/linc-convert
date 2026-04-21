import shutil
from pathlib import Path

import pytest
import zarr

from utils.compare_file import assert_zarr_equal

from linc_convert.modalities.lsm import spool


def test_lsm_spool_convert(tmp_path, spool_dat, zarr_version):
    expected_zarr = f"data/lsm_spool_zarr{zarr_version}.nii.zarr.zip"
    output = tmp_path / "spool.nii.zarr"
    spool.convert(inp=spool_dat, out=str(output), overlap=0)

    # assert_zarr_equal is assumed available from your test utils
    assert_zarr_equal(str(output), zarr.storage.ZipStore(expected_zarr, mode="r"))


@pytest.mark.golden
def test_lsm_spool_regen_golden(tmp_path, spool_dat, zarr_version):
    expected_zarr = f"data/lsm_spool_zarr{zarr_version}.nii.zarr.zip"
    output = tmp_path / "spool_output.nii.zarr"
    spool.convert(inp=spool_dat, out=str(output), overlap=0)

    # Write out new golden archive
    base = Path(expected_zarr).with_suffix("")
    shutil.make_archive(str(base), "zip", str(output))
