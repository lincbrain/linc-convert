import shutil
from pathlib import Path

import pytest
import zarr.storage
from utils.compare_file import assert_zarr_equal

from linc_convert.modalities.lsm import strip


def _get_first_strip_dir(root):
    """Return the first spool strip directory inside the synthetic root."""
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.endswith("_HR"):
            return p
    raise RuntimeError(f"No strip directory found under {root}")


def test_lsm_strip_convert(tmp_path, spool_dat, zarr_version):
    strip_dir = _get_first_strip_dir(spool_dat)

    output = tmp_path / "strip.nii.zarr"
    strip.convert(
        inp=str(strip_dir),
        info_file=None,
        out=str(output),
    )

    expected_zarr = f"data/lsm_strip_zarr{zarr_version}.nii.zarr.zip"
    assert_zarr_equal(
        str(output),
        zarr.storage.ZipStore(expected_zarr, mode="r"),
    )


@pytest.mark.golden
def test_lsm_strip_regen_golden(tmp_path, spool_dat, zarr_version):
    strip_dir = _get_first_strip_dir(spool_dat)

    output = tmp_path / "strip_output.nii.zarr"
    strip.convert(
        inp=str(strip_dir),
        info_file=None,
        out=str(output),
    )

    expected_zarr = f"data/lsm_strip_zarr{zarr_version}.nii.zarr.zip"
    base = Path(expected_zarr).with_suffix("")
    shutil.make_archive(str(base), "zip", str(output))
