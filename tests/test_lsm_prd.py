import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr
from utils.compare_file import assert_zarr_equal

from linc_convert.modalities.lsm import prd
from linc_convert.utils.io.prd import PrdSetInterpreter

# Directory holding the .prd input files that the conversion reads from.
PRD_INPUT_DIR = "data/lsm_prd"


def build_prd_subset(dest: Path) -> None:
    """Fetch a small subset of a Kinetix .prd assets to build the golden test input.

    Data: https://dandiarchive.org/dandiset/001769/draft/files?location=rawdata/HG9_Z1_Y49
    """
    import requests
    from dandi.dandiapi import DandiAPIClient

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    PRD_DANDISET = "001769"
    PRD_FILE_PATH = "rawdata/HG9_Z1_Y49"
    PRD_N_FRAMES = 10  # frames pulled from the front of each file
    PRD_N_FILES = 2  # number of files to subset

    with DandiAPIClient.for_dandi_instance("dandi") as client:
        dandiset = client.get_dandiset(PRD_DANDISET, "draft")

        def content_url(name: str) -> str:
            asset = dandiset.get_asset_by_path(f"{PRD_FILE_PATH}/{name}")
            return asset.get_content_url(follow_redirects=1, strip_query=True)

        # Download the metadata file
        (dest / "kinetixMetadata.txt").write_bytes(
            requests.get(content_url("kinetixMetadata.txt")).content
        )
        meta = PrdSetInterpreter.read_kinetix_metadata(str(dest))
        if meta["dataType"] != 16:
            raise ValueError(f"Unsupported dataType: {meta['dataType']}")
        bytes_per_pixel = np.dtype("uint16").itemsize
        stride = meta["width"] * meta["height"] * bytes_per_pixel + meta["gapBytes"]
        prefix_bytes = meta["headerBytes"] + PRD_N_FRAMES * stride

        # Download the first PRD_N_FILES files and only the first PRD_N_FRAMES frames of each
        for i in range(0, PRD_N_FILES):
            resp = requests.get(
                content_url(f"ss_stack_{i}.prd"),
                headers={"Range": f"bytes=0-{prefix_bytes - 1}"},
            )
            resp.raise_for_status()
            if resp.status_code != 206:
                raise RuntimeError(
                    f"Server ignored the Range request (status {resp.status_code}); "
                    "refusing to download the file."
                )
            (dest / f"ss_stack_{i}.prd").write_bytes(resp.content)


def test_lsm_prd(tmp_path, zarr_version):
    expected_zarr = f"data/lsm_prd_zarr{zarr_version}.nii.zarr.zip"
    output = tmp_path / "prd.nii.zarr"
    prd.convert(
        PRD_INPUT_DIR,
        out=str(output),
        zarr_version=zarr_version,
    )
    assert_zarr_equal(str(output), zarr.storage.ZipStore(expected_zarr, mode="r"))


@pytest.mark.golden
def test_lsm_prd_regen_golden(tmp_path, zarr_version):
    build_prd_subset(PRD_INPUT_DIR)
    base = f"data/lsm_prd_zarr{zarr_version}.nii.zarr"
    output = tmp_path / "prd.nii.zarr"
    prd.convert(
        PRD_INPUT_DIR,
        out=str(output),
        zarr_version=zarr_version,
    )
    shutil.make_archive(base, "zip", str(output))