import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import skimage
import zarr
from utils.compare_file import assert_zarr_equal

from linc_convert.modalities.lsm import spool


@pytest.fixture
def spool_dat(tmp_path):
    brain = skimage.data.brain()
    brain = brain.reshape(10, 4, 64, 256)
    for z in range(1, 11):
        for y in range(1, 4):
            out_path = tmp_path / f"test_run{y:02d}_y{y:02d}_z{z:02d}_HR"
            image = brain[z - y, y - 1][None, ...]

            write_zyla_spool_set(image, out_path, images_per_file=128)

    return tmp_path


def write_zyla_spool_set(
    image: np.ndarray,
    spool_set_path: str,
    *,
    pixel_encoding: str = "Mono16",
    images_per_file: int = None,
):
    """
    Given `image` of shape (numDepths, numColumns, numFrames),
    create a Zyla‑compatible spool directory at `spool_set_path`
    such that SpoolSetInterpreter(...).assemble_cropped() == image.

    Parameters
    ----------
    image : np.ndarray
        3D array with shape (numDepths, numColumns, numFrames).  Must be dtype uint16 if
        pixel_encoding='Mono16' (or uint8 if 'Mono8').
    spool_set_path : str
        Path to the directory to create.  Will be mkdir'd if necessary.
    pixel_encoding : {'Mono16','Mono8'}
        Selects dtype and controls AOIStride = numColumns * itemsize.
    images_per_file : int, optional
        How many frames to pack into each spool file.  Defaults to all frames in one
        file.
    """
    numDepths, numColumns, numFrames = image.shape
    if images_per_file is None:
        images_per_file = numFrames
    if pixel_encoding == "Mono16":
        dtype = np.uint16
    elif pixel_encoding == "Mono8":
        dtype = np.uint8
    else:
        raise ValueError("pixel_encoding must be 'Mono16' or 'Mono8'")
    if image.dtype != dtype:
        raise ValueError(f"Expected image.dtype={dtype}, got {image.dtype}")

    # compute padded rows: odd heights get +1, even get +2
    if numDepths % 2:
        numRows = numDepths + 1
    else:
        numRows = numDepths + 2

    # stride in bytes
    stride = numColumns * dtype().nbytes
    # how many bytes per frame
    frame_bytes = numRows * numColumns * dtype().nbytes

    # make directory
    os.makedirs(spool_set_path, exist_ok=True)

    # marker for format detection
    open(os.path.join(spool_set_path, "Spooled files.sifx"), "wb").close()

    # write acquisitionmetadata.ini
    ini = f"""
[data]
AOIHeight = {numDepths}
AOIWidth  = {numColumns}
AOIStride = {stride}
PixelEncoding = {pixel_encoding}
ImageSizeBytes = {frame_bytes}

[multiimage]
ImagesPerFile = {images_per_file}
"""
    with open(
        os.path.join(spool_set_path, "acquisitionmetadata.ini"), "w", encoding="utf-8"
    ) as f:
        f.write(ini)

    # split into as many spool files as needed
    n_files = (numFrames + images_per_file - 1) // images_per_file
    for file_idx in range(n_files):
        start_f = file_idx * images_per_file
        end_f = min(start_f + images_per_file, numFrames)
        this_count = end_f - start_f

        # prepare buffer: shape (this_count, numRows, numColumns)
        buf = np.zeros((this_count, numRows, numColumns), dtype=dtype)
        # fill the real depths into row 0..numDepths-1
        # image has shape (numDepths, numColumns, numFrames)
        # buf[f, d, c] = image[d, c, f]
        for f in range(this_count):
            buf[f, :numDepths, :] = image[:, :, start_f + f]

        # name = reversed zero‑padded index + "spool.dat"
        rev = str(file_idx).zfill(10)[::-1]
        fn = f"{rev}spool.dat"
        with open(os.path.join(spool_set_path, fn), "wb") as f:
            f.write(buf.tobytes())


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
