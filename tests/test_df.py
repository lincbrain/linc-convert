import glob
import os
import zipfile

import glymur
import numpy as np
from helper import _cmp_zarr_archives

from linc_convert.modalities.df import multi_slice


def _write_test_data(directory: str) -> None:
    for level in range(5):
        image = np.zeros((1024, 768, 3), dtype=np.uint8) + level * 10
        glymur.Jp2k(
            f"{directory}/{level}.jp2",
            data=image,
        )


def test_df(tmp_path):
    # _write_test_data(tmp_path)
    with zipfile.ZipFile("data/df_input.zip", "r") as z:
        z.extractall(tmp_path)
    output_zarr = tmp_path / "output.zarr"
    files = glob.glob(os.path.join(tmp_path, "*.jp2"))
    files.sort()
    multi_slice.convert(files, str(output_zarr))
    assert _cmp_zarr_archives(str(output_zarr), "data/df.zarr.zip")
