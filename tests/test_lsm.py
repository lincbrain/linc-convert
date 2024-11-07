from pathlib import Path

import numpy as np
import tifffile
from helper import _cmp_zarr_archives

from linc_convert.modalities.lsm import mosaic


def _write_test_data(directory: str) -> None:
    root_path = Path(directory)
    for z in range(1, 2, 1):
        for y in range(1, 3, 1):
            folder = root_path / f"test_z{z}_y{y}"
            folder.mkdir(parents=True, exist_ok=True)
            for plane in range(1, 4, 1):
                for c in range(1, 3, 1):
                    image = np.zeros((1024, 768)) + z * y * plane * c
                    tifffile.imwrite(
                        folder / f"test_z{z}_y{y}_plane{plane}_c{c}.tiff", image
                    )


def test_lsm(tmp_path):
    _write_test_data(tmp_path)
    output_zarr = tmp_path / "output.zarr"
    mosaic.convert(str(tmp_path), str(output_zarr))
    assert _cmp_zarr_archives(str(output_zarr), "data/lsm.zarr.zip")
