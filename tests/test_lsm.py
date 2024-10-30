import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import tifffile
import zarr

from linc_convert.modalities.lsm import mosaic


def _compare_zarr_archives(path1, path2):
    """
    Compare two Zarr archives to check if they contain the same data.

    Parameters
    ----------
    - path1 (str): Path to the first Zarr archive.
    - path2 (str): Path to the second Zarr archive.

    Returns
    -------
    - bool: True if both archives contain the same data, False otherwise.
    """
    # Open both Zarr groups
    zarr1 = zarr.open(path1, mode="r")
    zarr2 = zarr.open(path2, mode="r")

    # Compare keys (dataset structure)
    if zarr1.keys() != zarr2.keys():
        return False
    if zarr1.attrs != zarr2.attrs:
        return False

    # Compare each array in both archives
    for key in zarr1.keys():
        array1 = zarr1[key][:]
        array2 = zarr2[key][:]

        # Check for equality of the arrays
        if not np.array_equal(array1, array2):
            print(f"Mismatch found in dataset: {key}")
            return False
        if zarr1[key].attrs != zarr2[key].attrs:
            return False

    # If all checks pass
    print("The Zarr archives are identical.")
    return True


def _write_test_data(directory):
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


class Test(TestCase):
    def test_convert(self):
        with TemporaryDirectory() as tmp_dir:
            _write_test_data(tmp_dir)
            output_zarr = os.path.join(tmp_dir, "output.zarr")
            mosaic.convert(tmp_dir, output_zarr)
            self.assertTrue(
                _compare_zarr_archives(output_zarr, "tests/data/lsm.zarr.zip")
            )
