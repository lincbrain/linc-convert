import os
import tempfile

import test_lsm
import zarr

from linc_convert.modalities.lsm import mosaic

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_lsm._write_test_data(tmp_dir)
        output_zarr = os.path.join(tmp_dir, "output.zarr")
        mosaic.convert(tmp_dir, output_zarr)
        zarr.copy_all(zarr.open(output_zarr), zarr.open("data/lsm.zarr.zip", "w"))
