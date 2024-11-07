import glob
import os
import tempfile
import zipfile

import test_df
import test_lsm
import zarr

from linc_convert.modalities.df import multi_slice
from linc_convert.modalities.lsm import mosaic

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_df._write_test_data(tmp_dir)
        output_zarr = os.path.join(tmp_dir, "output.zarr")
        files = glob.glob(os.path.join(tmp_dir, "*.jp2"))
        files.sort()
        with zipfile.ZipFile("data/df_input.zip", "w") as z:
            for file in files:
                z.write(file, os.path.basename(file))
        multi_slice.convert(files, output_zarr)
        zarr.copy_all(zarr.open(output_zarr), zarr.open("data/df.zarr.zip", "w"))

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_lsm._write_test_data(tmp_dir)
        output_zarr = os.path.join(tmp_dir, "output.zarr")
        mosaic.convert(tmp_dir, output_zarr)
        zarr.copy_all(zarr.open(output_zarr), zarr.open("data/lsm.zarr.zip", "w"))
