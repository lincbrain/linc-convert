import os
import tempfile
from pathlib import Path

import zarr

import test_lsm
import test_wk
from linc_convert.modalities.lsm import mosaic
from linc_convert.modalities.wk import webknossos_annotation

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_lsm._write_test_data(tmp_dir)
        output_zarr = os.path.join(tmp_dir, "output.zarr")
        mosaic.convert(tmp_dir, output_zarr)
        zarr.copy_all(zarr.open(output_zarr), zarr.open("data/lsm.zarr.zip", "w"))

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_wk._write_test_data(tmp_dir)

        tmp_dir = Path(tmp_dir)
        wkw_dir = str(tmp_dir / "wkw")
        ome_dir = str(tmp_dir / "ome")

        basename = os.path.basename(ome_dir)[:-9]
        initials = wkw_dir.split("/")[-2][:2]
        output_zarr = os.path.join(
            tmp_dir, basename + "_dsec_" + initials + ".ome.zarr"
        )

        webknossos_annotation.convert(wkw_dir, ome_dir, tmp_dir, "{}")
        zarr.copy_all(zarr.open(output_zarr), zarr.open("data/wk.zarr.zip", "w"))
