import logging
import os.path as op
from typing import Optional, Tuple, Dict, List

import cyclopts
import dask
import dask.array as da
import imageio
import nibabel as nib
import numpy as np
import scipy.io as sio
import tensorstore as ts
import zarr
from niizarr import write_ome_metadata

from linc_convert.modalities.psoct.cli import psoct
from linc_convert.modalities.psoct._utils import struct_arr_to_dict, \
    atleast_2d_trailing, find_experiment_params, load_mat
from linc_convert.utils.zarr.zarr_io.drivers.tensorstore import default_write_config
from linc_convert.utils.zarr import ZarrConfig, from_config
from linc_convert.utils.zarr.zarr_io.drivers.zarr_python import compute_zarr_layout

logger = logging.getLogger(__name__)
z_stitching = cyclopts.App(name="z_stitching", help_format="markdown")
psoct.command(z_stitching)

@z_stitching.default
def stitch(
        inp: List[str],
        *,
        overlap:int = 0,
        offset:int = 0,
        zarr_config: ZarrConfig = None,
        **kwargs
)-> None:
    # load slices to dask
    dask_slices = [da.from_zarr(op.join(inp[i],"0"),) for i in range(len(inp))]
    # crop slices
    for i in range(len(dask_slices)):
        sl = dask_slices[i]
        if i != 0:
            sl = sl[overlap//2:]
        if i != len(dask_slices) - 1:
            sl = sl[:overlap-overlap//2]
        dask_slices[i] = sl
    vol = da.concatenate(dask_slices, axis=0)
    # output
    zgroup = from_config(zarr_config)
    arr = zgroup.create_array("0",vol.shape,dtype= vol.dtype, zarr_config=zarr_config)
    vol = vol.rechunk(zarr_config.chunk*3)
    vol.store(arr)


    zgroup.generate_pyramid()
    # zgroup = zarr.open_group(zgroup.store_path, mode= "a")
    write_ome_metadata(zgroup, ["z", "y", "x"])



