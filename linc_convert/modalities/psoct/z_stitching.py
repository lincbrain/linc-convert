import logging
import os.path as op
from colorsys import hsv_to_rgb
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
from linc_convert.modalities.psoct.cli import psoct
from linc_convert.modalities.psoct._utils import struct_arr_to_dict, \
    atleast_2d_trailing, find_experiment_params, load_mat
from linc_convert.utils.zarr.zarr_io.drivers.tensorstore import default_write_config
from linc_convert.utils.zarr import ZarrConfig
from linc_convert.utils.zarr.zarr_io.drivers.zarr_python import compute_zarr_layout

logger = logging.getLogger(__name__)
z_stitching = cyclopts.App(name="z_stitching", help_format="markdown")
psoct.command(z_stitching)

@z_stitching.default
def stitch(
        inp: List[str],
        *,
        zarr_config: ZarrConfig = None,
        **kwargs
):

    pass




