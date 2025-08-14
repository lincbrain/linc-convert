import logging
from typing import List

import cyclopts
import zarr
from niizarr import default_nifti_header, write_nifti_header, write_ome_metadata

from linc_convert.modalities.psoct.cli import psoct
from linc_convert.utils.zarr_config import ZarrConfig

logger = logging.getLogger(__name__)
pyramid = cyclopts.App(name="pyramid", help_format="markdown")
psoct.command(pyramid)


@pyramid.default
def convert(
        inp: List[str],
        *,
        zarr_config: ZarrConfig = None,
        **kwargs
        ) -> None:
    for i in inp:
        print(i)
        # zg = zarr.open_group(i, mode="r+")
        # zg.generate
        write_ome_metadata(zg, ["z", "y", "x"], [0.0025, 0.01, 0.01],
                           space_unit="millimeter")
        nii_header = default_nifti_header(zg["0"], zg.attrs["multiscales"])
        nii_header.set_xyzt_units("mm")
        write_nifti_header(zg, nii_header)

    pass
