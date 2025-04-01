"""Configuration related to output Zarr Archive."""
import dataclasses
import os
from dataclasses import dataclass, replace
from typing import Annotated, Literal

import numpy as np
from cyclopts import Parameter
from typing_extensions import Unpack
import zarr

from linc_convert.utils.zarr.compressor import make_compressor


@dataclass
class _ZarrConfig:
    """
    Configuration related to output Zarr Archive.

    Parameters
    ----------
    chunk
        Output chunk size.
        Behavior depends on the number of values provided:
        * one:   used for all spatial dimensions
        * three: used for spatial dimensions ([z, y, x])
        * four:  used for channels and spatial dimensions ([c, z, y, x])
    shard
        Output shard size.
        If `"auto"`, find shard size that ensures files smaller than 2TB,
        assuming a compression ratio or 2.
    version
        Zarr version to use. If `shard` is used, 3 is required.
    compressor : {blosc, zlib|gzip, raw}
        Compression method
    compressor_opt
        Compression options
    nii
        Convert to nifti-zarr. True if path ends in ".nii.zarr".

    overwrite
        when no name is supplied and using default output name, if overwrite is set,
        it won't ask if overwrite
    driver : {"zarr-python", "tensorstore", "zarrita"}
        library used for Zarr IO Operation

    """
    out: str = None
    zarr_version: Literal[2, 3] = 3
    chunk: tuple[int] = (128,)
    chunk_channels: bool = False
    chunk_time: bool = False
    shard: tuple[int | str] | None = None
    shard_channel: bool = False
    shard_time: bool = False
    dimension_separator: Literal[".", "/"] = "/"
    order: Literal["C", "F"] = "C"
    compressor: Literal["blosc", "zlib", None] = "blosc"
    compressor_opt: dict = dataclasses.field(default_factory=dict)
    no_pyramid_axis: Literal["x", "y", "z", None] = None
    levels: int = -1
    ome_version: Literal["0.4","0.5"] = "0.4"
    nii: bool = False
    max_load: int = 512
    overwrite: bool = False
    # driver: Literal["zarr-python", "tensorstore", "zarrita"] = "zarr-python"

    def __post_init__(self) -> None:
        self.nii |= self.out.endswith(".nii.zarr")

    def set_default_name(self, name: str):
        if self.out is not None:
            return
        self.out = name
        self.out += ".nii.zarr" if self.nii else ".ome.zarr"
        if os.path.exists(self.out) and not self.overwrite:
            answer = input(
                f"The output path '{self.out}' already exists. Do you want to overwrite it? (y/n): ")
            if answer.lower() not in ("y", "yes"):
                raise FileExistsError(
                    f"Output path '{self.out}' exists and overwrite was not confirmed.")

    def update(self, **kwargs):
        replace(self, **kwargs)
        return self

ZarrConfig = Annotated[_ZarrConfig, Parameter(name="*")]


def update(zarr_config: ZarrConfig|None, **kwargs: Unpack[ZarrConfig]) -> ZarrConfig:
    if zarr_config is None:
        zarr_config = _ZarrConfig()
    replace(zarr_config, **kwargs)
    return zarr_config

def open_zarr_group(zarr_config:ZarrConfig):
    store = zarr.storage.LocalStore(zarr_config.out)
    return zarr.group(store=store, overwrite=True, zarr_format=zarr_config.zarr_version)


def create_array(omz: zarr.Group, name:str, shape:tuple, zarr_config:ZarrConfig, data=None, dtype=None) -> zarr.Array:
    compressor = zarr_config.compressor
    compressor_opt = zarr_config.compressor_opt
    # if isinstance(compressor_opt, str):
    #     compressor_opt = ast.literal_eval(compressor_opt)
    opt = {
        "chunks": [*zarr_config.chunk] * 3,
        "order": zarr_config.order,
        "dtype": np.dtype(dtype).str,
        "fill_value": None,
        "compressors": make_compressor(compressor, **compressor_opt),
    }

    dimension_separator = zarr_config.dimension_separator
    if dimension_separator == '.' and omz.metadata.zarr_format == 2:
        pass
    elif dimension_separator == '/' and omz.metadata.zarr_format == 3:
        pass
    else:
        from zarr.core.chunk_key_encodings import ChunkKeyEncodingParams
        dimension_separator = ChunkKeyEncodingParams(
            name="default" if omz.metadata.zarr_format == 3 else "v2",
            separator=dimension_separator)

        opt["chunk_key_encoding"] = dimension_separator

    arr= omz.create_array(name=name,
                          shape=shape,
                          **opt)
    if data:
        arr[:] = data
    return arr