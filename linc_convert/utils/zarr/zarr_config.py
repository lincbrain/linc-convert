"""Configuration related to output Zarr Archive."""

from dataclasses import dataclass
from typing import Annotated, Literal, Optional

from cyclopts import Parameter


@dataclass
class _ZarrConfig:
    """
    Configuration related to output Zarr Archive.

    Parameters
    ----------
    out
        Path to the output Zarr directory [<INP>.ome.zarr]
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
    driver : {"zarr-python", "tensorstore", "zarrita"}
        library used for Zarr IO Operation

    """

    out: Optional[str] = None
    chunk: tuple[int] = (128,)
    shard: list[int | str] | None = None
    version: Literal[2, 3] = 3
    compressor: str = "blosc"
    compressor_opt: str = "{}"
    nii: bool = False
    driver: Literal["zarr-python", "tensorstore", "zarrita"] = "zarr-python"

    def __post_init__(self) -> None:
        print(self)
        self.nii |= self.out.endswith(".nii.zarr")


ZarrConfig = Annotated[_ZarrConfig, Parameter(name="*")]
