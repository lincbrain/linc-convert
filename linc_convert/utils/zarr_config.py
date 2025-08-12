"""Configuration related to output Zarr Archive."""

import logging
from dataclasses import dataclass, field, replace
from typing import Annotated, Literal, TypeAlias

from cyclopts import Parameter
from typing_extensions import Unpack

logger = logging.getLogger(__name__)

DriverLike: TypeAlias = Literal["zarr-python", "tensorstore", "zarrita"]


@Parameter(name="*")
@dataclass
class ZarrConfig:
    """
    Configuration related to output Zarr Archive.

    Parameters
    ----------
    out
        Output path
    chunk
        Output chunk size.
        Behavior depends on the number of values provided:
        * one:   used for all spatial dimensions
        * three: used for spatial dimensions ([z, y, x])
        * four+:  used for channels and spatial dimensions ([c, z, y, x])
        If `"auto"`, find chunk size smaller than 1 MB (TODO: not implemented)
    zarr_version
        Zarr version to use. If `shard` is used, 3 is required.
    chunk_channels:
        Put channels in different chunk.
        If False, combine all channels in a single chunk.
    chunk_time :
        Put timepoints in different chunk.
        If False, combine all timepoints in a single chunk.
    shard
        Output shard size.
        Behavior same as chunk.
        If `"auto"`, find shard size that ensures files smaller than 2TB,
        assuming a compression ratio or 2.
    shard_channels:
        Put channels in different shards.
        If False, combine all channels in a single shard.
    shard_time:
        Put timepoints in different shards.
        If False, combine all timepoints in a single shard.
    dimension_separator:
        The separator placed between the dimensions of a chunk.
    order:
        Memory layout order for the data array.
    compressor
        Compression method
    compressor_opt
        Compression options
    no_time
        If True, indicates that the dataset does not have a time dimension.
        In such cases, any fourth dimension is interpreted as the channel dimension.
    no_pyramid_axis
        Spatial axis that should not be downsampled when generating pyramid levels.
        If None, downsampling is applied across all spatial axes.
    levels : int, optional
        Number of pyramid levels to generate.
        If set to -1, all possible levels are generated until the smallest level
        fits into one chunk.
    ome_version
        Version of the OME-Zarr specification to use
    nii
        Convert the output to nifti-zarr format.
        This is automatically enabled if the output path ends with ".nii.zarr".
    max_load:
        Maximum amount of data to load into memory at once during processing.
    overwrite
        when no name is supplied and using default output name, if overwrite is set,
        it won't ask if overwrite
    driver : {"zarr-python", "tensorstore", "zarrita"}
        library used for Zarr IO Operation
    """

    out: Annotated[str, Parameter(name=["--out", "-o"])] = None
    zarr_version: Literal[2, 3] = 3
    chunk: tuple[int] = (128,)
    chunk_channels: bool = False
    chunk_time: bool = True
    shard: tuple[int] | Literal["auto"] | None = None
    shard_channels: bool = False
    shard_time: bool = False
    dimension_separator: Literal[".", "/"] = "/"
    order: Literal["C", "F"] = "C"
    compressor: Literal["blosc", "zlib", None] = "blosc"
    compressor_opt: dict[str, float | str] = field(default_factory=dict)
    no_time: bool = False
    no_pyramid_axis: Literal["x", "y", "z", None] = None
    levels: int = -1
    ome_version: Literal["0.4", "0.5"] = "0.4"
    nii: bool = False
    max_load: int = 512
    overwrite: bool = False
    driver: DriverLike = "zarr-python"

    def __post_init__(self) -> None:
        """
        Perform post-initialization checks and adjustments.

        - Automatically enable NIfTI mode if the output path ends with ".nii.zarr".
        - Ensure that sharding options (shard, shard_channels, shard_time) are only
          used when zarr_version == 3; otherwise raise NotImplementedError.
        """
        if self.out:
            self.nii |= str(self.out).endswith(".nii.zarr")
        if self.zarr_version == 2:
            if self.shard or self.shard_channels or self.shard_time:
                raise NotImplementedError("Shard is not supported for Zarr 2.")

    def set_default_name(self, name: str) -> None:
        """
        Assign a default output name if none was specified.

        Parameters
        ----------
        name : str
            Base filename (without extension) to use for the output archive.

        Returns
        -------
        None
        --------
        - If `self.out` is already set, does nothing.
        - Otherwise sets `self.out` to `name + ".nii.zarr"` if NIfTI mode
          is active, or `name + ".ome.zarr"` otherwise.
        - If the resulting path exists and `overwrite` is False, prompts
          the user for confirmation and raises FileExistsError if not confirmed.
        """
        if self.out is not None:
            return
        self.out = name
        self.out += ".nii.zarr" if self.nii else ".ome.zarr"


def update_default_config(
    zarr_config: ZarrConfig | None, **kwargs: Unpack[ZarrConfig]
) -> ZarrConfig:
    """
    Merge user overrides into an existing ZarrConfig or create a new one.

    Parameters
    ----------
    zarr_config : ZarrConfig or None
        Existing configuration to update. If None, a fresh ZarrConfig is created.
    **kwargs
        Fields of ZarrConfig to override in the returned instance.

    Returns
    -------
    ZarrConfig
        A new ZarrConfig instance with the given overrides applied.
    """
    if zarr_config is None:
        zarr_config = ZarrConfig()
    return replace(zarr_config, **kwargs)
