"""Configuration related to output Zarr Archive."""
import inspect
from dataclasses import dataclass, field, fields, is_dataclass, replace
from functools import wraps
from os import PathLike
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Literal,
    Type,
    TypeAlias,
    Union,
    get_args,
    get_origin,
)

from cyclopts import Parameter

DriverLike: TypeAlias = Literal["zarr-python", "tensorstore", "zarrita"]

@Parameter(name="*")
@dataclass
class ZarrConfig:
    """
    Configuration related to output Zarr Archive.

    Parameters
    ----------
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
    overwrite
        when no name is supplied and using default output name, if overwrite is set,
        it won't ask if overwrite
    driver : {"zarr-python", "tensorstore", "zarrita"}
        library used for Zarr IO Operation
    """

    zarr_version: Literal[2, 3] = 3
    chunk: tuple[int] = (128,)
    chunk_channels: bool = False
    chunk_time: bool = True
    shard: tuple[int] | Literal["auto"] | None = None
    shard_channels: bool = False
    shard_time: bool = False
    dimension_separator: Literal[".", "/"] = "/"
    order: Literal["C", "F"] = "C"
    compressor: Literal["blosc", "zlib", "none"] = "blosc"
    compressor_opt: dict[str, float | str] = field(default_factory=dict)
    no_time: bool = False
    no_pyramid_axis: Literal["x", "y", "z"] | None = None
    levels: int = -1
    ome_version: Literal["0.4", "0.5"] = "0.4"
    overwrite: bool = False
    driver: DriverLike = "zarr-python"

    def __post_init__(self) -> None:
        """
        Perform post-initialization checks and adjustments.

        - Ensure that sharding options (shard, shard_channels, shard_time) are only
          used when zarr_version == 3; otherwise raise NotImplementedError.
        """
        if self.zarr_version == 2:
            if self.shard or self.shard_channels or self.shard_time:
                raise ValueError("Shard is not supported for Zarr 2.")


@Parameter(name="*")
@dataclass
class GeneralConfig:
    """
    General configuration for the conversion process.

    Parameters
    ----------
    out : str | None
        Output path for the converted data.
    max_load : int
        Maximum number of items to load into memory at once.
    log_level : {"debug", "info", "warning", "error", "critical"}
        Logging level for the conversion process.
    verbose : bool
        If True, set log_level to "debug".
    """

    out: Annotated[str, Parameter(name=["--out", "-o"])] = None
    max_load: int = 1024
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    verbose: Annotated[bool, Parameter(name=["--verbose", "-v"])] = False

    def __post_init__(self) -> None:
        """
        Perform post-initialization checks and adjustments.

        - Ensure that max_load is a positive integer; otherwise raise ValueError.
        - If verbose is True, set log_level to "debug".
        """
        if not isinstance(self.max_load, int) or self.max_load <= 0:
            raise ValueError("max_load must be a positive integer")
        if self.verbose:
            self.log_level = "debug"

    def set_default_name(self, name: str, nii: bool = True) -> None:
        """
        Assign a default output name if none was specified.

        Parameters
        ----------
        name : str
            Base filename (without extension) to use for the output archive.
        nii : bool
            If True, the output will be in NIfTI-Zarr format (with `.nii.zarr` extension).
            If False, the output will be in OME-Zarr format (with `.ome.zarr`
            extension).

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
        self.out += ".nii.zarr" if nii else ".ome.zarr"


@Parameter(name="*")
@dataclass
class NiftiConfig:
    """
    Configuration related to output nifti-zarr.

    Parameters
    ----------
    nii
        Convert the output to nifti-zarr format.
        This is automatically enabled if the output path ends with ".nii.zarr".
    orientation
        Orientation of the slice
    center
        Set RAS[0, 0, 0] at FOV center
    nifti_header
        Path to the nifti header file to use.
        Can be a .nii[.gz] file, a binary header file, or a .nii.zarr archive.

    Orientation
    -----------
    The anatomical orientation of the slice is given in terms of RAS axes.

    It is a combination of two letters from the set
    `{"L", "R", "A", "P", "I", "S"}`, where

    * the first letter corresponds to the horizontal dimension and
        indicates the anatomical meaning of the _right_ of the input image,
    * the second letter corresponds to the vertical dimension and
        indicates the anatomical meaning of the _bottom_ of the input image.
     * the third letter corresponds to the slice dimension and
      indicates the anatomical meaning of the _end_ of the volume.

    We also provide the aliases

    * `"coronal"` == `"LI"`
    * `"axial"` == `"LP"`
    * `"sagittal"` == `"PI"`

    When the aliases are used, the third dimension will be popped from "RAS"
    """

    nii: bool = True
    orientation: str = "RAS"
    center: bool = True
    nifti_header: PathLike[str] | str | None = None

    def __post_init__(self) -> None:
        """Perform post-initialization checks and adjustments."""
        if self.nifti_header:
            self.nii = True


def autoconfig(func: Callable) -> Callable:
    """Use as @autoconfig only. Infers config params from dataclass annotations."""

    def _unwrap_dataclass_type(tp: dataclass) -> dataclass:
        # Handle Annotated[T, ...]
        if get_origin(tp) is Annotated:
            tp = get_args(tp)[0]
        # Handle Optional[T] / Union[T, None]
        if get_origin(tp) is Union:
            cand = [t for t in get_args(tp) if t is not type(None)]  # noqa: E721
            if len(cand) == 1:
                tp = cand[0]
        return tp if isinstance(tp, type) and is_dataclass(tp) else None

    sig = inspect.signature(func)

    # Build the config map by reading dataclass-typed parameters
    config_map: Dict[str, Type] = {}
    for name, p in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        dt = _unwrap_dataclass_type(p.annotation)
        if dt is not None:
            config_map[name] = dt

    # Collision check across config field names
    field_owner: Dict[str, str] = {}
    for pname, cls in config_map.items():
        if not is_dataclass(cls):
            raise TypeError(f"{pname} -> {cls} must be a dataclass type")
        for f in fields(cls):
            if f.name in field_owner:
                other = field_owner[f.name]
                raise ValueError(
                    f"Field '{f.name}' appears in both '{other}' and '{pname}'. "
                    "Rename to avoid ambiguity."
                )
            field_owner[f.name] = pname

    func_params = sig.parameters
    accepted_names = {
        n for n, p in func_params.items()
        if p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
           and n not in config_map
    }
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in func_params.values())

    @wraps(func)
    def wrapper(*args, **kwargs) -> Callable:  # noqa: ANN002, ANN003
        # Start from explicit instances (if provided)
        cfg_instances: Dict[str, Any] = {}
        for pname, cls in config_map.items():
            inst = kwargs.pop(pname, None)
            if inst is not None and not is_dataclass(inst):
                raise TypeError(
                    f"Parameter '{pname}' must be a {cls.__name__} instance")
            cfg_instances[pname] = inst or cls()

        # Route flat kwargs by field name into the right config
        consumed = set()
        for k in list(kwargs.keys()):
            owner = field_owner.get(k)
            if owner is not None:
                inst = cfg_instances[owner]
                cfg_instances[owner] = replace(inst, **{k: kwargs[k]})
                consumed.add(k)

        # Leftovers: allow real function params (e.g., key, meta) or **kwargs
        leftovers = {k: v for k, v in kwargs.items() if k not in consumed}
        if not accepts_var_kw:
            unknown = [k for k in leftovers.keys() if k not in accepted_names]
            if unknown:
                raise TypeError(f"Unknown options: {', '.join(sorted(unknown))}")

        return func(*args, **cfg_instances, **leftovers)

    return wrapper
