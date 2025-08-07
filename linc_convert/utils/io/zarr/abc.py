"""Abstract base classes for ZarrIO interfaces."""

import logging
from abc import ABC, abstractmethod
from numbers import Number
from os import PathLike
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
    Unpack,
)

import niizarr
import numpy as np
import tqdm
import zarr
from dask import array as da
from dask.diagnostics import ProgressBar
from nibabel import Nifti1Header, Nifti2Header
from numpy.typing import ArrayLike, DTypeLike

from linc_convert.utils.io.zarr.generate_pyramid import (
    compute_next_level,
    default_levels,
    next_level_shape,
)
from linc_convert.utils.zarr_config import ZarrConfig

NiftiHeaderLike = Union[Nifti1Header, Nifti2Header]


class ZarrArrayConfig(TypedDict):
    """Configuration for creating a Zarr Array."""

    chunks: tuple[int, ...]
    shards: Optional[tuple[int, ...]]
    compressors: Literal["blosc", "zlib", None]
    compressor_options: dict[str, Any]
    dimension_separator: Literal[".", "/"]
    order: Literal["C", "F"]
    fill_value: Number | None


class ZarrNode(ABC):
    """Base class for any Zarr-like object (group or array)."""

    def __init__(self, store_path: Union[str, PathLike]) -> None:
        self._store_path = store_path

    @property
    def store_path(self) -> Union[str, PathLike]:
        """Path to the Zarr store for this node."""
        return self._store_path

    @property
    @abstractmethod
    def attrs(self) -> Mapping[str, Any]:
        """Access metadata/attributes for this node."""
        ...

    @property
    @abstractmethod
    def zarr_version(self) -> Literal[2, 3]:
        """Get the Zarr format version."""
        ...

    # @abstractmethod
    # def __repr__(self) -> str:
    #     """Return the string representation of the Zarr node."""
    #     ...

    # @abstractmethod
    # def __str__(self) -> str:
    #     """Return the string representation of the Zarr node."""
    #     ...


class ZarrArray(ZarrNode):
    """Abstract interface for a Zarr array (n-dimensional data)."""

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Number of dimensions of the array."""
        ...

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """Shape of the array."""
        ...

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Data type of the array."""
        ...

    @property
    @abstractmethod
    def chunks(self) -> Tuple[int, ...]:
        """Chunk shape for the array."""
        ...

    @property
    @abstractmethod
    def shards(self) -> Tuple[int, ...] | None:
        """Shard shape, if supported; otherwise None."""
        ...

    @abstractmethod
    def __getitem__(self, key: str) -> ArrayLike:
        """Read data from the array."""
        ...

    @abstractmethod
    def __setitem__(self, key: str, value: ArrayLike | Number) -> None:
        """Write data to the array."""
        ...


class ZarrGroup(ZarrNode):
    """Abstract interface for a Zarr group (container of arrays/subgroups)."""

    @classmethod
    @abstractmethod
    def from_config(cls, zarr_config: ZarrConfig) -> "ZarrGroup":
        """Create a Zarr group from a configuration object."""
        ...

    @abstractmethod
    def __getitem__(self, key: str) -> ZarrNode:
        """Get a subgroup or array by name within this group."""
        ...

    @abstractmethod
    def __setitem__(self, key: str, value: ZarrNode) -> None:
        """Set a subgroup or array by name within this group."""

    @abstractmethod
    def __delitem__(self, key: str) -> None:
        """Delete a subgroup or array by name within this group."""
        ...

    @abstractmethod
    def _get_zarr_python_group(self) -> zarr.Group:
        """Get the underlying Zarr Python group object."""
        ...

    # @abstractmethod
    # def _create_array(self, **kwargs: Unpack[ZarrArrayConfig]) -> ArrayLike:
    #     ...

    @abstractmethod
    def create_group(self, name: str, overwrite: bool = False) -> "ZarrGroup":
        """Create or open a subgroup within this group."""
        ...

    @abstractmethod
    def create_array(
        self,
        name: str,
        shape: Sequence[int],
        dtype: DTypeLike,
        *,
        zarr_config: ZarrConfig = None,
        **kwargs: Unpack[ZarrArrayConfig],
    ) -> ZarrArray:
        """Create a new array within this group."""
        ...

    @abstractmethod
    def create_array_from_base(
        self,
        name: str,
        shape: Sequence[int],
        data: ArrayLike = None,
        **kwargs: Unpack[ZarrArrayConfig],
    ) -> ZarrArray:
        """Create a new array using metadata of an existing base-level array."""
        ...

    def generate_pyramid(
        self,
        levels: int = -1,
        ndim: int = 3,
        mode: Literal["mean", "median"] | Callable = "mean",
        no_pyramid_axis: Optional[int] = None,
    ) -> list[list[int]]:
        """
        Generate the levels of a pyramid in an existing Zarr.

        Parameters
        ----------
        levels : int
            Number of additional levels to generate. By default, stop when
            all dimensions are smaller than their corresponding chunk size.
        ndim : int
            Number of spatial dimensions.
        mode : {"mean", "median"}
            Function to be used for down-sampling, either a callable or mean or median.
        no_pyramid_axis : int | None
            Axis to leave unsampled.

        Returns
        -------
        shapes : list[list[int]]
            Shapes of each level, from finest to coarsest.
        """
        logger = logging.getLogger(__name__)
        base = self["0"]
        batch_shape, spatial_shape = base.shape[:-ndim], base.shape[-ndim:]
        all_shapes = [spatial_shape]
        chunk_size = base.chunks[-ndim:]
        if isinstance(mode, Callable):
            window = mode
        else:
            window_func = {"median": da.median, "mean": da.mean}
            if mode not in window_func:
                raise ValueError(f"Unsupported mode: {mode}")
            window = window_func[mode]

        if levels == -1:
            levels = default_levels(spatial_shape, chunk_size, no_pyramid_axis)

        for lvl in tqdm.tqdm(range(1, levels + 1)):
            spatial_shape = next_level_shape(spatial_shape, no_pyramid_axis)
            all_shapes.append(spatial_shape)
            logger.info(f"Compute level {lvl} with shape {spatial_shape}")
            arr = self.create_array_from_base(str(lvl), (*batch_shape, *spatial_shape))
            dat = da.from_array(self[str(lvl - 1)], chunks=self[str(lvl - 1)].chunks)
            dat = compute_next_level(dat, ndim, no_pyramid_axis, window)
            dat = dat.rechunk(arr.shards or arr.chunks).persist()
            with ProgressBar():
                dat.store(arr)

        return all_shapes

    def write_ome_metadata(
        self,
        axes: List[str],
        space_scale: Union[float, List[float]] = 1.0,
        time_scale: float = 1.0,
        space_unit: str = "micrometer",
        time_unit: str = "second",
        name: str = "",
        pyramid_aligns: Union[str, int, List[str], List[int]] = 2,
        levels: Optional[int] = None,
        no_pool: Optional[int] = None,
        multiscales_type: str = "",
        ome_version: Literal["0.4", "0.5"] = "0.4",
    ) -> None:
        """
        Write OME-compatible metadata into this group.

        Parameters
        ----------
        axes : list[str]
            Name of each dimension, in Zarr order (t, c, z, y, x)
        space_scale : float | list[float]
            Finest-level voxel size, in Zarr order (z, y, x)
        time_scale : float
            Time scale
        space_unit : str
            Unit of spatial scale (assumed identical across dimensions)
        time_unit : str
            Unit of timescale
        name : str
            Name attribute
        pyramid_aligns : float | list[float] | {"center", "edge"}
            Whether the pyramid construction aligns the edges or the centers
            of the corner voxels. If a (list of) number, assume that a moving
            window of that size was used.
        levels : int
            Number of existing levels. Default: find out automatically.
        no_pool: int
            Index of the spatial dimension that was not down-sampled
            when generating pyramid levels.
        multiscales_type: str
            Override the type field in multiscale attribute.
        ome_version: {"0.4", "0.5"}
            Version of the OME-Zarr specification to use

        Returns
        -------
        None.
        """
        niizarr.write_ome_metadata(
            self._get_zarr_python_group(),
            space_scale=space_scale,
            time_scale=time_scale,
            space_unit=space_unit,
            time_unit=time_unit,
            axes=axes,
            name=name,
            pyramid_aligns=pyramid_aligns,
            levels=levels,
            no_pool=no_pool,
            multiscales_type=multiscales_type,
            ome_version=ome_version,
        )

    def write_nifti_header(self, header: NiftiHeaderLike) -> None:
        """Write a NIfTI header to the Zarr group."""
        niizarr.write_nifti_header(self._get_zarr_python_group(), header)
