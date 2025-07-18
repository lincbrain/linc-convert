"""Abstract base classes for ZarrIO interfaces."""

from abc import ABC, ABCMeta, abstractmethod
from numbers import Number
from os import PathLike
from typing import Any, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import niizarr
import numpy as np
import tqdm
import zarr
from dask import array as da
from dask.diagnostics import ProgressBar
from nibabel import Nifti1Header, Nifti2Header
from numpy.typing import ArrayLike, DTypeLike

from linc_convert.utils.io.generate_pyramid import (
    compute_next_level,
    default_levels,
    next_level_shape,
)
from linc_convert.utils.io.zarr import logger
from linc_convert.utils.zarr_config import ZarrConfig

NiftiHeaderLike = Union[Nifti1Header, Nifti2Header]


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

    @abstractmethod
    def __repr__(self) -> str:
        """Return the string representation of the Zarr node."""

    @abstractmethod
    def __str__(self) -> str:
        """Return the string representation of the Zarr node."""


class ZarrArray(ZarrNode):
    """Abstract interface for a Zarr array (n-dimensional data)."""

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """Shape of the array."""

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Data type of the array."""

    @property
    @abstractmethod
    def chunks(self) -> Tuple[int, ...]:
        """Chunk shape for the array."""

    @property
    @abstractmethod
    def shards(self) -> Tuple[int, ...] | None:
        """Shard shape, if supported; otherwise None."""

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Number of dimensions of the array."""

    @abstractmethod
    def __getitem__(self, key: str) -> ArrayLike:
        """Read data from the array."""

    @abstractmethod
    def __setitem__(self, key: str, value: ArrayLike | Number) -> None:
        """Write data to the array."""


class ZarrGroup(ZarrNode, metaclass=ABCMeta):
    """Abstract interface for a Zarr group (container of arrays/subgroups)."""

    @classmethod
    @abstractmethod
    def from_config(cls, zarr_config: ZarrConfig) -> "ZarrGroup":
        """Create a Zarr group from a configuration object."""

    @abstractmethod
    def __getitem__(self, name: str) -> ZarrNode:
        """Get a subgroup or array by name within this group."""

    @abstractmethod
    def __delitem__(self, name: str) -> None:
        """Delete a subgroup or array by name within this group."""

    @abstractmethod
    def create_group(self, name: str, overwrite: bool = False) -> "ZarrGroup":
        """Create or open a subgroup within this group."""

    @abstractmethod
    def create_array(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: DTypeLike,
        *,
        zarr_config: ZarrConfig = None,
        **kwargs,
    ) -> ZarrArray:
        """Create a new array within this group."""
        ...

    @abstractmethod
    def create_array_from_base(
        self, name: str, shape: Sequence[int], data: ArrayLike = None, **kwargs
    ) -> ZarrArray:
        """Create a new array using metadata of an existing base-level array."""

    def generate_pyramid(
        self,
        levels: int = -1,
        ndim: int = 3,
        mode: Literal["mean", "median"] = "median",
        no_pyramid_axis: Optional[int] = None,
    ) -> list[list[int]]:
        """
        Generate the levels of a pyramid in an existing Zarr.

        Parameters
        ----------
        omz : zarr.Group
            Zarr group object
        levels : int
            Number of additional levels to generate.
            By default, stop when all dimensions are smaller than their
            corresponding chunk size.
        ndim : int
            Number of spatial dimensions.
        mode : {"mean", "median"}
            Whether to use a mean or median moving window.
        no_pyramid_axis : int | None
            Axis that should not be downsampled. If None, downsample
            across all three dimensions.

        Returns
        -------
        shapes : list[list[int]]
            Shapes of all levels, from finest to coarsest, including the
            existing top level.
        """
        base_level = self["0"]
        base_shape = list(base_level.shape)
        chunk_size = base_level.chunks

        window_func = {"median": da.median, "mean": da.mean}[mode]

        batch_shape, spatial_shape = base_shape[:-ndim], base_shape[-ndim:]
        all_shapes = [spatial_shape]

        if levels == -1:
            levels = default_levels(spatial_shape, chunk_size[-ndim:], no_pyramid_axis)

        for lvl in tqdm.tqdm(range(1, levels + 1)):
            prev_shape = spatial_shape
            spatial_shape = next_level_shape(prev_shape, no_pyramid_axis)
            all_shapes.append(spatial_shape)
            logger.info(f"Compute level {lvl} with shape {spatial_shape}")
            arr = self.create_array_from_base(
                str(lvl), shape=batch_shape + spatial_shape
            )
            # dat = da.from_zarr(self[str(lvl - 1)]._array)
            dat = da.from_array(self[str(lvl - 1)], chunks=self[str(lvl - 1)].chunks)
            dat = compute_next_level(dat, ndim, no_pyramid_axis, window_func)
            if arr.shards:
                dat = dat.rechunk(arr.shards)
            else:
                dat = dat.rechunk(arr.chunks)
            dat = dat.persist()
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

    def write_nifti_header(self, header: Union[Nifti1Header, Nifti2Header]) -> None:
        """Write a NIfTI header to the Zarr group."""
        niizarr.write_nifti_header(self._get_zarr_python_group(), header)

    @abstractmethod
    def _get_zarr_python_group(self) -> "zarr.ZarrGroup":
        """Get the underlying Zarr Python group object."""
