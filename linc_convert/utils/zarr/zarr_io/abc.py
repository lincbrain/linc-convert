from abc import ABC, abstractmethod, ABCMeta
from os import PathLike
from typing import Union, Tuple, Optional, Any, Literal

import numpy as np
import tqdm
from dask import array as da
from dask.diagnostics import ProgressBar

from linc_convert.utils.zarr import ZarrConfig
from linc_convert.utils.zarr.generate_pyramid import default_levels, next_level_shape, \
    compute_next_level
from linc_convert.utils.zarr.zarr_io import logger


class ZarrNode(ABC):
    """
    Base class for any Zarr-like object (group or array).
    """

    def __init__(self, store_path: Union[str, PathLike]):
        # record where on disk (or out‐of core store) this node lives
        self._store_path = store_path

    @property
    def store_path(self) -> str:
        return self._store_path

    @property
    @abstractmethod
    def attrs(self):
        """Access metadata/attributes for this node."""
        ...

    @property
    @abstractmethod
    def zarr_version(self) -> int:
        """
        Return the Zarr format version (e.g., 2 or 3).
        """
        ...


class ZarrArray(ZarrNode):
    """
    Abstract interface for a Zarr array (n-dimensional data).
    """

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of the array.
        """
        ...

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """
        Data type of the array.
        """
        ...

    @property
    @abstractmethod
    def chunks(self) -> Tuple[int, ...]:
        """
        Chunk shape for the array.
        """
        ...

    @property
    @abstractmethod
    def shards(self) -> Optional[Tuple[int, ...]]:
        """
        Shard shape, if supported; otherwise None.
        """
        ...

    @property
    @abstractmethod
    def ndim(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, key: Any) -> Any:
        """
        Read data from the array.
        """
        ...

    @abstractmethod
    def __setitem__(self, key: Any, value: Any) -> None:
        """
        Write data to the array.
        """
        ...


class ZarrGroup(ZarrNode, metaclass=ABCMeta):
    """
    Abstract interface for a Zarr group (container of arrays and/or subgroups).
    """

    @classmethod
    @abstractmethod
    def from_config(cls, zarr_config: ZarrConfig) -> 'ZarrGroup':
        ...

    @abstractmethod
    def __getitem__(self, name: str) -> ZarrNode:
        """
        Get a subgroup or array by name within this group.
        """
        ...

    @abstractmethod
    def __delitem__(self, key):
        ...

    @abstractmethod
    def create_group(self, name: str, overwrite: bool = False) -> 'ZarrGroup':
        """
        Create or open a subgroup within this group.
        """
        ...

    @abstractmethod
    def create_array(
            self,
            name: str,
            shape: Tuple[int, ...],
            **kwargs
    ) -> 'ZarrArray':
        """
        Create a new array within this group.
        """
        ...

    @abstractmethod
    def create_array_from_base(
            self,
            name: str,
            shape: Tuple[int, ...],
            data: Optional[Any] = None,
            **kwargs
    ) -> 'ZarrArray':
        """
        Create a new array using metadata of an existing base-level array.
        """
        ...

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

        base_level = self['0']
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
                str(lvl), shape=batch_shape + spatial_shape)
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
