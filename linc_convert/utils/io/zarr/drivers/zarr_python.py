"""ZarrIO Implementation using the zarr-python library."""

import gc
import json
import logging
from math import ceil
from numbers import Number
from os import PathLike
import os
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    Unpack,
)
from urllib.parse import urlparse

import dask.array as da
import numpy as np
import tqdm
import zarr
from dask.diagnostics import ProgressBar
from numpy.typing import ArrayLike, DTypeLike
from zarr.core.array import CompressorsLike
from zarr.core.chunk_key_encodings import ChunkKeyEncodingLike, ChunkKeyEncodingParams

from linc_convert.utils.io.zarr.abc import ZarrArray, ZarrArrayConfig, ZarrGroup
from linc_convert.utils.io.zarr.dandi import open_dandi_zarr_group
from linc_convert.utils.io.zarr.generate_pyramid import (
    compute_next_level,
    default_levels,
    next_level_shape,
)
from linc_convert.utils.io.zarr.helpers import _compute_zarr_layout
from linc_convert.utils.zarr_config import GeneralConfig, ZarrConfig


class ZarrPythonArray(ZarrArray):
    """Zarr Array implementation using the zarr-python library."""

    def __init__(self, array: zarr.Array) -> None:
        """
        Initialize the ZarrPythonArray with a zarr.Array.

        Parameters
        ----------
        array : zarr.Array
            Underlying Zarr array.
        """
        super().__init__(str(array.store_path))
        self._array = array

    @property
    def ndim(self) -> int:
        """Number of dimensions of the array."""
        return self._array.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the array."""
        return self._array.shape

    @property
    def dtype(self) -> np.dtype:
        """Data type of the array."""
        return self._array.dtype

    @property
    def chunks(self) -> Tuple[int, ...]:
        """Chunk shape for the array."""
        return self._array.chunks

    @property
    def shards(self) -> Optional[Tuple[int, ...]]:
        """Shard shape, if supported; otherwise None."""
        return getattr(self._array, "shards", None)

    @property
    def attrs(self) -> Mapping[str, Any]:
        """Access metadata/attributes for this node."""
        return self._array.attrs

    @property
    def zarr_version(self) -> int:
        """Get the Zarr format version."""
        return self._array.metadata.zarr_format

    def __getitem__(self, key: str) -> ArrayLike:
        """Read data from the array."""
        return self._array[key]

    def __setitem__(self, key: str, value: ArrayLike | Number) -> None:
        """Write data to the array."""
        self._array[key] = value

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Delegate any unknown attributes to the underlying array."""
        if name == "_array":
            return self._array
        if hasattr(self._array, name):
            return getattr(self._array, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    @classmethod
    def open(cls, *args: Any, **kwargs: Any) -> "ZarrPythonArray":  # noqa: ANN401
        """Open a Zarr array."""
        return cls(zarr.open_array(*args, **kwargs))

    @classmethod
    def open_array(cls, *args: Any, **kwargs: Any) -> "ZarrPythonArray":  # noqa: ANN401
        """Open a Zarr array."""
        return cls(zarr.open_array(*args, **kwargs))


class ZarrPythonGroup(ZarrGroup):
    """Zarr Group implementation using the zarr-python library."""

    def __init__(self, zarr_group: zarr.Group) -> None:
        """
        Initialize the ZarrPythonGroup with a zarr.Group.

        Parameters
        ----------
        zarr_group : zarr.Group
            Underlying Zarr Python group.
        """
        super().__init__(str(zarr_group.store_path))

        self._zgroup = zarr_group

    @classmethod
    def from_config(
        cls, out: str | PathLike[str], zarr_config: ZarrConfig
    ) -> "ZarrPythonGroup":
        """Create a Zarr group from a configuration object."""
        store = zarr.storage.LocalStore(out)
        return cls(
            zarr.group(
                store=store,
                overwrite=zarr_config.overwrite,
                zarr_format=zarr_config.zarr_version,
            )
        )

    @property
    def attrs(self) -> Mapping[str, Any]:
        """Access metadata/attributes for this node."""
        return self._zgroup.attrs

    @property
    def zarr_version(self) -> Literal[2, 3]:
        """Get the Zarr format version."""
        return self._zgroup.metadata.zarr_format

    def keys(self) -> Iterator[str]:
        """Get the names of all subgroups and arrays in this group."""
        yield from self._zgroup.keys()

    def __getitem__(self, key: str) -> Union[ZarrPythonArray, "ZarrPythonGroup"]:
        """Get a subgroup or array by name within this group."""
        if key not in self._zgroup:
            raise KeyError(
                f"Key '{key}' not found in group '{self.store_path}'")
        item = self._zgroup[key]
        if isinstance(item, zarr.Group):
            return ZarrPythonGroup(item)
        elif isinstance(item, zarr.Array):
            return ZarrPythonArray(item)
        else:
            raise TypeError(f"Unsupported item type: {type(item)}")

    def __setitem__(
        self, key: str, value: Union[ZarrPythonArray, "ZarrPythonGroup"]
    ) -> None:
        """Set a subgroup or array by name within this group."""
        if isinstance(value, ZarrPythonGroup):
            self._zgroup[key] = value._zgroup
        elif isinstance(value, ZarrPythonArray):
            self._zgroup[key] = value._array
        else:
            raise TypeError(f"Unsupported item type: {type(value)}")

    def __delitem__(self, key: str) -> None:
        """Delete a subgroup or array by name within this group."""
        del self._zgroup[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the names of all subgroups and arrays in this group."""
        yield from self.keys()

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Delegate attribute access to the underlying Zarr group."""
        return getattr(self._zgroup, name)

    def create_group(self, name: str, overwrite: bool = False) -> "ZarrPythonGroup":
        """Create or open a subgroup within this group."""
        subgroup = self._zgroup.create_group(name, overwrite=overwrite)
        return ZarrPythonGroup(subgroup)

    def create_array(
        self,
        name: str,
        shape: Sequence[int],
        dtype: DTypeLike,
        *,
        zarr_config: ZarrConfig = None,
        data: Optional[ArrayLike] = None,
        **kwargs: Unpack[ZarrArrayConfig],
    ) -> ZarrPythonArray:
        """Create a new array within this group."""
        if zarr_config is None:
            arr = self._zgroup.create_array(
                name, shape=shape, dtype=dtype, **kwargs)
            if data is not None:
                if type(data) is da.Array:
                    with ProgressBar():
                        da.to_zarr(data, arr)
                else:
                    arr[:] = data
            return ZarrPythonArray(arr)

        compressor = zarr_config.compressor
        compressor_opt = zarr_config.compressor_opt
        chunk, shard = _compute_zarr_layout(shape, dtype, zarr_config)
        # TODO: implement fill_value
        opt = {
            "chunks": chunk,
            "shards": shard,
            "order": zarr_config.order,
            "dtype": np.dtype(dtype).str,
            "fill_value": None,
            "compressors": _make_compressor(
                compressor, zarr_config.zarr_version, **compressor_opt
            ),
            "overwrite": zarr_config.overwrite
        }

        chunk_key_encoding = _dimension_separator_to_chunk_key_encoding(
            zarr_config.dimension_separator, zarr_config.zarr_version
        )
        if chunk_key_encoding:
            opt["chunk_key_encoding"] = chunk_key_encoding
        arr = self._zgroup.create_array(name=name, shape=shape, **opt)
        if data is not None:
            if type(data) is da.Array:
                with ProgressBar():
                    da.to_zarr(data, arr)
            else:
                arr[:] = data
        return ZarrPythonArray(arr)

    def create_array_from_base(
        self,
        name: str,
        shape: Sequence[int],
        data: ArrayLike = None,
        **kwargs: Unpack[ZarrArrayConfig],
    ) -> ZarrPythonArray:
        """Create a new array using the properties from a base_level object."""
        # this is very hacky, otherwise the inherited class will use their override
        base_level = ZarrPythonGroup.__getitem__(self, "0")
        opts = dict(
            dtype=base_level.dtype,
            chunks=base_level.chunks,
            shards=getattr(base_level, "shards", None),
            filters=getattr(base_level._array, "filters", None),
            compressors=getattr(base_level._array, "compressors", None),
            fill_value=getattr(base_level._array, "fill_value", None),
            order=getattr(base_level._array, "order", None),
            attributes=getattr(
                getattr(base_level._array, "metadata",
                        None), "attributes", None
            ),
            overwrite=True,
        )
        # Handle extra options based on metadata type
        meta = getattr(base_level, "metadata", None)
        if meta is not None:
            if hasattr(meta, "dimension_separator"):
                opts["chunk_key_encoding"] = _dimension_separator_to_chunk_key_encoding(
                    meta.dimension_separator, 2
                )
            if hasattr(meta, "chunk_key_encoding"):
                opts["chunk_key_encoding"] = getattr(
                    meta, "chunk_key_encoding", None)
            if hasattr(base_level, "serializer"):
                opts["serializer"] = getattr(base_level, "serializer", None)
            if hasattr(meta, "dimension_names"):
                opts["dimension_names"] = getattr(
                    meta, "dimension_names", None)
        # Remove None values
        opts = {k: v for k, v in opts.items() if v is not None}
        opts.update(kwargs)
        arr = self._zgroup.create_array(name=name, shape=shape, **opts)
        if data is not None:
            arr[:] = data
        return ZarrPythonArray(arr)

    @classmethod
    def open(cls, *args: Any, **kwargs: Any) -> "ZarrPythonGroup":  # noqa: ANN401
        """Open a Zarr group."""
        return cls(zarr.open_group(*args, **kwargs))

    @classmethod
    def open_group(cls, *args: Any, **kwargs: Any) -> "ZarrPythonGroup":  # noqa: ANN401
        """Open a Zarr group."""
        return cls(zarr.open_group(*args, **kwargs))

    def _read_pyramid_checkpoint(self, checkpoint_file: str) -> dict:
        """Load pyramid-generation checkpoint state, or return a fresh state
        if no checkpoint file exists yet (or it can't be parsed).
        """
        default_state = {
            "done_x_windows": [],
            "staged_fallback_done": False,
            "remaining_levels_done": False,
        }
        try:
            with open(checkpoint_file, "r") as f:
                state = json.load(f)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            return default_state

        # Be defensive about a partially-written or older-format checkpoint.
        for key, default_value in default_state.items():
            state.setdefault(key, default_value)
        return state

    def _write_pyramid_checkpoint(self, checkpoint_file: str, state: dict) -> None:
        """Persist pyramid-generation checkpoint state."""
        with open(checkpoint_file, "w") as f:
            json.dump(state, f)

    def generate_pyramid_staged(
        self,
        levels: int = -1,
        ndim: int = 3,
        mode: Literal["mean", "median"] | Callable = "mean",
        no_pyramid_axis: Optional[int] = None,
        copy_config: Optional[GeneralConfig] = None,
        copy_zarr_config: Optional[ZarrConfig] = None,
        checkpoint_file: Optional[str] = None,
    ) -> list[list[int]]:
        """
        Generate a pyramid, writing the first two levels 4 x-chunks at a
        time and the remaining levels in a single pass each.

        Levels 1 and 2 are written in windows of 4 chunks along x (using
        `generate_pyramid`'s region-by-region path via `copy_config`/
        `copy_zarr_config`, `x_min`, and `x_max`), one window at a time,
        covering both levels per window. Any remaining levels (3 and
        beyond) are then generated all at once, with no x-windowing.

        If interrupted and re-run, this resumes from a checkpoint file:
        x-windows already written are skipped, and either of the two
        single-shot passes (the no-`copy_config` levels-1-2 fallback, and
        the final "remaining levels" pass) are skipped if they already
        completed. Skipping a completed unit of work here is safe because
        `create_array`/`create_array_from_base` no longer overwrite an array
        that already exists -- the checkpoint only avoids redoing work, it
        isn't needed for correctness.

        Parameters
        ----------
        levels : int
            Total number of pyramid levels to generate. By default, stop
            when all dimensions are smaller than their corresponding
            chunk size.
        ndim : int
            Number of spatial dimensions.
        mode : {"mean", "median"}
            Function to be used for down-sampling, either a callable or
            mean or median.
        no_pyramid_axis : int | None
            Axis to leave unsampled.
        copy_config : GeneralConfig, optional
            Output configuration used for the windowed (levels 1-2) path.
            Required to actually get x-chunked writes; if omitted, levels
            1-2 fall back to a single full-width call.
        copy_zarr_config : ZarrConfig, optional
            Zarr configuration paired with `copy_config`.
        checkpoint_file : str, optional
            Path to a checkpoint file used to resume an interrupted run.
            Defaults to a file derived from this group's own store path
            if not given.

        Returns
        -------
        shapes : list[list[int]]
            Shapes of each level, from finest to coarsest.
        """
        logger = logging.getLogger("PyramidGenerationStaged")
        base = self["0"]
        chunk_size = base.chunks[-ndim:]
        x_chunk = chunk_size[2]
        full_x = base.shape[-1]

        if levels == -1:
            levels = 10

        staged_levels = min(2, levels)
        spatial_shape = list(base.shape[-ndim:])
        all_shapes: list[list[int]] = [spatial_shape]
        any_work_done = False

        if checkpoint_file is None:
            store_path = urlparse(str(self.store_path)
                                  ).path or str(self.store_path)
            checkpoint_file = f"{store_path.rstrip('/')}_pyramid.checkpoint"

        state = self._read_pyramid_checkpoint(checkpoint_file)

        if copy_config is not None and staged_levels > 0:
            window = 64 * x_chunk
            for x_min in range(0, full_x, window):
                x_max = min(full_x, x_min + window)

                if x_min in state["done_x_windows"]:
                    logger.info(
                        f"Skipping x-window {x_min}-{x_max} "
                        f"(already completed per checkpoint)"
                    )
                    continue

                all_shapes = self.generate_pyramid(
                    levels=staged_levels,
                    ndim=ndim,
                    mode=mode,
                    no_pyramid_axis=no_pyramid_axis,
                    copy_config=copy_config,
                    copy_zarr_config=copy_zarr_config,
                    x_min=x_min,
                    x_max=x_max,
                    level_start=1,
                )
                any_work_done = True
                gc.collect()

                state["done_x_windows"].append(x_min)
                self._write_pyramid_checkpoint(checkpoint_file, state)

        elif staged_levels > 0:
            if not state["staged_fallback_done"]:
                # No copy_config provided: can't take the windowed path,
                # so just generate levels 1-2 in one go.
                all_shapes = self.generate_pyramid(
                    levels=staged_levels,
                    ndim=ndim,
                    mode=mode,
                    no_pyramid_axis=no_pyramid_axis,
                    level_start=1,
                )
                any_work_done = True
                state["staged_fallback_done"] = True
                self._write_pyramid_checkpoint(checkpoint_file, state)
            else:
                logger.info(
                    "Skipping levels 1-2 fallback pass "
                    "(already completed per checkpoint)"
                )

        if levels > staged_levels:
            if not state["remaining_levels_done"]:
                all_shapes = self.generate_pyramid(
                    levels=levels,
                    ndim=ndim,
                    mode=mode,
                    no_pyramid_axis=no_pyramid_axis,
                    level_start=staged_levels + 1,
                )
                any_work_done = True
                state["remaining_levels_done"] = True
                self._write_pyramid_checkpoint(checkpoint_file, state)
            else:
                logger.info(
                    "Skipping remaining-levels pass "
                    "(already completed per checkpoint)"
                )

        if not any_work_done:
            # Every unit of work was already done on a previous run, so
            # `all_shapes` was never reassigned from a real call -- rebuild
            # it ourselves using the same per-level shape progression
            # `generate_pyramid` relies on, so the return value is still
            # correct.
            shape = spatial_shape
            all_shapes = [shape]
            for _ in range(1, levels + 1):
                shape = next_level_shape(shape, no_pyramid_axis)
                all_shapes.append(shape)

        return all_shapes

    def generate_pyramid(
        self,
        levels: int = -1,
        ndim: int = 3,
        mode: Literal["mean", "median"] | Callable = "mean",
        no_pyramid_axis: Optional[int] = None,
        copy_config: Optional[GeneralConfig] = None,
        copy_zarr_config: Optional[ZarrConfig] = None,
        x_min: Optional[int] = None,
        x_max: Optional[int] = None,
        level_start: int = 1
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
        overwrite_save = False
        if copy_zarr_config is not None:
            overwrite_save = copy_zarr_config.overwrite
            copy_zarr_config.overwrite = False
        logger = logging.getLogger("PyramidGeneration")
        base = self["0"]
        batch_shape, spatial_shape = base.shape[:-ndim], base.shape[-ndim:]
        all_shapes = [spatial_shape]
        chunk_size = base.chunks[-ndim:]
        if isinstance(mode, Callable):
            window = mode
        else:
            window_func = {"median": da.nanmedian, "mean": da.nanmean}
            if mode not in window_func:
                raise ValueError(f"Unsupported mode: {mode}")
            window = window_func[mode]

        if levels == -1:
            levels = default_levels(spatial_shape, chunk_size, no_pyramid_axis)

        for lvl in range(1, level_start):
            if copy_config is not None:
                x_max = ceil(x_max/2)
                x_min = ceil(x_min/2)
            spatial_shape = next_level_shape(spatial_shape, no_pyramid_axis)

        for lvl in tqdm.tqdm(range(level_start, levels + 1)):
            spatial_shape = next_level_shape(spatial_shape, no_pyramid_axis)
            all_shapes.append(spatial_shape)
            logger.info(f"Compute level {lvl} with shape {spatial_shape}")
            try:
                arr = self.create_array(
                    str(lvl),
                    shape=(*batch_shape, *spatial_shape),
                    zarr_config=copy_zarr_config,
                    dtype=np.uint16,
                )
            except:
                arr = self[str(lvl)]
            dat = da.from_array(self[str(lvl - 1)],
                                chunks=self[str(lvl - 1)].chunks)
            if copy_config is not None:
                x_max = min(dat.shape[2], x_max)
                chunks = list(dat.chunks)
                if isinstance(dat.chunks[2], Tuple):
                    chunks[2] = chunks[2][0]
                if isinstance(dat.chunks[1], Tuple):
                    chunks[1] = chunks[1][0]
                if isinstance(dat.chunks[0], Tuple):
                    chunks[0] = chunks[0][0]
                for x in range(x_min, x_max, chunks[2]*64):
                    for y in range(0, dat.shape[1], chunks[1]*128):
                        for z in range(0, dat.shape[0], chunks[0]*8):
                            logger.info(
                                f"writting pyramid level {lvl}, chunks starting at {z} {y} {x}")
                            x2 = min(x_max, x + chunks[2]*64)
                            y2 = min(dat.shape[1], y + chunks[1]*128)
                            z2 = min(dat.shape[0], z + chunks[0]*8)
                            dat2 = da.from_array(ZarrPythonGroup.from_config(
                                copy_config.out, copy_zarr_config)[str(lvl - 1)],
                                chunks=self[str(lvl - 1)].chunks)
                            dat2 = dat2[z:z2, y:y2, x:x2]
                            dat2 = compute_next_level(
                                dat2, ndim, no_pyramid_axis, window)
                            dat2 = dat2.rechunk(
                                arr.shards or arr.chunks).persist()
                            with ProgressBar():
                                arr._array[ceil(
                                    z/2):ceil(z2/2), ceil(y/2):ceil(y2/2), ceil(x/2):ceil(x2/2)] = dat2.compute()
                x_max = ceil(x_max/2)
                x_min = ceil(x_min/2)
            else:
                dat = compute_next_level(dat, ndim, no_pyramid_axis, window)
                dat = dat.rechunk(arr.shards or arr.chunks).persist()
                with ProgressBar():
                    dat.store(arr)
        if copy_zarr_config is not None:
            copy_zarr_config.overwrite = overwrite_save

        return all_shapes

    @classmethod
    def open_dandi(
        cls,
        dandiset_id: str,
        asset_path: str,
        api_key: str,
        *,
        api_url: str = "https://api.dandiarchive.org/api",
        dandiset_version: str = "draft",
    ) -> "ZarrPythonGroup":
        """Open a Zarr group backed by a DANDI asset.

        Parameters
        ----------
        dandiset_id : str
            DANDI dataset identifier.
        asset_path : str
            Path to the Zarr asset within the dandiset.
        api_key : str
            API token used for authentication.
        api_url : str, optional
            Base URL of the DANDI/LINC Brain API.
        dandiset_version : str, optional
            Dandiset version to access (default: ``"draft"``).
        """
        zgroup = open_dandi_zarr_group(
            dandiset_id=dandiset_id,
            asset_path=asset_path,
            api_key=api_key,
            api_url=api_url,
            version=dandiset_version,
        )
        return cls(zgroup)


def _make_compressor(
    name: str | None, zarr_version: Literal[2, 3], **prm: dict
) -> CompressorsLike:
    """Build compressor object from name and options."""
    if not isinstance(name, str):
        return name
    if name == 'none':
        return None

    if zarr_version == 2:
        import numcodecs

        compressor_map = {
            "blosc": numcodecs.Blosc,
            "zlib": numcodecs.Zstd
        }
    elif zarr_version == 3:
        import zarr.codecs

        compressor_map = {
            "blosc": zarr.codecs.BloscCodec,
            "zlib": zarr.codecs.ZstdCodec,
        }
    else:
        raise ValueError()
    name = name.lower()

    if name not in compressor_map:
        raise ValueError("Unknown compressor", name)
    Compressor = compressor_map[name]

    return Compressor(**prm)


def _dimension_separator_to_chunk_key_encoding(
    dimension_separator: Literal[".", "/"], zarr_version: Literal[2, 3]
) -> ChunkKeyEncodingLike:
    dimension_separator = dimension_separator
    if dimension_separator == "." and zarr_version == 2:
        pass
    elif dimension_separator == "/" and zarr_version == 3:
        pass
    else:
        dimension_separator = ChunkKeyEncodingParams(
            name="default" if zarr_version == 3 else "v2", separator=dimension_separator
        )
        return dimension_separator
