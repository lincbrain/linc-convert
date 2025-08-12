"""ZarrIO Implementation using the zarr-python library."""

from numbers import Number
from typing import (
    Any,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    Unpack,
)

import numpy as np
import zarr
import zarr.codecs
from numpy.typing import ArrayLike, DTypeLike
from zarr.core.array import CompressorsLike
from zarr.core.chunk_key_encodings import ChunkKeyEncodingLike, ChunkKeyEncodingParams

from linc_convert.utils.io.zarr.abc import ZarrArray, ZarrArrayConfig, ZarrGroup
from linc_convert.utils.io.zarr.helpers import _compute_zarr_layout
from linc_convert.utils.zarr_config import ZarrConfig


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
    def from_config(cls, zarr_config: ZarrConfig) -> "ZarrPythonGroup":
        """Create a Zarr group from a configuration object."""
        if zarr_config.out.startswith("/") or zarr_config.out.startswith("\\"):
            store = zarr.storage.LocalStore(zarr_config.out)
        else:
            store = zarr.storage.FsspecStore(zarr_config.out)
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
            raise KeyError(f"Key '{key}' not found in group '{self.store_path}'")
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
            arr = self._zgroup.create_array(name, shape, dtype, **kwargs)
            if data is not None:
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
        }

        chunk_key_encoding = _dimension_separator_to_chunk_key_encoding(
            zarr_config.dimension_separator, zarr_config.zarr_version
        )
        if chunk_key_encoding:
            opt["chunk_key_encoding"] = chunk_key_encoding
        arr = self._zgroup.create_array(name=name, shape=shape, **opt)
        if data:
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
                getattr(base_level._array, "metadata", None), "attributes", None
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
                opts["chunk_key_encoding"] = getattr(meta, "chunk_key_encoding", None)
            if hasattr(base_level, "serializer"):
                opts["serializer"] = getattr(base_level, "serializer", None)
            if hasattr(meta, "dimension_names"):
                opts["dimension_names"] = getattr(meta, "dimension_names", None)
        # Remove None values
        opts = {k: v for k, v in opts.items() if v is not None}
        opts.update(kwargs)
        arr = self._zgroup.create_array(name=name, shape=shape, **opts)
        if data is not None:
            arr[:] = data
        return ZarrPythonArray(arr)


def _make_compressor(
    name: str | None, zarr_version: Literal[2, 3], **prm: dict
) -> CompressorsLike:
    """Build compressor object from name and options."""
    if not isinstance(name, str):
        return name

    if zarr_version == 2:
        import numcodecs

        compressor_map = {
            "blosc": numcodecs.Blosc,
            "zlib": numcodecs.Zstd,
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
