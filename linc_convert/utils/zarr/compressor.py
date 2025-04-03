"""Functions for zarr compression."""
from typing import Any, Literal

import zarr.codecs


def make_compressor(name: str | None, zarr_version: Literal[2, 3],**prm: dict) -> Any:
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

    name = name.lower()

    if name not in compressor_map:
        raise ValueError('Unknown compressor', name)
    Compressor = compressor_map[name]

    return Compressor(**prm)
