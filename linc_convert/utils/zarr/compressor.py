"""Functions for zarr compression."""
from typing import Any

import zarr.codecs
from zarr import abc

def make_compressor(name: str, **prm: dict) -> Any:
    """Build compressor object from name and options."""
    # TODO: we should use `numcodecs.get_codec` instead`
    if not isinstance(name, str):
        return name

    compressor_map = {
        "blosc": zarr.codecs.BloscCodec,
        "zlib": zarr.codecs.ZstdCodec,
    }

    name = name.lower()

    if name not in compressor_map:
        raise ValueError('Unknown compressor', name)
    Compressor = compressor_map[name]

    return Compressor(**prm)
