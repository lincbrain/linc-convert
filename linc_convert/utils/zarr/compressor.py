"""Functions for zarr compression."""
from typing import Any

import zarr.codecs


def make_compressor(name: str | None, **prm: dict) -> Any:
    """Build compressor object from name and options."""
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
