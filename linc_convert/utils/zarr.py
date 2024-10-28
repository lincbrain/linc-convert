"""Zarr utilities."""

import numcodecs
import numcodecs.abc


def make_compressor(name: str, **prm: dict) -> numcodecs.abc.Codec:
    """Build compressor object from name and options."""
    # TODO: we should use `numcodecs.get_codec` instead`
    if not isinstance(name, str):
        return name
    name = name.lower()
    if name == "blosc":
        Compressor = numcodecs.Blosc
    elif name == "zlib":
        Compressor = numcodecs.Zlib
    else:
        raise ValueError("Unknown compressor", name)
    return Compressor(**prm)


def make_compressor_v2(name: str | None, **prm: dict) -> dict:
    """Build compressor dictionary for Zarr v2."""
    name = name.lower()
    if name not in ('blosc', 'zlib'):
        raise ValueError('Unknown compressor', name)
    return {"id": name, **prm}


def make_compressor_v3(name: str | None, **prm: dict) -> dict:
    """Build compressor dictionary for Zarr v3."""
    name = name.lower()
    if name not in ('blosc', 'zlib'):
        raise ValueError('Unknown compressor', name)
    return {"name": name, "configuration": prm}
