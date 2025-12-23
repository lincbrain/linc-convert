"""Utilities for reading MATLAB .mat files and wrapping arrays."""
import os
from typing import Optional, Union

import h5py

from linc_convert.utils.io.matlab_array_wrapper import (
    ArrayWrapper,
    H5ArrayWrapper,
    MatArraywrapper,
)


def make_wrapper(fname: str, key: Optional[str] = None) -> ArrayWrapper:
    """Create an ArrayWrapper for a MATLAB .mat file."""
    try:
        # "New" .mat file
        f = h5py.File(fname, "r")
        return H5ArrayWrapper(f, key)
    except Exception:
        # "Old" .mat file
        return MatArraywrapper(fname, key)


def as_arraywrapper(
    inp: Union[str, os.PathLike, ArrayWrapper],
    key: Optional[str] = None,
) -> ArrayWrapper:
    """Convert input to an ArrayWrapper if it is not already one."""
    if isinstance(inp, ArrayWrapper):
        return inp
    return make_wrapper(str(inp), key)
