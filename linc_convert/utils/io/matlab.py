import os
from typing import Optional, Union

import h5py
from linc_convert.utils.io.matlab_array_wrapper import (ArrayWrapper, H5arraywrapper,
                                                        Matarraywrapper)


def make_wrapper(fname: str, key) -> ArrayWrapper:
    try:
        # "New" .mat file
        f = h5py.File(fname, "r")
        return H5arraywrapper(f, key)
    except Exception:
        # "Old" .mat file
        return Matarraywrapper(fname, key)


def as_arraywrapper(
    inp: Union[str, os.PathLike, ArrayWrapper],
    key: Optional[str]
) -> ArrayWrapper:
    if isinstance(inp, ArrayWrapper):
        return inp
    return make_wrapper(str(inp), key)
