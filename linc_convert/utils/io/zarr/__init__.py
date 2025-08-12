"""ZarrIO module for handling Zarr data structures."""

import warnings

from .abc import ZarrArray, ZarrGroup, ZarrNode
from .factory import from_config, open_array, open_group

__all__ = [
    ZarrArray,
    ZarrGroup,
    ZarrNode,
    from_config,
    open_array,
    open_group,
]

try:
    import zarr  # noqa: F401

    from .drivers.zarr_python import ZarrPythonArray, ZarrPythonGroup

    __all__ += ['ZarrPythonArray', 'ZarrPythonGroup']
except ImportError:
    warnings.warn("zarr-python is not installed, driver disabled")

try:
    import tensorstore as TS  # noqa: F401

    from .drivers.tensorstore import ZarrTSArray, ZarrTSGroup

    __all__ += ["ZarrTSArray", "ZarrTSGroup"]
except ImportError:
    warnings.warn("Tensorstore is not installed, driver disabled")
