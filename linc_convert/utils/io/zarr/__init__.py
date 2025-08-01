"""ZarrIO module for handling Zarr data structures."""

import warnings

from .abc import ZarrArray, ZarrGroup, ZarrNode
from .drivers.zarr_python import ZarrPythonArray, ZarrPythonGroup
from .factory import from_config, open, open_group

try:
    import tensorstore as TS  # noqa: F401

    from .drivers.tensorstore import ZarrTSArray, ZarrTSGroup
except ImportError:
    warnings.warn("Tensorstore is not installed, driver disabled")

__all__ = [
    ZarrArray,
    ZarrGroup,
    ZarrNode,
    ZarrPythonArray,
    ZarrPythonGroup,
    from_config,
    open,
    open_group,
    "ZarrTSArray",
    "ZarrTSGroup",
]
