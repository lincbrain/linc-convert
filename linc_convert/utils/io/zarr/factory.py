"""Factory module for creating/opening Zarr Nodes with different drivers."""

import warnings
from os import PathLike
from typing import Literal

import zarr

from linc_convert.utils.io.zarr.abc import ZarrGroup, ZarrNode
from linc_convert.utils.io.zarr.drivers.zarr_python import (
    ZarrPythonArray,
    ZarrPythonGroup,
)
from linc_convert.utils.zarr_config import DriverLike, ZarrConfig

_DRIVER_ARRAY = {"zarr-python": ZarrPythonArray}
_DRIVER_GROUP = {"zarr-python": ZarrPythonGroup}

try:
    from linc_convert.utils.io.zarr.drivers.tensorstore import ZarrTSArray, ZarrTSGroup

    _DRIVER_ARRAY["tensorstore"] = ZarrTSArray
    _DRIVER_GROUP["tensorstore"] = ZarrTSGroup
except ImportError as e:
    warnings.warn(f"Tensorstore driver not available: {e}.")


class UnsupportedDriverError(ValueError):
    """Exception raised when an unsupported driver is specified."""

    def __init__(self, driver: DriverLike) -> None:
        super().__init__(f"Unsupported driver: {driver}")
        self.driver = driver


def _wrap_array(zarr_array: zarr.Array, driver: DriverLike) -> ZarrNode:
    base = ZarrPythonArray(zarr_array)
    if driver == "zarr-python":
        return base
    if driver in _DRIVER_ARRAY and hasattr(_DRIVER_ARRAY[driver],
                                           "from_zarr_python_array"):
        return _DRIVER_ARRAY[driver].from_zarr_python_array(base)
    raise UnsupportedDriverError(driver)


def _wrap_group(zarr_group: zarr.Group, driver: DriverLike) -> ZarrGroup:
    base = ZarrPythonGroup(zarr_group)
    if driver == "zarr-python":
        return base
    if driver in _DRIVER_GROUP and hasattr(_DRIVER_GROUP[driver],
                                           "from_zarr_python_group"):
        return _DRIVER_GROUP[driver].from_zarr_python_group(base)
    raise UnsupportedDriverError(driver)


def open(
    path: str | PathLike[str],
    mode: Literal["r", "r+", "a", "w", "w-"] = "a",
    zarr_version: Literal[2, 3] = 3,
    driver: DriverLike = "zarr-python",
) -> ZarrNode:
    """Open a Zarr Node (Array or Group) based on the specified driver."""
    node = zarr.open(path, mode=mode, zarr_format=zarr_version)
    if isinstance(node, zarr.Array):
        return _wrap_array(node, driver)
    elif isinstance(node, zarr.Group):
        return _wrap_group(node, driver)
    raise TypeError(f"Unsupported Zarr node type: {type(node)}")


def open_group(
    path: str | PathLike[str],
    mode: Literal["r", "r+", "a", "w", "w-"] = "a",
    zarr_version: Literal[2, 3] = 3,
    driver: DriverLike = "zarr-python",
) -> ZarrGroup:
    """Open a Zarr Group based on the specified driver."""
    return _wrap_group(zarr.open_group(path, mode=mode, zarr_format=zarr_version),
                       driver)


def from_config(zarr_config: ZarrConfig) -> ZarrGroup:
    """Create a ZarrGroup from a ZarrConfig."""
    group_cls = _DRIVER_GROUP.get(zarr_config.driver)
    if group_cls is None:
        raise UnsupportedDriverError(zarr_config.driver)
    return group_cls.from_config(zarr_config)
