"""Factory module for creating/opening Zarr Nodes with different drivers."""

import warnings
from os import PathLike
from typing import Literal

from linc_convert.utils.io.zarr import ZarrNode
from linc_convert.utils.io.zarr.abc import ZarrGroup
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


def open(
    path: str | PathLike[str],
    mode: Literal["r", "r+", "a", "w", "w-"] = "a",
    zarr_version: Literal[2, 3] = 3,
    driver: DriverLike = "zarr-python",
) -> ZarrNode:
    """
    Open a Zarr Node (Array or Group) based on the specified driver.

    Parameters
    ----------
    path : str | PathLike[str]
        Path to the Zarr Node.
    mode : Literal["r", "r+", "a", "w", "w-]
        Mode in which to open the Zarr Node.
    zarr_version : Literal[2, 3]
        Zarr version to use (default is 3).
    driver : DriverLike
        Driver to use for opening the Zarr Node (default is "zarr-python").

    Returns
    -------
    ZarrNode
        An instance of ZarrNode, which can be either a ZarrArray or Zarr
    """
    raise NotImplementedError


def open_group(
    path: str | PathLike[str],
    mode: Literal["r", "r+", "a", "w", "w-"] = "a",
    zarr_version: Literal[2, 3] = 3,
    driver: DriverLike = "zarr-python",
) -> ZarrGroup:
    """
    Open a Zarr Group based on the specified driver.

    Parameters
    ----------
    path : str | PathLike[str]
        Path to the Zarr Group.
    mode : Literal["r", "r+", "a", "w", "w-"]
        Mode in which to open the Zarr Group.
    zarr_version : Literal[2, 3]
        Zarr version to use (default is 3).
    driver : DriverLike
        Driver to use for opening the Zarr Group (default is "zarr-python").

    Returns
    -------
    ZarrGroup
        An instance of ZarrGroup based on the specified driver.
    """
    raise NotImplementedError


def from_config(zarr_config: ZarrConfig) -> ZarrGroup:
    """
    Create a ZarrGroup from a ZarrConfig.

    Parameters
    ----------
    zarr_config : ZarrConfig
        Configuration for the Zarr group.

    Returns
    -------
    ZarrGroup
        An instance of ZarrGroup based on the configuration.
    """
    if zarr_config.driver not in _DRIVER_GROUP:
        raise ValueError(f"{zarr_config.driver} is not supported.")
    return _DRIVER_GROUP[zarr_config.driver].from_config(zarr_config)
