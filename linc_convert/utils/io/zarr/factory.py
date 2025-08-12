"""Factory module for creating/opening Zarr Nodes with different drivers."""
import importlib
import warnings
from os import PathLike
from typing import Literal

from linc_convert.utils.io.zarr.abc import ZarrGroup, ZarrNode
from linc_convert.utils.zarr_config import DriverLike, ZarrConfig


class UnsupportedDriverError(ValueError):
    """Exception raised when an unsupported driver is specified."""

    def __init__(self, driver: DriverLike) -> None:
        super().__init__(f"Unsupported driver: {driver}")
        self.driver = driver


_DRIVER_ARRAY: dict[str, type] = {}
_DRIVER_GROUP: dict[str, type] = {}

# name -> (probe_module, array_path, group_path)
# where *_path is "pkg.module:ClassName"
_DRIVERS: dict[str, tuple[str, str, str]] = {
    "zarr-python": (
        "zarr",
        "linc_convert.utils.io.zarr.drivers.zarr_python:ZarrPythonArray",
        "linc_convert.utils.io.zarr.drivers.zarr_python:ZarrPythonGroup",
    ),
    "tensorstore": (
        "tensorstore",
        "linc_convert.utils.io.zarr.drivers.tensorstore:ZarrTSArray",
        "linc_convert.utils.io.zarr.drivers.tensorstore:ZarrTSGroup",
    ),
}


def _import_symbol(path: str) -> type:
    """Import a symbol given its full path as 'module:attr'."""
    mod_path, _, attr = path.partition(":")
    if not attr:
        raise ValueError(f"Expected 'module:attr' path, got {path!r}")
    module = importlib.import_module(mod_path)
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise ImportError(f"Cannot import '{attr}' from '{mod_path}'") from e


def _register_available_drivers() -> None:
    """Populate _DRIVER_ARRAY and _DRIVER_GROUP with available drivers."""
    for name, (probe_mod, array_path, group_path) in _DRIVERS.items():
        if importlib.util.find_spec(probe_mod) is None:
            warnings.warn(
                f"{name} driver not available: missing dependency '{probe_mod}'.")
            continue
        try:
            arr_cls = _import_symbol(array_path)
            grp_cls = _import_symbol(group_path)
        except Exception as e:
            # If the driver module imports but its own deps fail, surface that clearly.
            warnings.warn(f"{name} driver failed to load: {e}")
            continue
        _DRIVER_ARRAY[name] = arr_cls
        _DRIVER_GROUP[name] = grp_cls


_register_available_drivers()


def open_array(
    path: str | PathLike[str],
    mode: Literal["r", "r+", "a", "w", "w-"] = "a",
    zarr_version: Literal[2, 3] = 3,
    driver: DriverLike = "zarr-python",
) -> ZarrNode:
    """Open a Zarr Node (Array or Group) based on the specified driver."""
    array_cls = _DRIVER_ARRAY.get(driver)
    if array_cls is None:
        raise UnsupportedDriverError(driver)
    return array_cls.open(path, mode, zarr_version=zarr_version)


def open_group(
    path: str | PathLike[str],
    mode: Literal["r", "r+", "a", "w", "w-"] = "a",
    zarr_version: Literal[2, 3] = 3,
    driver: DriverLike = "zarr-python",
) -> ZarrGroup:
    """Open a Zarr Group based on the specified driver."""
    group_cls = _DRIVER_GROUP.get(driver)
    if group_cls is None:
        raise UnsupportedDriverError(driver)
    return group_cls.open(path, mode, zarr_version=zarr_version)


def from_config(zarr_config: ZarrConfig) -> ZarrGroup:
    """Create a ZarrGroup from a ZarrConfig."""
    group_cls = _DRIVER_GROUP.get(zarr_config.driver)
    if group_cls is None:
        raise UnsupportedDriverError(zarr_config.driver)
    return group_cls.from_config(zarr_config)
