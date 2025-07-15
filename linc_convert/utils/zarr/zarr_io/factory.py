import warnings

from linc_convert.utils.zarr import ZarrConfig
from linc_convert.utils.zarr.zarr_config import DriverLike
from linc_convert.utils.zarr.zarr_io.abc import ZarrGroup
from linc_convert.utils.zarr.zarr_io.drivers.zarr_python import ZarrPythonArray, \
    ZarrPythonGroup

_DRIVER_ARRAY = {
    "zarr-python": ZarrPythonArray
}
_DRIVER_GROUP = {
    "zarr-python": ZarrPythonGroup
}

try:
    from linc_convert.utils.zarr.zarr_io.drivers.tensorstore import ZarrTSArray, \
        ZarrTSGroup

    _DRIVER_ARRAY["tensorstore"] = ZarrTSArray
    _DRIVER_GROUP["tensorstore"] = ZarrTSGroup
except ImportError as e:
    warnings.warn(f"Tensorstore driver not available: {e}.")


def open(driver: DriverLike):
    raise NotImplementedError


def open_group(driver: DriverLike) -> ZarrGroup:
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
