import itertools
import re
from typing import Any, Literal, Generator

import nibabel as nib
import numpy as np
import zarr
from numpy._typing import ArrayLike
from scipy import io as sio

from linc_convert.utils.math import ceildiv
from linc_convert.utils.unit import convert_unit


def make_json(oct_meta: str) -> dict:
    """
    Make json from OCT metadata.

    Expected input:
    ---------------
    Image medium: 60% TDE
    Center Wavelength: 1294.84nm
    Axial resolution: 4.9um
    Lateral resolution: 4.92um
    FOV: 3x3mm
    Voxel size: 3x3x3um
    Depth focus range: 225um
    Number of focuses: 2
    Focus #: 2
    Slice thickness: 450um.
    Number of slices: 75
    Slice #:23
    Modality: dBI
    """

    def _parse_value_unit(
        string: str, n: int = None
    ) -> tuple[float | list[float], str | Any]:
        number = r"-?(\d+\.?\d*|\d*\.?\d+)(E-?\d+)?"
        value = "x".join([number] * (n or 1))
        match = re.fullmatch(r"(?P<value>" + value + r")(?P<unit>\w*)", string)
        value, unit = match.group("value"), match.group("unit")
        value = list(map(float, value.split("x")))
        if n is None:
            value = value[0]
        return value, unit

    meta = {
        "BodyPart": "BRAIN",
        "Environment": "exvivo",
        "SampleStaining": "none",
    }

    for line in oct_meta.split("\n"):
        if ":" not in line:
            continue

        key, value = line.split(":")
        key, value = key.strip(), value.strip()

        if key == "Image medium":
            parts = value.split()
            if "TDE" in parts:
                parts[parts.index("TDE")] = "2,2' Thiodiethanol (TDE)"
            meta["SampleMedium"] = " ".join(parts)

        elif key == "Center Wavelength":
            value, unit = _parse_value_unit(value)
            meta["Wavelength"] = value
            meta["WavelengthUnit"] = unit

        elif key == "Axial resolution":
            value, unit = _parse_value_unit(value)
            meta["ResolutionAxial"] = value
            meta["ResolutionAxialUnit"] = unit

        elif key == "Lateral resolution":
            value, unit = _parse_value_unit(value)
            meta["ResolutionLateral"] = value
            meta["ResolutionLateralUnit"] = unit

        elif key == "Voxel size":
            value, unit = _parse_value_unit(value, n=3)
            meta["PixelSize"] = value
            meta["PixelSizeUnits"] = unit

        elif key == "Depth focus range":
            value, unit = _parse_value_unit(value)
            meta["DepthFocusRange"] = value
            meta["DepthFocusRangeUnit"] = unit

        elif key == "Number of focuses":
            value, unit = _parse_value_unit(value)
            meta["FocusCount"] = int(value)

        elif key == "Slice thickness":
            value, unit = _parse_value_unit(value)
            unit = convert_unit(value, unit[:-1], "u")
            meta["SliceThickness"] = value

        elif key == "Number of slices":
            value, unit = _parse_value_unit(value)
            meta["SliceCount"] = int(value)

        elif key == "Modality":
            meta["OCTModality"] = value

        else:
            continue

    return meta


def struct_arr_to_dict(arr: np.void) -> dict:
    """
    Convert a NumPy structured array (single record) to a dictionary.

    Parameters:
        arr (np.void): A structured array element.

    Returns:
        dict: Dictionary mapping field names to their values.
    """
    return {name: arr[name].item() for name in arr.dtype.names}


def find_experiment_params(exp_file: str) -> tuple[dict, bool]:
    """
    Load experiment parameters from a .mat file, detecting if it's a Fiji experiment.

    Parameters:
        exp_file (str): Path to the .mat file.

    Returns:
        tuple:
            - dict: Experiment parameters.
            - bool: True if it's a Fiji experiment, False otherwise.

    Raises:
        ValueError: If no experiment key is found in the file.
    """
    is_fiji = False
    exp_key = None

    for key in mat_vars(exp_file):
        if 'Experiment_Fiji' in key:
            exp_key = key
            is_fiji = True
            break
        if 'Experiment' in key:
            exp_key = key

    if not exp_key:
        raise ValueError("No Experiment found in .mat file")

    exp_data = sio.loadmat(exp_file, squeeze_me=True)[exp_key]
    return struct_arr_to_dict(exp_data), is_fiji


def mat_vars(mat_file: str) -> Generator[str, None, None]:
    """
    Yield variable names from a .mat file, excluding internal variables.

    Parameters:
        mat_file (str): Path to the .mat file.

    Yields:
        str: Variable names not starting with '__'.
    """
    yield from (name for name, *_ in sio.whosmat(mat_file) if not name.startswith('__'))


def atleast_2d_trailing(arr: ArrayLike) -> np.ndarray:
    """
    Ensure the input is at least 2D by adding a new axis at the end if needed.

    If the input is 1D, it becomes shape (N, 1).
    If the input is 0D (scalar), it becomes shape (1, 1).
    If it's already 2D or more, it's returned unchanged.

    Parameters:
        arr (ArrayLike): Input array-like object.

    Returns:
        np.ndarray: A 2D or higher NumPy array with at least two dimensions.
    """
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    elif arr.ndim == 1:
        return arr[:, np.newaxis]
    return arr


def load_mat(mat_path, varname):
    data = sio.loadmat(mat_path, squeeze_me=True)
    return data[varname]
