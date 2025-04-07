import itertools
import re
from typing import Any, Literal

import nibabel as nib
import numpy as np
import zarr

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

