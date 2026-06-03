import json
import logging
import re
from pathlib import Path
from typing import Optional

import cyclopts
import numpy as np
import tifffile
from skimage.registration import phase_cross_correlation

from linc_convert.modalities.lsm.cli import lsm
from linc_convert.modalities.lsm.convert_spool_or_zarr import discover_tile_paths, prompt_dandi_api_key
from linc_convert.utils.zarr_config import autoconfig

_TILE_PATTERN = re.compile(
    r"^(?P<prefix>\w*)"
    r"_run(?P<run>[0-9]+)"
    r"_y(?P<y>[0-9]+)"
    r"(?:_z(?P<z>[0-9]+))?"
    r"(?P<suffix>\w*)$"
)

_TILE_PATTERN2 = re.compile(
    r"^(?P<prefix>[\w-]*)"
    r"_chunk-(?P<y>[0-9]+)"
    r"(?:_z(?P<z>[0-9]+))?"
    r"(?P<suffix>[\w-]*)$"
)


def parse_y(filename, file_pattern):
    stem = Path(filename).stem

    match = file_pattern.match(stem)

    if not match:
        raise ValueError(
            f"Could not parse y from '{filename}' using pattern "
            f"'{file_pattern.pattern}'"
        )

    try:
        return int(match.group("y"))
    except IndexError:
        raise ValueError(
            "file_pattern must contain a named capture group "
            "'(?P<y>...)'"
        )


def estimate_pairwise_shift(
    img1,
    img2,
    overlap_pixels,
    upsample_factor=20,
):
    """
    img2 is assumed below img1
    """

    region1 = img1[-overlap_pixels:, :]
    region2 = img2[:overlap_pixels, :]

    shift, error, _ = phase_cross_correlation(
        region1,
        region2,
        upsample_factor=upsample_factor,
    )

    dy, dx = shift

    return dx, dy, error


def estimate_strip_coordinates(
    tif_files,
    overlap_pixels,
    file_pattern,
    upsample_factor=20,
    *,
    y_start: Optional[int] = None,
    y_end: Optional[int] = None,
):
    """
    Returns
    -------
    ordered_files : list
    coordinates : list[(x, y)]

    coordinates[i] corresponds to ordered_files[i]
    """

    ordered_files = sorted(
        tif_files,
        key=lambda x: parse_y(x, file_pattern),
    )

    # first tile at origin
    coordinates = [(0.0, 0.0)]

    current_x = 0.0
    current_y = 0.0

    for f1, f2 in zip(ordered_files[:-1], ordered_files[1:]):

        img1 = tifffile.imread(f1)
        img2 = tifffile.imread(f2)

        if y_end is not None:
            img1 = img1[:y_end, :]
            img2 = img2[:y_end, :]
        if y_start is not None:
            img1 = img1[y_start:, :]
            img2 = img2[y_start:, :]

        dx, dy, error = estimate_pairwise_shift(
            img1,
            img2,
            overlap_pixels,
            upsample_factor,
        )

        height = img1.shape[0]

        # actual movement from tile1 -> tile2
        step_y = height - overlap_pixels + dy
        step_x = dx

        current_x += step_x
        current_y += step_y

        coordinates.append(
            (float(current_x), float(current_y))
        )

    return ordered_files, coordinates


logger = logging.getLogger(__name__)
coordinates = cyclopts.App(name="coordinates", help_format="markdown")
lsm.command(coordinates)


@coordinates.default
@autoconfig
def calculate_coordinates(
    input: str,
    output: str,
    overlap: int,
    file_pattern: str,
    *,
    dandiset_id: Optional[str] = None,
    y_start: Optional[int] = None,
    y_end: Optional[int] = None,
):
    compiled_pattern = re.compile(file_pattern)
    if "y" not in compiled_pattern.groupindex:
        raise ValueError(
            "file_pattern must contain a named group '(?P<y>...)'"
        )
    api_key = prompt_dandi_api_key() if dandiset_id else None
    tif_files = discover_tile_paths(
        input, api_key=api_key, dandiset_id=dandiset_id, filename_pattern=file_pattern)

    ordered_files, coordinates = estimate_strip_coordinates(
        tif_files=tif_files,
        overlap_pixels=overlap,
        file_pattern=compiled_pattern,
        y_start=y_start,
        y_end=y_end
    )

    data = {
        "overlap_pixels": overlap,
        "tiles": [
            {
                "file": str(path),
                "filename": Path(path).name,
                "x": float(x),
                "y": float(y),
            }
            for path, (x, y) in zip(
                ordered_files,
                coordinates,
            )
        ],
    }

    with open(output, "w") as f:
        json.dump(data, f, indent=2)

    return data
