import json
import logging
import re
from pathlib import Path

import cyclopts
import numpy as np
import tifffile
from skimage.registration import phase_cross_correlation

from linc_convert.modalities.lsm.cli import lsm
from linc_convert.modalities.lsm.convert_spool_or_zarr import discover_tile_paths
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


def parse_y(filename):
    stem = Path(filename).stem

    m = _TILE_PATTERN.match(stem)
    if m:
        return int(m.group("y"))

    m = _TILE_PATTERN2.match(stem)
    if m:
        return int(m.group("y"))

    raise ValueError(f"Could not parse y from {filename}")


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
    upsample_factor=20,
):
    """
    Returns
    -------
    ordered_files : list
    coordinates : list[(x, y)]

    coordinates[i] corresponds to ordered_files[i]
    """

    ordered_files = sorted(tif_files, key=parse_y)

    # first tile at origin
    coordinates = [(0.0, 0.0)]

    current_x = 0.0
    current_y = 0.0

    for f1, f2 in zip(ordered_files[:-1], ordered_files[1:]):

        img1 = tifffile.imread(f1)
        img2 = tifffile.imread(f2)

        if img1.ndim > 2:
            img1 = img1.max(axis=0)

        if img2.ndim > 2:
            img2 = img2.max(axis=0)

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
def calculate_coordiantes(
    input,
    output,
    overlap
):
    tif_files = discover_tile_paths(input)

    ordered_files, coordinates = estimate_strip_coordinates(
        tif_files=tif_files,
        overlap_pixels=overlap,
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
