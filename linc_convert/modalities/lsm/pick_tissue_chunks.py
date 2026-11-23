"""Pick the consecutive chunks with the most tissue (both cameras) and write them to a YAML.

Used to test a correction on a few chunks only: ``tissue_chunks.yaml`` lists the chunk numbers
(1-based, the same for both cameras, because the channels are registered between cameras) and
``first`` / ``last`` for the registration and stitching steps.
"""
import os
from typing import Dict, List, Optional

import cyclopts
import yaml

from linc_convert.modalities.lsm.cli import lsm
from linc_convert.modalities.lsm.stitching_utils.io import (
    crop_and_flip,
    discover_tile_paths,
    open_tile_array,
    write_yaml,
)
from linc_convert.modalities.lsm.stitching_utils.optix_shading import pick_tissue_run
from linc_convert.modalities.lsm.stitching_utils.scan_metadata import (
    get_channel_names,
    get_crops,
    get_vertical_flip,
)
from linc_convert.modalities.lsm.stitching_utils.tile_selection import (
    available_layer,
    tissue_row_fraction,
)

pick_tissue_chunks = cyclopts.App(name="pick_tissue_chunks", help_format="markdown")
lsm.command(pick_tissue_chunks)


def camera_fractions(input_path: str, scan_parameters: dict, camera_id: int,
                     slice_number: int, background_length: int, multiplier: float,
                     scan_layer: int, x_stride: Optional[int] = None) -> Dict[int, float]:
    """Tissue-row fraction of every chunk of a camera (the minimum over its channels)."""
    paths = discover_tile_paths(input_path, camera_id)
    channels = get_channel_names(scan_parameters, camera_id)
    crops = get_crops(scan_parameters, camera_id, slice_number, kind="stitchingCrop",
                      include_z=True)
    flip = get_vertical_flip(scan_parameters, camera_id)
    layer = available_layer(paths[0], scan_layer)
    scale = 2 ** layer
    bg = max(1, background_length // scale)
    stride = x_stride or max(1, 64 // scale)
    out = {}
    for number, path in enumerate(paths, start=1):
        tile = open_tile_array(path, layer, chunks=(128, 128, 128))
        out[number] = min(
            tissue_row_fraction(crop_and_flip(tile, crops[ch], flip, layer), bg,
                                multiplier, stride) for ch in channels)
        print(f"  camera {camera_id} chunk {number}: {out[number]:.0%} of rows in tissue",
              flush=True)
    return out


@pick_tissue_chunks.default
def pick_tissue_chunks_cmd(input_path: str, yaml_path: str, slice_number: int, *,
                           output: str, count: int = 5, background_length: int = 5000,
                           zy_threshold_multiplier: float = 1.1, scan_layer: int = 3,
                           x_stride: Optional[int] = None, force: bool = False) -> None:
    """Write the ``count`` consecutive chunks with the most tissue (both cameras) to a YAML.

    Parameters
    ----------
    input_path : str
        Folder with the slice's ``*.ome.zarr`` chunks.
    yaml_path : str
        Scan metadata YAML.
    slice_number : int
        Slice number (for the crops).
    output : str
        YAML file to write (made with its folder). Skipped if it exists, unless ``force``.
    count : int, default=5
        Number of consecutive chunks.
    background_length : int, default=5000
        Trailing x columns (full-resolution pixels) used for the noise floor.
    zy_threshold_multiplier : float, default=1.1
        Signal threshold multiplier.
    scan_layer : int, default=3
        Pyramid level scanned (a coarse one is enough).
    x_stride : int, optional
        Every this-many x columns are sampled per row (default 64 / 2^level, as in
        ``create_corrections``); only small test tiles need a smaller one.
    force : bool, default=False
        Overwrite an existing output.
    """
    if os.path.exists(output) and not force:
        print(f"{output} exists, skipping (--force to redo)")
        return
    with open(yaml_path) as f:
        params = yaml.safe_load(f)
    per_cam = {c: camera_fractions(input_path, params, c, slice_number, background_length,
                                   zy_threshold_multiplier, scan_layer, x_stride) for c in (1, 2)}
    n = min(len(v) for v in per_cam.values())
    scores = {k: min(per_cam[1][k], per_cam[2][k]) for k in range(1, n + 1)}
    tiles: List[int] = pick_tissue_run(scores, count)
    print(f"[tissue chunks] {tiles} (tissue fraction {min(scores[t] for t in tiles):.0%} "
          "at the weakest)")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    write_yaml(output, {"tiles": tiles, "first": tiles[0], "last": tiles[-1], "count": count,
                        "tissue_fraction": {int(k): round(v, 3) for k, v in scores.items()}})