"""Explicit scan-order expectations, using a rectangular acquisition grid."""

import pytest
import yaml

from linc_convert.modalities.psoct.generate_tile_config import generate_tile_config


@pytest.mark.parametrize(
    "columns, rows, grid_type, order, expected",
    [
        (3, 2, "snake-by-rows", "right-down",
         [(0, 0), (8, 0), (16, 0), (16, 8), (8, 8), (0, 8)]),
        (2, 3, "snake-by-columns", "down-left",
         [(8, 0), (8, 8), (8, 16), (0, 16), (0, 8), (0, 0)]),
    ],
)
def test_snake_coordinates(tmp_path, columns, rows, grid_type, order, expected):
    output = tmp_path / "tiles.yaml"
    generate_tile_config(
        columns=columns, rows=rows, tile_size=10,
        grid_type=grid_type, order=order, overlap_percentage=0.2,
        naming_format="tile_{tile_number:03d}.nii", out=str(output),
    )
    config = yaml.safe_load(output.read_text())
    assert [(t["x"], t["y"]) for t in config["tiles"]] == expected
    assert [t["tile_number"] for t in config["tiles"]] == list(range(1, 7))
    assert config["tiles"][0]["filepath"] == "tile_001.nii"
