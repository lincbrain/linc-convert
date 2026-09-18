"""Small local-file regressions for MIP loading and diagnostic JPEGs."""

import importlib

import nibabel as nib
import numpy as np
import pytest
import yaml
from PIL import Image

mosaic_module = importlib.import_module("linc_convert.modalities.psoct.mosaic")


@pytest.mark.parametrize("with_grid", [False, True])
def test_singleton_mip_jpeg_and_grid(tmp_path, monkeypatch, with_grid):
    data = np.arange(600, dtype=np.float32).reshape(30, 20, 1)
    nib.save(nib.Nifti1Image(data, np.eye(4)), tmp_path / "tile.nii")
    config = {
        "metadata": {"base_dir": str(tmp_path)},
        "tiles": [
            {"filepath": "missing.nii", "x": 0, "y": 0, "tile_number": 99},
            {"filepath": "tile.nii", "x": 100, "y": 200, "tile_number": 47},
            {"filepath": "tile.nii", "x": 124, "y": 200},
        ],
    }
    config_path = tmp_path / "tiles.yaml"
    config_path.write_text(yaml.safe_dump(config))
    calls = []
    save_grid = mosaic_module._save_tile_grid

    def record_grid(tiles, labels, shape, path):
        calls.append(([(t.x, t.y) for t in tiles], labels, shape))
        save_grid(tiles, labels, shape, path)

    monkeypatch.setattr(mosaic_module, "_save_tile_grid", record_grid)
    mosaic_module.mosaic2d(
        str(config_path),
        jpeg_output=str(tmp_path / "mosaic.jpg"),
        print_grid=str(tmp_path / "grid.jpg") if with_grid else None,
        tile_overlap=0.2,
        clip_x=2,
    )
    with Image.open(tmp_path / "mosaic.jpg") as image:
        assert image.format == "JPEG"
        assert image.size == (52, 20)
    if with_grid:
        assert calls == [([(0, 0), (24, 0)], ["47", "3"], (52, 20))]
        with Image.open(tmp_path / "grid.jpg") as image:
            assert image.format == "JPEG"
            assert image.mode == "RGB"
            assert image.width > image.height
    else:
        assert not calls
        assert not (tmp_path / "grid.jpg").exists()
