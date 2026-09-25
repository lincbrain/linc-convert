"""Small local-file regressions for MIP loading and diagnostic JPEGs."""

import importlib
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
import pytest
import yaml
from PIL import Image

mosaic_module = importlib.import_module("linc_convert.modalities.psoct.mosaic")


def test_mosaic2d_jpeg_tiles(tmp_path):
    """Stitch nine 3x3 JPEGs with one-pixel overlaps; remove all files afterward."""
    with TemporaryDirectory(dir=tmp_path) as temporary_directory:
        directory = Path(temporary_directory)
        tiles = []
        for row in range(3):
            for column in range(3):
                number = row * 3 + column + 1
                filename = f"tile_{number:02d}.jpg"
                pixels = np.full((3, 3), number * 20, dtype=np.uint8)
                Image.fromarray(pixels).save(directory / filename, quality=100)
                tiles.append(
                    {
                        "filepath": filename,
                        "tile_number": number,
                        "x": column * 2,
                        "y": row * 2,
                    }
                )
        config = directory / "tiles.yaml"
        config.write_text(
            yaml.safe_dump({"metadata": {"base_dir": str(directory)}, "tiles": tiles})
        )
        output = directory / "mosaic.jpg"
        mosaic_module.mosaic2d(
            str(config), jpeg_output=str(output), tile_overlap=1 / 3,
            circular_mean=False,
        )

        # Equal edge weights average neighbors at every overlapping row/column.
        # Asymmetric tile intensities also detect transposed tile placement.
        positions = np.array([0, 0, 0.5, 1, 1.5, 2, 2])
        expected = 20 + 60 * positions[:, None] + 20 * positions[None, :]
        expected = ((expected - 20) / 160 * 255).astype(np.uint8)
        with Image.open(output) as image:
            assert image.format == "JPEG"
            assert image.mode == "L"
            assert image.size == (7, 7)
            # Allow small differences from lossy JPEG encoding.
            np.testing.assert_allclose(np.array(image), expected, atol=8, rtol=0)
    assert not directory.exists()


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
