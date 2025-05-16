from pathlib import Path

import dandi.download
import filecmp
import numpy as np
import os
import subprocess
import tarfile
import tifffile

from helper import _cmp_zarr_archives
from linc_convert.modalities.lsm import mosaic


def _write_test_data(directory: str) -> None:
    root_path = Path(directory)
    for z in range(1, 2, 1):
        for y in range(1, 3, 1):
            folder = root_path / f"test_z{z}_y{y}"
            folder.mkdir(parents=True, exist_ok=True)
            for plane in range(1, 4, 1):
                for c in range(1, 3, 1):
                    image = np.zeros((1024, 768)) + z * y * plane * c
                    tifffile.imwrite(
                        folder / f"test_z{z}_y{y}_plane{plane}_c{c}.tiff", image
                    )


def test_lsm(tmp_path):
    _write_test_data(tmp_path)
    output_zarr = tmp_path / "output.zarr"
    mosaic.convert(str(tmp_path), out=str(output_zarr))
    assert _cmp_zarr_archives(str(output_zarr), "data/lsm.zarr.zip")

def test_transfer():
        
    input_dir = './000051/sourcedata/sub-test1'
    subprocess.Popen(f"linc-convert lsm transfer --input-dir '{input_dir}' --dandiset-url 'https://lincbrain.org/dandiset/000051' --dandi-instance 'linc' --subject 'test1' --output-dir '.' --max-size-gb 0.02 --no-upload", env=os.environ.copy(), shell=True)

    extract_dir = './sub-test1'
    os.mkdir(extract_dir)
    tar_files = list(Path(input_dir).glob("*.tar"))
    for tar_file in tar_files:
        with tarfile.open(tar_file, "r") as tar:
            tar.extractall(path=extract_dir)
        os.remove(tar_file)

    dirs_cmp = filecmp.dircmp(input_dir, extract_dir)
    
    assert len(dirs_cmp.left_only)==0 and len(dirs_cmp.right_only)==0, "Files do not match"

    input_dir_size = sum(os.path.getsize(f) for f in os.listdir(input_dir) if os.path.isfile(f))
    extract_dir_size = sum(os.path.getsize(f) for f in os.listdir(extract_dir) if os.path.isfile(f))

    assert input_dir_size == extract_dir_size, "File sizes do not match"
