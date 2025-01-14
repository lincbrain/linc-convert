
import os

import numpy as np
import wkw
import zarr
from helper import _cmp_zarr_archives

from linc_convert.modalities.wk import webknossos_annotation


def _write_test_data(directory: str) -> None:
    wkw_dir = f'{directory}/wkw'
    ome_dir = f'{directory}/ome'

    store = zarr.storage.DirectoryStore(ome_dir)
    omz = zarr.group(store=store, overwrite=True)

    for level in range(5):
        size = 2**(4-level)
        wkw_array = np.zeros((size, size, 5), dtype=np.uint8)
        ome_array = np.zeros((1, 5, size, size), dtype=np.uint8)

        wkw_filepath = os.path.join(wkw_dir, get_mask_name(level)) 
        with wkw.Dataset.create(wkw_filepath, wkw.Header(np.uint8)) as dataset:
            dataset.write((0, 0, 0), wkw_array)

        omz.create_dataset(f'{level}', shape=[1, 5, size, size])
        array = omz[f'{level}']
        array[...] = ome_array 

    multiscales = [{
            'version': '0.4',
            'axes': [
                {"name": "c", "type": "space", "unit": "millimeter"},
                {"name": "z", "type": "space", "unit": "millimeter"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ],
            'datasets': [],
            'type': 'jpeg2000',
            'name': '',
        }] 
    for n in range(5):
        multiscales[0]['datasets'].append({})
        level = multiscales[0]['datasets'][-1]
        level["path"] = str(n)

        level["coordinateTransformations"] = [
            {
                "type": "scale",
                "scale": [
                    1.0, 
                    1.0, 
                    float(2**n),
                    float(2**n),
                ]
            },
            {
                "type": "translation",
                "translation": [
                    0.0, 
                    0.0, 
                    float(2**n - 1) *0.5,
                    float(2**n - 1) *0.5,
                ]
            }
        ]
    omz.attrs["multiscales"] = multiscales




def test_wk(tmp_path):
    _write_test_data(tmp_path)

    wkw_dir = str(tmp_path / "wkw")
    ome_dir = str(tmp_path / "ome")
    basename = os.path.basename(ome_dir)[:-9] 
    initials = wkw_dir.split('/')[-2][:2]
    output_zarr = os.path.join(tmp_path, basename + '_dsec_' + initials + '.ome.zarr')

    print("starting the convert process")
    webknossos_annotation.convert(wkw_dir, ome_dir, tmp_path, "{}")

    z = zarr.open(output_zarr, mode='r')
    for level in range(5):
        print("output_zarr has", np.shape(z[level]), np.unique(z[level]))

    z = zarr.open('data/wk.zarr.zip', mode='r')
    for level in range(5):
        print("trusted result has", np.shape(z[level]), np.unique(z[level]))
  

    assert _cmp_zarr_archives(str(output_zarr), "data/wk.zarr.zip")



def get_mask_name(level):
    if level == 0:
        return '1'
    else:
        return f'{2**level}-{2**level}-1'

