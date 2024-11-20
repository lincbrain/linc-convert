"""
Convert annotations downloaded from webknossos into ome.zarr format following czyx direction 

"""

# stdlib
import os
import ast

# externals
import wkw
import json
import zarr
import shutil
import cyclopts
import numpy as np
from cyclopts import App

# internals
from linc_convert.modalities.wk.cli import wk
from linc_convert.utils.math import ceildiv
from linc_convert.utils.zarr import make_compressor


webknossos = cyclopts.App(name="webknossos", help_format="markdown")
wk.command(webknossos)


@webknossos.default
def convert(
    wkw_dir: str = None,
    ome_dir: str = None,
    dst: str = None,
    dic: str = None,
    *,
    chunk: int = 1024,
    compressor: str = 'blosc',
    compressor_opt: str = "{}",
    max_load: int = 16384,
) -> None:
    """
    Converts annotations(in .wkw format) from webknossos to ome.zarr format following czyx direction
    which is the same as underlying dataset. 

    It calculates offset from low-res images and set offset for other resolution levels accordingly. 

    Parameters
    ----------
    wkw_dir 
        Path to unzipped manual annotation folder, for example: .../annotation_folder/data_Volume
    ome_dir 
        Path to underlying ome.zarr dataset
    dst 
        Path to output directory [<INP_{_dsec_}_{initials}>.ome.zarr]
    dic 
        Dictionary of mapping annotation value to standard value, in case the annotation doesn't follow the standard of 
        0: background
        1: Light Bundle
        2: Moderate Bundle
        3: Dense Bundle
        4: Light Terminal
        5: Moderate Terminal
        6: Dense Terminal
        7: Single Fiber 
    """

    dic = json.loads(dic)

    # load underlying dataset info to get size info 
    omz_data = zarr.open_group(ome_dir, mode='r')
    nblevel = len([i for i in os.listdir(ome_dir) if i.isdigit()])
    wkw_dataset_path = os.path.join(wkw_dir, get_mask_name(nblevel-1))
    wkw_dataset = wkw.Dataset.open(wkw_dataset_path)

    low_res_offsets = []
    omz_res = omz_data[nblevel-1]
    n = omz_res.shape[1] 
    size = omz_res.shape[-2:]
    for idx in range(n):
        offset_x, offset_y = 0, 0
        data = wkw_dataset.read(off = (offset_y, offset_x, idx), shape = [size[1], size[0], 1])
        data = data[0, :, :, 0]
        data = np.transpose(data, (1, 0))
        [t,b,l,r] = find_borders(data)
        low_res_offsets.append([t,b,l,r])

    # setup save info 
    basename = os.path.basename(ome_dir)[:-9] 
    initials = wkw_dir.split('/')[-2][:2]
    out = os.path.join(dst, basename + '_dsec_' + initials + '.ome.zarr')
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    if isinstance(compressor_opt, str):
        compressor_opt = ast.literal_eval(compressor_opt)

    # Prepare Zarr group
    store = zarr.storage.DirectoryStore(out)
    omz = zarr.group(store=store, overwrite=True)


    # Prepare chunking options
    opt = {
        'chunks': [1, 1] + [chunk, chunk],
        'dimension_separator': r'/',
        'order': 'F',
        'dtype': 'uint8',
        'fill_value': None,
        'compressor': make_compressor(compressor, **compressor_opt),
    }
    print(opt)

    
    # Write each level
    for level in range(nblevel):
        omz_res = omz_data[level]
        size = omz_res.shape[-2:]
        shape = [1, n] + [i for i in size] 
        
        wkw_dataset_path = os.path.join(wkw_dir, get_mask_name(level))
        wkw_dataset = wkw.Dataset.open(wkw_dataset_path)

        omz.create_dataset(f'{level}', shape=shape, **opt)
        array = omz[f'{level}']

        # Write each slice
        for idx in range(n):
            if -1 in low_res_offsets[idx]:
                array[0, idx, :1, :1] = np.zeros((1, 1), dtype=np.uint8)
                continue 
            
            t, b, l, r = [k*2**(nblevel-level-1) for k in low_res_offsets[idx]]
            height, width = size[0]-t-b, size[1]-l-r 

            data = wkw_dataset.read(off = (l, t, idx), shape = [width, height, 1])
            data = data[0, :, :, 0]
            data = np.transpose(data, (1, 0))
            if dic:
                data = np.array([[dic[data[i][j]] for j in range(data.shape[1])] for i in range(data.shape[0])])
            subdat_size = data.shape 
            
            print('Convert level', level, 'with shape', shape, 'and slice', idx, 'with size', subdat_size)
            if max_load is None or (subdat_size[-2] < max_load and subdat_size[-1] < max_load):
                array[0, idx, t: t+subdat_size[-2], l: l+subdat_size[-1]] = data[...]
            else:
                ni = ceildiv(subdat_size[-2], max_load)
                nj = ceildiv(subdat_size[-1], max_load)
                
                for i in range(ni):
                    for j in range(nj):
                        print(f'\r{i+1}/{ni}, {j+1}/{nj}', end=' ')
                        start_x, end_x = i*max_load, min((i+1)*max_load, subdat_size[-2])
                        start_y, end_y = j*max_load, min((j+1)*max_load, subdat_size[-1])
                        array[0, idx,  t + start_x: t + end_x, l + start_y: l + end_y] = data[start_x: end_x, start_y: end_y]
                print('')
                
        
    # Write OME-Zarr multiscale metadata
    print('Write metadata')
    omz.attrs["multiscales"] = omz_data.attrs["multiscales"]



def get_mask_name(level):
    if level == 0:
        return '1'
    else:
        return f'{2**level}-{2**level}-1'


def cal_distance(img):
    m = img.shape[0]
    for i in range(m):
        cnt = np.sum(img[i, :])
        if cnt > 0:
            return i
    return m  


def find_borders(img):
    if np.max(img) == 0:
        return [-1, -1, -1, -1]
    t = cal_distance(img)
    b = cal_distance(img[::-1]) 
    l = cal_distance(np.rot90(img, k=3)) 
    r = cal_distance(np.rot90(img, k=1))

    return [max(0, k-1) for k in [t, b, l, r]]
