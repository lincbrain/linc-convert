import os
import cv2
import ast
import wkw
import zarr
import json
import uuid
import time
import math
import shutil
import glymur
import skimage
import cyclopts
import numcodecs
import numpy as np
from glob import glob
import nibabel as nib
from typing import Optional
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from skimage import measure
from skimage.draw import polygon2mask
from scipy.ndimage import binary_fill_holes

""" 
This script resave downloaded annotations from webknossos in ome.zarr format following direction czyx 
which is the same with underlying dataset. 

It calculates offset from low-res images and set offset for other resolution accordingly. 

If the annotation is contour rather than mask, set is_contour as True else False. 

wkw_dir is the path to unzipped annotation 
jp2_dir is the path to a jpeg2000 image of same subject to get voxel size information 
ome_dir is the path to underlying ome.zarr dataset
dst is the path to saving annotation mask 
"""

app = cyclopts.App(help_format="markdown")
@app.default
def convert(
    wkw_dir: str = None,
    jp2_dir: str = None, 
    ome_dir: str = None,
    dst: str = None,
    is_contour: bool = True, 
    *,
    chunk: int = 1024,
    compressor: str = 'blosc',
    compressor_opt: str = "{}",
    max_load: int = 16384,
    nii: bool = False,
    has_channel: int = 1,
):
    # load underlying dataset info to get size info 
    omz_data = zarr.open_group(ome_dir, mode='r')
    wkw_dataset_path = os.path.join(wkw_dir, get_mask_name(8))
    wkw_dataset = wkw.Dataset.open(wkw_dataset_path)

    low_res_offsets = []
    omz_res = omz_data[8]
    size = np.shape(omz_res)
    size = [i for i in omz_res.shape[-2:]] + [3]
    for idx in range(20):
        offset_x, offset_y = 0, 0
        data = wkw_dataset.read(off = (offset_y, offset_x, idx), shape = [size[1], size[0], 1])
        data = data[0, :, :, 0]
        data = np.transpose(data, (1, 0))
        [t,b,l,r] = find_borders(data)
        low_res_offsets.append([t,b,l,r])

    # load jp2 image to get voxel size info 
    j2k = glymur.Jp2k(jp2_dir)
    vxw, vxh = get_pixelsize(j2k)


    # setup save info 
    basename = os.path.basename(ome_dir)[:-9] 
    initials = wkw_dir.split('/')[-2][:2]
    out = os.path.join(dst, basename + '_dsec_' + initials + '.ome.zarr')
    print(out)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    if isinstance(compressor_opt, str):
        compressor_opt = ast.literal_eval(compressor_opt)

    # Prepare Zarr group
    store = zarr.storage.DirectoryStore(out)
    omz = zarr.group(store=store, overwrite=True)


    dic_EB = {0:0, 1:1, 9:2, 2:3, 4:4, 10:5, 11:6, 8:7, 3:8}
    dic_JS = {0:0, 4:2, 5:3, 9:4, 6:5, 2:6, 7:7, 3:8}
    dic_JW = {0:0, 1:1, 2:2, 3:3, 4:4, 6:5, 7:6, 8:7, 9:8}

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


    nblevel = 9
    # Write each level
    for level in range(nblevel):
        omz_res = omz_data[level]
        size = omz_res.shape[-2:]
        shape = [1, 20] + [i for i in size] 
        
        wkw_dataset_path = os.path.join(wkw_dir, get_mask_name(level))
        wkw_dataset = wkw.Dataset.open(wkw_dataset_path)

        omz.create_dataset(f'{level}', shape=shape, **opt)
        array = omz[f'{level}']

        # Write each slice
        for idx in range(20):
            if -1 in low_res_offsets[idx]:
                continue 
            
            t, b, l, r = [k*2**(8-level) for k in low_res_offsets[idx]]
            height, width = size[0]-t-b, size[1]-l-r 

            data = wkw_dataset.read(off = (l, t, idx), shape = [width, height, 1])
            data = data[0, :, :, 0]
            data = np.transpose(data, (1, 0))
            if is_contour and level > 3: 
                if initials == 'EB':
                    mapped_img = np.array([[dic_EB[data[i][j]] for j in range(data.shape[1])] for i in range(data.shape[0])])
                elif initials == 'JW': 
                    mapped_img = np.array([[dic_JW[data[i][j]] for j in range(data.shape[1])] for i in range(data.shape[0])])
                elif initials == 'JS': 
                    mapped_img = np.array([[dic_JS[data[i][j]] for j in range(data.shape[1])] for i in range(data.shape[0])])
                subdat = generate_mask(mapped_img)

            else: 
                subdat = data 
            subdat_size = subdat.shape 
            
            print('Convert level', level, 'with shape', shape, 'and slice', idx, 'with size', subdat_size)
            if max_load is None or (subdat_size[-2] < max_load and subdat_size[-1] < max_load):
                array[0, idx, t: t+subdat_size[-2], l: l+subdat_size[-1]] = subdat[...]
            else:
                ni = ceildiv(subdat_size[-2], max_load)
                nj = ceildiv(subdat_size[-1], max_load)
                
                for i in range(ni):
                    for j in range(nj):
                        print(f'\r{i+1}/{ni}, {j+1}/{nj}', end=' ')
                        start_x, end_x = i*max_load, min((i+1)*max_load, subdat_size[-2])
                        start_y, end_y = j*max_load, min((j+1)*max_load, subdat_size[-1])
                        array[0, idx,  t + start_x: t + end_x, l + start_y: l + end_y] = subdat[start_x: end_x, start_y: end_y]
                print('')
                
        
    # Write OME-Zarr multiscale metadata
    print('Write metadata')
    multiscales = [{
        'version': '0.4',
        'axes': [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}
        ],
        'datasets': [],
        'type': 'jpeg2000',
        'name': '',
    }]
    
    if has_channel:
        multiscales[0]['axes'].insert(0, {"name": "c", "type": "channel"})
    for n in range(nblevel):
        shape0 = omz['0'].shape[-2:]
        shape = omz[str(n)].shape[-2:]
        multiscales[0]['datasets'].append({})
        level = multiscales[0]['datasets'][-1]
        level["path"] = str(n)


        level["coordinateTransformations"] = [
            {
                "type": "scale",
                "scale": [1.0] * has_channel + [
                    1.0, 
                    (shape0[0]/shape[0])*vxh,
                    (shape0[1]/shape[1])*vxw,
                ]
            },
            {
                "type": "translation",
                "translation": [0.0] * has_channel + [
                    0.0, 
                    (shape0[0]/shape[0] - 1)*vxh*0.5,
                    (shape0[1]/shape[1] - 1)*vxw*0.5,
                ]
            }
        ]
    multiscales[0]["coordinateTransformations"] = [
        {
            "scale": [1.0] * (3 + has_channel),
            "type": "scale"
        }
    ]
    omz.attrs["multiscales"] = multiscales



def get_mask_name(level):
    if level == 0:
        return '1'
    else:
        return f'{2**level}-{2**level}-1'


def ceildiv(x, y):
    return int(math.ceil(x / y))


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

    return [t, b, l, r]


def contour_to_mask(mask, value):
    h, w = mask.shape[:2]
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 
    # closed_mask = skimage.morphology.binary_closing(mask) 
    # plt.figure()
    # plt.subplot(1,3,1)
    # plt.imshow(mask)
    # plt.subplot(1,3,2)
    # plt.imshow(closed_mask)
    single_mask = binary_fill_holes(closed_mask)

    # contours = measure.find_contours(closed_mask) 
    # single_mask = np.zeros_like(mask) 
    # print(len(contours))

    # for contour in contours:
    #     cur_mask = polygon2mask((h, w), contour)  
    #     single_mask |= cur_mask 
    single_mask = np.where(single_mask, value, 0)
    # plt.subplot(1,3,3)
    # plt.imshow(single_mask)
    # plt.show()
    return single_mask


def generate_mask(mask):
    final_mask = np.zeros_like(mask).astype(np.uint8)
    for value in range(1, 10): 
        if value not in mask:
            continue
        binary_mask = np.where(mask == value, 255, 0).astype(np.uint8)
        single_mask = contour_to_mask(binary_mask, value)
        final_mask = np.where(final_mask < single_mask, single_mask, final_mask)
    return final_mask


def make_compressor(name, **prm):
    if not isinstance(name, str):
        return name
    name = name.lower()
    if name == 'blosc':
        Compressor = numcodecs.Blosc
    elif name == 'zlib':
        Compressor = numcodecs.Zlib
    else:
        raise ValueError('Unknown compressor', name)
    return Compressor(**prm)


def get_pixelsize(j2k):
    # Adobe XMP metadata
    # https://en.wikipedia.org/wiki/Extensible_Metadata_Platform
    XMP_UUID = 'BE7ACFCB97A942E89C71999491E3AFAC'
    TAG_Images = '{http://ns.adobe.com/xap/1.0/}Images'
    Tag_Desc = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description'
    Tag_PixelWidth = '{http://ns.adobe.com/xap/1.0/}PixelWidth'
    Tag_PixelHeight = '{http://ns.adobe.com/xap/1.0/}PixelHeight'

    vxw = vxh = 1.0
    for box in j2k.box:
        if getattr(box, 'uuid', None) == uuid.UUID(XMP_UUID):
            try:
                images = list(box.data.iter(TAG_Images))[0]
                desc = list(images.iter(Tag_Desc))[0]
                vxw = float(desc.attrib[Tag_PixelWidth])
                vxh = float(desc.attrib[Tag_PixelHeight])
            except Exception:
                pass
    return vxw, vxh


if __name__ == "__main__":
    app()
