"""Convert annotation downloaded from webknossos into ome.zarr format."""

# stdlib
import ast
import json
import os
import shutil

import cyclopts
import numpy as np

# externals
import wkw
import zarr

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
    out: str = None,
    dic: str = None,
    *,
    chunk: int = 1024,
    compressor: str = "blosc",
    compressor_opt: str = "{}",
    max_load: int = 16384,
) -> None:
    """
    Convert annotations (in .wkw format) from webknossos to ome.zarr format.

    This script converts annotations from webknossos, following the czyx direction,
    to the ome.zarr format.
    The conversion ensures that the annotations match the underlying dataset.

    Parameters
    ----------
    wkw_dir : str
        Path to the unzipped manual annotation folder downloaded from webknossos
        in .wkw format. For example: .../annotation_folder/data_Volume.
    ome_dir : str
        Path to the underlying ome.zarr dataset, following the BIDS naming standard.
    out : str
        Path to the output directory for saving the converted ome.zarr.
        The ome.zarr file name is generated automatically based on ome_dir
        and the initials of the annotator.
    dic : dict
        A dictionary mapping annotation values to the following standard values
        if annotation doesn't match the standard.
        The dictionary should be in single quotes, with keys in double quotes,
        for example: dic = '{"2": 1, "4": 2}'.
        The standard values are:
        - 0: background
        - 1: Light Bundle
        - 2: Moderate Bundle
        - 3: Dense Bundle
        - 4: Light Terminal
        - 5: Moderate Terminal
        - 6: Dense Terminal
        - 7: Single Fiber
    """
    dic = json.loads(dic)
    dic = {int(key): int(value) for key, value in dic.items()}

    # load underlying dataset info to get size info
    omz_data = zarr.open_group(ome_dir, mode="r")
    nblevel = len([i for i in os.listdir(ome_dir) if i.isdigit()])
    wkw_dataset_path = os.path.join(wkw_dir, get_mask_name(nblevel - 1))
    wkw_dataset = wkw.Dataset.open(wkw_dataset_path)

    low_res_offsets = []
    omz_res = omz_data[nblevel - 1]
    n = omz_res.shape[1]
    size = omz_res.shape[-2:]
    for idx in range(n):
        offset_x, offset_y = 0, 0
        data = wkw_dataset.read(
            off=(offset_y, offset_x, idx), shape=[size[1], size[0], 1]
        )
        data = data[0, :, :, 0]
        data = np.transpose(data, (1, 0))
        [t0, b0, l0, r0] = find_borders(data)
        low_res_offsets.append([t0, b0, l0, r0])

    # setup save info
    basename = os.path.basename(ome_dir)[:-9]
    initials = wkw_dir.split("/")[-2][:2]
    out = os.path.join(out, basename + "_dsec_" + initials + ".ome.zarr")
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
        "chunks": [1, 1] + [chunk, chunk],
        "dimension_separator": r"/",
        "order": "F",
        "dtype": "uint8",
        "fill_value": None,
        "compressor": make_compressor(compressor, **compressor_opt),
    }
    print(opt)

    # Write each level
    for level in range(nblevel):
        omz_res = omz_data[level]
        size = omz_res.shape[-2:]
        shape = [1, n] + [i for i in size]

        wkw_dataset_path = os.path.join(wkw_dir, get_mask_name(level))
        wkw_dataset = wkw.Dataset.open(wkw_dataset_path)

        omz.create_dataset(f"{level}", shape=shape, **opt)
        array = omz[f"{level}"]

        # Write each slice
        for idx in range(n):
            if -1 in low_res_offsets[idx]:
                array[0, idx, :1, :1] = np.zeros((1, 1), dtype=np.uint8)
                continue

            top, bottom, left, right = [
                k * 2 ** (nblevel - level - 1) for k in low_res_offsets[idx]
            ]
            height, width = size[0] - top - bottom, size[1] - left - right

            data = wkw_dataset.read(off=(left, top, idx), shape=[width, height, 1])
            data = data[0, :, :, 0]
            data = np.transpose(data, (1, 0))
            if dic:
                data = np.array(
                    [
                        [dic[data[i][j]] for j in range(data.shape[1])]
                        for i in range(data.shape[0])
                    ]
                )
            subdat_size = data.shape

            print(
                "Convert level",
                level,
                "with shape",
                shape,
                "and slice",
                idx,
                "with size",
                subdat_size,
            )
            if max_load is None or (
                subdat_size[-2] < max_load and subdat_size[-1] < max_load
            ):
                array[
                    0, idx, top : top + subdat_size[-2], left : left + subdat_size[-1]
                ] = data[...]
            else:
                ni = ceildiv(subdat_size[-2], max_load)
                nj = ceildiv(subdat_size[-1], max_load)

                for i in range(ni):
                    for j in range(nj):
                        print(f"\r{i+1}/{ni}, {j+1}/{nj}", end=" ")
                        start_x, end_x = (i * max_load,)
                        min((i + 1) * max_load, subdat_size[-2])

                        start_y, end_y = (j * max_load,)
                        min((j + 1) * max_load, subdat_size[-1])
                        array[
                            0,
                            idx,
                            top + start_x : top + end_x,
                            left + start_y : left + end_y,
                        ] = data[start_x:end_x, start_y:end_y]
                print("")

    # Write OME-Zarr multiscale metadata
    print("Write metadata")
    omz.attrs["multiscales"] = omz_data.attrs["multiscales"]


def get_mask_name(level: int) -> str:
    """
    Return the name of the mask for a given resolution level.

    Parameters
    ----------
    level : int
        The resolution level for which to return the mask name.

    Returns
    -------
    str
        The name of the mask for the given level.
    """
    if level == 0:
        return "1"
    else:
        return f"{2**level}-{2**level}-1"


def cal_distance(img: np.ndarray) -> int:
    """
    Return the distance of non-zero values to the top border.

    Parameters
    ----------
    img : np.ndarray
        The array to calculate distance of object inside to border

    Returns
    -------
    int
        The distance of non-zero to the top border
    """
    m = img.shape[0]
    for i in range(m):
        cnt = np.sum(img[i, :])
        if cnt > 0:
            return i
    return m


def find_borders(img: np.ndarray) -> np.ndarray:
    """
    Return the distances of non-zero values to four borders.

    Parameters
    ----------
    img : np.ndarray
        The array to calculate distance of object inside to border

    Returns
    -------
    int
        The distance of non-zero values to four borders
    """
    if np.max(img) == 0:
        return [-1, -1, -1, -1]
    top = cal_distance(img)
    bottom = cal_distance(img[::-1])
    left = cal_distance(np.rot90(img, k=3))
    right = cal_distance(np.rot90(img, k=1))

    return [max(0, k - 1) for k in [top, bottom, left, right]]
