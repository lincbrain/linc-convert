"""
This module includes components that are based on the
https://github.com/CarolineMagnain/OCTAnalysis repository (currently private).  We
have permission from the owner to include the code here.
"""

import logging
import os.path as op
import warnings
from collections import defaultdict
from typing import Annotated

import cyclopts
import dask
import dask.array as da
import nibabel as nib
import numpy as np
import scipy.io as sio
import tqdm
from cyclopts import Parameter
from dask import delayed
from dask.diagnostics import ProgressBar
from niizarr import default_nifti_header

from linc_convert.modalities.psoct._utils import (atleast_2d_trailing,
                                                  find_experiment_params,
                                                  struct_arr_to_dict)
from linc_convert.modalities.psoct.cli import psoct
from linc_convert.modalities.psoct.mosaic2d import (_compute_blending_ramp,
                                                    _load_nifti_tile, _load_tile,
                                                    _normalize_tile_coords)
from linc_convert.utils.io.zarr import from_config
from linc_convert.utils.io.zarr.drivers.zarr_python import _compute_zarr_layout
from linc_convert.utils.zarr_config import ZarrConfig

logger = logging.getLogger(__name__)
single_tile = cyclopts.App(name="single_tile", help_format="markdown")
psoct.command(single_tile)

# 3d data has pixdim incorrectly set and cause nibabel keep logging warning
nib.imageglobals.logger.setLevel(40)


@single_tile.default
def single_tile(
    parameter_file: str,
    *,
    tile_idx: int,
    slice_index: int,
    dbi_output: Annotated[str, Parameter(name=["--dBI", "-d"])],
    o3d_output: Annotated[str, Parameter(name=["--O3D", "-o"])],
    r3d_output: Annotated[str, Parameter(name=["--R3D", "-r"])],
    tilted_illumination: bool = False,
    downsample: bool = False,
    zarr_config: ZarrConfig = None,
) -> None:
    """
    Parameters
    ----------
    dbi_output : str
        Output path for dBI volume.
    o3d_output : str
        Output path for O3D volume.
    r3d_output : str
        Output path for R3D volume.
    """
    logger.info("started")
    tilt_postfix = '_tilt' if tilted_illumination else ''
    # Load parameters
    raw_mat = sio.loadmat(parameter_file, squeeze_me=True)
    params = struct_arr_to_dict(raw_mat['Parameters'])
    scan_info = struct_arr_to_dict(raw_mat['Scan'])
    mosaic_info = struct_arr_to_dict(raw_mat['Mosaic3D'])
    exp_params, is_fiji = find_experiment_params(mosaic_info['Exp'])
    slice_indices = atleast_2d_trailing(mosaic_info['sliceidx'])
    scan_resolution = np.atleast_1d(scan_info['Resolution'])
    if len(scan_resolution) < 3:
        raise ValueError
    clip_x, clip_y = int(params['XPixClip']), int(params['YPixClip'])
    if scan_info['System'].lower() == 'octopus':
        clip_x, clip_y = clip_y, clip_x
    flip_z = bool(mosaic_info.get('invert_Z', True))

    tile_size = int(exp_params['NbPix'])
    tile_width, tile_height = tile_size - clip_x, tile_size - clip_y
    if tilted_illumination:
        tile_width = int(exp_params['NbPix' + tilt_postfix]) - clip_x
    x_coords, y_coords = _normalize_tile_coords(exp_params['X_Mean' + tilt_postfix],
                                                exp_params['Y_Mean' + tilt_postfix],
                                                )
    full_width = int(np.nanmax(x_coords) + tile_width)
    full_height = int(np.nanmax(y_coords) + tile_height)
    depth = int(mosaic_info['MZL'])

    blend_ramp = _compute_blending_ramp(tile_width, tile_height, x_coords, y_coords)

    map_indices = exp_params['MapIndex_Tot_offset' + tilt_postfix] + exp_params[
        'First_Tile' + tilt_postfix] - 1

    no_tilted_illumination_scan = scan_info['TiltedIllumination'] != 'Yes'

    focus = _load_nifti_tile(scan_info['FocusFile' + tilt_postfix])
    raw_tile_width = int(scan_info['NbPixels' + tilt_postfix])

    # slice_idx_in, slice_idx_out, slice_idx_run = slice_indices[:, slice_index]
    slice_idx_in = slice_index
    mosaic_idx = 2 * slice_idx_in - 1
    if tilted_illumination:
        mosaic_idx += 1
    if no_tilted_illumination_scan:
        mosaic_idx = slice_idx_in

    file_prefix = [s for s in np.atleast_1d(scan_info['FilePrefix']) if
                   'cropped' in str(s).lower()]
    if file_prefix:
        file_prefix = file_prefix[0]
    else:
        raise ValueError("No complex file prefix found in scan_info['FilePrefix']")

    raw_data_dir = scan_info['RawDataDir']
    input_file_template = scan_info['FileNameFormat']
    input_file_template = input_file_template.replace('[modality]', file_prefix)
    input_file_template = op.join(raw_data_dir, input_file_template)

    if mosaic_idx > 8 and tilted_illumination:
        temp_compensate = True
    else:
        temp_compensate = False

    chunk, shard = _compute_zarr_layout((depth, full_height, full_width), np.float32,
                                        zarr_config)

    dBI_result, R3D_result, O3D_result = build_slice(mosaic_idx, tile_idx,
                                                     input_file_template,
                                                     blend_ramp, clip_x, clip_y,
                                                     full_width, full_height, depth,
                                                     flip_z, map_indices, scan_info,
                                                     tile_width, tile_height, x_coords,
                                                     y_coords,
                                                     raw_tile_width,
                                                     focus,
                                                     temp_compensate,
                                                     chunk=chunk if not shard else
                                                     shard)

    writers = []
    results = []
    zgroups = []
    for out, res in zip([dbi_output, r3d_output, o3d_output],
                        [dBI_result, R3D_result, O3D_result]):
        img = nib.Nifti1Image(res, np.eye(4))
        nib.save(img, out)


# For I80 slab1 use only
def temporary_compensation(dBI, R3D, O3D, depth, focus, scan_info):
    Nz = depth
    # 1) build a mask of pixels needing correction
    mask_crop = focus <= scan_info['Focus_CropStart']  # (Ny, Nx)
    # 2) find, for each (y,x), the last index along z where dBI3D == -inf
    mask_inf = np.isneginf(dBI)  # (Ny, Nx, Nz)

    depth_idx = np.arange(Nz)[None, None, :]  # (1, 1, Nz)
    last_inf = np.max(np.where(mask_inf, depth_idx, -1), axis=2)
    # last_inf[y,x] is -1 if there was no -inf, or the maximum index otherwise
    # 3) decide which pixels actually get shifted
    do_shift = (mask_crop & (last_inf >= 0))  # (Ny, Nx)
    # 4) build the "shifted" index array for every (y,x,z)
    #    new_idx[y,x,z] = last_inf[y,x] + z
    new_idx = last_inf[..., None] + depth_idx  # (Ny, Nx, Nz)
    # 5) clip to valid range so we can safely gather,
    #    then gather each volume
    clipped = np.clip(new_idx, 0, Nz - 1)
    dBI3D_shift = np.take_along_axis(dBI, clipped, axis=2)
    R3D_shift = np.take_along_axis(R3D, clipped, axis=2)
    O3D_shift = np.take_along_axis(O3D, clipped, axis=2)
    # 6) zero-out any positions that “fell off the end” (new_idx >= Nz)
    overflow = new_idx >= Nz
    dBI3D_shift[overflow] = 0
    R3D_shift[overflow] = 0
    O3D_shift[overflow] = 0
    # 7) select per-pixel whether to keep the original or use the shifted version
    dBI = np.where(do_shift[..., None], dBI3D_shift, dBI)
    R3D = np.where(do_shift[..., None], R3D_shift, R3D)
    O3D = np.where(do_shift[..., None], O3D_shift, O3D)
    return da.stack([dBI, R3D, O3D], 3)


def build_slice(
    mosaic_idx, want_tile_idx, input_file_template, blend_ramp, clip_x, clip_y,
    full_width,
    full_height, depth, flip_z, map_indices, scan_info, tile_width,
    tile_height, x_coords, y_coords, raw_tile_width, focus, temp_compensate,
    chunk
    ):
    coords = []
    tiles = []
    for (i, j), tile_idx in tqdm.tqdm(np.ndenumerate(map_indices), "Loading Tiles"):
        if tile_idx <= 0 or np.isnan(x_coords[i, j]) or np.isnan(
            y_coords[i, j]):
            continue
        if tile_idx != want_tile_idx:
            continue
        x0, y0 = int(x_coords[i, j]), int(y_coords[i, j])

        input_file_path = input_file_template % (mosaic_idx, tile_idx)

        complex3d = _load_tile(input_file_path)
        if complex3d.shape[0] > 4 * raw_tile_width:
            warnings.warn(
                f"Complex3D shape {complex3d.shape} is larger than expected "
                f"{4 * raw_tile_width}. "
                "Trimming to 4*raw_tile_width."
            )
            complex3d = complex3d[:4 * raw_tile_width, :, :]
        elif complex3d.shape[0] < 4 * raw_tile_width:
            raise ValueError(
                f"Complex3D shape {complex3d.shape} is smaller than expected "
                f"{4 * raw_tile_width}. "
                "Check the input file."
            )

        dBI3D, R3D, O3D = process_complex3d(complex3d, raw_tile_width)

        if False and temp_compensate:
            results = delayed(temporary_compensation)(dBI3D, R3D, O3D, depth, focus,
                                                      scan_info)
            results = da.from_delayed(results, shape=(*dBI3D.shape, 3),
                                      dtype=dBI3D.dtype)
        else:
            results = da.stack([dBI3D, R3D, O3D], axis=3)

        if flip_z:
            results = da.flip(results, axis=2)
        if scan_info['System'].lower() == 'octopus':
            results = da.swapaxes(results, 0, 1)

        results = results[clip_x:, clip_y:] * blend_ramp[:, :, None, None]

        tiles.append(results)
        coords.append((x0, y0))
    tiles[0] = tiles[0].astype("float32")
    return tiles[0][..., 0], tiles[0][..., 1], tiles[0][..., 2]


def stitch_whole_canvas_padding(
    coords, tiles,
    full_width, full_height, depth,
    blend_ramp, tile_size, **_
    ):
    """
    Build a full‐size zero‐canvas per tile and slice‐assign into it,
    then stack & sum (build_slice_whole_canvas_padding).
    """
    pw, ph = tile_size
    canvases = []
    weights = []

    for (x0, y0), t in zip(coords, tiles):
        canvas_tile = da.zeros((full_width, full_height, depth, 3),
                               chunks=(pw, ph, depth, 3),
                               dtype=np.float32)
        weight_tile = da.zeros((full_width, full_height),
                               chunks=(pw, ph),
                               dtype=np.float32)

        rs = slice(x0, x0 + pw)
        cs = slice(y0, y0 + ph)
        canvas_tile[rs, cs, ...] = t
        weight_tile[rs, cs] = blend_ramp

        canvases.append(canvas_tile)
        weights.append(weight_tile)

    canvas = da.sum(da.stack(canvases, axis=0), axis=0)
    weight = da.sum(da.stack(weights, axis=0), axis=0)

    canvas /= weight[..., None, None]
    canvas = da.nan_to_num(canvas)
    return [canvas[..., i] for i in range(3)]


def process_complex3d(complex3d, raw_tile_width):
    complex3d = complex3d.rechunk({0: raw_tile_width})
    comp = complex3d.reshape(
        (4, raw_tile_width, complex3d.shape[1], complex3d.shape[2]))
    j1r, j1i, j2r, j2i = comp[0], comp[1], comp[2], comp[3]

    j1 = j1r + 1j * j1i
    j2 = j2r + 1j * j2i
    mag1 = da.abs(j1)
    mag2 = da.abs(j2)

    dBI3D = da.flip(10 * da.log10(mag1 ** 2 + mag2 ** 2), axis=2)
    R3D = da.flip(
        da.arctan(mag1 / mag2) / da.pi * 180,
        axis=2
    )
    offset = 100 / 180 * da.pi
    phi = da.angle(j1) - da.angle(j2) + offset * 2
    # wrap into [-π, π]
    phi = da.where(phi > da.pi, phi - 2 * da.pi, phi)
    phi = da.where(phi < -da.pi, phi + 2 * da.pi, phi)
    O3D = da.flip(phi / (2 * da.pi) * 180, axis=2)

    return dBI3D, R3D, O3D
