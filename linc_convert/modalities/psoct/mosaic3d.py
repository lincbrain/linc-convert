"""
This module includes components that are based on the https://github.com/CarolineMagnain/OCTAnalysis repository (currently private).  We have permission from the owner to include the code here.
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
import tensorstore as ts
import tqdm
import zarr
from cyclopts import Parameter
from dask import delayed
from dask.diagnostics import ProgressBar
from niizarr import default_nifti_header, write_ome_metadata

from linc_convert.modalities.psoct.cli import psoct
from linc_convert.modalities.psoct.mosaic2d import _normalize_tile_coords, \
    _compute_blending_ramp, _load_nifti_tile, _load_tile
from linc_convert.modalities.psoct._utils import struct_arr_to_dict, \
    find_experiment_params, atleast_2d_trailing
from linc_convert.utils.zarr.zarr_io.drivers.tensorstore import default_write_config
from linc_convert.utils.zarr.zarr_io.drivers.zarr_python import compute_zarr_layout
from linc_convert.utils.zarr.generate_pyramid import generate_pyramid
from linc_convert.utils.zarr.zarr_config import ZarrConfig

logger = logging.getLogger(__name__)
mosaic3d = cyclopts.App(name="mosaic3d", help_format="markdown")
psoct.command(mosaic3d)

# 3d data has pixdim incorrectly set and cause nibabel keep logging warning
nib.imageglobals.logger.setLevel(40)

@mosaic3d.default
def mosaic3d_telesto(
        parameter_file: str,
        *,
        slice_index: int,
        dbi_output: Annotated[str, Parameter(name=["--dBI", "-d"])],
        o3d_output: Annotated[str, Parameter(name=["--O3D", "-o"])],
        r3d_output: Annotated[str, Parameter(name=["--R3D", "-r"])],
        tilted_illumination: bool = False,
        downsample: bool = False,
        use_gpu: bool = False,
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
    x_coords, y_coords = _normalize_tile_coords(exp_params['X_Mean' + tilt_postfix], exp_params['Y_Mean' + tilt_postfix],
    )
    full_width = int(np.nanmax(x_coords) + tile_width)
    full_height = int(np.nanmax(y_coords) + tile_height)
    depth = int(mosaic_info['MZL'])

    blend_ramp = _compute_blending_ramp(tile_width, tile_height, x_coords, y_coords)

    map_indices = exp_params['MapIndex_Tot_offset' + tilt_postfix] + exp_params['First_Tile' + tilt_postfix] - 1

    no_tilted_illumination_scan = scan_info['TiltedIllumination'] != 'Yes'

    focus = _load_nifti_tile(scan_info['FocusFile'+ tilt_postfix] )
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

    chunk, shard = compute_zarr_layout((depth, full_height, full_width), np.float32,
                                       zarr_config)

    dBI_result, R3D_result, O3D_result = build_slice(mosaic_idx,
                                                                          input_file_template,
                                                     blend_ramp, clip_x, clip_y,
                                                     full_width, full_height, depth,
                                                     flip_z, map_indices, scan_info,
                                                     tile_width, tile_height, x_coords,
                                                                          y_coords,
                                                                          raw_tile_width,
                                                                          focus,
                                                                          temp_compensate,
                                                                          chunk=chunk if not shard else shard)


    dBI_result = dBI_result.transpose(2,1,0)
    R3D_result = R3D_result.transpose(2,1,0)
    O3D_result = O3D_result.transpose(2,1,0)

    writers = []
    results = []
    zgroups = []
    for out, res in zip([dbi_output, r3d_output, o3d_output],
                        [dBI_result, R3D_result, O3D_result]):

        if shard:
            res = da.rechunk(res, chunks=shard)
        else:
            res = da.rechunk(res, chunks=chunk)
        zgroup = zarr.create_group(out, overwrite=True, zarr_format=zarr_config.zarr_version)
        zgroups.append(zgroup)
        wconfig = default_write_config(op.join(out, '0'), shape=res.shape,
                                       dtype=np.float32, chunk=chunk, shard=shard)
        wconfig["create"] = True
        wconfig["delete_existing"] = True
        tswriter = ts.open(wconfig).result()
        writers.append(tswriter)
        results.append(res)
    task = da.store(results, writers, compute=False)

    with ProgressBar():
        task.compute()

    scan_resolution = scan_resolution[:3][::-1]
    logger.info("finished")
    for zgroup in zgroups:
        generate_pyramid(zgroup)
        write_ome_metadata(zgroup, ["z", "y", "x"], scan_resolution, space_unit="millimeter")
        nii_header = default_nifti_header(zgroup["0"], zgroup.attrs["multiscales"])
        nii_header.set_xyzt_units("mm")
        # write_nifti_header(zgroup, nii_header)
    logger.info("finished generating pyramid")


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
    return da.stack([dBI,R3D,O3D], 3)


def build_slice(mosaic_idx, input_file_template, blend_ramp, clip_x, clip_y, full_width,
                full_height, depth, flip_z, map_indices, scan_info, tile_width,
                tile_height, x_coords, y_coords, raw_tile_width, focus, temp_compensate,
                chunk):
    coords = []
    tiles = []
    for (i, j), tile_idx in tqdm.tqdm(np.ndenumerate(map_indices)):
        if tile_idx <= 0 or np.isnan(x_coords[i, j]) or np.isnan(
                y_coords[i, j]):
            continue
        x0, y0 = int(x_coords[i, j]), int(y_coords[i, j])
        row_slice = slice(x0, x0 + tile_width)
        col_slice = slice(y0, y0 + tile_height)

        input_file_path = input_file_template % (mosaic_idx, tile_idx)
        complex3d = _load_tile(input_file_path)
        if complex3d.shape[0] > 4 * raw_tile_width:
            warnings.warn(
                f"Complex3D shape {complex3d.shape} is larger than expected {4 * raw_tile_width}. "
                "Trimming to 4*raw_tile_width."
            )
            complex3d = complex3d[:4 * raw_tile_width, :, :]
        elif complex3d.shape[0] < 4 * raw_tile_width:
            raise ValueError(
                f"Complex3D shape {complex3d.shape} is smaller than expected {4 * raw_tile_width}. "
                "Check the input file."
            )

        dBI3D, R3D, O3D = process_complex3d(complex3d, raw_tile_width)

        if temp_compensate:
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

    return stitch_tiles(
        coords, tiles,
        full_width=full_width, full_height=full_height,
        depth=depth, blend_ramp=blend_ramp,
        tile_size=(tile_width, tile_height),
        chunk=chunk
    )


def stitch_whole_canvas_padding(coords, tiles,
                        full_width, full_height, depth,
                        blend_ramp, tile_size, **_):
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

def _combine_block(_, block_tiles, block_weights, *args, block_info=None, **kwargs):
    chunk_id = tuple(block_info[None]['chunk-location'][:2])
    paints = block_tiles[chunk_id]
    weights = block_weights[chunk_id]
    shape = block_info[None]['chunk-shape']

    if not paints:
        return np.broadcast_to(np.zeros((), dtype=np.float32), shape)

    total_paint = da.sum(da.stack(paints, axis=0), axis=0)
    total_weight = da.sum(da.stack(weights, axis=0), axis=0)
    return total_paint / total_weight[...,None,None]


def stitch_tiles(coords, tiles,
                 full_width, full_height, depth,
                 blend_ramp, tile_size, **_):
    """
    Chunk‐aligned padding + per‐block summation (build_slice_chunked_padding).
    """
    pw, ph = tile_size
    canvas = da.zeros((full_width, full_height, depth, 3),
                      chunks=(pw, ph, depth, 3),
                      dtype=np.float32)
    weight = da.zeros((full_width, full_height),
                      chunks=(pw, ph),
                      dtype=np.float32)

    # collect per‐chunk pieces
    block_tiles = defaultdict(list)
    block_weights = defaultdict(list)

    for (x0, y0), t in zip(coords, tiles):
        # which tile‐chunks this falls into?
        x0c = x0 // pw
        y0c = y0 // ph
        x1c = (x0 + pw - 1) // pw
        y1c = (y0 + ph - 1) // ph

        # pad region covering those chunks
        x_start = x0c * pw
        y_start = y0c * ph
        x_end = (x1c + 1) * pw
        y_end = (y1c + 1) * ph

        block_canvas = da.zeros((x_end - x_start, y_end - y_start, depth, 3),
                                chunks=(pw, ph, depth, 3),
                                dtype=np.float32)
        block_weight = da.zeros((x_end - x_start, y_end - y_start),
                                chunks=(pw, ph),
                                dtype=np.float32)

        # place tile into that big block
        xs = slice(x0 - x_start, x0 - x_start + pw)
        ys = slice(y0 - y_start, y0 - y_start + ph)
        block_canvas[xs, ys, ...] = t
        block_weight[xs, ys] = blend_ramp

        # chop into per‐chunk pieces
        for cx in range(x0c, x1c + 1):
            for cy in range(y0c, y1c + 1):
                bid = (cx, cy)
                sub_x = slice((cx - x0c) * pw, (cx - x0c + 1) * pw)
                sub_y = slice((cy - y0c) * ph, (cy - y0c + 1) * ph)
                block_tiles[bid].append(block_canvas[sub_x, sub_y, ...])
                block_weights[bid].append(block_weight[sub_x, sub_y])


    canvas = da.map_blocks(_combine_block,canvas, block_tiles, block_weights,
        dtype=canvas.dtype,
        chunks=(pw, ph, depth, 3))

    return [canvas[..., i] for i in range(3)]


def process_complex3d(complex3d, raw_tile_width):
    complex3d = complex3d.rechunk({0:raw_tile_width})
    comp = complex3d.reshape(
        (4, raw_tile_width, complex3d.shape[1], complex3d.shape[2]))
    j1r, j1i, j2r, j2i = comp[0], comp[1], comp[2], comp[3]

    j1 = j1r + 1j * j1i
    j2 = j2r + 1j * j2i
    mag1 = da.abs(j1)
    mag2 = da.abs(j2)

    dBI3D = da.flip(10 * da.log10(mag1**2 + mag2**2), axis=2)
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


def do_downsample(volume: da.Array, factors: tuple) -> da.Array:
    fx, fy, fz = factors
    return da.coarsen(np.mean, volume, {0: fx, 1: fy, 2: fz}, trim_excess=True)

def stack_slices(mosaic_info, slice_indices, slices):
    return da.concatenate(slices, axis=2)
    num_slices = len(slices)
    z_off, z_sm, z_s = mosaic_info["z_parameters"][:3]
    # z_off: every slice is going to remove this much from beginning
    z_off, z_sm, z_s = int(z_off), int(z_sm), int(z_s)
    z_sms = z_sm + z_s
    z_m = z_sm - z_s
    # --- Load one slice to get block size and build tissue profile ---
    Id0 = slices[0]
    if modality.lower() == 'mus':
        # (If you want the 'mus' branch, you'd need skimage.filters.threshold_multiotsu, etc.)
        raise NotImplementedError("The 'mus' branch is not shown here.")
    else:
        # dBI: average a small tissue-only block
        tissue0 = Id0[:200, :200, z_off:].mean(axis=(0, 1))
    # only keep the next z_sms values, offset by zoff
    tissue = tissue0[z_off: z_off + z_sms]
    # --- compute blending weights ---
    s = tissue[z_s] / tissue[:z_s]  # Top overlapping skirt
    ms = tissue[z_s] / tissue[z_s: z_sms]  # non-overlap + bottom skirt
    degree = 1  # both dBI and mus use degree=1
    # w1 = s * np.linspace(0, 1, z_s) ** degree
    # w2 = ms[z_m:] * np.linspace(1, 0, z_s) ** degree
    # w3 = ms[:z_m]
    # TODO: omit the normalizing weight here for now to avoid computation
    w1 = np.linspace(0, 1, z_s) ** degree
    w2 = np.linspace(1, 0, z_s) ** degree
    w3 = np.ones(z_m)
    row_pxblock, col_pxblock = Id0.shape[:2]
    tot_z_pix = int(z_sm * num_slices)
    Ma = dask.array.zeros((row_pxblock,
                           col_pxblock,
                           tot_z_pix),
                          dtype=np.float32)
    for i in range(num_slices):
        si = int(slice_indices[0, i])  # incoming slice index
        print(f'\tstitching slice {i + 1}/{num_slices} (MAT idx {si:03d})')
        Id = slices[i]
        nx, ny, _ = Id.shape

        if s == 0:
            # first slice: apply fresh data
            # pattern = np.array([1, w3, w2])  # MATLAB’s [w1*0+1, w3, w2]
            # W = np.broadcast_to(pattern[np.newaxis, np.newaxis, :],  # shape (1,1,3)
            #                     (nx, ny, pattern.size))  # → (nx, ny, 3)

            W = np.concatenate([np.ones_like(w1), w3, w2])
            zr1 = np.arange(z_sms)  # 0 .. z_sms-1
            zr2 = zr1 + z_off  # zoff .. zoff+z_sms-1

        elif s == num_slices - 1:
            # last slice: add onto bottom-most z_sm planes
            # pattern = np.array([w1, w3])  # MATLAB’s [w1, w3]
            # W = np.broadcast_to(pattern[np.newaxis, np.newaxis, :],
            #                     (nx, ny, pattern.size))  # → (nx, ny, 2)
            W = np.concatenate([w1, w3, np.ones_like(w2)])
            zr1 = np.arange(z_sm) + (tot_z_pix - z_sm)  # targets top of Ma’s z-axis
            zr2 = np.arange(z_sm) + z_off  # picks from Id
        else:
            # middle slices: accumulate
            # pattern = np.array([w1, w3, w2])  # MATLAB’s [w1, w3, w2]
            # W = np.broadcast_to(pattern[np.newaxis, np.newaxis, :],
            #                     (nx, ny, pattern.size))  # → (nx, ny, 3)
            W = np.concatenate([w1, w3, w2])
            zr1 = np.arange(z_sms) + (s - 1) * z_sm  # where to write in Ma
            zr2 = np.arange(z_sms) + z_off  # where to read in Id
        # if i == 0:
        #     # first slice: only top skirt=1 + body + bottom skirt
        #     vec = np.concatenate([np.ones(z_s), w3, w2])
        #     W = np.tile(vec, (row_pxblock * col_pxblock, 1)) \
        #         .reshape(row_pxblock, col_pxblock, z_sms)
        #     z1 = np.arange(z_sms)
        #     z2 = z1 + int(z_off)
        # elif i == n_slices - 1:
        #     # last slice: only top skirt + body
        #     vec = np.concatenate([w1, w3])
        #     W = np.tile(vec, (row_pxblock * col_pxblock, 1)) \
        #         .reshape(row_pxblock, col_pxblock, z_sm)
        #     z1 = np.arange(tot_z_pix - z_sm, tot_z_pix)
        #     z2 = int(z_off) + np.arange(z_sm)
        # else:
        #     # middle slices: skirt/body/skirt
        #     vec = np.concatenate([w1, w3, w2])
        #     W = np.tile(vec, (row_pxblock * col_pxblock, 1)) \
        #         .reshape(row_pxblock, col_pxblock, z_sms)
        #     start_z = i * z_sm
        #     z1 = np.arange(start_z, start_z + z_sms)
        #     z2 = int(z_off) + np.arange(z_sms)

        Ma[:, :, zr1] += Id[:, :, zr2] * W
    return Ma

