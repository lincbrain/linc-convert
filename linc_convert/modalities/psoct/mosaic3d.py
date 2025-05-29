import logging
import os.path as op
import warnings
from typing import Annotated

import cyclopts
import dask
import dask.array as da
import nibabel as nib
import numpy as np
import scipy.io as sio
import tensorstore as ts
import tqdm
from cyclopts import Parameter

from linc_convert.modalities.psoct.cli import psoct
from linc_convert.modalities.psoct.mosaic2d import _normalize_tile_coords, \
    _compute_blending_ramp, _load_mat_tile, _load_nifti_tile
from linc_convert.modalities.psoct.utils._utils import struct_arr_to_dict, \
    find_experiment_params, atleast_2d_trailing
from linc_convert.modalities.psoct.utils._zarr import default_write_config
from linc_convert.utils.zarr.create_array import compute_zarr_layout
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
    tilt_postfix = '_tilt' if tilted_illumination else ''
    # Load parameters
    raw_mat = sio.loadmat(parameter_file, squeeze_me=True)
    params = struct_arr_to_dict(raw_mat['Parameters'])
    scan_info = struct_arr_to_dict(raw_mat['Scan'])
    mosaic_info = struct_arr_to_dict(raw_mat['Mosaic3D'])
    # Load Experiment parameters
    exp_params, is_fiji = find_experiment_params(mosaic_info['Exp'])

    # load variables from parameters
    # modality = mosaic_info['modality']
    slice_indices = atleast_2d_trailing(mosaic_info['sliceidx'])
    input_dirs = np.atleast_1d(mosaic_info['indir'])
    file_path_template = mosaic_info['file_format']
    file_type = mosaic_info['InFileType'].lower()

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

    # Map indices
    map_indices = exp_params['MapIndex_Tot_offset' + tilt_postfix] + exp_params['First_Tile' + tilt_postfix] - 1
    num_slices = slice_indices.shape[1]

    no_tilted_illumination_scan = scan_info['TiltedIllumination'] != 'Yes'

    focus = _load_nifti_tile(scan_info['FocusFile'] + tilt_postfix)
    raw_tile_width = int(scan_info['NbPixels' + tilt_postfix])
    # Build full volume graph
    slices = [build_slice(s, blend_ramp, clip_x, clip_y, full_width, full_height, depth,
                          flip_z, is_fiji, map_indices, scan_info, slice_indices,
                          tile_width, tile_height, x_coords, y_coords,
                          tilted_illumination, no_tilted_illumination_scan, focus,
                          raw_tile_width) for s in range(1)]  # num_slices)]

    # if len(slices) > 1:
    #     volume = stack_slices(mosaic_info, slice_indices, slices)
    # else:
    #     volume = slices[0]
    volume= slices[0]

    chunk, shard = compute_zarr_layout(volume[0].shape, volume[0].dtype, zarr_config)
    writers = []
    results = []
    for out, res in zip([dbi_output, r3d_output, o3d_output], volume):

        if shard:
            res = da.rechunk(res, chunks=shard)
        else:
            res = da.rechunk(res, chunks=chunk)
        wconfig = default_write_config(op.join(out, '0'), shape=res.shape,
                                       dtype=np.float32, chunk=chunk, shard=shard)
        wconfig["create"] = True
        wconfig["delete_existing"] = True
        tswriter = ts.open(wconfig).result()
        # tswriter = da.from_array(tswriter, chunks=res.chunks)
        writers.append(tswriter)
        results.append(res)
    task = da.store(results, writers, compute=False)
    # task.visualize(filename="dag.svg", format="svg")
    task.compute()
    # Store the volumes
    # writer = da.stack(writers, axis=0)
    # result = da.stack(results, axis=0)
    # result.store(writer, lock=False, compute=True)

    #
    # tswriter = ts.open(wconfig).result()
    # volume.store(tswriter)

def build_slice(slice_idx, blend_ramp, clip_x, clip_y, full_width, full_height, depth,
                flip_z, is_fiji, map_indices, scan_info, slice_indices, tile_width,
                tile_height, x_coords, y_coords, tilted_illumination,
                no_tilted_illumination_scan, focus, raw_tile_width):
    raw_data_dir = scan_info['RawDataDir']
    input_file_format = scan_info['FileNameFormat']
    if '.nii' in input_file_format:
        file_type = 'nifti'
    elif '.mat' in input_file_format:
        file_type = 'mat'
    else:
        raise ValueError(f"Unsupported file type: {input_file_format}")

    file_prefix = [s for s in np.atleast_1d(scan_info['FilePrefix']) if
                   'cropped' in str(s).lower()]
    if file_prefix:
        file_prefix = file_prefix[0]
    else:
        raise ValueError("No complex file prefix found in scan_info['FilePrefix']")

    slice_idx_in, slice_idx_out, slice_idx_run = slice_indices[:, slice_idx]

    mosaic_idx = 2 * slice_idx_in - 1
    if tilted_illumination:
        mosaic_idx += 1
    if no_tilted_illumination_scan:
        mosaic_idx = slice_idx_in

    dBI_canvas, R3D_canvas, O3D_canvas = [da.zeros((full_width, full_height, depth),
                                                   chunks=(
                                                   tile_width * 2, tile_height * 2,
                                                   depth), dtype=np.float32) for _ in
                                          range(3)]
    canvas = da.zeros((full_width, full_height, depth,3), chunks=(
                                                   tile_width, tile_height,
                                                   depth, 3), dtype=np.float32)
    weight = da.zeros((full_width, full_height), chunks=(tile_width, tile_height),
                     dtype=np.float32)
    if is_fiji:
        pass
        # following line is commented out in original code
        # MapIndex = Exp['MapIndex_Tot'][:,:, sl_out]

    dBI_record, R3D_record, O3D_record, weight_record = [], [], [], []
    for (i, j), tile_idx in tqdm.tqdm(np.ndenumerate(map_indices)):
        
        if tile_idx <= 0 or np.isnan(x_coords[i, j]) or np.isnan(
                y_coords[i, j]):
            continue
        x0, y0 = int(x_coords[i, j]), int(y_coords[i, j])
        row_slice = slice(x0, x0 + tile_width)
        col_slice = slice(y0, y0 + tile_height)

        input_file_name = input_file_format % (mosaic_idx, tile_idx)
        input_file_name = input_file_name.replace('[modality]', file_prefix)
        input_file_path = op.join(raw_data_dir, input_file_name)

        if file_type == 'mat':
            complex3d = _load_mat_tile(input_file_path)
        elif file_type == 'nifti':
            complex3d = _load_nifti_tile(input_file_path)


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

        results = da.stack([dBI3D, R3D, O3D], axis=3)
        results = results[clip_x:, clip_y:]*blend_ramp[:,:,None,None]

        canvas[row_slice, col_slice] += results
        weight[row_slice, col_slice] += blend_ramp

        #
        # if mosaic_idx > 8 and tilted_illumination:
        #     Nz = depth
        #     # 1) build a mask of pixels needing correction
        #     mask_crop = focus <= scan_info['Focus_CropStart']  # (Ny, Nx)
        #
        #     # 2) find, for each (y,x), the last index along z where dBI3D == -inf
        #     mask_inf = np.isneginf(dBI3D)  # (Ny, Nx, Nz)
        #     depth_idx = np.arange(Nz)[None, None, :]  # (1, 1, Nz)
        #     last_inf = np.max(np.where(mask_inf, depth_idx, -1), axis=2)
        #     # last_inf[y,x] is -1 if there was no -inf, or the maximum index otherwise
        #
        #     # 3) decide which pixels actually get shifted
        #     do_shift = (mask_crop & (last_inf >= 0))  # (Ny, Nx)
        #
        #     # 4) build the "shifted" index array for every (y,x,z)
        #     #    new_idx[y,x,z] = last_inf[y,x] + z
        #     new_idx = last_inf[..., None] + depth_idx  # (Ny, Nx, Nz)
        #
        #     # 5) clip to valid range so we can safely gather,
        #     #    then gather each volume
        #     clipped = np.clip(new_idx, 0, Nz - 1)
        #     dBI3D_shift = np.take_along_axis(dBI3D, clipped, axis=2)
        #     R3D_shift = np.take_along_axis(R3D, clipped, axis=2)
        #     O3D_shift = np.take_along_axis(O3D, clipped, axis=2)
        #
        #     # 6) zero-out any positions that “fell off the end” (new_idx >= Nz)
        #     overflow = new_idx >= Nz
        #     dBI3D_shift[overflow] = 0
        #     R3D_shift[overflow] = 0
        #     O3D_shift[overflow] = 0
        #
        #     # 7) select per-pixel whether to keep the original or use the shifted version
        #     dBI3D = np.where(do_shift[..., None], dBI3D_shift, dBI3D)
        #     R3D = np.where(do_shift[..., None], R3D_shift, R3D)
        #     O3D = np.where(do_shift[..., None], O3D_shift, O3D)
        #
        # for arr, canvas in zip([dBI3D, R3D, O3D], [dBI_record, R3D_record, O3D_record]):
        #     if scan_info['System'].lower() == 'octopus':
        #         arr = np.swapaxes(arr, 0, 1)
        #     if flip_z:
        #         arr = arr[:, :, ::-1]
        #     arr = arr[clip_x:, clip_y:]
        #     if modality == "mus" and file_type == 'nifti':
        #         raise NotImplementedError("mus 3d stitching not updated")
        #         arr = process_mus_nifti(arr, depth)
        #     else:
        #         arr = arr[:, :, :depth]
        #     arr *= blend_ramp[..., None]

            #
            # adding = da.zeros_like(dBI_canvas)
            # adding[row_slice, col_slice, :] =
            # canvas.append(adding)
        #
        # adding = da.zeros_like(weight)
        # adding[row_slice, col_slice] = blend_ramp
        # weight_record.append(adding)


    # dBI_canvas = da.sum(da.stack(dBI_record, axis=0), axis=0)
    # R3D_canvas = da.sum(da.stack(R3D_record, axis=0), axis=0)
    # O3D_canvas = da.sum(da.stack(O3D_record, axis=0), axis=0)
    # weight = da.sum(da.stack(weight_record, axis=0), axis=0)
    # Normalize the canvases by the weight
    # dBI_canvas /= weight[..., None]
    # R3D_canvas /= weight[..., None]
    # O3D_canvas /= weight[..., None]
    # dBI_canvas = da.nan_to_num(dBI_canvas)
    # R3D_canvas = da.nan_to_num(R3D_canvas)
    # O3D_canvas = da.nan_to_num(O3D_canvas)
    canvas /= weight[..., None, None]
    canvas = canvas.rechunk({3:1})
    dBI_canvas, R3D_canvas, O3D_canvas = canvas[..., 0], canvas[..., 1], canvas[..., 2]
    return [dBI_canvas, R3D_canvas, O3D_canvas]


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

