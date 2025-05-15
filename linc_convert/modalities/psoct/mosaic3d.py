import logging
import os.path as op
import warnings

import cyclopts
import dask.array as da
import dask
import nibabel as nib
import numpy as np
import scipy.io as sio
import tensorstore as ts
import tqdm

from linc_convert.modalities.psoct.cli import psoct
from linc_convert.modalities.psoct.mosaic2d import _normalize_tile_coords, \
    _compute_blending_ramp, process_mus_nifti, _load_mat_tile, _load_nifti_tile, \
    get_volname
from linc_convert.modalities.psoct.utils._utils import struct_arr_to_dict, \
    find_experiment_params, atleast_2d_trailing, load_mat
from linc_convert.modalities.psoct.utils._zarr import default_write_config
from linc_convert.utils.zarr.create_array import compute_zarr_layout
from linc_convert.utils.zarr.zarr_config import ZarrConfig

logger = logging.getLogger(__name__)
mosaic3d = cyclopts.App(name="mosaic3d", help_format="markdown")
psoct.command(mosaic3d)


@mosaic3d.default
def mosaic3d_telesto(
        parameter_file: str,
        *,
        modality:str,
        tilted_illumination: bool = False,
        downsample: bool = False,
        use_gpu: bool = False,
        zarr_config: ZarrConfig = None,
) -> None:
    logger.info(f"modality is {modality}")
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

    modality_base = raw_mat['Processed3D']['save'].item().flatten()
    substr = modality[:3]
    matches = [s for s in modality_base if substr.lower() in str(s).lower()]
    if matches:
        modality_token = matches[0]
    else:
        modality_token = substr
        warnings.warn(
            f"{modality} (current) modality is not included in Enface struct. "
            "Mosaic3D might fail."
        )


    # Map indices
    map_indices = exp_params['MapIndex_Tot_offset' + tilt_postfix] + exp_params['First_Tile' + tilt_postfix] - 1
    num_slices = slice_indices.shape[1]

    no_tilted_illumination_scan = scan_info['TiltedIllumination'] != 'Yes'
    # Build full volume graph
    slices = [build_slice(s,blend_ramp, clip_x, clip_y, full_width, full_height, depth,
                         exp_params, file_path_template, file_type, flip_z,
                         input_dirs, is_fiji, map_indices, modality, modality_token,
                         scan_info, slice_indices, tile_width, tile_height, x_coords,
                         y_coords, tilted_illumination, no_tilted_illumination_scan) for s in range(num_slices)]

    # if len(slices) > 1:
    #     volume = stack_slices(modality, mosaic_info, slice_indices, slices)
    # else:
    #     volume = slices[0]
    volume= slices[0]

    chunk, shard = compute_zarr_layout(volume.shape, volume.dtype, zarr_config)

    if shard:
        volume = da.rechunk(volume,chunks=shard)
    else:
        volume = da.rechunk(volume,chunks=chunk)

    wconfig = default_write_config(op.join(zarr_config.out, '0'),shape=volume.shape,dtype = np.float32, chunk=chunk,shard=shard)
    wconfig["create"] = True
    wconfig["delete_existing"] = True

    tswriter = ts.open(wconfig).result()
    volume.store(tswriter)


def stack_slices(modality, mosaic_info, slice_indices, slices):
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


def build_slice(slice_idx, blend_ramp, clip_x, clip_y, full_width, full_height, depth, exp_params,
                file_path_template, file_type, flip_z, input_dirs, is_fiji,
                map_indices, modality, modality_token, scan_info, slice_indices, tile_width,
                tile_height, x_coords, y_coords, tilted_illumination, no_tilted_illumination_scan):
    
    slice_idx_in, slice_idx_out, slice_idx_run = slice_indices[:, slice_idx]

    mosaic_idx = 2 * slice_idx + 1
    if tilted_illumination:
        mosaic_idx += 1
    if no_tilted_illumination_scan:
        mosaic_idx = slice_idx_in

    input_dir = input_dirs[slice_idx_run - 1]

    canvas = da.zeros((full_width, full_height, depth),
                     chunks=(tile_width*2, tile_height*2, depth), dtype=np.float32)
    weight = da.zeros((full_width, full_height), chunks=(tile_width*2, tile_height*2),
                     dtype=np.float32)
    # canvas = da.zeros((full_width, full_height, depth),
    #                  chunks=(full_width, full_height, depth), dtype=np.float32)
    # weight = da.zeros((full_width, full_height), chunks=(full_width, full_height),
    #                  dtype=np.float32)
    if is_fiji:
        pass
        # following line is commented out in original code
        # MapIndex = Exp['MapIndex_Tot'][:,:, sl_out]
    file_path_template = op.join(input_dir, file_path_template)
    for (index_row, index_column), tile_idx in tqdm.tqdm(np.ndenumerate(map_indices)):
        if tile_idx <= 0 or np.isnan(x_coords[index_row, index_column]) or np.isnan(
                y_coords[index_row, index_column]):
            continue
        r0, c0 = int(x_coords[index_row, index_column]), int(y_coords[index_row, index_column])
        row_slice = slice(r0, r0 + tile_width)
        col_slice = slice(c0, c0 + tile_height)
        tile_id = tile_idx

        # Load data lazily
        if file_type == 'mat':
            arr = _load_mat_tile(get_volname(file_path_template, mosaic_idx, tile_id, modality_token))
        elif file_type == 'nifti':
            arr = _load_nifti_tile(get_volname(file_path_template, mosaic_idx, tile_id, modality_token))
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        if scan_info['System'].lower() == 'octopus':
            arr = np.swapaxes(arr, 0, 1)
        if flip_z:
            arr = arr[:, :, ::-1]

        arr = arr[clip_x:, clip_y:, :]

        if modality == "mus" and file_type == 'nifti':
            raise NotImplementedError("mus 3d stitching not updated")
            arr = process_mus_nifti(arr, depth)
        else:
            arr = arr[:, :, :depth]
        canvas[row_slice, col_slice, :] += arr * blend_ramp[..., None]
        weight[row_slice, col_slice] += blend_ramp

    result = canvas / weight[..., None]
    result = da.nan_to_num(result)
    return result


def do_downsample(volume: da.Array, factors: tuple) -> da.Array:
    fx, fy, fz = factors
    return da.coarsen(np.mean, volume, {0: fx, 1: fy, 2: fz}, trim_excess=True)


