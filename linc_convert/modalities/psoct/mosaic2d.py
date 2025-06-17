"""
This module includes components that are based on the https://github.com/CarolineMagnain/OCTAnalysis repository (currently private).  We have permission from the owner to include the code here.
"""

import logging
import os.path as op
from colorsys import hsv_to_rgb
from typing import Optional, Tuple, Dict

import cyclopts
import dask
import dask.array as da
import imageio
import nibabel as nib
import numpy as np
import scipy.io as sio
import tensorstore as ts
from skimage.exposure import exposure

from linc_convert.modalities.psoct.cli import psoct
from linc_convert.modalities.psoct.utils._utils import struct_arr_to_dict, \
    atleast_2d_trailing, find_experiment_params, load_mat
from linc_convert.modalities.psoct.utils._zarr import default_write_config
from linc_convert.utils.zarr import ZarrConfig
from linc_convert.utils.zarr.create_array import compute_zarr_layout

logger = logging.getLogger(__name__)
mosaic2d = cyclopts.App(name="mosaic2d", help_format="markdown")
psoct.command(mosaic2d)


@mosaic2d.default
def mosaic2d_telesto(
        parameter_file: str,
        *,
        modality: str,
        method: str = None,
        tilted_illumination: bool = False,
        tiff_output_dir: str = None,
        zarr_config: ZarrConfig = None,
) -> None:
    """
    Python translation of Mosaic2D_Telesto.

    Parameters
    ----------
    inp : str
        Paths

    """
    logger.info(f"--Modality is {modality}--")
    tilt_postfix = '_tilt' if tilted_illumination else ''

    # Load .mat containing Parameters, Scan, and Mosaic2D
    raw_mat = sio.loadmat(parameter_file, squeeze_me=True)
    params = struct_arr_to_dict(raw_mat['Parameters'])
    scan_info = struct_arr_to_dict(raw_mat['Scan'])
    mosaic_info = struct_arr_to_dict(raw_mat['Mosaic2D'])
    exp_params, is_fiji = find_experiment_params(mosaic_info['Exp'])

    slice_indices = atleast_2d_trailing(mosaic_info['sliceidx'])
    input_dirs = np.atleast_1d(mosaic_info['indir'])
    file_format_template = mosaic_info['file_format']
    file_type = mosaic_info['InFileType'].lower()
    transpose_needed = bool(params['transpose'])
    no_tilted_illumination_scan = scan_info['TiltedIllumination'] != 'Yes'

    clip_x, clip_y = int(params['XPixClip']), int(params['YPixClip'])
    if scan_info['System'].lower() == 'octopus':
        clip_x, clip_y = clip_y, clip_x

    tile_size = int(exp_params['NbPix'])
    tile_width, tile_height = tile_size - clip_x, tile_size - clip_y
    if tilted_illumination:
        tile_width = int(exp_params['NbPix' + tilt_postfix]) - clip_x
    x_coords, y_coords = _normalize_tile_coords(
        exp_params[method]['X_Mean'] if method else exp_params['X_Mean' + tilt_postfix],
        exp_params[method]['Y_Mean'] if method else exp_params['Y_Mean' + tilt_postfix],
    )
    full_width = int(np.nanmax(x_coords) + tile_width)
    full_height = int(np.nanmax(y_coords) + tile_height)
    depth = 4 if modality.lower() == 'orientation' else 1
    blend_ramp = _compute_blending_ramp(tile_width, tile_height, x_coords, y_coords)

    gray_range, save_tiff = _get_gray_range(params, modality)
    if not save_tiff:
        tiff_output_dir = None

    modality_token = _select_modality_token(raw_mat, modality)
    map_indices = exp_params['MapIndex_Tot_offset' + tilt_postfix] + exp_params[
        'First_Tile' + tilt_postfix] - 1

    num_slices = slice_indices.shape[1]
    slices = [
        build_slice(s, slice_indices=slice_indices, scan_info=scan_info,
                    exp_params=exp_params, input_dirs=input_dirs,
                    file_format_template=file_format_template, file_type=file_type,
                    modality=modality, modality_token=modality_token,
                    blend_ramp=blend_ramp, map_indices=map_indices,
                    tile_width=tile_width, tile_height=tile_height,
                    full_width=full_width, full_height=full_height, depth=depth,
                    x_coords=x_coords, y_coords=y_coords, clip_x=clip_x, clip_y=clip_y,
                    transpose_needed=transpose_needed, tiff_output_dir=tiff_output_dir,
                    gray_range=gray_range, tilted_illumination=tilted_illumination,
                    no_tilted_illumination_scan = no_tilted_illumination_scan) for s in
        range(num_slices)]
    arr = da.stack(slices,axis=-1)
    #TODO: stack 2d -> nifti = arr.rot90.swapaxes(0,1)
    chunk, shard = compute_zarr_layout(arr.shape, arr.dtype, zarr_config)
    write_cfg = default_write_config(op.join(zarr_config.out, '0'), arr.shape, dtype=np.float32,
                                   chunk=chunk, shard=shard)
    if shard:
        arr = da.rechunk(arr,chunks=shard)
    else:
        arr = da.rechunk(arr,chunks=chunk)
    write_cfg["create"] = True
    write_cfg["delete_existing"] = True

    tswriter = ts.open(write_cfg).result()
    arr.store(tswriter)

    return None


def build_slice(slice_idx, slice_indices, scan_info, exp_params, input_dirs,
                file_format_template, file_type, modality, modality_token, blend_ramp,
                map_indices, tile_width, tile_height, full_width, full_height, depth,
                x_coords, y_coords, clip_x, clip_y, transpose_needed, tiff_output_dir,
                gray_range, tilted_illumination, no_tilted_illumination_scan):
    lower_mod = modality.lower()
    slice_idx_in, slice_idx_out, slice_idx_run = slice_indices[:, slice_idx]
    mosaic_idx = 2 * slice_idx_in - 1
    if tilted_illumination:
        mosaic_idx += 1
    if no_tilted_illumination_scan:
        mosaic_idx = slice_idx_in

    input_dir = input_dirs[int(slice_idx_run) - 1]
    file_path_template = op.join(input_dir, file_format_template)
    canvas = da.zeros((full_width, full_height, depth), dtype=np.float32,
                      chunks='auto')
    weight = da.zeros((full_width, full_height), dtype=np.float32, chunks='auto')
    for (index_row, index_column), tile_idx in np.ndenumerate(map_indices):
        if tile_idx <= 0 or np.isnan(x_coords[index_row, index_column]) or np.isnan(
                y_coords[index_row, index_column]):
            continue
        r0 = int(x_coords[index_row, index_column])
        c0 = int(y_coords[index_row, index_column])
        row_slice = slice(r0, r0 + tile_width)
        col_slice = slice(c0, c0 + tile_height)
        tile_id = tile_idx

        if file_type == 'mat':
            arr = _load_mat_tile(
                get_volname(file_path_template, mosaic_idx, tile_id, modality_token))
        elif file_type == 'nifti':
            if modality == "mus":
                arr = _load_nifti_tile(
                    get_volname(file_path_template, mosaic_idx, tile_id, "cropped"))
            else:
                arr = _load_nifti_tile(
                    get_volname(file_path_template, mosaic_idx, tile_id, modality_token))
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        if scan_info['System'].lower() == 'octopus':
            arr = np.swapaxes(arr, 0, 1)
        if transpose_needed:
            arr = np.swapaxes(arr, 0, 1)
        arr = arr[clip_x:, clip_y:]

        if modality == "mus" and file_type == 'nifti':
            # TODO: is this necessary in Mosaic2D?
            # as total_depth will be 1 and we will crop 1 depth
            # TODO: in the updated version, it was said 2d mus is not updated yet so skip it
            raise NotImplementedError
            arr = process_mus_nifti(arr, depth)

        if lower_mod == 'orientation':
            rad = np.deg2rad(arr)
            c, s = np.cos(rad), np.sin(rad)
            arr = np.stack([c * c, c * s, c * s, s * s], axis=-1)

        if arr.ndim == 2:
            canvas[row_slice, col_slice] += (arr * blend_ramp)[..., None]
        else:
            canvas[row_slice, col_slice] += arr * blend_ramp[..., None]
        weight[row_slice, col_slice] += blend_ramp

    canvas /= weight[:, :, None]

    tilt_postfix = '_tilt' if tilted_illumination else ''
    if lower_mod == 'orientation':
        print("Starting orientation angles eigen decomp...")
        h, w = canvas.shape[:2]
        a_x = canvas.reshape((h * w, 2, 2))
        eigvals, eigvecs = np.linalg.eigh(a_x)
        x = eigvecs[:, 0, 1]
        y = eigvecs[:, 1, 1]
        canvas = np.arctan2(y, x) / np.pi * 180
        canvas = canvas.reshape((h, w))
        canvas[canvas < -90] += 180
        canvas[canvas > 90] -= 180
        canvas = np.rot90(canvas, k=-1)
    else:
        canvas = np.squeeze(canvas)
        canvas = np.nan_to_num(canvas)
        canvas = np.rot90(canvas, k=-1)
        if tiff_output_dir:
            logging.info("Save .tiff mosaic")
            normed = exposure.rescale_intensity(
                canvas,
                in_range=gray_range if gray_range is not None else 'image')
            imageio.imwrite(
                op.join(tiff_output_dir, f"{modality}{tilt_postfix}_slice{slice_idx_out:03d}.tiff"),
                (normed * 255).astype(np.uint8))

        # ref = loadmat(op.join(
        #     "/local_mount/space/megaera/1/users/kchai/psoct/process_data/StitchingFiji",
        #     f"{modality}_slice{1:03d}.mat"))['MosaicFinal']
        # diff = np.abs(ref-canvas).compute()
        # i,j = np.where(diff==np.max(diff))
        # i,j = ref.shape[-1]-j-1, i
        # print(i,j)
        # np.testing.assert_array_almost_equal(ref, canvas,decimal=6)

    return canvas


def _load_nifti_tile(path: str) -> da.Array:
    img = nib.load(path, mmap=True)
    # return da.from_delayed(dask.delayed(img.dataobj), shape=img.shape, dtype=img.get_data_dtype())
    return da.from_array(img.dataobj, chunks=img.shape)
    delayed = dask.delayed(img.get_fdata)()
    return da.from_delayed(delayed, shape=img.shape, dtype=img.get_data_dtype())


def _load_mat_tile(path: str) -> da.Array:
    var_name, shape, dtype = next(
        (n, s, dt) for n, s, dt in sio.whosmat(path) if not n.startswith("__")
    )
    delayed = dask.delayed(load_mat)(path, var_name)
    return da.from_delayed(delayed, shape=shape, dtype=dtype)

def _load_tile(path:str):
    if path.endswith('.mat'):
        return _load_mat_tile(path)
    if path.endswith('.nii' or '.nii.gz'):
        return _load_nifti_tile(path)
    raise ValueError(f"Unsupported file type: {path}")

def _compute_blending_ramp(
        tile_width: int,
        tile_height: int,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
) -> da.Array:
    """
    Create a 2D blending weight map for overlapping tiles.
    """
    dx = np.nanmedian(np.diff(x_coords, axis=0))
    dy = np.nanmedian(np.diff(y_coords, axis=1))
    rx, ry = tile_width - round(dx), tile_height - round(dy)

    xv = np.linspace(0, 1, rx)
    yv = np.linspace(0, 1, ry)
    wx = np.ones(tile_width)
    wy = np.ones(tile_height)
    if rx > 0:
        wx[:rx], wx[-rx:] = xv, xv[::-1]
    if ry > 0:
        wy[:ry], wy[-ry:] = yv, yv[::-1]
    ramp = da.from_array(np.outer(wx, wy)[:, :], chunks=(tile_width, tile_height))
    return ramp


def _normalize_tile_coords(x_arr: np.ndarray, y_arr: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray]:
    x0, y0 = np.nanmin(x_arr), np.nanmin(y_arr)
    return x_arr - x0, y_arr - y0


def _get_gray_range(
        params_dict: Dict[str, any], modality: str
) -> Tuple[Optional[tuple | str], bool]:
    """
    Determine gray-level range for a modality; return (range, save_tiff_flag).
    """
    key_map = {
        'aip': 'AipGrayRange',
        'mip': 'MipGrayRange',
        'retardance': 'RetGrayRange',
        'birefringence': 'BirefGrayRange',
        'mus': 'musGrayRange',
        'surf': 'surfGrayRange',
    }
    low = modality.lower()
    if low in key_map:
        gray_range = params_dict.get(key_map[low])
    else:
        gray_range = params_dict.get(f"{modality}GrayRange")
    if gray_range is None:
        logger.warning(
            f"{modality} grayscale range not found. TIFF output disabled."
        )
        return None, False
    if isinstance(gray_range, np.ndarray):
        gray_range = tuple(gray_range[:2])
    return gray_range, True


def _select_modality_token(raw_mat: dict[str, any], modality: str) -> str:
    """
    Choose the filename token matching the modality from Enface.inputstr.
    """
    candidates = list(raw_mat['Enface']['save'].item())
    prefix = modality[:3].lower()
    for token in candidates:
        if prefix in token.lower():
            return token
    logger.warning(
        f"Modality '{modality}' not found in Enface.save; using first 3 letters."
    )
    return modality[:3]


def get_rgb_4d(ori, value_range):
    """
    Convert an orientation volume (in degrees) into an RGB 4D volume via HSV mapping.
    value_range: [min_angle, max_angle] (e.g. [-90, 90])
    """
    r1, r2 = value_range
    ori = ori.astype(np.float64)

    # wrap values as in MATLAB version
    ori = np.where(ori > r2, ori - 180, ori)
    ori = np.where(ori < r1, ori + 180, ori)

    # normalize to [0,1] for hue
    hue = (ori - r1) / (r2 - r1)

    nx, ny, nz = ori.shape
    hsv = np.ones((nx, ny, nz, 3), dtype=np.float64)
    hsv[..., 0] = hue  # H
    hsv[..., 1] = 1.0  # S
    hsv[..., 2] = 1.0  # V

    rgb = hsv_to_rgb(hsv)
    rgb = (rgb * 255).astype(np.uint8)
    return rgb


def get_volname(base_file_name: str, mosaic_num: int, tile_num: int,
                modality: str) -> str:
    """
    Generate a volume name by replacing placeholders in the base file name.

    This function provides flexibility for atypical processing where input file
    naming deviates from the standard. It replaces '%tileID' with a zero-padded
    three-digit number and '%modality' with the specified modality.

    Args:
        base_file_name (str): The template file name containing placeholders.
        num (int): The tile ID number to insert (zero-padded to three digits).
        modality (str): The modality string to insert.

    Returns:
        str: The formatted volume name with placeholders replaced.
    """
    # vol_name = base_file_name
    # vol_name = vol_name.replace("%tileID", f"{num:03d}")
    vol_name = base_file_name % (mosaic_num, tile_num)
    vol_name = vol_name.replace("[modality]", modality)
    return vol_name


def process_mus_nifti(arr, total_depth):
    I = 10.0 ** (arr / 10.0)
    I_rev = I[:, :, ::-1]
    # cumulative sum in reversed order → (H,W,N)
    cumsum_rev = np.cumsum(I_rev, axis=2)
    # drop the very first (no “next” beyond the last) → (H,W,N-1)
    sum_excl_rev = cumsum_rev[:, :, :-1]
    # flip back to original order → (H,W,N-1)
    sum_excl = sum_excl_rev[:, :, ::-1]
    # Now sum_excl[..., k] == sum of I[..., k+1:] exactly as in MATLAB’s sum(I(:,:,z+1:end),3)
    sum_excl = sum_excl[:, :,
               :total_depth]  # keep only the first MZL sums → (H,W,MZL)
    # divide elementwise and apply the constant factors
    I = I[:, :, :total_depth] / sum_excl / (2.0 * 0.0025)
    arr = np.squeeze(np.mean(I, axis=2))
    return arr
