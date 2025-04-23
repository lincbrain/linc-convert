import os.path as op
import logging
from colorsys import hsv_to_rgb

import cyclopts
import imageio
import numpy as np
import scipy.io as sio
import nibabel as nib
import dask
import dask.array as da
import tensorstore as ts
from scipy.io import loadmat
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
        # inp:str,
        parameter_file: str,
        *,

        modality: str,
        method: str = None,
        zarr_config: ZarrConfig = None,
) -> None:
    """
    Python translation of Mosaic2D_Telesto.

    Parameters
    ----------
    inp : str
        Paths

    """
    # Load .mat containing Parameters, Scan, and Mosaic2D
    params = sio.loadmat(parameter_file, squeeze_me=True)
    Parameters = struct_arr_to_dict(params['Parameters'])
    Scan = struct_arr_to_dict(params['Scan'])
    Mosaic2D = struct_arr_to_dict(params['Mosaic2D'])

    use_method = method is not None
    logger.info(f"--Modality is {modality}--")

    sliceidx = atleast_2d_trailing(Mosaic2D['sliceidx'])

    Exp, is_fiji = find_experiment_params(Mosaic2D['Exp'])

    if use_method:
        Exp['X_Mean'] = Exp[method]['X_Mean']
        Exp['Y_Mean'] = Exp[method]['Y_Mean']

    indir = np.atleast_1d(Mosaic2D['indir'])

    file_format = Mosaic2D['file_format']
    filetype = Mosaic2D['InFileType'].lower()
    transpose_flag = bool(Parameters['transpose'])

    if Scan['System'] == 'Octopus':
        xp, yp = Parameters['YPixClip'], Parameters['XPixClip']
    else:
        xp, yp = Parameters['XPixClip'], Parameters['YPixClip']
    xp, yp = int(xp), int(yp)

    # Mosaic geometry
    nb_pix = int(Exp['NbPix'])
    size_row, size_column = nb_pix - xp, nb_pix - yp
    # need to copy here
    X, Y = Exp['X_Mean'], Exp['Y_Mean']
    X -= np.nanmin(X)
    Y -= np.nanmin(Y)
    MXL = int(np.nanmax(X) + size_row)
    MYL = int(np.nanmax(Y) + size_column)
    MZL = 4 if modality =='Orientation' else 1

    # Precompute blending ramps
    dx = np.nanmedian(np.diff(X, axis=0))
    dy = np.nanmedian(np.diff(Y, axis=1))
    rx, ry = size_row - round(dx), size_column - round(dy)
    xv = np.linspace(0, 1, int(rx))
    yv = np.linspace(0, 1, int(ry))
    x = np.ones(size_row)
    x[:rx], x[-rx:] = xv, xv[::-1]
    y = np.ones(size_column)
    y[:ry], y[-ry:] = yv, yv[::-1]
    RampOrig = da.from_array(np.outer(x, y)[:, :], chunks=(size_row, size_column))

    GrayRange = None
    savetiff = True
    noscale = False
    key_map = {
        'aip': 'AipGrayRange',
        'mip': 'MipGrayRange',
        'retardance': 'RetGrayRange',
        'mus': 'musGrayRange',
        'surf': 'surfGrayRange'
    }
    lower_mod = modality.lower()
    if lower_mod in key_map:
        GrayRange = Parameters.get(key_map[lower_mod])
        if GrayRange is None:
            noscale = True
    else:
        dyn_key = f"{modality}GrayRange"
        GrayRange = Parameters.get(dyn_key)
        if GrayRange is None:
            noscale = True
    if noscale:
        savetiff = False
        logger.warning(
            f"{modality} grayscale range not found. Only MAT file will be saved.")

    # ----- Modality string when inconsistency in file names -----
    modality_base = params['Enface']['inputstr'].item()  # assumed list of strings
    # find entry containing first 3 letters of modality
    modality_str = next((s for s in modality_base if lower_mod[:3] in s.lower()),
                        None)
    if modality_str is None:
        modality_str = modality[:3]
        logger.warning(f"{modality} not in Enface.inputstr. Mosaic2D might fail.")

    # using float32 sample
    sample_dtype = np.float32

    MapIndex = Exp['MapIndex_Tot_offset'] + Exp['First_Tile'] - 1
    num_slices = sliceidx.shape[1]
    mosaics = {}

    def build_slice(s: int) -> da.Array:
        sl_in, sl_out, sl_run = sliceidx[:, s]
        indir_curr = indir[int(sl_run) - 1]
        base = op.join(indir_curr, file_format)

        M = da.zeros((MXL, MYL, MZL), dtype=np.float32, chunks='auto')
        Ma = da.zeros((MXL, MYL), dtype=np.float32, chunks='auto')


        for (jj, ii), idx in np.ndenumerate(MapIndex):
            if idx <= 0 or np.isnan(X[jj, ii]):
                continue
            r0 = int(X[jj, ii])
            c0 = int(Y[jj, ii])
            row_slice = slice(r0, r0 + size_row)
            col_slice = slice(c0, c0 + size_column)
            tile_id = (sl_in - 1) * int(Exp['TilesPerSlice']) + idx
            if filetype == 'nifti':
                if modality == "mus":
                    path = get_volname(base, tile_id, "cropped")
                else:
                    path = get_volname(base, tile_id, modality_str)
                header = nib.load(path)
                shape = header.shape
                dtype = header.get_data_dtype()
                delayed_arr = dask.delayed(header.get_fdata)()

            elif filetype == 'mat':
                mat_path = get_volname(base, tile_id, modality_str)
                name, shape, dtype = None, None, None
                for name, shape, dtype in sio.whosmat(mat_path):
                    if not name.startswith("__"):
                        break
                if not name:
                    raise ValueError("Variable not found")
                delayed_arr = dask.delayed(load_mat)(mat_path, name)
            arr = da.from_delayed(delayed_arr,
                                  shape=shape,
                                  dtype=dtype)
            if Scan['System'].lower() == 'octopus':
                arr = np.swapaxes(arr, 0,1)
            if transpose_flag:
                arr = np.swapaxes(arr, 0,1)
            arr = arr[xp:,yp:]
            if modality == "mus" and filetype == 'nifti':
                # shape (H, W, N)
                I = 10.0 ** (arr / 10.0)

                # compute, for each k=0..MZL-1, the sum I[:, :, k+1:]
                # via a reverse cumulative-sum trick:
                # reverse along the 3rd axis  → shape (H,W,N)
                I_rev = I[:, :, ::-1]
                # cumulative sum in reversed order → (H,W,N)
                cumsum_rev = np.cumsum(I_rev,
                                       axis=2)
                # drop the very first (no “next” beyond the last) → (H,W,N-1)
                sum_excl_rev = cumsum_rev[:, :, :-1]
                # flip back to original order → (H,W,N-1)
                sum_excl = sum_excl_rev[:, :, ::-1]

                # Now sum_excl[..., k] == sum of I[..., k+1:] exactly as in MATLAB’s sum(I(:,:,z+1:end),3)
                sum_excl = sum_excl[:, :,
                           :MZL]  # keep only the first MZL sums → (H,W,MZL)

                # divide elementwise and apply the constant factors
                I = I[:, :, :MZL] / sum_excl / (2.0 * 0.0025)
                arr = np.squeeze(np.mean(I, axis=2))


            # blend
            if lower_mod == 'orientation':
                rad = np.deg2rad(arr)
                c, s = np.cos(rad), np.sin(rad)
                SLO_OC = np.stack([c * c, c * s, c * s, s*s], axis=-1)
                # add weighted outer products
                M[row_slice, col_slice] += SLO_OC * RampOrig[...,None]
                Ma[row_slice, col_slice] += RampOrig
            else:
                if arr.ndim ==2:
                    M[row_slice, col_slice] += (arr* RampOrig)[..., None]
                else:
                    M[row_slice, col_slice] += arr * RampOrig
                Ma[row_slice, col_slice] += RampOrig

        modalstr_map = {
            'mip': 'MIP',
            'aip': 'AIP',
            'retardance': 'Retardance',
            'orientation': 'Orientation',
            'mus': 'mus'
        }
        modalstr = modalstr_map.get(lower_mod, modality)
        avg = M / Ma[:, :, None]

        if lower_mod == 'orientation':
            avg = np.array(avg)
            ref = loadmat(op.join(
                "/local_mount/space/megaera/1/users/kchai/psoct/process_data/StitchingFiji",
                f"{modalstr}_slice{1:03d}.raw.mat"))['M']
            np.testing.assert_array_almost_equal(ref, avg)
            print("Starting orientation angles eigen decomp...")
            h, w = M.shape[:2]
            a_x = avg.reshape((h * w, 2, 2))
            eigvals, eigvecs  = np.linalg.eigh(a_x)
            x = eigvecs[:, 0, 1]
            y = eigvecs[:, 1, 1]
            O = np.arctan2(y, x) / np.pi * 180
            O = O.reshape((h, w))
            # comp0, comp1, comp2, comp3 = avg[:, :, 0], avg[:, :, 1], avg[:, :, 2], avg[
            #                                                                        :, :,
            #                                                                        3]
            # O = np.zeros((MXL, MYL))
            # for i in range(MXL):
            #     for j in range(MYL):
            #         mat2 = np.array(
            #             [[comp0[i, j], comp1[i, j]], [comp2[i, j], comp3[i, j]]])
            #         w, v = np.linalg.eigh(mat2)
            #         principal = v[:, np.argmax(w)]
            #         O[i, j] = np.degrees(np.arctan2(principal[1], principal[0]))
            O[O < -90] += 180
            O[O > 90] -= 180
            MosaicFinal = np.rot90(O, k=-1)
            # ref = loadmat(op.join(
            #     "/local_mount/space/megaera/1/users/kchai/psoct/process_data/StitchingFiji",
            #     f"{modalstr}_slice{1:03d}.mat"))['MosaicFinal']
            # np.testing.assert_array_almost_equal(ref, MosaicFinal)

            #
            # # save orientation mosaic

            #
            # # masking orientation
            # RetSliceTiff = os.path.join(outdir,
            #                             f"Retardance_slice{sliceid_out:03d}.tiff")
            # AipSliceTiff = os.path.join(outdir, f"AIP_slice{sliceid_out:03d}.tiff")
            # data1 = imageio.imread(RetSliceTiff)
            # data2 = imageio.imread(AipSliceTiff)
            # data4 = wiener(data2, (5, 5))
            #
            # O_norm = (O + 90) / 180
            #
            # # Orientation1
            # I1 = exposure.rescale_intensity(data1.astype(np.float64),
            #                                 in_range='image') / (1 - 0.4)
            # I1 = np.clip(I1, 0, 1)
            # map3D = np.ones((*O_norm.shape, 3))
            # map3D[..., 0] = O_norm
            # map3D[..., 2] = I1
            # maprgb = hsv2rgb(map3D)
            # imageio.imwrite(
            #     op.join("/local_mount/space/megaera/1/users/kchai/psoct", f"{modalstr}1_slice{1:03d}.tiff"),
            #     (maprgb * 255).astype(np.uint8))

            return MosaicFinal
            #
            # # Orientation2
            # I2 = (-exposure.rescale_intensity(data2.astype(np.float64),
            #                                   in_range='image') + 1 - 0.2) / (
            #                  (1 - 0.2) * 0.5)
            # I2 = np.clip(I2, 0, 1)
            # I2[data4 <= 20] = 0
            # map3D2 = np.ones((*O_norm.shape, 3))
            # map3D2[..., 0] = O_norm
            # map3D2[..., 2] = I2
            # maprgb2 = hsv2rgb(map3D2)
            # imageio.imwrite(
            #     os.path.join(outdir, f"{modalstr}2_slice{sliceid_out:03d}.tiff"),
            #     (maprgb2 * 255).astype(np.uint8))
            #
            # # Orientation3
            # map3D3 = np.ones((*O_norm.shape, 3))
            # map3D3[..., 0] = O_norm
            # maprgb3 = hsv2rgb(map3D3)
            # imageio.imwrite(
            #     os.path.join(outdir, f"{modalstr}3_slice{sliceid_out:03d}.tiff"),
            #     (maprgb3 * 255).astype(np.uint8))

        else:
            avg = np.squeeze(avg)
            MosaicFinal = np.rot90(avg, k=-1)
            MosaicFinal = np.nan_to_num(MosaicFinal)

            # if savetiff:
            #     print("Saving .tiff mosaic...")
            nonlocal GrayRange
            if isinstance(GrayRange,np.ndarray):
                GrayRange =tuple(GrayRange[:])
            normed = exposure.rescale_intensity(MosaicFinal,
                                                in_range=GrayRange if GrayRange is not None else 'image')
            imageio.imwrite(
                op.join("/local_mount/space/megaera/1/users/kchai/psoct", f"{modalstr}_slice{1:03d}.tiff"),
                (normed * 255).astype(np.uint8))
            # print("Saving .mat mosaic...")
            # savemat(os.path.join(outdir, f"{modalstr}_slice{sliceid_out:03d}.mat"),
            #         {'MosaicFinal': MosaicFinal})
            ref = loadmat(op.join(
                "/local_mount/space/megaera/1/users/kchai/psoct/process_data/StitchingFiji",
                f"{modalstr}_slice{1:03d}.mat"))['MosaicFinal']
            diff = np.abs(ref-MosaicFinal).compute()
            i,j = np.where(diff==np.max(diff))
            i,j = ref.shape[-1]-j-1, i
            print(i,j)
            np.testing.assert_array_almost_equal(ref, MosaicFinal,decimal=4)
            return MosaicFinal

    slices = [build_slice(s) for s in range(num_slices)]
    arr = da.stack(slices,axis=-1)
    chunk, shard = compute_zarr_layout(arr.shape, arr.dtype, zarr_config)
    wconfig = default_write_config(zarr_config.out, arr.shape, dtype=np.float32,
                                   chunk=chunk, shard=shard)
    wconfig["create"] = True
    wconfig["delete_existing"] = True

    tswriter = ts.open(wconfig).result()
    arr.store(tswriter)

    return None


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

def get_volname(base_file_name: str, num: int, modality: str) -> str:
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
    vol_name = base_file_name
    vol_name = vol_name.replace("%tileID", f"{num:03d}")
    vol_name = vol_name.replace("%modality", modality)
    return vol_name