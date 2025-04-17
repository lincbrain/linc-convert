import os
import numpy as np
import scipy.io as sio
import nibabel as nib
import imageio
import warnings

import dask.array as da

from linc_convert.utils.zarr import ZarrConfig


def mosaic2d_telesto(
        inp:str,
        *,
        parameter_file: str,
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
    Parameters = params['Parameters']
    Scan = params['Scan']
    Mosaic2D = params['Mosaic2D']

    use_method = method is not None
    print(f"--Modality is {modality}--")

    sliceidx = Mosaic2D['sliceidx']  # 3xN array

    exp_file = Mosaic2D['Exp']
    mat_contents = sio.whosmat(exp_file)
    names = [m[0] for m in mat_contents]
    key = next((n for n in names if 'Experiment_Fiji' in n),
               next((n for n in names if 'Experiment' in n), None))
    if key is None:
        raise ValueError("No Experiment found in .mat")
    Experiment = sio.loadmat(exp_file, squeeze_me=True)[key]

    if use_method:
        Experiment['X_Mean'] = Experiment[method]['X_Mean']
        Experiment['Y_Mean'] = Experiment[method]['Y_Mean']

    indir = list(Mosaic2D['indir'].flat)
    file_format = Mosaic2D['file_format']
    filetype = Mosaic2D['InFileType']
    transpose_flag = bool(Parameters['transpose'])

    system = Scan['System']
    if system == 'Octopus':
        XPixClip = int(Parameters['YPixClip'])
        YPixClip = int(Parameters['XPixClip'])
    else:
        XPixClip = int(Parameters['XPixClip'])
        YPixClip = int(Parameters['YPixClip'])

    X = da.from_array(Experiment['X_Mean'], chunks='auto').astype(float)
    Y = da.from_array(Experiment['Y_Mean'], chunks='auto').astype(float)
    X = X - X.min() + 1
    Y = Y - Y.min() + 1
    NbPix = int(Experiment['NbPix'])
    sizerow = NbPix - XPixClip
    sizecol = NbPix - YPixClip
    MXL = int(da.nanmax(X).compute() + sizerow - 1)
    MYL = int(da.nanmax(Y).compute() + sizecol - 1)
    MZL = 4 if modality.lower() == 'orientation' else 1

    dx = da.nanmedian(X.diff(axis=0)).compute()
    dy = da.nanmedian(Y.diff(axis=1)).compute()
    ramp_x = sizerow - int(round(dx))
    ramp_y = sizecol - int(round(dy))
    x = da.ones((sizerow,), chunks=sizerow)
    y = da.ones((sizecol,), chunks=sizecol)
    xv = da.linspace(0, 1, ramp_x)
    yv = da.linspace(0, 1, ramp_y)
    x = da.concatenate([xv, da.ones((sizerow - ramp_x,), chunks=sizerow - ramp_x)])
    y = da.concatenate([yv, da.ones((sizecol - ramp_y,), chunks=sizecol - ramp_y)])
    RampOrig = x[:, None] * y[None, :]

    MapIndex = Experiment['MapIndex_Tot_offset'] + Experiment['First_Tile'] - 1
    num_slices = sliceidx.shape[1]
    mosaics = {}

    for s in range(num_slices):
        sl_in, sl_out, sl_run = sliceidx[:, s]
        indir_curr = indir[int(sl_run) - 1]
        base = os.path.join(indir_curr, file_format)

        M = da.zeros((MXL, MYL, MZL), dtype=np.float32, chunks='auto')
        Ma = da.zeros((MXL, MYL), dtype=np.float32, chunks='auto')

        rows, cols = MapIndex.shape
        for ii in range(cols):
            for jj in range(rows):
                idx = MapIndex[jj, ii]
                if idx > 0 and not np.isnan(Experiment['X_Mean'][jj, ii]):
                    r0 = int(Experiment['X_Mean'][jj, ii])
                    c0 = int(Experiment['Y_Mean'][jj, ii])
                    rowslice = slice(r0, r0 + sizerow)
                    colslice = slice(c0, c0 + sizecol)
                    currtile = int((sl_in - 1) * Experiment['TilesPerSlice'] + idx)

                    # lazy load with dask
                    if filetype.lower() == 'nifti':
                        img = nib.load(get_volname(base, currtile, modality[:3]))
                        proxy = img.dataobj
                        I = da.from_array(proxy, chunks='auto')
                    elif filetype.lower() == 'mat':
                        matname = get_volname(base, currtile, modality[:3])
                        matcontent = sio.loadmat(matname, squeeze_me=True)
                        varname = next(k for k in matcontent if not k.startswith('__'))
                        arr = matcontent[varname]
                        I = da.from_array(arr, chunks='auto')
                    else:
                        warnings.warn(f"Unknown filetype: {filetype}")
                        continue

                    if system.lower() == 'octopus':
                        I = I.T
                    if transpose_flag:
                        I = I.T

                    I = I[XPixClip:, YPixClip:]

                    if modality.lower() == 'orientation':
                        ang = da.deg2rad(I)
                        SLO_OC = da.stack([da.cos(ang) ** 2,
                                           da.cos(ang) * da.sin(ang),
                                           da.cos(ang) * da.sin(ang),
                                           da.sin(ang) ** 2], axis=-1)
                        M[rowslice, colslice, :] += SLO_OC * RampOrig[..., None]
                        Ma[rowslice, colslice] += RampOrig
                    else:
                        M[rowslice, colslice, 0] += I * RampOrig
                        Ma[rowslice, colslice] += RampOrig

        if modality.lower() == 'orientation':
            T = M / Ma[..., None]
            H = T.reshape((-1, 2, 2))
            vals, vecs = da.linalg.eig(H)
            idx_max = da.argmax(vals, axis=1)
            v = vecs[da.arange(H.shape[0]), :, idx_max]
            angles = da.rad2deg(da.arctan2(v[:, 1], v[:, 0])).reshape((MXL, MYL))
            O = da.rot90(angles, -1)
            mosaics[int(sl_out)] = O
        else:
            Z = M[..., 0] / Ma
            Z = da.rot90(Z, -1)
            Z = da.where(da.isnan(Z), 0, Z)
            mosaics[int(sl_out)] = Z

    return mosaics


def get_volname(base: str, num: int, modality: str) -> str:
    """
    Replace placeholders in the base filename.
    """
    vol = base.replace('%tileID', f"{num:03d}")
    vol = vol.replace('%modality', modality)
    return vol
