import os
import numpy as np
import dask.array as da
import scipy.io as sio
import nibabel as nib
import tensorstore as ts
import warnings



def mosaic3d_telesto(
        parameter_file: str,
        out_zarr: str,
        downsample_zarr: str = None
) -> None:
    """


    """
    # Load parameters
    params = sio.loadmat(parameter_file, squeeze_me=True)
    Parameters = params['Parameters']
    Scan = params['Scan']
    Mosaic3D = params['Mosaic3D']

    modality = Mosaic3D['modality']
    print(f"--Modality is {modality}--")

    # Input volumes index
    sliceidx = Mosaic3D['sliceidx']  # shape (3, N)

    # Load Experiment struct
    exp_file = Mosaic3D['Exp']
    content = sio.whosmat(exp_file)
    keys = [k[0] for k in content]
    exp_key = next((k for k in keys if 'Experiment_Fiji' in k),
                   next((k for k in keys if 'Experiment' in k), None))
    if exp_key is None:
        raise ValueError("No Experiment found in .mat file")
    Experiment = sio.loadmat(exp_file, squeeze_me=True)[exp_key]

    # Directories and file settings
    indir = list(Mosaic3D['indir'].flat)
    file_format = Mosaic3D['FileNameFormat'][0]
    filetype = Mosaic3D['InFileType']
    out_dir = os.path.dirname(out_zarr)
    os.makedirs(out_dir, exist_ok=True)

    # Clipping
    system = Scan['System']
    if system == 'Octopus':
        XPixClip = int(Parameters['YPixClip'])
        YPixClip = int(Parameters['XPixClip'])
    else:
        XPixClip = int(Parameters['XPixClip'])
        YPixClip = int(Parameters['YPixClip'])

    # Flip in Z?
    flag_flip = bool(Mosaic3D.get('invert_Z', True))

    # Mosaic geometry
    NbPix = int(Experiment['NbPix'])
    X = np.array(Experiment['X_Mean'], dtype=float)
    Y = np.array(Experiment['Y_Mean'], dtype=float)
    X -= X.min() - 1
    Y -= Y.min() - 1
    sizerow = NbPix - XPixClip
    sizecol = NbPix - YPixClip
    MXL = int(np.nanmax(X) + sizerow - 1)
    MYL = int(np.nanmax(Y) + sizecol - 1)
    MZL = int(Mosaic3D['MZL'])

    # Blending ramp
    dx = np.nanmedian(np.diff(Experiment['X_Mean'], axis=0))
    dy = np.nanmedian(np.diff(Experiment['Y_Mean'], axis=1))
    ramp_x = sizerow - int(round(dx))
    ramp_y = sizecol - int(round(dy))
    xv = np.linspace(0, 1, ramp_x)
    yv = np.linspace(0, 1, ramp_y)
    x = np.ones(sizerow)
    y = np.ones(sizecol)
    x[:ramp_x] = xv
    x[-ramp_x:] = xv
    y[:ramp_y] = yv
    y[-ramp_y:] = yv
    Ramp2D = np.outer(x, y)
    Ramp3D = np.broadcast_to(Ramp2D[:, :, None], (MXL, MYL, MZL))

    # Map indices
    MapIndex = Experiment['MapIndex_Tot_offset'] + Experiment['First_Tile'] - 1

    # Number of slices
    num_slices = sliceidx.shape[1]

    # Prepare list of lazy slice arrays
    slices = []
    for s in range(num_slices):
        sl_in, sl_out, sl_run = map(int, sliceidx[:, s])
        indir_curr = indir[sl_run - 1]
        base = os.path.join(indir_curr, file_format)

        # Accumulators as dask arrays
        M = da.zeros((MXL, MYL, MZL), dtype=np.float32, chunks=(MXL, MYL, MZL))
        Ma = da.zeros((MXL, MYL), dtype=np.float32, chunks=(MXL, MYL))

        # Loop tiles
        rows, cols = MapIndex.shape
        for jj in range(rows):
            for ii in range(cols):
                idx = int(MapIndex[jj, ii])
                if idx <= 0 or np.isnan(X[jj, ii]):
                    continue
                r0 = int(X[jj, ii])
                c0 = int(Y[jj, ii])
                tile_slice = slice(r0, r0 + sizerow)
                tile_col   = slice(c0, c0 + sizecol)
                currtile = (sl_in - 1) * int(Experiment['TilesPerSlice']) + idx

                # Lazy load array
                if filetype.lower() == 'nifti':
                    volname = get_volname(base, currtile, 'cropped')
                    img = nib.load(volname)
                    data = da.from_array(img.dataobj, chunks='auto')
                else:  # .mat
                    fname = os.path.join(indir_curr, file_format.replace('%tileID', f"{currtile:03d}"))
                    mat = sio.loadmat(fname, squeeze_me=True)
                    var = next(k for k in mat if not k.startswith('__'))
                    data = da.from_array(mat[var], chunks='auto')

                # Clip and flip
                data = data[XPixClip:, YPixClip:, :MZL]
                if system == 'Octopus':
                    data = data.transpose((1, 0, 2))
                if flag_flip:
                    data = data[:, :, ::-1]

                # Convert dtype
                data = data.astype(np.float32)

                # Compute modality-specific data (e.g., mus or dBI)
                # Assuming direct use of first MZL frames
                patch = data

                # Blend
                M = M.at[tile_slice, tile_col, :].add(patch * Ramp3D)
                Ma = Ma.at[tile_slice, tile_col].add(Ramp2D)

        # Finalize slice
        Ma3 = da.broadcast_to(Ma[:, :, None], (MXL, MYL, MZL))
        mosaic = M / Ma3
        mosaic = da.where(da.isfinite(mosaic), mosaic, 0)
        slices.append(mosaic)

    # Stack into full volume: shape (num_slices, MXL, MYL, MZL)
    volume = da.stack(slices, axis=0)

    # Open TensorStore for output
    spec = {
        'driver': 'zarr',
        'kvstore': {'driver': 'file', 'path': out_zarr},
        'metadata': {
            'dtype': 'float32',
            'shape': list(volume.shape),
            'chunks': list(volume.chunksize)
        }
    }
    store = ts.open(spec, create=True).result()
    # Write Dask array into TensorStore
    ts.copy(volume, store).result()

    # Optional downsampling
    if downsample_zarr:
        sxyz = Mosaic3D['sxyz']
        # factors: floor(sxyz / voxel_size) e.g., (2,2,1)
        factors = (int(sxyz[0]), int(sxyz[1]), 1)
        down = do_downsample(volume, factors)
        ds_spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': downsample_zarr},
            'metadata': {
                'dtype': 'float32',
                'shape': list(down.shape),
                'chunks': list(down.chunksize)
            }
        }
        ds_store = ts.open(ds_spec, create=True).result()
        ts.copy(down, ds_store).result()

    print(f"Wrote {out_zarr}{' and ' + downsample_zarr if downsample_zarr else ''}")

def get_volname(base: str, num: int, modality: str) -> str:
    """
    Replace placeholders in the base filename.
    """
    vol = base.replace('%tileID', f"{num:03d}")
    vol = vol.replace('%modality', modality)
    return vol


def do_downsample(volume: da.Array, factors: tuple) -> da.Array:
    """
    Downsample a Dask array by taking block-wise mean over given integer factors.
    factors: (fx, fy, fz)
    """
    fx, fy, fz = factors
    return da.coarsen(np.mean, volume, {0: fx, 1: fy, 2: fz}, trim_excess=True)
