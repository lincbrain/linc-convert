import logging
import os.path as op

import cyclopts
import dask.array as da
import dask
import nibabel as nib
import numpy as np
import scipy.io as sio
import tensorstore as ts

from linc_convert.modalities.psoct.cli import psoct
from linc_convert.modalities.psoct.utils._utils import struct_arr_to_dict, \
    find_experiment_params, mat_vars, atleast_2d_trailing, load_mat
from linc_convert.modalities.psoct.utils._zarr import default_write_config
from linc_convert.utils.zarr.create_array import compute_zarr_layout
from linc_convert.utils.zarr.zarr_config import ZarrConfig

logger = logging.getLogger(__name__)
mosaic3d = cyclopts.App(name="mosaic3d", help_format="markdown")
psoct.command(mosaic3d)


@mosaic3d.default
def mosaic3d_telesto(
        # inp: str,
        parameter_file: str,
        *,
        # parameter_file: str,
        downsample: bool = False,
        use_gpu: bool = False,
        zarr_config: ZarrConfig = None,
) -> None:

    # Load parameters
    params = sio.loadmat(parameter_file, squeeze_me=True)
    Parameters, Scan, Mosaic3D = params['Parameters'], params['Scan'], params[
        'Mosaic3D']
    Parameters = struct_arr_to_dict(Parameters)
    Scan = struct_arr_to_dict(Scan)
    Mosaic3D = struct_arr_to_dict(Mosaic3D)
    # Load Experiment parameters
    Exp, is_fiji = find_experiment_params(Mosaic3D['Exp'])

    # load variables from parameters
    modality = Mosaic3D['modality']
    sliceidx = atleast_2d_trailing(Mosaic3D['sliceidx'])
    FileNameFormat = atleast_2d_trailing(Scan['FileNameFormat'])

    indir = np.atleast_1d(Mosaic3D['indir'])
    ftype = Mosaic3D['InFileType'].lower()

    # Clipping parameters
    if Scan['System'] == 'Octopus':
        xp, yp = Parameters['YPixClip'], Parameters['XPixClip']
    else:
        xp, yp = Parameters['XPixClip'], Parameters['YPixClip']
    xp, yp = int(xp), int(yp)
    flipZ = bool(Mosaic3D.get('invert_Z', True))

    # Mosaic geometry
    nb_pix = int(Exp['NbPix'])
    size_row, size_column = nb_pix - xp, nb_pix - yp
    X, Y = Exp['X_Mean'], Exp['Y_Mean']
    X -= np.nanmin(X)
    Y -= np.nanmin(Y)
    MXL = int(np.nanmax(X) + size_row)
    MYL = int(np.nanmax(Y) + size_column)
    MZL = int(Mosaic3D['MZL'])

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
    Ramp2D = da.from_array(np.outer(x, y)[:, :], chunks=(size_row, size_column))
    Ramp3D = Ramp2D[...,None]

    # Map indices
    MapIdx = Exp['MapIndex_Tot_offset'] + Exp['First_Tile'] - 1
    n_slices = sliceidx.shape[1]

    def build_slice(s: int) -> da.Array:
        sl_in, sl_out, run = sliceidx[:, s]
        base_dir = indir[run - 1]
        M_acc = da.zeros((MXL, MYL,MZL), chunks=(MXL, MYL, MZL), dtype=np.float32)
        W_acc = da.zeros((MXL, MYL), chunks=(MXL, MYL), dtype=np.float32)
        if is_fiji:
            pass
            # following line is commented out in original code
            # MapIndex = Exp['MapIndex_Tot'][:,:, sl_out]

        for (jj, ii), idx in np.ndenumerate(MapIdx):
            if idx <= 0 or np.isnan(X[jj, ii]):
                continue
            r0, c0 = int(X[jj, ii]), int(Y[jj, ii])
            row_range = slice(r0, r0 + size_row)
            column_range = slice(c0, c0 + size_column)
            tile_id = (sl_in - 1) * int(Exp['TilesPerSlice']) + idx

            # Load data lazily
            if ftype == 'nifti':
                if tile_id>120:
                    # do not know why, from matlab code
                    tile_id -= 1
                path = op.join(base_dir, f"test_processed_{tile_id:03d}_cropped.nii")
                header = nib.load(path)
                shape = header.shape
                dtype = header.get_data_dtype()
                delayed_arr = dask.delayed(header.get_fdata)()

            else:
                mat_path = op.join(base_dir, FileNameFormat[0,0]%tile_id)
                # arr = sio.loadmat(mat_path, squeeze_me=True)[next(mat_vars(mat_path))]
                # arr = da.from_array(arr, chunks='auto')
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

            arr = arr.astype(np.float32)
            if Scan['System'] == 'Octopus':
                arr = arr.transpose((1, 0, 2))
            if flipZ:
                arr = arr[:, :, ::-1]

            arr = arr[xp:, yp:, :]

            if modality == "mus" and ftype == 'nifti':
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
                sum_excl_rev = cumsum_rev[:, :,:-1]
                # flip back to original order → (H,W,N-1)
                sum_excl = sum_excl_rev[:, :, ::-1]

                # Now sum_excl[..., k] == sum of I[..., k+1:] exactly as in MATLAB’s sum(I(:,:,z+1:end),3)
                sum_excl = sum_excl[:, :, :MZL]  # keep only the first MZL sums → (H,W,MZL)

                # divide elementwise and apply the constant factors
                arr = I[:, :, :MZL] / sum_excl / (2.0 * 0.0025)

            else:
                arr = arr[:,:,:MZL]
            M_acc[row_range, column_range, :] += arr * Ramp3D
            W_acc[row_range, column_range] += Ramp2D

            # arr = arr.astype(np.float32)[None, row_range, column_range]
            # Blend into accumulators
            # M_acc = da.overlap.overlap(M_acc, {0: 0, 1: 0,
            #                                    2: 0})  # no-op but ensures dask consistency
            # M_acc = M_acc.map_blocks(
            #     lambda block, patch=arr, ramp=Ramp3D: block + patch * ramp,
            #     dtype=np.float32)
            # W_acc = W_acc.map_blocks(
            #     lambda block, ramp2=Ramp2D: block + ramp2[None, :, :],
            #     dtype=np.float32)

        # Normalize and replace NaNs
        result = M_acc / W_acc[...,None]
        result = da.nan_to_num(result)
        return result

    # Build full volume graph
    slices = [build_slice(s) for s in range(n_slices)]


    def stitch_slices(slices):
        z_off, z_sm, z_s = Mosaic3D["z_parameters"][:3]
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
        #TODO: omit the normalizing weight here for now to avoid computation
        w1 = np.linspace(0, 1, z_s) ** degree
        w2 = np.linspace(1, 0, z_s) ** degree
        w3 = np.ones(z_m)

        row_pxblock, col_pxblock = Id0.shape[:2]
        tot_z_pix = int(z_sm * n_slices)
        Ma = dask.array.zeros((row_pxblock,
                       col_pxblock,
                       tot_z_pix),
                      dtype=np.float32)
        for i in range(n_slices):
            si = int(sliceidx[0, i])  # incoming slice index
            print(f'\tstitching slice {i + 1}/{n_slices} (MAT idx {si:03d})')
            Id = slices[i]
            nx, ny, _ = Id.shape

            if s == 0:
                # first slice: apply fresh data
                # pattern = np.array([1, w3, w2])  # MATLAB’s [w1*0+1, w3, w2]
                # W = np.broadcast_to(pattern[np.newaxis, np.newaxis, :],  # shape (1,1,3)
                #                     (nx, ny, pattern.size))  # → (nx, ny, 3)

                W = np.concatenate([np.ones_like(w1),w3,w2])
                zr1 = np.arange(z_sms)  # 0 .. z_sms-1
                zr2 = zr1 + z_off  # zoff .. zoff+z_sms-1

            elif s == n_slices-1:
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
    if len(slices) > 1:
        volume = stitch_slices(slices)
    else:
        volume = slices[0]


    chunk, shard = compute_zarr_layout(volume.shape, volume.dtype, zarr_config)
    wconfig = default_write_config(zarr_config.out,volume.shape,dtype = np.float32, chunk=chunk,shard=shard)
    wconfig["create"] = True
    wconfig["delete_existing"] = True

    tswriter = ts.open(wconfig).result()
    # spec = {
    #     'driver': 'zarr',
    #     'kvstore': {'driver': 'file', 'path': zarr_config.out},
    #     'metadata': {
    #         'dtype': np.float32,
    #         'shape': volume.shape,
    #         'chunks': chunk,
    #     }
    # }
    # if shard:
    #     spec['metadata']['shard'] = shard
    #     volume = volume.rechunk(shard)
    # else:
    #     volume = volume.rechunk(chunk)
    # store = ts.open(spec, create=True).result()
    volume.store(tswriter)
    # ts.copy(volume, store).result()

    # # Optional downsample
    # if downsample:
    #     factors = tuple(int(x) for x in Mosaic3D['sxyz'][:2]) + (1,)
    #     down = do_downsample(volume, factors)
    #     ds_path = out_zarr.replace('.zarr', '_ds.zarr')
    #     ds_spec = {**spec, 'kvstore': {'driver':'file', 'path': ds_path},
    #                'metadata': {'shape': list(down.shape), 'chunks': list(down.chunksize)}}
    #     ds_store = ts.open(ds_spec, create=True).result()
    #     ts.copy(down, ds_store).result()
    #     print(f"Wrote {out_zarr} and {ds_path}")
    # else:
    #     print(f"Wrote {out_zarr}")



def do_downsample(volume: da.Array, factors: tuple) -> da.Array:
    fx, fy, fz = factors
    return da.coarsen(np.mean, volume, {0: fx, 1: fy, 2: fz}, trim_excess=True)


