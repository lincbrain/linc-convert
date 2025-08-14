import logging
import os.path as op
from typing import List

import cyclopts
import dask.array as da
from niizarr import write_ome_metadata

from linc_convert.modalities.psoct.cli import psoct
from linc_convert.utils.io.zarr import from_config
from linc_convert.utils.zarr_config import ZarrConfig

logger = logging.getLogger(__name__)
z_stitching = cyclopts.App(name="z_stitching", help_format="markdown")
psoct.command(z_stitching)


@z_stitching.default
def stitch(
        inp: List[str],
        *,
        overlap: int = 0,
        offset: int = 0,
        zarr_config: ZarrConfig = None,
        **kwargs
        ) -> None:
    # load slices to dask
    dask_slices = [da.from_zarr(op.join(inp[i], "0"), ) for i in range(len(inp))]
    # crop slices
    for i in range(len(dask_slices)):
        sl = dask_slices[i]
        if i != 0:
            sl = sl[overlap // 2:]
        if i != len(dask_slices) - 1:
            sl = sl[:overlap - overlap // 2]
        dask_slices[i] = sl
    vol = da.concatenate(dask_slices, axis=0)
    # output
    zgroup = from_config(zarr_config)
    arr = zgroup.create_array("0", vol.shape, dtype=vol.dtype, zarr_config=zarr_config)
    vol = vol.rechunk(zarr_config.chunk * 3)
    vol.store(arr)

    zgroup.generate_pyramid()
    # zgroup = zarr.open_group(zgroup.store_path, mode= "a")
    write_ome_metadata(zgroup, ["z", "y", "x"])

#
# def stack_slices(mosaic_info, slice_indices, slices):
#     num_slices = len(slices)
#     z_off, z_sm, z_s = mosaic_info["z_parameters"][:3]
#     # z_off: every slice is going to remove this much from beginning
#     z_off, z_sm, z_s = int(z_off), int(z_sm), int(z_s)
#     z_sms = z_sm + z_s
#     z_m = z_sm - z_s
#     # --- Load one slice to get block size and build tissue profile ---
#     Id0 = slices[0]
#     if modality.lower() == 'mus':
#         # (If you want the 'mus' branch, you'd need
#         # skimage.filters.threshold_multiotsu, etc.)
#         raise NotImplementedError("The 'mus' branch is not shown here.")
#     else:
#         # dBI: average a small tissue-only block
#         tissue0 = Id0[:200, :200, z_off:].mean(axis=(0, 1))
#     # only keep the next z_sms values, offset by zoff
#     tissue = tissue0[z_off: z_off + z_sms]
#     # --- compute blending weights ---
#     s = tissue[z_s] / tissue[:z_s]  # Top overlapping skirt
#     ms = tissue[z_s] / tissue[z_s: z_sms]  # non-overlap + bottom skirt
#     degree = 1  # both dBI and mus use degree=1
#     # w1 = s * np.linspace(0, 1, z_s) ** degree
#     # w2 = ms[z_m:] * np.linspace(1, 0, z_s) ** degree
#     # w3 = ms[:z_m]
#     # TODO: omit the normalizing weight here for now to avoid computation
#     w1 = np.linspace(0, 1, z_s) ** degree
#     w2 = np.linspace(1, 0, z_s) ** degree
#     w3 = np.ones(z_m)
#     row_pxblock, col_pxblock = Id0.shape[:2]
#     tot_z_pix = int(z_sm * num_slices)
#     Ma = dask.array.zeros((row_pxblock,
#                            col_pxblock,
#                            tot_z_pix),
#                           dtype=np.float32)
#     for i in range(num_slices):
#         si = int(slice_indices[0, i])  # incoming slice index
#         print(f'\tstitching slice {i + 1}/{num_slices} (MAT idx {si:03d})')
#         Id = slices[i]
#         nx, ny, _ = Id.shape
#
#         if s == 0:
#             # first slice: apply fresh data
#             # pattern = np.array([1, w3, w2])  # MATLAB’s [w1*0+1, w3, w2]
#             # W = np.broadcast_to(pattern[np.newaxis, np.newaxis, :],  # shape (1,1,3)
#             #                     (nx, ny, pattern.size))  # → (nx, ny, 3)
#
#             W = np.concatenate([np.ones_like(w1), w3, w2])
#             zr1 = np.arange(z_sms)  # 0 .. z_sms-1
#             zr2 = zr1 + z_off  # zoff .. zoff+z_sms-1
#
#         elif s == num_slices - 1:
#             # last slice: add onto bottom-most z_sm planes
#             # pattern = np.array([w1, w3])  # MATLAB’s [w1, w3]
#             # W = np.broadcast_to(pattern[np.newaxis, np.newaxis, :],
#             #                     (nx, ny, pattern.size))  # → (nx, ny, 2)
#             W = np.concatenate([w1, w3, np.ones_like(w2)])
#             zr1 = np.arange(z_sm) + (tot_z_pix - z_sm)  # targets top of Ma’s z-axis
#             zr2 = np.arange(z_sm) + z_off  # picks from Id
#         else:
#             # middle slices: accumulate
#             # pattern = np.array([w1, w3, w2])  # MATLAB’s [w1, w3, w2]
#             # W = np.broadcast_to(pattern[np.newaxis, np.newaxis, :],
#             #                     (nx, ny, pattern.size))  # → (nx, ny, 3)
#             W = np.concatenate([w1, w3, w2])
#             zr1 = np.arange(z_sms) + (s - 1) * z_sm  # where to write in Ma
#             zr2 = np.arange(z_sms) + z_off  # where to read in Id
#         # if i == 0:
#         #     # first slice: only top skirt=1 + body + bottom skirt
#         #     vec = np.concatenate([np.ones(z_s), w3, w2])
#         #     W = np.tile(vec, (row_pxblock * col_pxblock, 1)) \
#         #         .reshape(row_pxblock, col_pxblock, z_sms)
#         #     z1 = np.arange(z_sms)
#         #     z2 = z1 + int(z_off)
#         # elif i == n_slices - 1:
#         #     # last slice: only top skirt + body
#         #     vec = np.concatenate([w1, w3])
#         #     W = np.tile(vec, (row_pxblock * col_pxblock, 1)) \
#         #         .reshape(row_pxblock, col_pxblock, z_sm)
#         #     z1 = np.arange(tot_z_pix - z_sm, tot_z_pix)
#         #     z2 = int(z_off) + np.arange(z_sm)
#         # else:
#         #     # middle slices: skirt/body/skirt
#         #     vec = np.concatenate([w1, w3, w2])
#         #     W = np.tile(vec, (row_pxblock * col_pxblock, 1)) \
#         #         .reshape(row_pxblock, col_pxblock, z_sms)
#         #     start_z = i * z_sm
#         #     z1 = np.arange(start_z, start_z + z_sms)
#         #     z2 = int(z_off) + np.arange(z_sms)
#
#         Ma[:, :, zr1] += Id[:, :, zr2] * W
#     return Ma
