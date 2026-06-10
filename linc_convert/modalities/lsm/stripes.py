"""Generate stripe .tff files from ome zarr and MIP projection."""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional

# externals
import cyclopts
import dask.array as da
import numpy as np
import SimpleITK as sitk
import tifffile as tiff
import yaml
from dask.diagnostics import ProgressBar
from dask_image.ndinterp import affine_transform
from scipy.ndimage import binary_dilation, binary_erosion, convolve, shift
from skimage import io
from skimage.filters import threshold_otsu
from skimage.registration import phase_cross_correlation

# internals
from linc_convert.modalities.lsm.cli import lsm
from linc_convert.modalities.lsm.convert_spool_or_zarr import (
    Deskewed_Tile,
    discover_tile_paths,
    open_tile_reader,
    prompt_dandi_api_key,
)
from linc_convert.utils.io.zarr.drivers.zarr_python import ZarrPythonGroup
from linc_convert.utils.zarr_config import (
    GeneralConfig,
    ZarrConfig,
    autoconfig,
)

logger = logging.getLogger(__name__)
stripes = cyclopts.App(name="stripes", help_format="markdown")
lsm.command(stripes)

camera_channel_map = {
    1: ["594", "660"],
    2: ["488", "561"],
}

# --------------------------------------------------
# Functions
# --------------------------------------------------


def clean_mask(mask, erode_size=5, dilate_size=5):
    """
    Remove small outliers from a binary mask using erosion + dilation.

    Parameters
    ----------
    mask : np.ndarray (bool)
        Input mask (Y, X)
    erode_size : int
        Size of erosion structuring element
    dilate_size : int
        Size of dilation structuring element

    Returns
    -------
    cleaned_mask : np.ndarray (bool)
    """

    # Structuring elements
    erode_struct = np.ones((erode_size, erode_size), dtype=bool)
    dilate_struct = np.ones((dilate_size, dilate_size), dtype=bool)

    mask_eroded = binary_erosion(mask, structure=erode_struct)

    mask_clean = binary_dilation(mask_eroded, structure=dilate_struct)

    mask_clean = binary_dilation(mask_clean, structure=dilate_struct)

    mask_clean = binary_erosion(mask_clean, structure=erode_struct)

    return mask_clean


def load_scan_parameters(yaml_path: Path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def get_camera_info(scan_parameters: dict, camera_id: int):
    cam_key = f"Camera{camera_id}"
    crop = scan_parameters["crop"][cam_key]
    channels = camera_channel_map[camera_id]

    info = []
    for ch_idx, ch_name in enumerate(channels, start=1):
        ch_key = f"Ch{ch_idx}"
        info.append({
            "channel": ch_name,
            "camera_id": camera_id,
            "y_start": int(crop[ch_key]["yStart"]),
            "y_end": int(crop[ch_key]["yEnd"]),
            "x_start": int(crop[ch_key]["xStart"]),
            "x_end": int(crop[ch_key]["xEnd"]),
            "vertical_flip": bool(crop["verticalFlip"]),
        })
    return info


def compute_tissue_mask_otsu(img_u16: np.ndarray, ds: int = 8,
                             clip_hi_pct: float = 99.9,
                             fallback_pct: float = 70.0) -> np.ndarray:
    img = img_u16.astype(np.float32, copy=False)
    small = img[::ds, ::ds]

    hi = np.percentile(small, clip_hi_pct)
    small_c = np.minimum(small, hi)

    try:
        thr = threshold_otsu(small_c)
    except Exception:
        thr = np.percentile(small_c, fallback_pct)

    # NEW: check threshold vs max
    max_val = small_c.max()
    if max_val > 0 and thr >= (max_val / 1.2):
        return np.zeros_like(img, dtype=bool), max_val*10

    tissue_small = small_c > thr
    if tissue_small.mean() < 0.002:
        thr = np.percentile(small_c, fallback_pct)

        # check again after fallback
        if max_val > 0 and thr >= (max_val / 1.2):
            return np.zeros_like(img, dtype=bool), max_val*10

        tissue_small = small_c > thr

    tissue = np.repeat(np.repeat(tissue_small, ds, axis=0), ds, axis=1)
    return tissue[:img.shape[0], :img.shape[1]], thr


def row_keep_from_mask(mask_pix: np.ndarray,
                       row_frac_thresh: float,
                       dilate_rows: int) -> np.ndarray:
    row_frac = mask_pix.mean(axis=1)
    row_keep = row_frac >= row_frac_thresh
    if dilate_rows and dilate_rows > 0:
        row_keep = binary_dilation(
            row_keep, structure=np.ones(dilate_rows, dtype=bool))
    return row_keep


def resample_mask_y_nearest(mask_src: np.ndarray, Y_dst: int) -> np.ndarray:
    Y_src, X = mask_src.shape
    src_rows = np.round(np.linspace(0, Y_src - 1, Y_dst)).astype(int)
    return mask_src[src_rows, :]


def smooth_1d(v: np.ndarray, win: int) -> np.ndarray:
    win = int(win)
    if win <= 1:
        return v.astype(np.float32)
    if win % 2 == 0:
        win += 1
    if v.size <= win:
        return v.astype(np.float32)
    k = np.ones(win, dtype=np.float32) / win
    return np.convolve(v.astype(np.float32), k, mode="same")


def compute_corr_zy_from_pixel_mask(
    vol_zyx: da.Array,     # (Z, Y, X)
    mask_pix: np.ndarray,  # (Y, X) or (Z, Y, X)
    tissue_frac_min: float,
    thr: float,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Compute (Z,Y) correction map from a 3D Dask volume, using 2D smoothing.

    Returns
    -------
    corr_zy : np.ndarray (Z, Y)
    """

    vol = vol_zyx.astype(np.float32)
    Z, Y, X = vol.shape

    # -------------------------
    # Broadcast mask
    # -------------------------
    mask_da = da.from_array(mask_pix, chunks=vol.chunks[1:])

    if mask_pix.shape == (Y, X):
        mask_da = mask_da[None, :, :]
        mask_da = da.broadcast_to(mask_da, (Z, Y, X))
    elif mask_pix.shape != (Z, Y, X):
        raise ValueError(
            f"mask shape {mask_pix.shape} != volume shape {(Z, Y, X)}"
        )

    # -------------------------
    # Mask data
    # -------------------------
    masked = da.where(mask_da, vol, np.nan)
    masked = da.where(
        da.isfinite(masked) & (masked < thr),
        np.nan,
        masked
    )

    # -------------------------
    # Compute median across X → (Z,Y)
    # -------------------------
    corr_zy = da.nanmedian(masked, axis=2)
    counts = da.sum(da.isfinite(masked), axis=2)

    # Bring small arrays into memory
    corr_zy = corr_zy.compute()
    counts = counts.compute()

    # -------------------------
    # Remove low-quality rows
    # -------------------------
    min_n = int(tissue_frac_min * X)
    corr_zy[counts < min_n] = np.nan

    bad = np.sum(np.isfinite(corr_zy), axis=1) < 10

    median_val = np.nanmedian(corr_zy)

    if not np.isfinite(median_val):
        median_val = 1.0

    corr_zy[bad] = median_val

    # -------------------------
    # ✅ 2D smoothing (NEW)
    # -------------------------
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)

    # Handle NaNs during convolution
    valid = np.isfinite(corr_zy).astype(np.float32)
    corr_filled = np.nan_to_num(corr_zy, nan=0)

    smooth_num = convolve(corr_filled, kernel, mode="nearest")
    smooth_den = convolve(valid, kernel, mode="nearest")

    corr_zy_smooth = smooth_num / (smooth_den + 1e-6)

    # -------------------------
    # Normalize
    # -------------------------
    corr_zy_smooth /= 1000

    return corr_zy_smooth.astype(np.float32)


def apply_corr_zy_lazy(
    vol_zyx: da.Array,
    corr_zy: np.ndarray,   # (Z, Y)
    eps: float = 1e-6
) -> da.Array:
    """
    vol_zyx shape: (Z, Y, X)
    corr_zy shape: (Z, Y)
    Applies correction along Z and Y lazily.
    """

    # Convert volume
    vol_f = vol_zyx.astype(np.float32)

    # Convert correction to Dask
    corr_da = da.from_array(
        corr_zy.astype(np.float32),
        chunks=(corr_zy.shape[0], corr_zy.shape[1])
    )

    # Expand to (Z, Y, 1) for broadcasting
    corr_da = corr_da[:, :, None]

    # Apply correction
    vol_corr = vol_f / (corr_da + eps)

    return da.clip(vol_corr, 0, 65535).astype(np.uint16)


def apply_corr_y_lazy(vol_zyx: da.Array, corr_y: np.ndarray, eps: float = 1e-6) -> da.Array:
    """
    vol_zyx shape: (Z, Y, X)
    corr_y shape: (Y,)
    Applies correction along Y lazily.
    """
    vol_f = vol_zyx.astype(np.float32)
    corr_y_da = da.from_array(corr_y.astype(np.float32), chunks=(len(corr_y),))
    corr_y_da = corr_y_da[None, :, None]
    vol_corr = vol_f / (corr_y_da + eps)
    return da.clip(vol_corr, 0, 65535).astype(np.uint16)


def apply_corr_y(img_u16: np.ndarray, corr_y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply row-wise stripe correction to a 2D image
    Intended for debugging / validation on MIPs
    Not used in the volume correction pipeline
    """
    img32 = img_u16.astype(np.float32, copy=False)
    out32 = img32 / (corr_y[:, None] + eps)
    return np.clip(out32, 0, np.iinfo(img_u16.dtype).max).astype(img_u16.dtype)


def crop_mip_channels(mip_2d, cam_info, x_crop=None):
    out = {}
    for meta in cam_info:
        y1, y2 = meta["y_start"], meta["y_end"]
        cropped = mip_2d[y1:y2, :]
        if x_crop is not None:
            x1, x2 = x_crop
            cropped = cropped[:, x1:x2]
        out[meta["channel"]] = cropped
    return out


def crop_volume_channels(vol_zyx, cam_info):
    """
    vol_zyx shape: (Z, Y, X), dask or numpy
    Returns dict[channel] -> cropped volume (Z, Yc, X)
    """
    out = {}
    for meta in cam_info:
        y1, y2 = meta["y_start"], meta["y_end"]
        out[meta["channel"]] = vol_zyx[:, y1:y2, :]
    return out


def maybe_flip_z_lazy(vol_zyx, do_flip: bool):
    if do_flip:
        return vol_zyx[::-1, :, :]
    return vol_zyx


def skew_correction_affine_dask(
    scape_data_yzx: da.Array,
    conversion_factors: List[float],
    delta: float = 36.0,
):
    """
    Input shape: (Y, Z, X)
    Output shape: (Y, Z, X_out)
    """
    shear = np.tan(np.deg2rad(delta)) * \
        conversion_factors[1] / conversion_factors[2]
    y_in, z_in, x_in = scape_data_yzx.shape
    x_out = int(np.ceil(shear * z_in)) + x_in

    affine = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, shear, 1]
    ])
    affine_inv = np.linalg.inv(affine)

    skew_corrected = affine_transform(
        scape_data_yzx,
        matrix=affine_inv,
        offset=0,
        order=3,
        mode="constant",
        cval=0.0,
        output_shape=(y_in, z_in, x_out),
        output_chunks=scape_data_yzx.chunksize,
    )
    return skew_corrected


def apply_affine(
    scape_data_yzx: da.Array,
    affine: np.ndarray,
):
    affine_inv = np.linalg.inv(affine)

    applied = affine_transform(
        scape_data_yzx,
        matrix=affine_inv,
        order=3,
        mode="constant",
        cval=0.0,
        output_chunks=scape_data_yzx.chunks,
    )
    return applied


def skew_correct_volume_lazy(vol_zyx, scan_parameters, camera_id, force_flip=None):
    delta = scan_parameters["skewCorr"]["delta"]
    umps = scan_parameters["skewCorr"]["umPixelSize"]
    conversion_factors = [umps["y"], umps["z"], umps["x"]]

    if force_flip is None:
        do_flip = bool(scan_parameters["crop"]
                       [f"Camera{camera_id}"]["verticalFlip"])
    else:
        do_flip = force_flip

    vol_in = maybe_flip_z_lazy(vol_zyx, do_flip)
    vol_yzx = da.transpose(vol_in, (1, 0, 2))

    vol_skew_yzx = skew_correction_affine_dask(
        vol_yzx,
        conversion_factors=conversion_factors,
        delta=delta,
    )

    vol_skew_zyx = da.transpose(vol_skew_yzx, (1, 0, 2))

    return vol_skew_zyx


def register_affine(fixed_img, moving_img):
    """
    Compute affine transform that maps moving → fixed.
    """

    # Initial transform (center-based)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_img,
        moving_img,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # Registration method
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric (robust)
    registration_method.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=50
    )

    registration_method.SetMetricSamplingStrategy(
        registration_method.RANDOM
    )
    registration_method.SetMetricSamplingPercentage(0.25)

    # Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )

    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution pyramid
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Set initial transform
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Run registration
    final_transform = registration_method.Execute(fixed_img, moving_img)

    print("Optimizer stop condition:",
          registration_method.GetOptimizerStopConditionDescription())
    print("Final metric value:", registration_method.GetMetricValue())

    return final_transform


def affine_to_matrix(transform):
    matrix = np.array(transform.GetMatrix()).reshape(3, 3)
    translation = np.array(transform.GetTranslation())

    affine = np.eye(4)
    affine[:3, :3] = matrix
    affine[:3, 3] = translation
    return affine


def get_all_affines(path_cm1, path_cm2, scanParameters, fixed_idx=2):
    cam_info_1 = get_camera_info(scanParameters, 1)
    cam_info_2 = get_camera_info(scanParameters, 2)

    reader_1 = open_tile_reader(path_cm1)
    reader_2 = open_tile_reader(path_cm2)

    vol_channels_1 = crop_volume_channels(reader_1, cam_info_1)
    vol_channels_2 = crop_volume_channels(reader_2, cam_info_2)

    # Precompute flips
    do_flip_1 = bool(scanParameters["crop"]["Camera1"]["verticalFlip"])
    do_flip_2 = bool(scanParameters["crop"]["Camera2"]["verticalFlip"])

    channels = []
    channel_keys = []  # <-- track mapping

    # Camera 1
    for i in range(2):
        key = camera_channel_map[1][i]
        channels.append(
            maybe_flip_z_lazy(vol_channels_1[key], do_flip_1)
        )
        channel_keys.append((1, key))  # (camera, channel_key)

    # Camera 2
    for i in range(2):
        key = camera_channel_map[2][i]
        channels.append(
            maybe_flip_z_lazy(vol_channels_2[key], do_flip_2)
        )
        channel_keys.append((2, key))

    # Convert to SimpleITK
    for i in range(4):
        channels[i] = sitk.GetImageFromArray(channels[i].compute())

    # Compute affines
    affines = {}
    for i in range(4):
        cam, ch_key = channel_keys[i]

        if cam not in affines:
            affines[cam] = {}

        if i == fixed_idx:
            affine = np.eye(4)
        else:
            affine = affine_to_matrix(
                register_affine(channels[fixed_idx], channels[i])
            )

        affines[cam][ch_key] = affine

    # Debug print
    for cam in affines:
        for ch in affines[cam]:
            print(f"Camera {cam}, Channel {ch}:\n{affines[cam][ch]}")

    return affines


@stripes.default
@autoconfig
def create(
    inp_cm1: str,
    inp_cm2: str,
    mip_dir: str,
    yaml_path: str,
    camera_id: int,
    file_num: int,
    *,
    general_config: GeneralConfig = None,
    zarr_config: ZarrConfig = None,
    dandiset_id: Optional[str] = None,
    ds: int = 8,
    clip_hi_pct: float = 99.9,
    fallback_pct: float = 70.0,
    tissue_frac_min: float = 0.02,
    smooth_win: float = 1,
    row_frac_thresh_488: float = 0.005,
    dilate_rows: float = 15,
) -> None:
    """
    Generate ZY projection TIFF images from volumetric tile data.

    This function discovers tile datasets from a local or DANDI source, 
    loads each volume, optionally crops it along the Z and Y axes, computes a mask 
    from a corresponding YX MIP image, and applies this mask to the volume before 
    generating a ZY projection using a NaN-aware median. The result is written to disk 
    as a TIFF file.

    Parameters
    ----------
    inp : str
        Input path specifying where to discover tile datasets. Can refer to a local path
        or a remote dataset when `dandiset_id` is provided.
    mip_dir : str
        Directory containing precomputed YX MIP TIFF images used for masking. Each tile
        must have a corresponding file named ``{tile_name}_proc-mip.tiff``.
    general_config : GeneralConfig, optional
        Configuration object containing output settings. Must include an ``out`` attribute
        specifying the output directory.
    dandiset_id : Optional[str], default None
        If provided, tiles are loaded from the specified DANDI dataset using an API key.
    z_start : Optional[int], default None
        Starting index for cropping along the Z dimension (inclusive).
    z_end : Optional[int], default None
        Ending index for cropping along the Z dimension (exclusive).
    y_start : Optional[int], default None
        Starting index for cropping along the Y dimension (inclusive).
    y_end : Optional[int], default None
        Ending index for cropping along the Y dimension (exclusive).

    Returns
    -------
    None
        This function writes output files to disk and does not return a value.

    Raises
    ------
    FileNotFoundError
        If the corresponding YX MIP image for a tile cannot be found.

    Notes
    -----
    - The mask is computed from the YX MIP image using `_compute_mask` and is applied
      across all Z slices.
    - Masked-out regions are set to NaN before computing the median projection.
    - NaN values are replaced with a large sentinel value (9999999.0).
    - Output files are written atomically using a temporary file and then renamed.
    """
    scanParameters = load_scan_parameters(yaml_path)

    strip_ids = range(1, int(scanParameters["numStrips"]) + 1)

    api_key = prompt_dandi_api_key() if dandiset_id else None

    tile_paths_1 = discover_tile_paths(
        inp_cm1, dandiset_id=dandiset_id, api_key=api_key)

    tile_paths_2 = discover_tile_paths(
        inp_cm2, dandiset_id=dandiset_id, api_key=api_key)

    affines = get_all_affines(tile_paths_1[file_num],
                              tile_paths_2[file_num], scanParameters)

    tile_paths = tile_paths_1 if camera_id == 1 else tile_paths_2

    for index, path in enumerate(tile_paths):
        logger.info(path)
        logger.info(index)

        start_time = time.time()
        name = os.path.basename(path.rstrip("/").replace(".ome.zarr", ""))

        reader = open_tile_reader(
            path,
            dandiset_id=dandiset_id,
            api_key=api_key,
        )

        if any(
            not os.path.exists(f"{general_config.out}/{i}/{name}.ome.zarr")
            for i in camera_channel_map[camera_id]
        ):

            yx_path = os.path.join(
                mip_dir, f"{name}_proc-mip.tiff").replace("slice0", "slice")

            if not os.path.exists(yx_path):
                raise FileNotFoundError(f"Missing YX image: {yx_path}")

            raw_mip = tiff.imread(yx_path).astype(np.float32)
            raw_mip_channels = {}
            cam_info = get_camera_info(scanParameters, camera_id)
            raw_mip_channels.update(
                crop_mip_channels(raw_mip, cam_info)
            )
            cam_info = get_camera_info(scanParameters, camera_id)
            vol_channels = crop_volume_channels(reader, cam_info)
            for i in camera_channel_map[camera_id]:
                output_name = f"{general_config.out}/{i}/{name}.ome.zarr"
                if not os.path.exists(output_name):
                    mask, thr = compute_tissue_mask_otsu(
                        raw_mip_channels[i],
                        ds=ds,
                        clip_hi_pct=clip_hi_pct,
                        fallback_pct=fallback_pct,
                    )

                    row_keep = row_keep_from_mask(
                        mask,
                        row_frac_thresh=row_frac_thresh_488,
                        dilate_rows=dilate_rows,
                    )
                    mask = mask & row_keep[:, None]
                    mask = clean_mask(mask)
                    mask = mask[:, :]
                    chunk = zarr_config.chunk
                    if len(zarr_config.chunk) == 1:
                        chunk = tuple([zarr_config.chunk[0]]*3)
                    vol = vol_channels[i][:, :, :]
                    corr_zy = compute_corr_zy_from_pixel_mask(
                        vol, mask, tissue_frac_min, thr*0.65)

                    zy_max = np.max(corr_zy)
                    thr_map = corr_zy * thr / zy_max

                    # reshape for broadcasting to (Z, Y, X)
                    thr_map = thr_map[:, :, None]   # (Z, Y, 1)

                    # apply threshold lazily
                    vol = da.where(vol < thr_map*0.65, 0, vol)

                    vol = apply_corr_zy_lazy(vol, corr_zy)
                    vol = apply_affine(vol, affines[camera_id][i])
                    vol = skew_correct_volume_lazy(
                        vol, scanParameters, camera_id)

                    omz = ZarrPythonGroup.from_config(
                        output_name+".tmp", zarr_config)
                    out = omz.create_array("0", shape=vol.shape,
                                           zarr_config=zarr_config, dtype=np.uint16)

                    vol = da.rechunk(
                        vol, out._array.shards or chunk)
                    with ProgressBar():
                        da.to_zarr(vol, out._array)
                    omz.generate_pyramid(levels=zarr_config.levels)
                    omz.write_ome_metadata(
                        axes=["z", "y", "x"],
                    )

                    os.replace(output_name + ".tmp", output_name)
                    print("--- %s secs ---" % (time.time() - start_time))

        print("--- %s secs ---" % (time.time() - start_time))
