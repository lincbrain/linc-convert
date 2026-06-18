import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
from scipy.linalg import expm, logm

from linc_convert.modalities.lsm.convert_spool_or_zarr import open_tile_reader
from linc_convert.modalities.lsm.preprocessing_utils.corrections import (
    crop_volume_channels,
    stripe_skew_corr,
)
from linc_convert.modalities.lsm.preprocessing_utils.io import (
    camera_channel_map,
    get_camera_info,
    load_mask_and_thresholds,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Affine utilities
# ---------------------------------------------------------------------

def average_affines_log_euclidean(matrices: List[np.ndarray]) -> np.ndarray:
    """
    Compute the mean affine transform using Log-Euclidean averaging.

    Parameters
    ----------
    matrices : list of np.ndarray
        List of 4x4 affine matrices.

    Returns
    -------
    np.ndarray
        Averaged 4x4 affine matrix.
    """
    logs = [logm(M) for M in matrices]
    mean_log = sum(logs) / len(logs)
    return expm(mean_log)


def estimate_affine_multislice(
    ref_vol: np.ndarray,
    mov_vol: np.ndarray,
    z_indices: List[int],
) -> np.ndarray:
    """
    Estimate a 3D affine transform from multiple Z-slices.

    A 2D affine is estimated for each slice (Z,Y plane), then combined
    using Log-Euclidean averaging.

    Parameters
    ----------
    ref_vol : dask.array.Array
        Reference volume (Z, Y, X).
    mov_vol : dask.array.Array
        Moving volume (Z, Y, X).
    z_indices : list of int
        Indices of slices (along X) used for registration.

    Returns
    -------
    np.ndarray
        4x4 affine transform mapping moving → reference.
    """
    affines: List[np.ndarray] = []

    for z in z_indices:
        ref_slice = ref_vol[:, :, z].compute()
        mov_slice = mov_vol[:, :, z].compute()

        transform = estimate_affine_zy(ref_slice, mov_slice)
        affines.append(sitk_to_4x4(transform))

    return average_affines_log_euclidean(affines)


def estimate_affine_zy(ref_img: np.ndarray, mov_img: np.ndarray) -> sitk.Transform:
    """
    Estimate a 2D affine transform between two ZY slices using mutual information.

    Parameters
    ----------
    ref_img : np.ndarray
        Reference image (Z, Y).
    mov_img : np.ndarray
        Moving image (Z, Y).

    Returns
    -------
    sitk.Transform
        Estimated 2D affine transform.
    """
    ref = sitk.GetImageFromArray(ref_img.astype(np.float32))
    mov = sitk.GetImageFromArray(mov_img.astype(np.float32))

    transform = sitk.AffineTransform(2)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2)

    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    reg.SetInitialTransform(transform, inPlace=False)

    return reg.Execute(ref, mov)


def sitk_to_4x4(tx: sitk.Transform) -> np.ndarray:
    """
    Convert a SimpleITK 2D affine transform into a 4x4 matrix in (Z,Y,X) space.

    Parameters
    ----------
    tx : sitk.Transform
        Input transform (2D Affine or CompositeTransform).

    Returns
    -------
    np.ndarray
        4x4 affine matrix.
    """
    # Unwrap composite transforms
    if isinstance(tx, sitk.CompositeTransform):
        if tx.GetNumberOfTransforms() == 0:
            raise RuntimeError("Empty CompositeTransform")
        tx = tx.GetNthTransform(0)

    A = np.array(tx.GetMatrix()).reshape(2, 2)
    t = np.array(tx.GetTranslation())

    M = np.eye(4)

    # Z axis
    M[0, 0] = A[0, 0]
    M[0, 1] = A[0, 1]
    M[0, 3] = t[0]

    # Y axis
    M[1, 0] = A[1, 0]
    M[1, 1] = A[1, 1]
    M[1, 3] = t[1]

    # X unchanged (identity)

    M[3, :3] = 0.0
    return M


# ---------------------------------------------------------------------
# High-level affine pipeline
# ---------------------------------------------------------------------

def get_all_affines(
    path_cm1: str,
    path_cm2: str,
    scan_parameters: dict,
    mip_dir: str,
    fixed_idx: int = 2,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Compute inter-channel affines for two cameras.

    Preprocesses volumes (masking + correction + skew), crops in Y,
    then estimates affine transforms aligning all channels to a fixed reference.

    Parameters
    ----------
    path_cm1 : str
        Path to camera 1 tile.
    path_cm2 : str
        Path to camera 2 tile.
    scan_parameters : dict
        Acquisition metadata (from YAML).
    mip_dir : str
        Directory containing MIP images used for masking.
    fixed_idx : int, default=2
        Index of reference channel in the combined list.

    Returns
    -------
    dict
        Nested dictionary:
        {camera_id: {channel_name: 4×4 affine matrix}}
    """
    reader_1 = open_tile_reader(path_cm1)
    reader_2 = open_tile_reader(path_cm2)

    name_1 = os.path.basename(path_cm1.rstrip("/").replace(".ome.zarr", ""))
    name_2 = os.path.basename(path_cm2.rstrip("/").replace(".ome.zarr", ""))

    cam_info_1 = get_camera_info(scan_parameters, 1)
    cam_info_2 = get_camera_info(scan_parameters, 2)

    masks_1, thrs_1 = load_mask_and_thresholds(name_1, mip_dir, cam_info_1)
    masks_2, thrs_2 = load_mask_and_thresholds(name_2, mip_dir, cam_info_2)

    vol_1 = crop_volume_channels(reader_1, cam_info_1)
    vol_2 = crop_volume_channels(reader_2, cam_info_2)

    channels: List[np.ndarray] = []
    channel_keys: List[Tuple[int, str]] = []

    # -------------------------
    # Load and preprocess channels
    # -------------------------
    for cam_id, vol_dict, masks, thrs in [
        (1, vol_1, masks_1, thrs_1),
        (2, vol_2, masks_2, thrs_2),
    ]:
        for ch in camera_channel_map[cam_id]:
            vol = stripe_skew_corr(
                vol_dict[ch],
                masks[ch],
                thrs[ch],
                cam_id,
                scan_parameters,
            )[:, 50:-50, :]  # symmetric Y cropping

            channels.append(vol)
            channel_keys.append((cam_id, ch))

            logger.info(f"Loaded cam{cam_id} channel {ch}")

    # -------------------------
    # Compute affines
    # -------------------------
    affines: Dict[int, Dict[str, np.ndarray]] = {}

    for i, (cam_id, ch) in enumerate(channel_keys):
        affines.setdefault(cam_id, {})

        if i == fixed_idx:
            affine = np.eye(4)
        else:
            affine = estimate_affine_multislice(
                channels[fixed_idx],
                channels[i],
                [19884, 14076, 19786, 30327],
            )
            affine[3, :3] = 0.0

            # Correct for Y cropping
            y_offset = 50
            T = np.eye(4)
            T[1, 3] = y_offset

            affine = T @ affine @ np.linalg.inv(T)

        affines[cam_id][ch] = affine

    # Debug logging
    for cam_id, ch_dict in affines.items():
        for ch, mat in ch_dict.items():
            logger.info(f"Camera {cam_id}, Channel {ch}:\n{mat}")

    return affines
