import numpy as np
from scipy import ndimage


def compute_tissue_mask(
    img: np.ndarray,
    downsample: int = 8,
    clip_high_percentile: float = 99.9,
) -> tuple[np.ndarray, float]:
    """
    Generate a binary tissue mask from a 2D image using thresholding and largest-component filtering.

    The algorithm:
    1. Downsamples the image
    2. Computes a high-percentile clipping value
    3. Estimates a threshold from bright regions
    4. Thresholds and upsamples back to original size
    5. Keeps only the largest connected component and fills holes

    Parameters
    ----------
    img : np.ndarray
        Input image (2D), typically uint16.
    downsample : int, default=8
        Downsampling factor for faster mask estimation.
    clip_high_percentile : float, default=99.9
        Upper percentile used to clip bright outliers.

    Returns
    -------
    mask : np.ndarray (bool)
        Binary tissue mask of same shape as input.
    threshold : float
        Intensity threshold used for segmentation.
    """
    # Convert to float (no copy if already float)
    img_f = img.astype(np.float32, copy=False)

    # -------------------------
    # Estimate threshold
    # -------------------------
    # Use bright region at right edge (heuristic)
    edge_region = img_f[::downsample, -500:]
    threshold = np.percentile(edge_region, 98) * 1.1

    # Downsampled image
    small = img_f[::downsample, ::downsample]

    # Clip extreme intensities
    clip_val = np.percentile(small, clip_high_percentile)
    small_clipped = np.minimum(small, clip_val)

    # Threshold
    tissue_small = small_clipped > threshold

    # -------------------------
    # Upsample mask
    # -------------------------
    mask = np.repeat(
        np.repeat(tissue_small, downsample, axis=0),
        downsample,
        axis=1,
    )

    mask = mask[: img.shape[0], : img.shape[1]]

    # -------------------------
    # Post-process mask
    # -------------------------
    mask = largest_component_filled(mask)

    return mask, threshold


def largest_component_filled(
    mask: np.ndarray,
    connectivity: int = 1,
) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask and fill holes within it.

    Parameters
    ----------
    mask : np.ndarray
        2D boolean array.
    connectivity : int, default=1
        Connectivity definition:
        - 1 → 4-connectivity
        - 2 → 8-connectivity

    Returns
    -------
    np.ndarray
        Cleaned binary mask (largest component with holes filled).

    Raises
    ------
    ValueError
        If mask is not 2D or connectivity is invalid.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    # Define connectivity structure
    if connectivity == 1:
        structure = ndimage.generate_binary_structure(2, 1)
    elif connectivity == 2:
        structure = np.ones((3, 3), dtype=int)
    else:
        raise ValueError(
            "connectivity must be 1 (4-connectivity) or 2 (8-connectivity)")

    # Label connected components
    labeled, num_features = ndimage.label(mask, structure=structure)

    if num_features == 0:
        return mask.copy()

    # Find largest component
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_label = int(np.argmax(sizes) + 1)

    largest = labeled == largest_label

    # Fill holes
    return ndimage.binary_fill_holes(largest)
