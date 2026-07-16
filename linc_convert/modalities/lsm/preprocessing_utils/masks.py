import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu


def compute_tissue_mask_otsu(img_u16, downsample=8, clip_high_percentile=99,
                             bimodal_ratio=1.2):
    img = img_u16.astype(np.float32, copy=False)
    small = img[::downsample, ::downsample].ravel()

    # Check for bimodality before running Otsu.
    # On a unimodal (all-background) tile, p99 is close to the median.
    # On a real tile with tissue, p99 is in the bright tissue mode,
    # well above the median. A ratio below ~1.2 strongly suggests
    # a single background distribution with no real tissue.
    med = np.median(small)
    p98 = np.percentile(small, 98)
    if p98 / (med + 1e-6) < bimodal_ratio:
        return np.zeros(img_u16.shape, dtype=bool)

    small_2d = img[::downsample, ::downsample]
    hi = np.percentile(small_2d, clip_high_percentile)
    small_c = np.minimum(small_2d, hi)

    thr = threshold_otsu(small_c)
    tissue_small = small_c > thr

    tissue = np.repeat(np.repeat(tissue_small, downsample,
                       axis=0), downsample, axis=1)
    return tissue[:img.shape[0], :img.shape[1]], thr


def compute_tissue_mask(
    img: np.ndarray,
    downsample: int = 4,
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
    # Slice the small regions out of the ORIGINAL array first, and only
    # cast those small slices to float32 -- casting the whole image up
    # front would allocate a full-size float32 copy (4x the memory of a
    # uint16 input) just to immediately throw most of it away via
    # slicing.
    # -------------------------
    # Estimate threshold
    # -------------------------
    # Use bright region at right edge (heuristic)
    edge_region = img[::downsample, -5000::downsample].astype(np.float32)
    threshold = min(np.median(edge_region), 120)
    del edge_region

    # Downsampled image (small, cast only this slice -- and copy here,
    # since .astype() on a different dtype already copies, so this is
    # safe to mutate in place below without touching `img`)
    small = img[::downsample, ::downsample].astype(np.float32)

    # Clip extreme intensities, in place (safe: `small` is already an
    # owned copy from .astype() above, not a view into `img`)
    clip_val = np.percentile(small, clip_high_percentile)
    np.minimum(small, clip_val, out=small)

    # Threshold
    tissue_small = small > threshold*1.4
    del small

    # -------------------------
    # Upsample mask
    # -------------------------
    # A single broadcast+reshape instead of two chained np.repeat calls:
    # the chained version allocates one full-size-ish intermediate array
    # for the axis-0 repeat, then a second one for the axis-1 repeat, so
    # both are briefly resident at once. broadcast_to is a zero-copy
    # view; only the final reshape allocates, and only once.
    h, w = tissue_small.shape
    mask = np.broadcast_to(
        tissue_small[:, None, :, None], (h, downsample, w, downsample)
    ).reshape(h * downsample, w * downsample)
    del tissue_small

    mask = mask[: img.shape[0], : img.shape[1]]

    # -------------------------
    # Post-process mask
    # -------------------------
    mask = components_over_threshold_filled(mask)

    return mask, threshold


def components_over_threshold_filled(
    mask: np.ndarray,
    min_size: int = 1000,
    connectivity: int = 1,
) -> np.ndarray:
    """
    Keep all connected components larger than `min_size` and fill holes.

    Parameters
    ----------
    mask : np.ndarray
        2D boolean array.
    min_size : int, default=1000
        Minimum component size to keep.
    connectivity : int, default=1
        Connectivity definition:
        - 1 → 4-connectivity
        - 2 → 8-connectivity

    Returns
    -------
    np.ndarray
        Cleaned binary mask (all components > min_size, holes filled).
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    # Define connectivity structure
    if connectivity == 1:
        structure = ndimage.generate_binary_structure(2, 1)
    elif connectivity == 2:
        structure = np.ones((3, 3), dtype=int)
    else:
        raise ValueError("connectivity must be 1 or 2")

    # Label connected components
    labeled, num_features = ndimage.label(mask, structure=structure)
    if num_features == 0:
        return mask.copy()

    # Compute component sizes. np.bincount on the label array is
    # significantly faster than ndimage.sum(mask, labeled, ...) here:
    # every labeled pixel is, by construction, a foreground (mask=True)
    # pixel, so a plain occurrence count over the label array gives the
    # exact same sizes as weighting by `mask`, without ndimage.sum's
    # more general (and slower) weighted-reduction machinery.
    sizes = np.bincount(labeled.ravel(), minlength=num_features + 1)[1:]

    # Keep only components larger than min_size, via a small lookup
    # table indexed by label value, rather than
    # np.isin(labeled, list(keep_labels)). The lookup table is sized to
    # the number of components, not the image -- much smaller than
    # anything proportional to `labeled` itself, and the final indexing
    # step (`keep_lookup[labeled]`) is a single fast fancy-index pass.
    keep_lookup = np.zeros(num_features + 1, dtype=bool)
    keep_lookup[1:] = sizes > min_size
    filtered = keep_lookup[labeled]
    del labeled

    # Fill holes
    return ndimage.binary_fill_holes(filtered)
