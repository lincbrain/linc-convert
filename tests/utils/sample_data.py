from typing import List, Optional, Tuple

import numpy as np
import skimage


def sample_data_variation(
        param: float,
        image: Optional[np.ndarray] = None,
        output_dtype: Optional[np.dtype] = None
        ) -> np.ndarray:
    """
    Scale global intensity by (1+param), then wrap values modulo the output dtype range.

    Parameters
    ----------
    param : float
        Scale offset: actual scale = 1.0 + param.
    image : np.ndarray, optional
        Input volume; defaults to skimage.data.brain().
    output_dtype : np.dtype, optional
        Desired dtype; defaults to image.dtype.

    Returns
    -------
    np.ndarray
        Intensity‐scaled & wrapped image.
    """
    if image is None:
        image = skimage.data.brain()
    if output_dtype is None:
        output_dtype = image.dtype

    # 1) apply scale in float
    scale = 1.0 + param
    img_float = image.astype(np.float64) * scale

    # 2) compute wrap range for the output dtype
    if np.issubdtype(output_dtype, np.integer):
        info = np.iinfo(output_dtype)
    else:
        info = np.finfo(output_dtype)
    rng = info.max - info.min + 1

    # 3) wrap (modulo) then shift back into [min, max]
    img_wrapped = ((img_float - info.min) % rng) + info.min

    # 4) cast back
    return img_wrapped.astype(output_dtype)


def generate_sample_data_variation(
        n: int,
        param_range: Tuple[float, float] = (-0.95, 0.95),
        image: Optional[np.ndarray] = None,
        output_dtype: Optional[np.dtype] = None
        ) -> List[np.ndarray]:
    """
    Generate `n` deterministic, wrapped‐intensity variations.

    Parameters
    ----------
    n : int
        Number of variations.
    param_range : (float, float)
        Range of `param` to sample linearly.
    image : np.ndarray, optional
        Base volume.
    output_dtype : np.dtype, optional
        Desired dtype of outputs.

    Returns
    -------
    List[np.ndarray]
        List of `n` wrapped‐intensity volumes.
    """
    if image is None:
        image = skimage.data.brain()
    params = np.linspace(param_range[0], param_range[1], n)
    return [
        sample_data_variation(p, image=image, output_dtype=output_dtype)
        for p in params
        ]
