"""Shading corrections from optix (CIDRE, IlluminationFunction, GammaMixture) as pipeline maps.

The pipeline corrects a chunk with two maps per channel, indexed by (z, y) and shared by
every x: ``corrected = (raw - noise).clip(0) * scaler``. The frames optix works on are
exactly these (z, y) planes, one per x, so an estimated flat field ``f`` and dark ``d`` map
directly: ``noise = d`` and ``scaler = mean(f) / f``.

optix (``pip install git+https://github.com/neuroscales/optix``) and torch are imported only
when a fit is made.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

METHODS = ("cidre", "illumination", "gamma_mixture")


def pick_tissue_run(scores: Dict[int, float], count: int = 5) -> List[int]:
    """The ``count`` consecutive tiles (1-based numbers) with the most tissue.

    The window with the largest minimum score wins; ties go to the larger sum.

    Raises
    ------
    ValueError
        If there are fewer than ``count`` tiles or ``count`` < 1.
    """
    numbers = sorted(scores)
    if count < 1 or len(numbers) < count:
        raise ValueError(f"need at least {count} tiles, have {len(numbers)}")
    best, best_key = None, None
    for i in range(len(numbers) - count + 1):
        window = numbers[i:i + count]
        if window[-1] - window[0] != count - 1:        # not consecutive
            continue
        vals = [scores[n] for n in window]
        key = (min(vals), sum(vals))
        if best_key is None or key > best_key:
            best, best_key = window, key
    if best is None:
        raise ValueError("the tile numbers have gaps: no consecutive run found")
    return best


def frame_columns(vol, noise: np.ndarray, multiplier: float, count: int,
                  min_signal: float, background_columns: int) -> np.ndarray:
    """x indices of ``count`` evenly spread frames that contain tissue.

    A frame (the (z, y) plane at x) counts if at least ``min_signal`` of its pixels are
    above ``noise * multiplier``. The trailing ``background_columns`` columns, which the
    noise map is made from, are not used.

    Parameters
    ----------
    vol : dask.array.Array
        Volume (z, y, x).
    noise : numpy.ndarray
        Noise (background) map (z, y).
    """
    n_x = vol.shape[2] - max(background_columns, 0)
    if n_x < 1:
        raise ValueError("the volume has no columns left after the background columns")
    probe = np.unique(np.linspace(0, n_x - 1, num=min(n_x, max(4 * count, 64))).astype(int))
    thr = noise[:, :, None] * multiplier
    keep = []
    for x in probe:
        frame = np.asarray(vol[:, :, int(x)].compute(), dtype=np.float32)
        if float((frame > thr[:, :, 0]).mean()) >= min_signal:
            keep.append(int(x))
    if not keep:
        return np.zeros(0, dtype=int)
    keep = np.asarray(keep)
    take = np.unique(np.linspace(0, len(keep) - 1, num=min(count, len(keep))).astype(int))
    return keep[take]


def _resize(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a smooth 2-D map to ``shape``."""
    if tuple(arr.shape) == tuple(shape):
        return arr.astype(np.float32)
    import torch
    import torch.nn.functional as fn
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None, None]
    out = fn.interpolate(t, size=tuple(shape), mode="bilinear", align_corners=False)
    return out[0, 0].numpy()


def fit_maps(method: str, frames: np.ndarray, dark: Optional[np.ndarray] = None,
             gain: Optional[float] = None, max_iterations: Optional[int] = None,
             dark_source: str = "estimate") -> Tuple[np.ndarray, np.ndarray]:
    """Fit an optix estimator to frames (n, z, y) and return ``(noise, scaler)`` maps (z, y).

    Parameters
    ----------
    method : {"cidre", "illumination", "gamma_mixture"}
    frames : numpy.ndarray
        Raw frames, shape (n, z, y).
    dark : numpy.ndarray, optional
        Camera dark / background map (z, y). ``illumination`` needs it (its method assumes
        dark-subtracted frames); the other two estimate their own dark (``gamma_mixture``
        uses the offset of its sensor estimate) and fall back to this map when it has none.
    gain : float, optional
        Nominal gain (ADU per electron) for ``gamma_mixture``; default 1.
    max_iterations : int, optional
        Iteration limit of CIDRE / GammaMixture (the optix default if None).
    dark_source : {"estimate", "noise_map"}, default="estimate"
        Where ``noise`` comes from for ``cidre`` / ``gamma_mixture``: the estimator's own dark
        (offset) or the measured ``dark`` map. ``illumination`` always uses the map.

    Returns
    -------
    noise, scaler : numpy.ndarray
        float32, shape (z, y). ``scaler = mean(flat) / flat``.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if dark_source not in ("estimate", "noise_map"):
        raise ValueError(f"dark_source must be 'estimate' or 'noise_map', got {dark_source!r}")
    if dark_source == "noise_map" and dark is None:
        raise ValueError("dark_source='noise_map' needs a dark map")
    try:
        import torch
        from optix.shading import CIDRE, GammaMixture, IlluminationFunction
    except ImportError as exc:                       # pragma: no cover - environment dependent
        raise ImportError("optix and torch are needed: pip install "
                          "git+https://github.com/neuroscales/optix") from exc
    if frames.ndim != 3 or frames.shape[0] < 2:
        raise ValueError(f"frames must be (n >= 2, z, y), got {frames.shape}")
    kw = {} if max_iterations is None else {"max_iterations": int(max_iterations)}
    if method == "cidre":
        data = torch.from_numpy(frames.astype(np.float64))
        est = CIDRE(dtype=torch.float64, **kw).fit(data)
    elif method == "illumination":
        if dark is None:
            raise ValueError("illumination needs a dark / noise map")
        data = torch.from_numpy(frames.astype(np.float32))
        est = IlluminationFunction(size="auto").fit(
            data, dark=torch.from_numpy(np.asarray(dark, dtype=np.float32)))
    else:
        data = torch.from_numpy(frames.astype(np.float32))
        if gain is not None:
            kw["gain"] = float(gain)
        est = GammaMixture(quantization="auto", **kw).fit(data)
    shading = est.estimates["shading"]
    flat = shading.flat.detach().cpu().numpy().astype(np.float64)
    noise = None
    if shading.dark is not None:
        noise = np.asarray(shading.dark.detach().cpu().numpy(), dtype=np.float64)
    elif method == "gamma_mixture" and "sensor" in est.estimates:
        noise = np.asarray(est.estimates["sensor"].offset.detach().cpu().numpy(), np.float64)
    if method == "illumination" or noise is None or dark_source == "noise_map":
        noise = np.asarray(dark, dtype=np.float64) if dark is not None else np.zeros_like(flat)
    if noise.shape != flat.shape:
        noise = np.broadcast_to(noise, flat.shape)
    if not np.isfinite(flat).all() or (flat <= 0).any():
        raise ValueError(
            f"{method}: the estimated flat field has non-positive or non-finite values")
    scaler = flat.mean() / flat
    return noise.astype(np.float32), scaler.astype(np.float32)


def upsample_maps(noise: np.ndarray, scaler: np.ndarray, shape: Sequence[int]
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Resize both maps (smooth fields) to ``shape`` = the corrections' (z, y) shape."""
    shape = (int(shape[0]), int(shape[1]))
    return _resize(noise, shape), _resize(scaler, shape)
