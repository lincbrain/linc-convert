"""Read overlap volumes and register them with multiview-stitcher (z and y only)."""

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
import yaml

from linc_convert.modalities.lsm.stitching_utils.corrections import (
    compute_noise_map,
    subtract_noise,
)

if TYPE_CHECKING:
    import dask.array as da


def read_volume_rows(vol: "da.Array", y0: int, y1: int, x0: int, x1: int,
                     background_length: int, subtract_background: bool = True,
                     band: int = 64) -> np.ndarray:
    """Read a block of rows into memory, optionally removing the row background.

    The per-row background (median of the chunk's last ``background_length``
    x columns) is subtracted. Rows are read in bands to bound the temporaries.

    Parameters
    ----------
    vol : dask.array.Array
        Chunk volume, shape (Z, Y, X).
    y0, y1 : int
        Row range to read, ``y1`` exclusive.
    x0, x1 : int
        Column range to read, ``x1`` exclusive.
    background_length : int
        Number of trailing x columns used for the background.
    subtract_background : bool, default=True
        Subtract the background (clipped at 0).
    band : int, default=64
        Number of rows read at a time.

    Returns
    -------
    numpy.ndarray
        float32 sub-volume, shape (Z, y1 - y0, x1 - x0).
    """
    Z = vol.shape[0]
    out = np.zeros((Z, y1 - y0, x1 - x0), dtype=np.float32)
    for r0 in range(y0, y1, band):
        r1 = min(r0 + band, y1)
        slab = np.asarray(vol[:, r0:r1, x0:x1], dtype=np.float32)
        if subtract_background:
            noise = compute_noise_map(
                vol[:, r0:r1], background_length)   # (Z, rows)
            slab = np.asarray(subtract_noise(slab, noise))
        out[:, r0 - y0:r1 - y0] = slab
    return out


def register_volume_zy(vol_a: np.ndarray, vol_b: np.ndarray, a_y0: float,
                       init_zy: tuple[float, float],
                       transform_key: str = "stage_metadata",
                       ) -> tuple[tuple[float, float], float, dict[str, Any]]:
    """Register two (z, y, x) volumes with translation in z and y only.

    The stock phase-correlation step runs on the full 3-D data (so every x
    column contributes). The x component of its result is recorded as a
    diagnostic and then forced to 0, so the optimiser only ever moves chunk B
    in z and y. Both volumes share the same x grid.

    Parameters
    ----------
    vol_a, vol_b : numpy.ndarray
        Overlapping sub-volumes of chunks A and B, shape (Z, Y, X).
    a_y0 : float
        y position of volume A's first row, in this level's pixels.
        This is needed because vol_a is the bottom sub_volume of A.
        Most likely equal to init_zy[1].
    init_zy : tuple of float
        Starting ``(z, y)`` position of volume B.
    transform_key : str, default="stage_metadata"
        Name of the input transform in multiview-stitcher.

    Returns
    -------
    residual_zy : tuple of float
        ``(dz, dy)`` that moves B from its starting position to the registered
        one, in this level's pixels.
    quality : float
        Mean registration quality (NaN if none was reported).
    info : dict
        Diagnostics: ``params_a``, ``params_b``, ``unconstrained_x_shift_px``
        and ``x_applied``.
    """
    from multiview_stitcher import msi_utils, registration
    from multiview_stitcher import spatial_image_utils as si_utils

    def msim(vol: np.ndarray, tz: float, ty: float) -> Any:
        sim = si_utils.get_sim_from_array(
            vol, dims=["z", "y", "x"], scale={"z": 1.0, "y": 1.0, "x": 1.0},
            translation={"z": float(tz), "y": float(ty), "x": 0.0},
            transform_key=transform_key)
        return msi_utils.get_msim_from_sim(sim, scale_factors=[])

    x_shift: list[float] = []

    def pairwise_zy_only(fixed_data, moving_data, **kw) -> dict[str, Any]:
        # stock 3-D default is 2 (half-pixel steps)
        kw.setdefault("upsample_factor", 10)
        res = registration.phase_correlation_registration(
            fixed_data, moving_data, **kw)
        aff = np.array(res["affine_matrix"], dtype=float)
        # unconstrained x shift (diagnostic)
        x_shift.append(float(aff[2, 3]))
        aff[2, 3] = 0.0                        # x is never translated
        res = dict(res)
        res["affine_matrix"] = aff
        return res

    msims = [msim(vol_a, 0.0, a_y0), msim(vol_b, init_zy[0], init_zy[1])]
    out = registration.register(
        msims, transform_key=transform_key, new_transform_key="registered",
        reg_channel_index=0, pairwise_reg_func=pairwise_zy_only,
        groupwise_resolution_kwargs={"transform": "translation"},
        pre_registration_pruning_method=None, overlap_tolerance=0.0,
        return_dict=True)
    params = [np.asarray(p, dtype=float).reshape(-1, 4, 4)[0]
              for p in out["params"]]
    pb = params[1]
    residual = (float(pb[0, 3]), float(pb[1, 3]))                  # (dz, dy)
    quals = out["pairwise_registration"]["metrics"]["qualities"]
    quality = (float(np.mean(np.asarray(next(iter(quals.values())), dtype=float)))
               if quals else float("nan"))
    return residual, quality, {
        "params_b": pb.tolist(), "params_a": params[0].tolist(),
        "unconstrained_x_shift_px": x_shift[-1] if x_shift else None,
        "x_applied": float(pb[2, 3])}


Pair = Tuple[float, float, float]   # (dz, y_offset, quality), full-res px


def average_pairs(pairs: List[Pair]) -> Dict[str, Any]:
    """Average the results of consecutive chunk pairs, weighted by quality.

    Parameters
    ----------
    pairs : list of tuple
        ``(dz, y_offset, quality)`` per pair, in full-resolution pixels.

    Returns
    -------
    dict
        ``dz``, ``y_offset`` and ``mean_quality`` (plain mean of the finite
        qualities), plus ``weights``: the normalised weight of each pair
        (numpy array summing to 1).
    """
    dz = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    q = np.array([p[2] for p in pairs])
    w = np.where(np.isfinite(q) & (q > 0), q, 0.0)
    for j in np.flatnonzero(w == 0):
        print(f"WARNING: pair {j + 1} has quality {q[j]:g}; it gets no weight")
    if w.sum() == 0:
        print("WARNING: no pair has positive quality; using the plain average")
        w = np.ones_like(q)
    return {"dz": float(np.sum(w * dz) / w.sum()),
            "y_offset": float(np.sum(w * y) / w.sum()),
            "mean_quality": float(np.nanmean(q)),
            "weights": w / w.sum()}


def load_tile_positions(coords_path: str, n_tiles: int) -> Tuple[List[int], float]:
    """Turn the register_chunks result into one y start per chunk."""
    with open(coords_path) as f:
        result = yaml.safe_load(f) or {}
    try:
        y_offset = float(result["weighted_average"]["y_offset"])
        dz = float(result["weighted_average"].get("dz", 0.0))
    except (KeyError, TypeError):
        raise KeyError(f"{coords_path} has no weighted_average.y_offset "
                       "(was it written by register_chunks?)")
    if not np.isfinite(y_offset) or y_offset <= 0:
        raise ValueError(
            f"weighted_average.y_offset must be > 0, got {y_offset}")
    return [int(round(i * y_offset)) for i in range(n_tiles)], dz
