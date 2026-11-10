"""Read overlap volumes and register them with multiview-stitcher (z and y only)."""

from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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
                     band: int = 64, scaler: Optional[np.ndarray] = None,
                     noise: Optional[np.ndarray] = None) -> np.ndarray:
    """Read a block of rows into memory, optionally removing the row background.

    The per-row background is subtracted (clipped at 0), then the rows are
    multiplied by ``scaler`` if it is given. The background is taken from the
    ``noise`` map where that has a value, and otherwise computed as the median
    of the chunk's last ``background_length`` x columns. Rows are read in
    bands to bound the temporaries.

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
    scaler : numpy.ndarray, optional
        zy multiplier per (z, y) row of ``vol``, shape ``vol.shape[:2]`` (see
        ``create_corrections``); applied after the background subtraction.
    noise : numpy.ndarray, optional
        Background per (z, y) row of ``vol``, shape ``vol.shape[:2]`` (the
        ``create_corrections`` noise map). NaN entries are computed from the
        chunk instead. Ignored if ``subtract_background`` is False.

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
            rows = (np.full((Z, r1 - r0), np.nan, dtype=np.float32)
                    if noise is None else noise[:, r0:r1].astype(np.float32))
            if np.isnan(rows).any():           # no map for some rows: measure
                own = compute_noise_map(vol[:, r0:r1], background_length)
                rows = np.where(np.isnan(rows), own, rows)
            slab = np.asarray(subtract_noise(slab, rows))
        if scaler is not None:
            slab = slab * scaler[:, r0:r1, None]
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


def load_pair_registration(
        coords_path: str, tile_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Read the registration of every neighbouring pair of chunks.

    Pairs are matched to ``tile_names`` by file name. A pair that was not
    registered, or whose quality is missing or <= 0, falls back to the
    weighted average from register_chunks.

    Parameters
    ----------
    coords_path : str
        YAML written by ``register_chunks``.
    tile_names : list of str
        Chunk folder names in stitching order.

    Returns
    -------
    y_offsets, dzs : numpy.ndarray
        y offset and z shift of chunk ``i + 1`` relative to chunk ``i``
        (``len(tile_names) - 1`` values each, full-resolution pixels).
    y_average, dz_average : float
        The weighted averages.

    Raises
    ------
    KeyError
        If the file has no ``weighted_average``.
    """
    with open(coords_path) as f:
        result = yaml.safe_load(f) or {}
    try:
        y_avg = float(result["weighted_average"]["y_offset"])
        dz_avg = float(result["weighted_average"].get("dz", 0.0))
    except (KeyError, TypeError):
        raise KeyError(f"{coords_path} has no weighted_average.y_offset "
                       "(was it written by register_chunks?)")
    n = len(tile_names)
    y = np.full(max(n - 1, 0), y_avg)
    dz = np.full(max(n - 1, 0), dz_avg)
    position = {name: i for i, name in enumerate(tile_names)}
    own = 0
    for pair in result.get("pairs") or []:
        a = position.get(pair.get("chunk_a"))
        b = position.get(pair.get("chunk_b"))
        quality = pair.get("quality")
        if a is None or b is None or b != a + 1:
            continue
        if quality is None or not np.isfinite(quality) or quality <= 0:
            print(f"  pair {a + 1}->{b + 1}: quality {quality}; "
                  "using the weighted average")
            continue
        y[a], dz[a] = float(pair["y_offset"]), float(pair["dz"])
        own += 1
    print(f"  per-chunk registration: {own} of {max(n - 1, 0)} pairs have their own "
          "values, the rest use the average")
    return y, dz, y_avg, dz_avg


def register_volumes_3d(fixed: np.ndarray, moving: np.ndarray,
                        transform_key: str = "stage_metadata", fit_x: bool = False,
                        ) -> tuple[np.ndarray, float]:
    """Register ``moving`` to ``fixed`` with a translation (z and y; x if ``fit_x``).

    The phase correlation always runs on the full 3-D data (every x column
    contributes); unless ``fit_x`` the x component of its result is dropped.

    Both volumes are (z, y, x), on the same voxel grid and each starting at its
    own origin (0, 0, 0). The registration is multiview-stitcher's phase
    correlation, with a translation model.

    Parameters
    ----------
    fixed, moving : numpy.ndarray
        Volumes, shape (Z, Y, X) (shapes may differ).
    transform_key : str, default="stage_metadata"
        Name of the input transform in multiview-stitcher.
    fit_x : bool, default=False
        Also fit the x translation.

    Returns
    -------
    affine : numpy.ndarray
        4 x 4 homogeneous matrix, ``moving`` voxel (z, y, x) -> ``fixed`` voxel
        (rotation part: identity), in this level's voxels.
    quality : float
        Registration quality (NaN if none was reported).
    """
    from multiview_stitcher import msi_utils, registration
    from multiview_stitcher import spatial_image_utils as si_utils

    def msim(vol: np.ndarray) -> Any:
        sim = si_utils.get_sim_from_array(
            vol, dims=["z", "y", "x"], scale={"z": 1.0, "y": 1.0, "x": 1.0},
            translation={"z": 0.0, "y": 0.0, "x": 0.0},
            transform_key=transform_key)
        return msi_utils.get_msim_from_sim(sim, scale_factors=[])

    def pairwise(fixed_data, moving_data, **kw) -> dict[str, Any]:
        kw.setdefault("upsample_factor", 10)
        res = dict(registration.phase_correlation_registration(
            fixed_data, moving_data, **kw))
        if not fit_x:
            aff = np.array(res["affine_matrix"], dtype=float)
            aff[2, 3] = 0.0
            res["affine_matrix"] = aff
        return res

    out = registration.register(
        [msim(fixed), msim(moving)], transform_key=transform_key,
        new_transform_key="registered", reg_channel_index=0,
        pairwise_reg_func=pairwise,
        groupwise_resolution_kwargs={"transform": "translation"},
        pre_registration_pruning_method=None, overlap_tolerance=0.0,
        return_dict=True)
    affine = np.asarray(out["params"][1], dtype=float).reshape(-1, 4, 4)[0]
    quals = out["pairwise_registration"]["metrics"]["qualities"]
    quality = (float(np.mean(np.asarray(next(iter(quals.values())), dtype=float)))
               if quals else float("nan"))
    return affine, quality


ANTS_STAGES = {
    "rigid": ["Translation", "Rigid"],
    "similarity": ["Translation", "Rigid", "Similarity"],
    "affine": ["Translation", "Rigid", "Similarity", "Affine"],
}
# volumes at a very small layer are small: use every voxel in the metric, and a 3-level pyramid
ANTS_DEFAULTS: Dict[str, Any] = {"aff_random_sampling_rate": 1.0,
                                 "aff_iterations": (1000, 500, 250),
                                 "aff_smoothing_sigmas": (2, 1, 0),
                                 "aff_shrink_factors": (4, 2, 1)}
MODELS = ("translation",) + tuple(ANTS_STAGES)


def register_volumes_model(fixed: np.ndarray, moving: np.ndarray,
                           model: str = "translation", fit_x: bool = False,
                           transform_key: str = "stage_metadata",
                           ants_kwargs: Optional[Dict[str, Any]] = None,
                           ) -> tuple[np.ndarray, float]:
    """Register ``moving`` to ``fixed`` with a translation, rigid, similarity or affine model.

    ``translation`` is phase correlation (``register_volumes_3d``). The other models
    run ANTsPy through multiview-stitcher (needs ``antspyx``), in stages
    (translation, rigid, ... up to ``model``), starting from the phase-correlation
    translation in z and y. The translation is then refined by phase correlation
    of the warped volume (mutual information is only about a voxel accurate there).

    Without ``fit_x`` the transform acts in the (z, y) plane only (rotation about x,
    scaling and shear of z and y; x is left alone). It is found on the mean
    projections of the two volumes along x, which is exact for such a transform.
    With ``fit_x`` it is a full 3-D fit.

    Parameters
    ----------
    fixed, moving : numpy.ndarray
        Volumes (Z, Y, X) on the same voxel grid, each with its origin at (0, 0, 0).
    model : {"translation", "rigid", "similarity", "affine"}
        Transform family.
    fit_x : bool, default=False
        Include x (3-D fit).
    transform_key : str, default="stage_metadata"
        Name of the input transform in multiview-stitcher.
    ants_kwargs : dict, optional
        Overrides for the ANTsPy registration (``ANTS_DEFAULTS``).

    Returns
    -------
    affine : numpy.ndarray
        4 x 4 homogeneous matrix (z, y, x), ``moving`` voxel -> ``fixed`` voxel, in
        this level's voxels.
    quality : float
        Registration quality (NaN if none was reported).

    Raises
    ------
    ValueError
        If ``model`` is unknown.
    ImportError
        If a model other than ``translation`` is used without ``antspyx``.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    start, quality = register_volumes_3d(fixed, moving, transform_key, fit_x=fit_x)
    if model == "translation":
        return start, quality
    try:
        import ants  # noqa: F401
    except ImportError:
        raise ImportError(f"model {model!r} needs the antspyx package "
                          "(pip install antspyx)") from None
    from multiview_stitcher import msi_utils, registration
    from multiview_stitcher import spatial_image_utils as si_utils

    if fit_x:
        dims, fix_v, mov_v, axes = ["z", "y", "x"], fixed, moving, [0, 1, 2]
    else:                               # (z, y) as the two axes of a 2-D image
        dims, fix_v, mov_v, axes = ["y", "x"], fixed.mean(axis=2), moving.mean(axis=2), [0, 1]
    t0 = start[:3, 3][axes]

    def msim(vol: np.ndarray, shift: np.ndarray) -> Any:
        sim = si_utils.get_sim_from_array(
            vol, dims=dims, scale={d: 1.0 for d in dims},
            translation={d: float(v) for d, v in zip(dims, shift)},
            transform_key=transform_key)
        return msi_utils.get_msim_from_sim(sim, scale_factors=[])

    pairwise = partial(registration.registration_ANTsPy,
                       transform_types=ANTS_STAGES[model],
                       **{**ANTS_DEFAULTS, **(ants_kwargs or {})})

    out = registration.register(
        [msim(fix_v, np.zeros(len(dims))), msim(mov_v, t0)],
        transform_key=transform_key, new_transform_key="registered",
        reg_channel_index=0, pairwise_reg_func=pairwise,
        groupwise_resolution_kwargs={
            "transform": "rigid" if model == "rigid" else "affine"},
        pre_registration_pruning_method=None, overlap_tolerance=0.0,
        return_dict=True)
    n = len(dims)
    residual = np.asarray(out["params"][1], dtype=float).reshape(-1, n + 1, n + 1)[0]
    place = np.eye(n + 1)
    place[:n, n] = t0                   # moving voxel -> where phase correlation put it
    local = residual @ place            # moving voxel -> fixed voxel (axes fitted)
    affine = np.eye(4)
    idx = np.array(axes)
    affine[np.ix_(idx, idx)] = local[:n, :n]
    affine[idx, 3] = local[:n, n]
    quals = out["pairwise_registration"]["metrics"]["qualities"]
    quality = (float(np.mean(np.asarray(next(iter(quals.values())), dtype=float)))
               if quals else float("nan"))
    # the mutual-information fit is good for rotation / scale but only about a voxel
    # accurate in translation: finish with phase correlation of the warped volume
    # (a pure shift), which is exact for that part
    import scipy.ndimage
    for _ in range(2):
        inv = np.linalg.inv(affine)
        warped = scipy.ndimage.affine_transform(
            moving, inv[:3, :3], offset=inv[:3, 3], output_shape=fixed.shape,
            order=1, mode="constant", cval=0.0)
        delta, _q = register_volumes_3d(fixed, warped, transform_key, fit_x=fit_x)
        affine = delta @ affine
    return affine, quality