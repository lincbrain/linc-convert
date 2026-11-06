"""Create background-subtraction and zy-correction TIFFs from reference tiles.

Builds the background-subtraction ("noise") and zy-correction ("scaler")
TIFFs for one camera of one slice, from a fixed set of reference tiles.

Local-only: it reads existing tile folders and writes TIFFs, nothing else (no
``linc_convert`` needed). No tissue mask / MIP is used; instead every row of a tile
is measured on the same x columns (``column_mask``, see ``create_corrections``). The
maths lives in
``stitching_utils.corrections``.

    from create_corrections import create_corrections
    create_corrections(
        input_path="…/strips_raw/sample-slice019", yaml_path="LSM_metadata.yaml",
        camera_id=1, slice_number=19, reference_tiles=[23, 26, 29],
        output_dir="…/60_corrections", background_length=5000,
        zy_threshold_multiplier=1.1)

Per channel of the camera and per reference tile (full-resolution level 0 only):

1. The tile is cropped with the channel's ``stitchingCrop`` (y and z ranges
   from the metadata YAML) and z-flipped if the camera's ``verticalFlip``.
2. noise map (Z, Y) = median of each row's last ``background_length`` x columns.
3. The noise is subtracted (clipped at 0) -- always the full noise map.
4. threshold = median of the noise map. A pixel counts as signal only if its
   RAW value is >= ``threshold * zy_threshold_multiplier``.
5. raw scaler (Z, Y) = median over x (every 64th column) of the noise-subtracted
   signal pixels; rows where it is ~0 or undefined are marked invalid.
6. scaler = 1 / (raw scaler / median(raw scaler)): the multiplier that brings
   every row to the tile's typical row brightness. Invalid rows get 1e-9
   (suppressed).

The per-tile maps are combined with an element-wise median across the
reference tiles. Output per channel ``ch``, in ``{output_dir}/{ch}/``
(float32, shape (Z, Y)):

* ``{ch}_alt_zy_noise.tiff``  -- background-subtraction map
* ``{ch}_alt_zy_scaler.tiff`` -- zy-correction map, applied as
  ``(vol - noise).clip(0) * scaler``

With ``fit_offset`` the background that is subtracted becomes ``noise - offset``
with one offset shared by all rows, and the multipliers are fitted together with
it (see ``stitching_utils.percentile_fit``); the noise TIFF then holds
``noise - offset``, so ``stitch_chunks`` and ``register_chunks`` need no change.

``reference_tiles`` are 1-based positions in the camera's tile list (1 is the
first chunk, ``chunk-0001``): the
``*acq-camera-NN*.ome.zarr`` folders of ``input_path`` in sorted order.
With ``auto_reference_tiles`` they are picked from the tissue-rich middle of
the slice instead (see ``stitching_utils.tile_selection``).

Needs: cyclopts, dask, numpy, tifffile, zarr, pyyaml, and the ``stitching_utils``
folder next to this file.
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cyclopts
import numpy as np
import yaml

from linc_convert.modalities.lsm.cli import lsm
from linc_convert.modalities.lsm.stitching_utils import percentile_fit
from linc_convert.modalities.lsm.stitching_utils.corrections import (
    X_STRIDE,
    calibrate_tile,
    correction_paths,
)
from linc_convert.modalities.lsm.stitching_utils.io import (
    crop_and_flip,
    discover_tile_paths,
    open_tile_array,
    write_tiff,
    write_yaml,
)
from linc_convert.modalities.lsm.stitching_utils.scan_metadata import (
    Crop,
    get_channel_names,
    get_crops,
    get_vertical_flip,
)
from linc_convert.modalities.lsm.stitching_utils.tile_selection import (
    available_layer,
    choose_reference_tiles,
    tissue_row_fraction,
)
from linc_convert.utils.zarr_config import autoconfig


def _level_settings(background_length: int, layer: int) -> Tuple[int, int]:
    """Background length and x stride scaled from full resolution to ``layer``."""
    scale = 2 ** layer
    return max(1, background_length // scale), max(1, X_STRIDE // scale)


def _auto_reference_tiles(
    tile_paths: Sequence[str], crops: Dict[str, Crop], channels: List[str],
    vertical_flip: bool, background_length: int, zy_threshold_multiplier: float,
    scan_layer: int, count: int, min_tissue: float, edge_margin: int,
    report_path: str,
) -> List[int]:
    """Pick reference tiles from the tissue-rich middle; see ``create_corrections``.

    Parameters
    ----------
    tile_paths : sequence of str
        The camera's chunk folders in acquisition order.
    crops : dict of str to tuple
        Stitching crop of every channel of the camera.
    channels : list of str
        Channels that need maps; a tile must have tissue in all of them.
    vertical_flip : bool
        Whether the camera's volumes are flipped along z.
    background_length : int
        Trailing x columns (full-resolution pixels) used for the noise floor.
    zy_threshold_multiplier : float
        Signal threshold multiplier, the same one used for the calibration.
    scan_layer : int
        Pyramid level to scan (a coarse one is enough and fast; the highest
        level that exists, up to this one, is used).
    count : int
        Number of tiles to pick.
    min_tissue : float
        Minimum fraction of a tile's rows in tissue for it to be eligible.
    edge_margin : int
        Eligible tiles skipped at each end of the tissue run.
    report_path : str
        YAML file that records the tissue fraction of every tile and the pick.

    Returns
    -------
    list of int
        Chosen 1-based tile numbers.
    """
    layer = available_layer(tile_paths[0], scan_layer)
    bg_len, x_stride = _level_settings(background_length, layer)
    print(
        f"[auto tiles] scanning {len(tile_paths)} tiles at pyramid level {layer}")
    fractions: Dict[int, float] = {}
    for number, path in enumerate(tile_paths, start=1):
        tile = open_tile_array(path, layer, chunks=(128, 128, 128))
        fractions[number] = min(
            tissue_row_fraction(crop_and_flip(tile, crops[ch], vertical_flip, layer),
                                bg_len, zy_threshold_multiplier, x_stride)
            for ch in channels)
        print(
            f"  tile {number}: {fractions[number]:.0%} of rows in tissue", flush=True)
    chosen = choose_reference_tiles(fractions, count, min_tissue, edge_margin)
    if len(chosen) < count:
        print(
            f"  WARNING: only {len(chosen)} eligible tile(s) (wanted {count})")
    print(f"[auto tiles] reference tiles: {chosen}")
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    write_yaml(report_path, {
        "reference_tiles": chosen, "scan_layer": layer, "min_tissue": min_tissue,
        "edge_margin": edge_margin, "zy_threshold_multiplier": zy_threshold_multiplier,
        "tissue_fraction": {int(k): round(v, 3) for k, v in fractions.items()}})
    return chosen


def _offset_corrections(
    tile_paths: Sequence[str], reference_tiles: Sequence[int],
    crops: Dict[str, Crop], channels: List[str], vertical_flip: bool, layer: int,
    background_length: int, zy_threshold_multiplier: float,
    settings: "percentile_fit.PercentileSettings", report_path: str,
    fixed_offset: Optional[float] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Fit the shared offset and the multipliers; see ``create_corrections``.

    Parameters
    ----------
    tile_paths : sequence of str
        The camera's chunk folders in acquisition order.
    reference_tiles : sequence of int
        1-based reference tile numbers.
    crops : dict of str to tuple
        Stitching crop of every channel of the camera.
    channels : list of str
        Channels to process.
    vertical_flip : bool
        Whether the camera's volumes are flipped along z.
    layer : int
        Pyramid level of the raw chunks.
    background_length : int
        Trailing x columns (at this level) used for the row background.
    zy_threshold_multiplier : float
        Signal threshold multiplier.
    settings : PercentileSettings
        Percentile fit settings.
    report_path : str
        YAML file that records the fit.
    fixed_offset : float, optional
        Offset to use instead of estimating one.

    Returns
    -------
    dict of str to tuple
        Per channel ``(noise_map, scaler_map)``; the noise map already has the
        offset taken out (``background - offset``).
    """
    stats: Dict[str, List[percentile_fit.TileStats]] = {ch: [] for ch in channels}
    for ref_index in reference_tiles:
        path = tile_paths[ref_index - 1]
        print(f"[offset fit] tile {ref_index}: {os.path.basename(path)}", flush=True)
        tile = open_tile_array(path, layer, chunks=(128, 128, 128))
        for ch in channels:
            vol = crop_and_flip(tile, crops[ch], vertical_flip, layer)
            tile_stats = percentile_fit.sample_tile(
                vol, background_length, zy_threshold_multiplier, settings)
            stats[ch].append(tile_stats)
            used = (f"; {tile_stats.columns_kept} of {tile_stats.columns_total} "
                    "sampled columns used for every row"
                    if tile_stats.columns_total else "")
            print(f"  channel {ch}: tissue in {tile_stats.tissue_fraction:.0%} of "
                  f"the columns{used}", flush=True)
            if tile_stats.columns_total and \
                    tile_stats.columns_kept < 0.1 * tile_stats.columns_total:
                print(f"  WARNING: only {tile_stats.columns_kept} of "
                      f"{tile_stats.columns_total} columns pass the column mask "
                      f"(more than {settings.column_mask_fraction:.0%} of the rows "
                      "are below the tissue threshold in the others); raise "
                      "column_mask_fraction if the maps come out empty", flush=True)

    maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    report: Dict[str, Any] = {}
    for ch in channels:
        fit = percentile_fit.fit_offset(stats[ch], settings, fixed_offset=fixed_offset)
        noise = np.median(np.stack([t.noise_map for t in stats[ch]]), axis=0)
        maps[ch] = (noise - fit.offset,
                    percentile_fit.multiplier_to_scaler(fit.multiplier))
        reason = ""
        if fixed_offset is not None:
            reason = " (fixed, not estimated)"
        elif fit.offset == 0.0 and abs(fit.estimate) > 0:
            reason = " (estimate not inside the search range, so 0)"
        elif not np.isfinite(fit.estimate):
            reason = " (no usable rows)"
        rounds = ", ".join(f"{h:+.2f}" for h in fit.offset_history)
        print(f"channel {ch}: offset {fit.estimate:+.2f} after {len(fit.offset_history)} "
              f"round(s) [{rounds}]{'' if fit.converged else ' (NOT converged)'}; "
              f"search range +-{fit.search_limit:.1f}; applied offset "
              f"{fit.offset:+.2f}{reason}", flush=True)
        report[ch] = {
            "offset_applied": float(fit.offset),
            "offset_estimate": float(fit.estimate),
            "converged": bool(fit.converged),
            "offset_history": [float(h) for h in fit.offset_history],
            "search_limit": float(fit.search_limit),
            "percentile_levels": [float(v) for v in fit.levels],
            "multiplier_level_weights": [float(v) for v in fit.multiplier_level_weights],
            "offset_level_weights": [float(v) for v in fit.offset_level_weights],
            "offset_score_at_zero": float(fit.score_zero),
            "offset_score_at_estimate": float(fit.score_estimate),
            "rows_used_per_tile": {
                int(t): round(u / max(fit.rows_total, 1), 3) for t, u in
                zip(reference_tiles, fit.rows_used)}}
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    write_yaml(report_path, {
        "convention": "noise TIFF = background - offset; corrected = "
                      "(value - noise TIFF).clip(0) * scaler; a positive offset "
                      "subtracts less",
        "reference_tiles": [int(t) for t in reference_tiles], "layer": layer,
        "zy_threshold_multiplier": zy_threshold_multiplier,
        "fixed_offset": None if fixed_offset is None else float(fixed_offset),
        "settings": {"percentiles": [float(v) for v in settings.levels()],
                     "column_mask_fraction": settings.column_mask_fraction,
                     "max_columns": settings.max_columns,
                     "max_offset_fraction": settings.max_offset_fraction,
                     "multiplier_weight_power": settings.multiplier_weight_power,
                     "offset_weight_power": settings.offset_weight_power,
                     "max_iterations": settings.max_iterations,
                     "offset_tol": settings.offset_tol},
        "channels": report})
    return maps


def _create_per_chunk(
    input_path: str, camera_id: int, slice_number: int, scan_parameters: dict,
    channels: List[str], chunk_numbers: Optional[List[int]], output_dir: str,
    background_length: int, zy_threshold_multiplier: float, force: bool,
    layer: int = 0, column_mask_fraction: Optional[float] = None,
) -> Dict[str, Any]:
    """Calibrate each chunk on its own; see ``create_corrections(per_chunk)``."""
    tile_paths = discover_tile_paths(input_path, camera_id)
    numbers = list(chunk_numbers) if chunk_numbers else list(
        range(1, len(tile_paths) + 1))
    bad = [i for i in numbers if not 1 <= i <= len(tile_paths)]
    if bad:
        raise ValueError(f"chunks {bad} out of range: camera {camera_id} has "
                         f"{len(tile_paths)} tiles (valid: 1..{len(tile_paths)})")
    crops = get_crops(scan_parameters, camera_id, slice_number)
    vertical_flip = get_vertical_flip(scan_parameters, camera_id)

    bg_len, x_stride = _level_settings(background_length, layer)
    results: Dict[str, Any] = {ch: {} for ch in channels}
    for number in numbers:
        todo = []
        for ch in channels:
            paths = correction_paths(output_dir, ch, number)
            results[ch][number] = paths
            if force or not all(os.path.exists(p) for p in paths.values()):
                todo.append(ch)
        if not todo:
            print(
                f"[calibration] chunk {number}: TIFFs already exist, skipping")
            continue
        path = tile_paths[number - 1]
        print(f"[calibration] chunk {number}: {os.path.basename(path)}")
        tile = open_tile_array(path, layer, chunks=(128, 128, 128))
        for ch in todo:
            vol = crop_and_flip(tile, crops[ch], vertical_flip, layer)
            noise_map, scaler_map = calibrate_tile(
                vol, bg_len, zy_threshold_multiplier, x_stride, column_mask_fraction)
            write_tiff(results[ch][number]["noise"], noise_map)
            write_tiff(results[ch][number]["scaler"], scaler_map)
            print(f"  channel {ch}: noise_map mean={noise_map.mean():.2f}")
    return results


create_corrections = cyclopts.App(
    name="create_corrections", help_format="markdown")
lsm.command(create_corrections)


@create_corrections.default
@autoconfig
def create_corrections(
    input_path: str,
    yaml_path: str,
    camera_id: int,
    slice_number: int,
    *,
    reference_tiles: Optional[List[int]] = None,
    auto_reference_tiles: bool = False,
    n_reference_tiles: int = 8,
    min_tissue_fraction: float = 0.9,
    edge_margin: int = 3,
    auto_scan_layer: int = 3,
    output_dir: str,
    per_chunk: bool = False,
    background_length: int = 5000,
    zy_threshold_multiplier: float = 1.1,
    only_channel: Optional[str] = None,
    force: bool = False,
    layer: int = 0,
    fit_offset: bool = False,
    offset_low_percentile: float = 30.0,
    offset_high_percentile: float = 70.0,
    offset_percentiles: Optional[List[float]] = None,
    fixed_offset: Optional[float] = None,
    offset_max_columns: int = 512,
    offset_max_fraction: float = 0.5,
    offset_weight_power: float = 3.0,
    multiplier_weight_power: float = 3.0,
    offset_max_iterations: int = 8,
    offset_tol: float = 0.1,
    column_mask: bool = True,
    column_mask_fraction: float = 0.2,
) -> None:
    """Create the background-subtraction and zy-correction TIFFs.

    Parameters
    ----------
    input_path : str
        Folder holding the camera's tile ``*.ome.zarr`` folders.
    yaml_path : str
        Scan metadata YAML (stitching crops, channel layout).
    camera_id : int
        Camera number (1 or 2).
    slice_number : int
        Physical slice number (selects the crop definitions).
    reference_tiles : list of int, optional
        1-based positions (1 = first chunk) of the tiles to calibrate from.
        Required unless ``auto_reference_tiles`` or ``per_chunk`` is set. With
        ``per_chunk`` it limits which chunks get their own maps (default: all
        chunks of the camera).
    auto_reference_tiles : bool, default=False
        Pick the reference tiles automatically instead: every tile is scanned
        (at the coarse ``auto_scan_layer``) for the fraction of its rows that
        cross tissue in all channels being processed; tiles below
        ``min_tissue_fraction`` and the ``edge_margin`` outermost eligible tiles
        are dropped, and ``n_reference_tiles`` are taken evenly spread over the
        rest. The tissue fractions and the pick are written to
        ``{output_dir}/reference_tiles_camera{camera_id}.yaml``. Not allowed
        together with ``reference_tiles`` or ``per_chunk``.
    n_reference_tiles : int, default=8
        Tiles to pick with ``auto_reference_tiles`` (4-8 explain the stripes of a
        camera; more adds nothing).
    min_tissue_fraction : float, default=0.9
        Fraction of a tile's rows that must cross tissue for it to be eligible.
    edge_margin : int, default=3
        Eligible tiles skipped at each end of the tissue run (slice edges are
        partly tissue and have unusually strong stripes).
    auto_scan_layer : int, default=3
        Pyramid level scanned for the tile selection (the highest level that
        exists, up to this one). Independent of ``layer``.
    output_dir : str
        TIFFs are written to ``{output_dir}/{channel}/``.
    per_chunk : bool, default=False
        Calibrate every chunk on its own instead of taking the median over
        reference tiles. Each chunk gets
        ``{ch}_chunk{NNNN}_alt_zy_noise.tiff`` and ``..._scaler.tiff`` (NNNN is
        the 1-based chunk number), to be used with ``stitch_chunks`` option
        ``per_chunk_corrections``. Existing chunk files are skipped unless
        ``force`` is set.
    background_length : int, default=5000
        Trailing x columns (full-resolution pixels) used for each row's noise
        floor.
    zy_threshold_multiplier : float, default=1.1
        A pixel counts toward the scaler only if its raw value is at least
        ``median(noise_map) * zy_threshold_multiplier``.
    only_channel : str, optional
        Process just this channel of the camera.
    force : bool, default=False
        Recompute channels whose TIFFs already exist.
    layer : int, default=0
        Pyramid level of the raw chunks to calibrate on (0 = full resolution).
        Each level halves every axis, so the maps are ``2**layer`` times
        smaller in z and y; ``background_length`` and the x sampling are
        scaled to the level. Use the same ``layer`` (and a separate
        ``output_dir``) with ``stitch_chunks``.
    fit_offset : bool, default=False
        Also fit one background offset shared by all rows of a channel (needs at
        least 2 reference tiles, 8 or more recommended; not with ``per_chunk``). The
        subtracted background becomes ``noise - offset`` and the noise TIFF holds
        that, so a positive offset subtracts less and a negative one more.
        For every reference tile (a whole chunk after the stitching crop) and
        (z, y) row, the ``offset_percentiles`` of the row's tissue pixels along x
        are taken (tissue: raw value >= ``median(noise) * zy_threshold_multiplier``,
        the same x columns for every row, see ``column_mask``). The offset and the
        per-(z, y) multipliers are then found by ALTERNATING, see
        ``stitching_utils.percentile_fit``: multipliers for the current offset with
        the HIGH percentiles weighted heavily, then the offset that minimises the
        differences between the rows' corrected percentiles of every tile for those
        multipliers, with the LOW percentiles weighted heavily; repeated until the
        offset moves by less than ``offset_tol`` counts or ``offset_max_iterations``
        rounds. The offset is applied only if it lies inside the search range
        (+-``offset_max_fraction`` x the typical lowest percentile above the
        background), otherwise 0 is used (the multipliers are still the ones fitted
        here). The offset after every round is written to
        ``{output_dir}/offset_fit_camera{camera_id}.yaml``. All tiles of a channel are
        held in memory as K x Z x Y floats each (K percentiles).
    offset_low_percentile, offset_high_percentile : float, default=30, 70
        Only used when ``offset_percentiles`` is not given.
    offset_percentiles : list of float, optional
        Two or more increasing percentiles (0-100) used instead of the low / high
        pair, for example ``10 15 20 30 40 50 60 70 80 90 95``.
    offset_weight_power : float, default=3
        Percentile ``p`` counts ``(1 - p / 100) ** power`` in the offset search
        (lower percentiles heavy). 0 weights all percentiles equally.
    multiplier_weight_power : float, default=3
        Percentile ``p`` counts ``(p / 100) ** power`` in the multiplier fit
        (higher percentiles heavy).
    offset_max_fraction : float, default=0.5
        Offset search range, as a share of the typical lowest percentile above the
        background.
    offset_max_iterations : int, default=8
        Rounds of (multipliers, offset) at most.
    offset_tol : float, default=0.1
        Counts; the offset is stable when a round moves it by less.
    fixed_offset : float, optional
        Use this offset (0 for none) instead of estimating one: no search, only
        the multipliers are fitted, with the percentile method. Needs ``fit_offset``. Use it when the
        offset is not worth estimating (``evaluate_corrections`` can show that).
    offset_max_columns : int, default=512
        At most this many x columns (evenly strided) are read per row.
    column_mask : bool, default=True
        Measure every row of a tile on the same x columns. A pixel is masked if it
        is below the tissue threshold; an x column is then masked for ALL rows if
        more than ``column_mask_fraction`` of the tile's rows are masked there, and
        kept for ALL rows otherwise. Without it every pixel is judged on its own,
        so a dim row (weaker laser) loses more pixels than a bright one and the
        percentiles of top and bottom rows are not taken over the same pixels.
        Applies to the offset fit (percentiles) and to the plain multiplier
        calibration (row median).
    column_mask_fraction : float, default=0.2
        Share of rows (0-1) that may be masked in a column that is kept.

    Notes
    -----
    Nothing is returned: ``linc-convert`` treats a returned value as the exit
    status, so a dict made the job look failed. Per-channel TIFFs are written
    to ``{output_dir}/{channel}/``; maps that already exist are reported and
    skipped unless ``force`` is True.

    Raises
    ------
    ValueError
        If ``camera_id`` is not 1 or 2, ``reference_tiles`` is empty or out
        of range, ``reference_tiles`` and ``auto_reference_tiles`` are both
        given (or ``auto_reference_tiles`` is combined with ``per_chunk``), no
        tile has enough tissue for the automatic pick, or ``only_channel`` is
        not a channel of the camera.
    """
    if camera_id not in (1, 2):
        raise ValueError(f"camera_id must be 1 or 2, got {camera_id}")
    if auto_reference_tiles and (reference_tiles or per_chunk):
        raise ValueError("auto_reference_tiles cannot be combined with "
                         "reference_tiles or per_chunk")
    if fixed_offset is not None and not fit_offset:
        raise ValueError("fixed_offset needs fit_offset (the multipliers are then "
                         "fitted with the percentile method at that offset)")
    if fit_offset and per_chunk:
        raise ValueError("fit_offset needs several reference tiles (stripes are "
                         "told from anatomy by repeating across tiles) and "
                         "cannot be combined with per_chunk")
    if not per_chunk and not reference_tiles and not auto_reference_tiles:
        raise ValueError(
            "reference_tiles must contain at least one tile index "
            "(or set auto_reference_tiles)")

    with open(yaml_path) as f:
        scan_parameters = yaml.safe_load(f)
    channels = get_channel_names(scan_parameters, camera_id)
    if only_channel is not None:
        if only_channel not in channels:
            raise ValueError(f"only_channel {only_channel!r} is not one of "
                             f"camera {camera_id}'s channels: {channels}")
        channels = [only_channel]

    if per_chunk:
        _create_per_chunk(
            input_path, camera_id, slice_number, scan_parameters, channels,
            reference_tiles, output_dir, background_length,
            zy_threshold_multiplier, force, layer,
            column_mask_fraction if column_mask else None)
        return

    results: Dict[str, Dict[str, str]] = {}
    todo: List[str] = []
    for ch in channels:
        results[ch] = correction_paths(output_dir, ch)
        if not force and all(os.path.exists(p) for p in results[ch].values()):
            print(
                f"channel {ch}: TIFFs already exist, skipping (force=True to recompute)"
            )
        else:
            todo.append(ch)
    if not todo:
        return

    tile_paths = discover_tile_paths(input_path, camera_id)
    bad = [i for i in (reference_tiles or []) if not 1 <= i <= len(tile_paths)]
    if bad:
        raise ValueError(f"reference_tiles {bad} out of range: camera {camera_id} has "
                         f"{len(tile_paths)} tiles (valid: 1..{len(tile_paths)})")

    crops = get_crops(scan_parameters, camera_id, slice_number)
    vertical_flip = get_vertical_flip(scan_parameters, camera_id)
    if auto_reference_tiles:
        reference_tiles = _auto_reference_tiles(
            tile_paths, crops, todo, vertical_flip, background_length,
            zy_threshold_multiplier, auto_scan_layer, n_reference_tiles,
            min_tissue_fraction, edge_margin,
            os.path.join(output_dir, f"reference_tiles_camera{camera_id}.yaml"))
    assert reference_tiles is not None

    bg_len, x_stride = _level_settings(background_length, layer)
    if fit_offset:
        if len(reference_tiles) < (1 if fixed_offset is not None else 2):
            raise ValueError("fit_offset needs at least 2 reference tiles "
                             f"(8 or more recommended), got {len(reference_tiles)}")
        settings = percentile_fit.PercentileSettings(
            low=offset_low_percentile, high=offset_high_percentile,
            percentiles=tuple(offset_percentiles) if offset_percentiles else None,
            max_columns=offset_max_columns, max_offset_fraction=offset_max_fraction,
            multiplier_weight_power=multiplier_weight_power,
            offset_weight_power=offset_weight_power,
            max_iterations=offset_max_iterations, offset_tol=offset_tol,
            column_mask_fraction=column_mask_fraction if column_mask else None)
        settings.levels()      # validates the percentiles
        maps = _offset_corrections(
            tile_paths, reference_tiles, crops, todo, vertical_flip, layer, bg_len,
            zy_threshold_multiplier, settings,
            os.path.join(output_dir, f"offset_fit_camera{camera_id}.yaml"),
            fixed_offset)
        for ch in todo:
            write_tiff(results[ch]["noise"], maps[ch][0])
            write_tiff(results[ch]["scaler"], maps[ch][1])
            print(f"channel {ch}: wrote {results[ch]['noise']}\n"
                  f"  wrote {results[ch]['scaler']}")
        return
    noises: Dict[str, List[np.ndarray]] = {ch: [] for ch in todo}
    scalers: Dict[str, List[np.ndarray]] = {ch: [] for ch in todo}
    for ref_index in reference_tiles:
        path = tile_paths[ref_index - 1]
        print(f"[calibration] tile {ref_index}: {os.path.basename(path)}")
        tile = open_tile_array(path, layer, chunks=(128, 128, 128))
        for ch in todo:
            vol = crop_and_flip(tile, crops[ch], vertical_flip, layer)
            noise_map, scaler_map = calibrate_tile(
                vol, bg_len, zy_threshold_multiplier, x_stride,
                column_mask_fraction if column_mask else None)
            noises[ch].append(noise_map)
            scalers[ch].append(scaler_map)
            print(f"  channel {ch}: noise_map mean={noise_map.mean():.2f}")

    for ch in todo:
        noise = np.median(np.stack(noises[ch], axis=0), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            scaler = np.nanmedian(np.stack(scalers[ch], axis=0), axis=0)
        scaler = np.nan_to_num(scaler, nan=1e-9)

        write_tiff(results[ch]["noise"], noise)
        write_tiff(results[ch]["scaler"], scaler)
        print(f"channel {ch}: averaged noise_map mean={noise.mean():.2f} over tiles "
              f"{list(reference_tiles)}\n  wrote {results[ch]['noise']}\n"
              f"  wrote {results[ch]['scaler']}")