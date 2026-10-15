"""Create background-subtraction and zy-correction TIFFs from reference tiles.

Builds the background-subtraction ("noise") and zy-correction ("scaler")
TIFFs for one camera of one slice, from a fixed set of reference tiles.

Local-only: it reads existing tile folders and writes TIFFs, nothing else (no
``linc_convert`` needed). No tissue mask / MIP is used. The maths lives in
``stitching_utils.corrections``.

    from create_corrections import create_corrections
    create_corrections(
        input_path="…/strips_raw/sample-slice019", yaml_path="LSM_metadata.yaml",
        camera_id=1, slice_number=19, reference_tiles=[23, 26, 29],
        output_dir="…/60_corrections", background_length=5000,
        zy_threshold_multiplier=1.05)

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

``reference_tiles`` are 1-based positions in the camera's tile list (1 is the
first chunk, ``chunk-0001``): the
``*acq-camera-NN*.ome.zarr`` folders of ``input_path`` in sorted order.

Needs: cyclopts, dask, numpy, tifffile, zarr, pyyaml, and the ``stitching_utils``
folder next to this file.
"""

import os
import warnings
from typing import Any, Dict, List, Optional

import cyclopts
import numpy as np
import yaml

from linc_convert.modalities.lsm.stitching_utils.corrections import (
    calibrate_tile,
    correction_paths,
)
from linc_convert.modalities.lsm.stitching_utils.io import (
    crop_and_flip,
    discover_tile_paths,
    open_tile_array,
    write_tiff,
)
from linc_convert.modalities.lsm.stitching_utils.scan_metadata import (
    get_channel_names,
    get_crops,
    get_vertical_flip,
)

app = cyclopts.App(name="create_corrections", help_format="markdown")


def _create_per_chunk(
    input_path: str, camera_id: int, slice_number: int, scan_parameters: dict,
    channels: List[str], chunk_numbers: Optional[List[int]], output_dir: str,
    background_length: int, zy_threshold_multiplier: float, force: bool,
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
        tile = open_tile_array(path, chunks=(128, 128, 128))
        for ch in todo:
            vol = crop_and_flip(tile, crops[ch], vertical_flip)
            noise_map, scaler_map = calibrate_tile(
                vol, background_length, zy_threshold_multiplier)
            write_tiff(results[ch][number]["noise"], noise_map)
            write_tiff(results[ch][number]["scaler"], scaler_map)
            print(f"  channel {ch}: noise_map mean={noise_map.mean():.2f}")
    return results


@app.default
def create_corrections(
    input_path: str,
    yaml_path: str,
    camera_id: int,
    slice_number: int,
    *,
    reference_tiles: Optional[List[int]] = None,
    output_dir: str,
    per_chunk: bool = False,
    background_length: int = 5000,
    zy_threshold_multiplier: float = 1.05,
    only_channel: Optional[str] = None,
    force: bool = False,
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
        Required unless ``per_chunk`` is set. With ``per_chunk`` it limits
        which chunks get their own maps (default: all chunks of the camera).
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
        Trailing x columns used for each row's noise floor.
    zy_threshold_multiplier : float, default=1.05
        A pixel counts toward the scaler only if its raw value is at least
        ``median(noise_map) * zy_threshold_multiplier``.
    only_channel : str, optional
        Process just this channel of the camera.
    force : bool, default=False
        Recompute channels whose TIFFs already exist.

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
        of range, or ``only_channel`` is not a channel of the camera.
    """
    if camera_id not in (1, 2):
        raise ValueError(f"camera_id must be 1 or 2, got {camera_id}")
    if not per_chunk and not reference_tiles:
        raise ValueError(
            "reference_tiles must contain at least one tile index")

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
            zy_threshold_multiplier, force)
        return

    assert reference_tiles is not None
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
    bad = [i for i in reference_tiles if not 1 <= i <= len(tile_paths)]
    if bad:
        raise ValueError(f"reference_tiles {bad} out of range: camera {camera_id} has "
                         f"{len(tile_paths)} tiles (valid: 1..{len(tile_paths)})")

    crops = get_crops(scan_parameters, camera_id, slice_number)
    vertical_flip = get_vertical_flip(scan_parameters, camera_id)

    noises: Dict[str, List[np.ndarray]] = {ch: [] for ch in todo}
    scalers: Dict[str, List[np.ndarray]] = {ch: [] for ch in todo}
    for ref_index in reference_tiles:
        path = tile_paths[ref_index - 1]
        print(f"[calibration] tile {ref_index}: {os.path.basename(path)}")
        tile = open_tile_array(path, chunks=(128, 128, 128))
        for ch in todo:
            vol = crop_and_flip(tile, crops[ch], vertical_flip)
            noise_map, scaler_map = calibrate_tile(
                vol, background_length, zy_threshold_multiplier)
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


if __name__ == "__main__":
    app()
