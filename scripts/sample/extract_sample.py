#!/usr/bin/env python
"""Extract a small, fast-iteration LSM stitching sample from the real dataset.

Feature: stitching-artifact-correction (FR-014). Crops a few *adjacent* strips
(full Y, so seams/overlaps are preserved; cropped Z and X for speed) from the real
``strips_raw`` OME-Zarr strips and writes them as small OME-Zarr groups into a
scratch directory. The extracted sample is NEVER committed (restricted data) — only
this script is versioned.

The output tiles keep the original ``..._chunk-NNNN_acq-camera-01.ome.zarr`` naming
so ``linc-convert lsm {mip,stripes,stitch} --alternate-pattern`` discovers and orders
them exactly like the full dataset.

Fails loudly (constitution IV): missing source, no matching strips, an empty crop
window, or a non-readable result all raise rather than producing a silent empty sample.

Example
-------
    uv run python scripts/sample/extract_sample.py \
        --src /orcd/data/linc/001/js1518/001412/derivatives/sub-MF283/strips_raw/sample-slice036 \
        --out $WORK/sample --n-strips 3 --z 64 --x 4096
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import zarr


def _find_strips(src: str, camera: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(src, f"*acq-{camera}.ome.zarr")))
    if not paths:
        # fall back to any ome.zarr if the camera suffix differs
        paths = sorted(glob.glob(os.path.join(src, "*.ome.zarr")))
    if not paths:
        raise FileNotFoundError(f"No *.ome.zarr strips found under {src!r}")
    return paths


def _level0(path: str) -> zarr.Array:
    grp = zarr.open_group(path, mode="r")
    if "0" not in grp:
        raise KeyError(f"{path!r} has no level-0 array '0' (keys: {list(grp.keys())})")
    return grp["0"]


def extract(
    src: str,
    out: str,
    *,
    n_strips: int,
    z: int,
    x: int,
    z_start: int,
    x_start: int,
    start_index: int,
    camera: str,
) -> list[str]:
    if not os.path.isdir(src):
        raise NotADirectoryError(f"--src is not a directory: {src!r}")

    strips = _find_strips(src, camera)
    chosen = strips[start_index : start_index + n_strips]
    if len(chosen) < n_strips:
        raise ValueError(
            f"Requested {n_strips} adjacent strips from index {start_index}, "
            f"but only {len(chosen)} available (total {len(strips)})."
        )
    if n_strips < 2:
        raise ValueError("Need >= 2 adjacent strips so the sample contains a seam.")

    os.makedirs(out, exist_ok=True)
    written: list[str] = []

    for path in chosen:
        name = os.path.basename(path.rstrip("/"))
        a = _level0(path)
        sz, sy, sx = a.shape
        zc = min(z_start + z, sz)
        xc = min(x_start + x, sx)
        if z_start >= zc or x_start >= xc:
            raise ValueError(
                f"Empty crop window for {name}: z[{z_start}:{zc}] x[{x_start}:{xc}] "
                f"(source {a.shape})"
            )

        data = np.asarray(a[z_start:zc, :, x_start:xc])  # full Y kept
        if not data.any():
            raise ValueError(
                f"Crop window for {name} is all-zero (no signal at "
                f"z[{z_start}:{zc}] x[{x_start}:{xc}]). Pick a different "
                f"--x-start/--z-start."
            )

        dst_path = os.path.join(out, name)
        dst = zarr.open_group(dst_path, mode="w")
        chunks = (min(data.shape[0], 64), min(data.shape[1], 512), min(data.shape[2], 512))
        arr = dst.create_array("0", shape=data.shape, dtype=data.dtype, chunks=chunks)
        arr[:] = data
        # minimal OME multiscale metadata (level "0" only)
        dst.attrs["multiscales"] = [
            {
                "version": "0.4",
                "axes": [
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [{"path": "0"}],
            }
        ]
        written.append(dst_path)
        print(
            f"[extract] {name}: src{a.shape} -> sample{data.shape} "
            f"max={int(data.max())}",
            flush=True,
        )

    return written


def _validate(paths: list[str]) -> None:
    """Re-open each tile through the project reader so failures surface here."""
    try:
        from linc_convert.modalities.lsm.convert_spool_or_zarr import open_tile_reader
    except Exception as exc:  # pragma: no cover - import guard
        print(f"[extract] WARNING: could not import open_tile_reader to validate: {exc}")
        return
    for p in paths:
        r = open_tile_reader(p, dandiset_id=None, api_key=None)
        _ = r.shape  # touch shape; raises if unreadable
        print(f"[extract] validated readable: {os.path.basename(p)} shape={r.shape}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="slice directory of *.ome.zarr strips")
    ap.add_argument("--out", required=True, help="output sample directory (scratch)")
    ap.add_argument("--n-strips", type=int, default=3)
    ap.add_argument("--z", type=int, default=64, help="Z extent to keep")
    ap.add_argument("--x", type=int, default=4096, help="X extent to keep")
    ap.add_argument("--z-start", type=int, default=0)
    ap.add_argument("--x-start", type=int, default=0)
    ap.add_argument("--start-index", type=int, default=0, help="first strip index")
    ap.add_argument("--camera", default="camera-01")
    args = ap.parse_args(argv)

    written = extract(
        args.src,
        args.out,
        n_strips=args.n_strips,
        z=args.z,
        x=args.x,
        z_start=args.z_start,
        x_start=args.x_start,
        start_index=args.start_index,
        camera=args.camera,
    )
    _validate(written)
    print(f"[extract] wrote {len(written)} strips to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
