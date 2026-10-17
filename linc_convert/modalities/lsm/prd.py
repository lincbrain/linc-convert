"""Convert Kinetix .prd image(s) to OME-Zarr/NIfTI-Zarr.

A .prd file is a stack of uint16 frames with a fixed header
and inter-frame gap, and padding at the end. The input is a directory of
`ss_stack_*.prd` files, which are concatenated along the acquisition order.

Example input files can be found at
https://dandiarchive.org/dandiset/001769/draft/files?location=rawdata/HG9_Z1_Y1
"""

import base64
import glob
import logging
import math
import os
import platform
import re
import sys
import sysconfig
from dataclasses import replace
from typing import Annotated

import cyclopts
import numpy as np
import tqdm

from linc_convert.modalities.lsm.cli import lsm
from linc_convert.utils.io.prd import (
    PrdCapture,
    PrdSetInterpreter,
    parse_prd_header,
)
from linc_convert.utils.io.zarr import from_config
from linc_convert.utils.io.zarr.helpers import _compute_zarr_layout
from linc_convert.utils.io.zarr.stream_pyramid import (
    StreamingPyramidWriter,
    plan_for_memory,
    stream_volume,
)
from linc_convert.utils.nifti_header import build_nifti_header
from linc_convert.utils.zarr_config import (
    GeneralConfig,
    NiftiConfig,
    ZarrConfig,
    autoconfig,
)

logger = logging.getLogger(__name__)
prd = cyclopts.App(name="prd", help_format="markdown")
lsm.command(prd)


@prd.default
@autoconfig
def convert(
    inp: str,
    *,
    voxel_size: Annotated[
        list[float] | None, cyclopts.Parameter(consume_multiple=True)
    ] = None,
    workers: int | None = None,
    max_memory: float | None = None,
    max_shard_size: str | None = "40GB",
    slab_depth: int | None = None,
    verify: bool = False,
    checksum: bool = True,
    general_config: GeneralConfig | None = None,
    zarr_config: ZarrConfig | None = None,
    nii_config: NiftiConfig | None = None,
) -> None:
    """
    Convert Kinetix .prd file(s) into a pyramidal OME-Zarr or NIfTI-Zarr.

    Parameters
    ----------
    inp
        Path to a directory of ss_stack_*.prd files.
    voxel_size
        Voxel size along the X, Y and Z dimensions (camera columns, camera
        rows, frames), in microns. Default: from the scan's `<scan>_info.mat`
        (GUIcalFactors); an error is raised if neither is available.
    workers
        Number of threads used to compress and write chunks.
        Default: number of available CPUs.
    max_memory
        Memory budget in GiB. Workers are reduced until the estimated peak
        memory fits; leftover budget deepens the slab (unless `slab_depth`
        is given).
    max_shard_size
        Cap on the uncompressed bytes per shard, used when `shard` is not
        given. A stored shard never exceeds its uncompressed size, so this
        also bounds each object (S3 objects are limited to 5 TB stored);
        stored shards are smaller by the compression ratio. The default
        "40GB" gives ~13 GB shard files at the ~3x compression of LSM data.
        Shards span the full frame in y and x and as many slices along z as
        fit, and are streamed into append-only files, so large shards need
        no extra memory. "none" writes unsharded chunks.
    slab_depth
        Slices read, compressed and written per batch (rounded up to the
        inner chunk, or to the shard for non-streamed layouts). Deeper slabs
        use more memory and keep more threads busy. Default: one chunk (or
        shard) along z.
    verify
        After writing, re-read level 0 and check that it is bit-identical
        to the .prd frames (costs one extra read + decode of level 0).
    checksum
        Record the SHA-256 of each `.prd` file (verified by `export`).
        Files are hashed by 4 extra threads that read them independently
        (usually from the page cache); `--no-checksum` skips it.
    general_config
        General configuration
    zarr_config
        Zarr related configuration
    nii_config
        NIfTI header related configuration
    """
    reader = PrdSetInterpreter(inp)
    reader.capture = PrdCapture(reader.prd_files, checksum=checksum)
    info_path = _find_info_mat(inp)
    info = _read_info_mat(info_path) if info_path else None
    if voxel_size is None:
        voxel_size = _voxel_size_from_info(info)

    # Set default output path if not provided
    general_config.set_default_name(os.path.basename(inp))

    if str(max_shard_size).lower() == "none":
        max_shard_size = None
    if max_shard_size and not zarr_config.shard:
        zarr_config = replace(zarr_config, shard=_shard_for_size(
            reader.shape, reader.dtype, zarr_config, _parse_size(max_shard_size)))
        logger.info("Shard shape %s", zarr_config.shard)

    zgroup = from_config(general_config.out, zarr_config)
    logger.info(general_config.out)

    if workers is None:
        workers = len(os.sched_getaffinity(0)) if hasattr(
            os, "sched_getaffinity") else os.cpu_count()
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    if sysconfig.get_config_var("Py_GIL_DISABLED") and gil:
        logger.warning("Free-threaded Python, but an extension re-enabled "
                       "the GIL; compression threads will not scale.")
    logger.info("Python %s, GIL %s, %d worker thread(s)",
                platform.python_version(), "on" if gil else "off", workers)
    logger.info(
        "Stream %d file(s) with shape: "
        "%d total_frames, %d height, %d width.",
        len(reader.prd_files), *reader.shape,
    )
    if max_memory:
        chunk, shard = _compute_zarr_layout(reader.shape, reader.dtype, zarr_config)
        appendable = shard is not None and zarr_config.driver == "zarr-python"
        native = (chunk if appendable else (shard or chunk))[-3:]
        planned, depth = plan_for_memory(
            int(max_memory * 2**30), reader.shape, reader.dtype.itemsize,
            native[0], native, workers)
        workers, slab_depth = planned, slab_depth or depth
        logger.info("Using %d worker(s), slab depth %d to fit %.1f GiB",
                    workers, slab_depth, max_memory)
    writer = StreamingPyramidWriter(
        zgroup, reader.shape, reader.dtype, zarr_config,
        levels=zarr_config.levels, max_workers=workers,
        slab_depth=slab_depth,
    )
    with tqdm.tqdm(total=reader.shape[0], unit="frame") as pbar:
        stream_volume(writer, reader.read_frames, pbar.update)

    reader.capture.finish()
    _write_prd_group(general_config.out, inp, reader, info_path, info)
    if verify:
        _verify_level0(reader, zgroup["0"])

    voxel_size = list(map(float, reversed(voxel_size)))

    logger.info("Write OME-Zarr multiscale metadata")
    zgroup.write_ome_metadata(
            axes=["z", "y", "x"],
            space_scale=voxel_size,
            ome_version=zarr_config.ome_version,
    )

    if nii_config.nii:
        header = build_nifti_header(
            zgroup=zgroup,
            voxel_size_zyx=tuple(voxel_size),
            unit="micrometer",
            nii_config=nii_config,
        )
        zgroup.write_nifti_header(header)

    logger.info("Conversion complete.")


# Microscope axes (as named in `<scan>_info.mat`) for each zarr axis. The
# flat face of the section is the lab x-y plane (zarr z-x).
LAB_AXES = {
    "z": "frames: stage step (lab x)",
    "y": "camera rows (lab z, section depth)",
    "x": "camera columns (lab y)",
}


def _find_info_mat(inp: str) -> str | None:
    """`<scan>_info.mat` next to (or inside) the scan directory."""
    scan = os.path.normpath(inp)
    inside = sorted(glob.glob(os.path.join(scan, "*_info.mat")))
    return next((p for p in [scan + "_info.mat", *inside] if os.path.isfile(p)), None)


def _read_info_mat(path: str) -> dict | None:
    """Scan description from a MATLAB `info` file, as JSON-compatible values."""
    try:
        from scipy.io import loadmat
    except ImportError:
        logger.warning("scipy is not installed: %s is stored but not decoded", path)
        return None

    def plain(v):  # noqa: ANN001, ANN202
        if hasattr(v, "_fieldnames"):
            return {k: plain(getattr(v, k)) for k in v._fieldnames}
        if isinstance(v, np.ndarray):
            return [plain(x) for x in v.tolist()] if v.ndim else plain(v.item())
        if isinstance(v, (np.generic,)):
            return v.item()
        if isinstance(v, list):
            return [plain(x) for x in v]
        return v

    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    return {k: plain(v) for k, v in mat.items() if not k.startswith("__")}


def _voxel_size_from_info(info: dict | None) -> list[float]:
    """Voxel size (X, Y, Z = camera columns, camera rows, frames) in microns."""
    cal = ((info or {}).get("info") or {}).get("GUIcalFactors") or {}
    try:
        return [float(cal["y_umPerPix"]), float(cal["z_umPerPix"]),
                float(cal["x_umPerPix"])]
    except KeyError:
        raise ValueError(
            "Voxel size unknown: no <scan>_info.mat with GUIcalFactors next to "
            "the scan directory. Pass --voxel-size X Y Z (camera columns, "
            "camera rows, frames; microns).") from None


def _write_prd_group(
    out: str, inp: str, reader: PrdSetInterpreter, info_path: str | None,
    info: dict | None,
) -> None:
    """Store what the pixels do not carry, enough to rebuild the `.prd` files.

    `prd/` holds a manifest (file names, frames, sizes, SHA-256), the decoded
    PRD header and per-frame records, the sidecar files verbatim and the
    non-pixel bytes of every file (headers, gaps, trailing bytes).
    """
    import zarr

    capture = reader.capture
    nonpixel = [b"".join(parts) for parts in capture.nonpixel]
    files = [
        dict(name=os.path.basename(path), frames=n, bytes=os.path.getsize(path),
             sha256=digest, nonpixel_bytes=len(np_bytes))
        for path, n, digest, np_bytes in zip(
            reader.prd_files, reader.frames_per_file, capture.sha256, nonpixel)
    ]
    scan = os.path.basename(os.path.normpath(inp))
    sidecars = {
        os.path.join(scan, name): os.path.join(inp, name)
        for name in sorted(os.listdir(inp))
        if not name.startswith("ss_stack_") and os.path.isfile(os.path.join(inp, name))
    }
    if info_path:
        parent = os.path.dirname(os.path.normpath(inp))
        sidecars[os.path.relpath(info_path, parent)] = info_path
    group = zarr.open_group(out, mode="r+").create_group("prd", overwrite=True)
    group.attrs.update(dict(
        format="PVCAM PRD",
        scan=scan,
        layout=dict(width=reader.width, height=reader.height, dtype=reader.dtype.str,
                    header_bytes=reader.header_bytes, gap_bytes=reader.gap_bytes),
        files=files,
        header=parse_prd_header(nonpixel[0]) if nonpixel else None,
        axes=LAB_AXES,
        sidecars={k: _b64(v) for k, v in sidecars.items()},
        info=info,
    ))
    data = np.frombuffer(b"".join(nonpixel), dtype="u1")
    group.create_array("nonpixel", data=data, chunks=(max(1, min(len(data), 2**24)),))
    for name, values in reader.frame_metadata(capture).items():
        group.create_array(name, data=values)


def _b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


@prd.command(name="export")
def export(inp: str, out: str, *, verify: bool = True) -> None:
    """
    Rebuild the original `.prd` scan from a converted zarr.

    Writes `OUT/<scan>/ss_stack_*.prd` and the sidecar files, byte for byte.

    Parameters
    ----------
    inp
        Converted NIfTI-Zarr / OME-Zarr (made by `lsm prd`).
    out
        Directory in which the scan directory is created.
    verify
        Check each rebuilt file against the SHA-256 recorded at conversion.
    """
    import hashlib

    import zarr

    group = zarr.open_group(inp, mode="r")
    meta = dict(group["prd"].attrs)
    nonpixel = group["prd/nonpixel"][:].tobytes() if meta["files"] else b""
    level0 = group["0"]
    hdr, gap = meta["layout"]["header_bytes"], meta["layout"]["gap_bytes"]
    for rel, b64 in meta["sidecars"].items():
        os.makedirs(os.path.dirname(os.path.join(out, rel)), exist_ok=True)
        with open(os.path.join(out, rel), "wb") as fh:
            fh.write(base64.b64decode(b64))
    os.makedirs(os.path.join(out, meta["scan"]), exist_ok=True)
    z0 = np_off = 0
    for f in meta["files"]:
        raw = nonpixel[np_off:np_off + f["nonpixel_bytes"]]
        np_off += f["nonpixel_bytes"]
        sha = hashlib.sha256()
        with open(os.path.join(out, meta["scan"], f["name"]), "wb") as fh:
            def put(b: bytes) -> None:
                fh.write(b)
                sha.update(b)

            put(raw[:hdr])
            for k0 in range(0, f["frames"], 128):
                frames = np.ascontiguousarray(
                    level0[z0 + k0:z0 + min(k0 + 128, f["frames"])],
                    dtype=meta["layout"]["dtype"])
                for k, frame in enumerate(frames, start=k0):
                    put(frame.tobytes())
                    pos = hdr + k * gap
                    put(raw[pos:pos + gap] if k < f["frames"] - 1 else raw[pos:])
        z0 += f["frames"]
        if verify and f["sha256"] and sha.hexdigest() != f["sha256"]:
            raise RuntimeError(f"{f['name']}: SHA-256 differs from the original")
    logger.info("Rebuilt %d file(s)%s.", len(meta["files"]),
                " (SHA-256 verified)" if verify else "")


def _parse_size(size: str | float) -> int:
    """Parse "5TB", "500 GiB", "1e12"... into bytes (decimal units)."""
    if isinstance(size, (int, float)):
        return int(size)
    m = re.fullmatch(r"\s*([0-9.eE+-]+)\s*([kKmMgGtTpP]?)(i?)[bB]?\s*", size)
    if not m:
        raise ValueError(f"Cannot parse size: {size!r}")
    value, prefix, binary = m.groups()
    base = 1024 if binary else 1000
    power = " kmgtp".index(prefix.lower()) if prefix else 0
    return int(float(value) * base**power)


def _shard_for_size(
    shape: tuple, dtype: np.dtype, zarr_config: ZarrConfig, max_bytes: int
) -> tuple[int, int, int]:
    """Shard spanning the frame in y/x and as many z slices as fit."""
    chunk, _ = _compute_zarr_layout(shape, dtype, replace(zarr_config, shard=None))
    cz, cy, cx = chunk[-3:]
    ny = math.ceil(shape[-2] / cy) * cy
    nx = math.ceil(shape[-1] / cx) * cx
    nz = math.ceil(shape[-3] / cz) * cz
    # 1% margin for incompressible chunks and the shard index
    slices = int(0.99 * max_bytes) // (ny * nx * np.dtype(dtype).itemsize)
    sz = min(nz, (slices // cz) * cz)
    if sz < cz:
        raise ValueError(f"max_shard_size {max_bytes} B is smaller than one "
                         f"chunk-thick slab ({cz * ny * nx * dtype.itemsize} B).")
    return (sz, ny, nx)


def _verify_level0(reader: PrdSetInterpreter, arr) -> None:  # noqa: ANN001
    """Check that level 0 decodes to exactly the .prd frames."""
    step = int((arr.shards or arr.chunks)[0])
    step = max(step, int(arr.chunks[0]))
    buf = np.empty((step, *reader.shape[1:]), reader.dtype)
    for z0 in range(0, reader.shape[0], step):
        z1 = min(z0 + step, reader.shape[0])
        want = reader.read_frames(z0, z1, buf)
        got = np.asarray(arr[z0:z1])
        if not np.array_equal(got, want):
            bad = int(np.count_nonzero(got != want))
            raise RuntimeError(
                f"Verification failed: {bad} voxel(s) differ in slices "
                f"[{z0}, {z1}) of level 0.")
    logger.info("Verified: level 0 is bit-identical to the .prd frames.")
