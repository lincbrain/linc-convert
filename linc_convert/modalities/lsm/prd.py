"""Convert Kinetix .prd image(s) to OME-Zarr/NIfTI-Zarr.

A .prd file is a stack of uint16 frames with a fixed header
and inter-frame gap, and padding at the end. The input is a directory of
`ss_stack_*.prd` files, which are concatenated along the acquisition order.

Example input files can be found at
https://dandiarchive.org/dandiset/001769/draft/files?location=rawdata/HG9_Z1_Y1
"""

import logging
import math
import os
import platform
import re
import sys
import sysconfig
from dataclasses import replace

import cyclopts
import numpy as np
import tqdm

from linc_convert.modalities.lsm.cli import lsm
from linc_convert.utils.io.prd import PrdSetInterpreter
from linc_convert.utils.io.zarr import from_config
from linc_convert.utils.io.zarr.generate_pyramid import default_levels
from linc_convert.utils.io.zarr.helpers import _compute_zarr_layout
from linc_convert.utils.io.zarr.stream_pyramid import (
    StreamingPyramidWriter,
    stream_volume,
    workers_for_memory,
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
    voxel_size: list[float] = (1, 1, 1),
    workers: int | None = None,
    max_memory: float | None = None,
    max_shard_size: str | None = None,
    slab_depth: int | None = None,
    verify: bool = False,
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
        Voxel size along the X, Y and Z dimensions, in microns.
    workers
        Number of threads used to compress and write chunks.
        Default: number of available CPUs.
    max_memory
        Memory budget in GiB. The number of workers is reduced until the
        estimated peak memory fits (useful during acquisition).
    max_shard_size
        Largest uncompressed shard size, e.g. "5TB" (the S3 object limit).
        Shards then span the full frame in y and x and as many slices along
        z as fit; this overrides `shard`. Sharded output is streamed into
        append-only shard files, so large shards need no extra memory.
    slab_depth
        Slices read, compressed and written per batch (rounded up to the
        inner chunk, or to the shard for non-streamed layouts). Deeper slabs
        use more memory and keep more threads busy. Default: one chunk (or
        shard) along z.
    verify
        After writing, re-read level 0 and check that it is bit-identical
        to the .prd frames (costs one extra read + decode of level 0).
    general_config
        General configuration
    zarr_config
        Zarr related configuration
    nii_config
        NIfTI header related configuration
    """
    reader = PrdSetInterpreter(inp)

    # Set default output path if not provided
    general_config.set_default_name(os.path.basename(inp))

    if max_shard_size:
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
        chunk, shard = _compute_zarr_layout(
            reader.shape, reader.dtype, zarr_config)
        levels = zarr_config.levels
        if levels == -1:
            levels = default_levels(reader.shape, chunk[-3:], None)
        appendable = shard is not None and zarr_config.driver == "zarr-python"
        native = chunk[-3] if appendable else (shard or chunk)[-3]
        unit_z = -(-(slab_depth or native) // native) * native
        tile = chunk[-3:] if appendable else (shard or chunk)[-3:]
        workers = workers_for_memory(
            int(max_memory * 2**30), reader.shape, reader.dtype.itemsize,
            unit_z, tile, levels, workers,
        )
        logger.info("Using %d worker(s) to fit %.1f GiB", workers, max_memory)
    writer = StreamingPyramidWriter(
        zgroup, reader.shape, reader.dtype, zarr_config,
        levels=zarr_config.levels, max_workers=workers,
        slab_depth=slab_depth,
    )
    with tqdm.tqdm(total=reader.shape[0], unit="frame") as pbar:
        stream_volume(writer, reader.read_frames, pbar.update)

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
