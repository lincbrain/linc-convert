"""Richardson-Lucy-type deconvolution (optix Snyder) of the (z, y) frames of a stitched volume.

Run on the output of ``stitch_chunks`` (a separate step, not part of the stitch): every frame
(the (z, y) plane at one x) is deconvolved with a Gaussian PSF, ``block_x`` frames at a time,
and written to a new OME-Zarr with one level. Use ``x_start`` / ``x_end`` to test a window.
"""
import os
from typing import Optional, Tuple

import cyclopts
import numpy as np

from linc_convert.modalities.lsm.cli import lsm

optix_deconvolve = cyclopts.App(name="optix_deconvolve", help_format="markdown")
lsm.command(optix_deconvolve)


def parse_sigma(text: str) -> Tuple[float, float]:
    """``"z,y"`` (or one number) in pixels -> two positive floats."""
    vals = [float(t) for t in str(text).split(",")]
    if len(vals) == 1:
        vals = vals * 2
    if len(vals) != 2 or min(vals) <= 0:
        raise ValueError(f"psf_sigma must be one number or 'z,y' (> 0), got {text!r}")
    return vals[0], vals[1]


def deconvolve_frames(frames: np.ndarray, sigma: Tuple[float, float], n_iter: Optional[int],
                      max_iter: int) -> np.ndarray:
    """Snyder deconvolution of frames (n, z, y) with a Gaussian PSF; returns float32."""
    import torch
    from optix.deconv import Snyder
    from optix.models import gaussian_psf
    size = tuple(2 * int(np.ceil(3 * s)) + 1 for s in sigma)
    psf = gaussian_psf(size, sigma, dtype=torch.float32)
    data = torch.from_numpy(np.ascontiguousarray(frames, dtype=np.float32))
    est = Snyder(psf, n_iter=n_iter, max_iter=max_iter).fit(data)
    return est.apply(data).detach().cpu().numpy().astype(np.float32)


@optix_deconvolve.default
def optix_deconvolve_cmd(input_path: str, output_path: str, *, psf_sigma: str = "2,2",
                         n_iter: Optional[int] = None, max_iter: int = 30,
                         block_x: int = 32, x_start: int = 0, x_end: int = 0,
                         group: str = "0", force: bool = False) -> None:
    """Deconvolve every (z, y) frame of a stitched OME-Zarr level with optix Snyder.

    Parameters
    ----------
    input_path : str
        Stitched ``.ome.zarr`` (axes z, y, x).
    output_path : str
        New ``.ome.zarr`` with the deconvolved level (and no other level).
    psf_sigma : str, default="2,2"
        Gaussian PSF standard deviation in pixels, ``"z,y"`` or one number.
    n_iter : int, optional
        Fixed number of iterations; otherwise the chi-square stopping rule (``max_iter``).
    max_iter : int, default=30
        Iteration cap of the chi-square rule.
    block_x : int, default=32
        Frames processed at a time.
    x_start, x_end : int, default=0
        Frame window (``x_end`` 0 = to the end); frames outside it are written as zeros.
    group : str, default="0"
        Level of the input to read.
    force : bool, default=False
        Overwrite an existing output.
    """
    import shutil

    import zarr
    sigma = parse_sigma(psf_sigma)
    src_group = zarr.open_group(input_path, mode="r")
    src = src_group[group]
    if src.ndim != 3:
        raise ValueError(f"expected a 3-D (z, y, x) array, got shape {src.shape}")
    if os.path.exists(output_path):
        if not force:
            print(f"{output_path} exists, skipping (--force to redo)")
            return
        shutil.rmtree(output_path)
    n_x = src.shape[2]
    x_end = n_x if x_end <= 0 else min(x_end, n_x)
    if not 0 <= x_start < x_end:
        raise ValueError(f"empty window x {x_start}:{x_end} of {n_x}")
    dst_group = zarr.open_group(output_path, mode="w")
    attrs = dict(src_group.attrs)
    ome = attrs.get("ome", attrs)
    for ms in ome.get("multiscales", []):
        ms["datasets"] = [d for d in ms.get("datasets", []) if str(d.get("path")) == group][:1]
    dst_group.attrs.update(attrs)
    dst = dst_group.create_array(group, shape=src.shape, chunks=src.chunks, dtype=src.dtype)
    top = float(np.iinfo(src.dtype).max) if np.issubdtype(src.dtype, np.integer) else None
    for a in range(x_start, x_end, block_x):
        b = min(a + block_x, x_end)
        frames = np.asarray(src[:, :, a:b], dtype=np.float32).transpose(2, 0, 1)   # (n, z, y)
        out = deconvolve_frames(frames, sigma, n_iter, max_iter).transpose(1, 2, 0)
        if top is not None:
            out = np.clip(np.rint(out), 0, top)
        dst[:, :, a:b] = out.astype(src.dtype)
        print(f"  frames {a}:{b} of {x_start}:{x_end}", flush=True)
    print(f"wrote {output_path}")