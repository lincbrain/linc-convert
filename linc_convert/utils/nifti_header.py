"""Utilities for loading, creating, and manipulating NIfTI headers."""
import io
import warnings
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
from niizarr import default_nifti_header
from niizarr._header import bin2nii, get_nibabel_klass

from linc_convert.utils.io.zarr import open_array
from linc_convert.utils.io.zarr.abc import NiftiHeader
from linc_convert.utils.orientation import center_affine, orientation_to_affine
from linc_convert.utils.unit import to_nifti_unit
from linc_convert.utils.zarr_config import NiiConfig


def _try_open_zarr_array(path: Path) -> "ZarrArray":  # noqa: F821
    """
    Attempt to open a Zarr array at `path`.

    Returns the opened array on success, or raises the last exception.
    """
    last_err: Exception | None = None
    for v in (2, 3):
        try:
            return open_array(path, mode="r", zarr_version=v)
        except Exception as e:  # noqa: BLE001 - we want to keep original exception
            # types
            last_err = e
    # If we get here, all attempts failed.
    assert last_err is not None
    raise last_err


def load_nifti_header_from_file(path: str | Path) -> "NiftiHeader":
    """
    Load a NIfTI header from a file path.

    Supports:
      - Standard NIfTI files: `.nii`, `.nii.gz`
      - Zarr containing Nifti Header Array: `.zarr`
      - A Zarr Array containing a NIfTI header

    Parameters
    ----------
    path : str | pathlib.Path
        Path to a NIfTI file or Zarr directory.

    Returns
    -------
    NiftiHeader
        A nibabel NIfTI header object.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the path exists but is not a recognized NIfTI file or Zarr directory.
    Exception
        If Zarr opening fails for all supported versions.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    suffix = "".join(p.suffixes).lower()

    # Standard NIfTI files
    if suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
        img = nib.load(str(p))
        return img.header.copy()

    if suffix.endswith(".zarr") or p.is_dir():
        zarr_target = p
        candidate = p / "nifti"
        if candidate.exists():
            zarr_target = candidate
        return load_nifti_header_from_zarray(zarr_target)

    warnings.warn(
        f"Cannot load NIfTI header from {p!r}: not a recognized NIfTI file or Zarr "
        f"directory.",
        stacklevel=2,
    )
    raise ValueError(f"Unrecognized NIfTI header source: {p!r}")


def load_nifti_header_from_zarray(path: str | Path) -> "NiftiHeader":
    """
    Load a NIfTI header from a Zarr array storing the raw NIfTI header bytes.

    This function attempts Zarr format v2 first, then v3 (or vice versa if you
    change the order in `_try_open_zarr_array`). It reads the header bytes once,
    infers the correct nibabel header class via `get_nibabel_klass`, and
    constructs the header with `from_fileobj(check=False)` (since the buffer
    contains exactly the header bytes, not a full image).

    Parameters
    ----------
    path : str | pathlib.Path
        Path to the Zarr array (directory) containing NIfTI header bytes, or a
        directory that contains a `nifti` Zarr array.

    Returns
    -------
    NiftiHeader
        A nibabel NIfTI header object parsed from the stored bytes.

    Raises
    ------
    Exception
        If the Zarr array cannot be opened using supported versions.
    ValueError
        If the Zarr array does not yield any bytes.
    """
    p = Path(path)

    arr = _try_open_zarr_array(p)

    np_arr = np.asarray(arr)
    raw_bytes = np_arr.tobytes()

    if not raw_bytes:
        raise ValueError(f"No header bytes found in Zarr array at: {p}")

    header_probe = bin2nii(raw_bytes)
    NiftiHeader, _NiftiImage = get_nibabel_klass(header_probe)

    # Build the header from the bytes buffer.
    header = NiftiHeader.from_fileobj(io.BytesIO(raw_bytes), check=False)
    return header


def _recompute_affine_if_requested(
    affine_loaded: Optional[np.ndarray],
    orientation: str,
    center: bool,
    shape_zyx: Tuple[int, int, int],
    voxel_size_xyz: Tuple[float, float, float],
) -> np.ndarray:
    """
    Decide which affine to use. 
    
    If we have an affine from a loaded header
    AND the user hasn't asked for a specific orientation/centering,
    keep it. Otherwise, compute from orientation and (optionally) center.
    """
    want_custom = (orientation is not None and orientation != "RAS") or center
    if (affine_loaded is not None) and not want_custom:
        return affine_loaded

    # Build from orientation + voxel size; your helpers already exist.
    # Note: your `orientation_to_affine` expects voxel sizes in XYZ order,
    # while the shape is ZYX; you previously used vx[::-1] for that call.
    aff = orientation_to_affine(orientation, *voxel_size_xyz)
    if center:
        # Your center_affine takes shape in XYZ order; you previously used
        # reversed_shape[:3] where reversed_shape = list(reversed(arr.shape)).
        # Here, shape_zyx is already in (Z, Y, X), so pass reversed to (X, Y, Z)
        aff = center_affine(aff, (shape_zyx[2], shape_zyx[1], shape_zyx[0]))
    return aff


def build_nifti_header(
    *,
    zgroup: 'ZarrGroup',  # noqa: F821
    voxel_size_zyx: Tuple[float, float, float],
    unit: str,
    nii_config: NiiConfig,
) -> NiftiHeader:
    """
    Build a NIfTI header for the data in `zgroup`.

    Behavior:
      - If `nii_config.nifti_header` is provided and loadable, use it as a template.
      - Otherwise, start from `default_nifti_header(...)` using OME multiscales.
      - In either case, we always align the header to the actual data:
          * set shape / dtype to match the stored array
          * pick affine based on either the loaded header or the requested
          orientation/centering
          * set units from project’s `to_nifti_unit(unit)`

    Notes
    -----
      - The data array is taken as `zgroup["0"]`, and we follow your existing
        convention: NIfTI shape is reversed (X, Y, Z) compared to stored (Z, Y, X).
    """
    arr = zgroup["0"]
    shape_zyx = tuple(arr.shape)
    dtype = np.dtype(arr.dtype)

    # Pull multiscales from OME metadata if present
    multiscales = zgroup.attrs.get("ome", zgroup.attrs).get("multiscales")

    # Start from loaded header if available; else default
    hdr_loaded = None
    loaded_affine = None

    if nii_config.nifti_header:
        hdr_loaded = load_nifti_header_from_file(nii_config.nifti_header)
        # Try to grab an affine if the loader exposes it
        try:
            if hasattr(hdr_loaded, "_nib"):
                # nibabel header shim: need an affine; not stored in header, so None
                loaded_affine = None
        except Exception:
            pass

    if hdr_loaded is None:
        # Fall back to your project's default header builder
        # Note: this function expects the OME multiscales metadata.
        hdr = default_nifti_header(arr, multiscales)
    else:
        hdr = hdr_loaded

    # Compute affine (possibly from requested orientation/center)
    # Your previous code used vx[::-1] when calling orientation_to_affine
    # because vx was [y, x, z] or similar; here we accept voxel_size_zyx
    # and convert to XYZ for that call.
    vx_xyz = (voxel_size_zyx[2], voxel_size_zyx[1], voxel_size_zyx[0])

    affine = _recompute_affine_if_requested(
        affine_loaded=loaded_affine,
        orientation=nii_config.orientation,
        center=nii_config.center,
        shape_zyx=shape_zyx,
        voxel_size_xyz=vx_xyz,
    )

    # NIfTI expects (X, Y, Z)
    shape_xyz = (shape_zyx[2], shape_zyx[1], shape_zyx[0])

    # Synchronize core fields with the actual data
    hdr.set_data_shape(shape_xyz)
    hdr.set_data_dtype(dtype)
    hdr.set_qform(affine)
    hdr.set_sform(affine)
    hdr.set_xyzt_units(to_nifti_unit(unit))

    return hdr
