"""Orientation of an array of voxels with respect to world space."""

import numpy as np


def orientation_ensure_3d(orientation: str) -> str:
    """
    Convert an ND orientation string to a 3D orientation string.

    Parameters
    ----------
    orientation : str
        A 2D or 3D orientation string, such as `"RA"` or `"RAS"`.

    Returns
    -------
    orientation : str
        A 3D orientation string compatible with the input orientaition
    """
    orientation = {
        "coronal": "LI",
        "axial": "LP",
        "sagittal": "PI",
    }.get(orientation.lower(), orientation).upper()
    if len(orientation) == 2:
        if "L" not in orientation and "R" not in orientation:
            orientation += "R"
        if "P" not in orientation and "A" not in orientation:
            orientation += "A"
        if "I" not in orientation and "S" not in orientation:
            orientation += "S"
    return orientation


def orientation_to_affine(
    orientation: str, vxw: float = 1, vxh: float = 1, vxd: float = 1
) -> np.ndarray:
    """
    Build an affine matrix from an orientation string and voxel size.

    Parameters
    ----------
    orientation : str
        Orientation string
    vxw : float
        Width voxel size
    vxh : float
        Height voxel size
    vxd : float
        Depth voxel size

    Returns
    -------
    affine : (4, 4) array
        Affine orientation matrix
    """
    orientation = orientation_ensure_3d(orientation)
    affine = np.zeros([4, 4])
    vx = np.asarray([vxw, vxh, vxd])
    for i in range(3):
        letter = orientation[i]
        sign = -1 if letter in "LPI" else 1
        letter = {"L": "R", "P": "A", "I": "S"}.get(letter, letter)
        index = list("RAS").index(letter)
        affine[index, i] = sign * vx[i]
    return affine


def center_affine(affine: np.ndarray, shape: list[int]) -> np.ndarray:
    """
    Ensure that the center of the field-of-view has world coordinate (0,0,0).

    !!! note "The input affine is NOT modified in-place"

    Parameters
    ----------
    affine : array
        Orientation affine matrix
    shape : list[int]
        Shape of the array of voxels

    Returns
    -------
    affine : array
        Modified affine matrix.
    """
    if len(shape) == 2:
        shape = [*shape, 1]
    shape = np.asarray(shape)
    affine = np.copy(affine)
    affine[:3, -1] = -0.5 * affine[:3, :3] @ (shape - 1)
    return affine
