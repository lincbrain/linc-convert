import math
import numcodecs
import numpy as np


def orientation_ensure_3d(orientation):
    orientation = {
        'coronal': 'LI',
        'axial': 'LP',
        'sagittal': 'PI',
    }.get(orientation.lower(), orientation).upper()
    if len(orientation) == 2:
        if 'L' not in orientation and 'R' not in orientation:
            orientation += 'R'
        if 'P' not in orientation and 'A' not in orientation:
            orientation += 'A'
        if 'I' not in orientation and 'S' not in orientation:
            orientation += 'S'
    return orientation


def orientation_to_affine(orientation, vxw=1, vxh=1, vxd=1):
    orientation = orientation_ensure_3d(orientation)
    affine = np.zeros([4, 4])
    vx = np.asarray([vxw, vxh, vxd])
    for i in range(3):
        letter = orientation[i]
        sign = -1 if letter in 'LPI' else 1
        letter = {'L': 'R', 'P': 'A', 'I': 'S'}.get(letter, letter)
        index = list('RAS').index(letter)
        affine[index, i] = sign * vx[i]
    return affine


def center_affine(affine, shape):
    if len(shape) == 2:
        shape = [*shape, 1]
    shape = np.asarray(shape)
    affine[:3, -1] = -0.5 * affine[:3, :3] @ (shape - 1)
    return affine


def ceildiv(x, y):
    return int(math.ceil(x / y))


def floordiv(x, y):
    return int(math.floor(x / y))


def make_compressor(name, **prm):
    if not isinstance(name, str):
        return name
    name = name.lower()
    if name == 'blosc':
        Compressor = numcodecs.Blosc
    elif name == 'zlib':
        Compressor = numcodecs.Zlib
    else:
        raise ValueError('Unknown compressor', name)
    return Compressor(**prm)
