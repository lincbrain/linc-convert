"""
JPEG2000 to OME-ZARR
====================

This script converts JPEG2000 files generated by MBF-Neurolucida into
a pyramidal OME-ZARR hierarchy. It does not recompute the image pyramid
but instead reuse the JPEG2000 levels (obtained by wavelet transform).

dependencies:
    numpy
    glymur
    zarr
    nibabel
    typer
"""
import cyclopts
import glymur
import zarr
import ast
import numcodecs
import uuid
import os
import math
import numpy as np
import nibabel as nib
from typing import Optional

app = cyclopts.App(help_format="markdown")


@app.default
def convert(
    inp: str,
    out: str = None,
    *,
    chunk: int = 1024,
    compressor: str = 'blosc',
    compressor_opt: str = "{}",
    max_load: int = 16384,
    nii: bool = False,
    orientation: str = 'coronal',
    center: bool = True,
    thickness: Optional[float] = None,
):
    """
    This command converts JPEG2000 files generated by MBF-Neurolucida
    into a pyramidal OME-ZARR (or NIfTI-Zarr) hierarchy.

    It does not recompute the image pyramid but instead reuse the
    JPEG2000 levels (obtained by wavelet transform).

    ## Orientation
    The anatomical orientation of the slice is given in terms of RAS axes.
    It is a combination of two letters from the set {L, R, A, P, I, S}, where

    - the first letter corresponds to the horizontal dimension and
        indicates the anatomical meaning of the _right_ of the jp2 image,
    - the second letter corresponds to the vertical dimension and
        indicates the anatomical meaning of the _bottom_ of the jp2 image.

    We also provide the aliases

    - coronal == LI
    - axial == LP
    - sagittal == PI

    The orientation flag is only usefule when converting to nifti-zarr.

    Parameters
    ----------
    inp
        Path to the input JP2 file
    out
        Path to the output Zarr directory [<INP>.ome.zarr]
    chunk
        Output chunk size
    compressor : {blosc, zlib, raw}
        Compression method
    compressor_opt
        Compression options
    max_load
        Maximum input chunk size
    nii
        Convert to nifti-zarr. True if path ends in ".nii.zarr"
    orientation
        Orientation of the slice
    center
        Set RAS[0, 0, 0] at FOV center
    thickness
        Slice thickness
    """
    if not out:
        out = os.path.splitext(inp)[0]
        out += '.nii.zarr' if nii else '.ome.zarr'

    nii = nii or out.endswith('.nii.zarr')

    if isinstance(compressor_opt, str):
        compressor_opt = ast.literal_eval(compressor_opt)

    j2k = glymur.Jp2k(inp)
    vxw, vxh = get_pixelsize(j2k)

    # Prepare Zarr group
    omz = zarr.storage.DirectoryStore(out)
    omz = zarr.group(store=omz, overwrite=True)

    # Prepare chunking options
    opt = {
        'chunks': list(j2k.shape[2:]) + [chunk, chunk],
        'dimension_separator': r'/',
        'order': 'F',
        'dtype': np.dtype(j2k.dtype).str,
        'fill_value': None,
        'compressor': make_compressor(compressor, **compressor_opt),
    }

    # Write each level
    nblevel = j2k.codestream.segment[2].num_res
    has_channel = j2k.ndim - 2
    for level in range(nblevel):
        subdat = WrappedJ2K(j2k, level=level)
        shape = subdat.shape
        print('Convert level', level, 'with shape', shape)
        omz.create_dataset(str(level), shape=shape, **opt)
        array = omz[str(level)]
        if max_load is None or (shape[-2] < max_load and shape[-1] < max_load):
            array[...] = subdat[...]
        else:
            ni = ceildiv(shape[-2], max_load)
            nj = ceildiv(shape[-1], max_load)
            for i in range(ni):
                for j in range(nj):
                    print(f'\r{i+1}/{ni}, {j+1}/{nj}', end='')
                    array[
                        ...,
                        i*max_load:min((i+1)*max_load, shape[-2]),
                        j*max_load:min((j+1)*max_load, shape[-1]),
                    ] = subdat[
                        ...,
                        i*max_load:min((i+1)*max_load, shape[-2]),
                        j*max_load:min((j+1)*max_load, shape[-1]),
                    ]
            print('')

    # Write OME-Zarr multiscale metadata
    print('Write metadata')
    multiscales = [{
        'version': '0.4',
        'axes': [
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}
        ],
        'datasets': [],
        'type': 'jpeg2000',
        'name': '',
    }]
    if has_channel:
        multiscales[0]['axes'].insert(0, {"name": "c", "type": "channel"})

    for n in range(nblevel):
        shape0 = omz['0'].shape[-2:]
        shape = omz[str(n)].shape[-2:]
        multiscales[0]['datasets'].append({})
        level = multiscales[0]['datasets'][-1]
        level["path"] = str(n)

        # I assume that wavelet transforms end up aligning voxel edges
        # across levels, so the effective scaling is the shape ratio,
        # and there is a half voxel shift wrt to the "center of first voxel"
        # frame
        level["coordinateTransformations"] = [
            {
                "type": "scale",
                "scale": [1.0] * has_channel + [
                    (shape0[0]/shape[0])*vxh,
                    (shape0[1]/shape[1])*vxw,
                ]
            },
            {
                "type": "translation",
                "translation": [0.0] * has_channel + [
                    (shape0[0]/shape[0] - 1)*vxh*0.5,
                    (shape0[1]/shape[1] - 1)*vxw*0.5,
                ]
            }
        ]
    multiscales[0]["coordinateTransformations"] = [
        {
            "scale": [1.0] * (2 + has_channel),
            "type": "scale"
        }
    ]
    omz.attrs["multiscales"] = multiscales

    if not nii:
        print('done.')
        return

    # Write NIfTI-Zarr header
    # NOTE: we use nifti1 because dimensions typically do not fit in a short
    # TODO: we do not write the json zattrs, but it should be added in
    #       once the nifti-zarr package is released
    shape = list(reversed(omz['0'].shape))
    if has_channel:
        shape = shape[:2] + [1, 1] + shape[2:]
    affine = orientation_to_affine(orientation, vxw, vxh, thickness or 1)
    if center:
        affine = center_affine(affine, shape[:2])
    header = nib.Nifti2Header()
    header.set_data_shape(shape)
    header.set_data_dtype(omz['0'].dtype)
    header.set_qform(affine)
    header.set_sform(affine)
    header.set_xyzt_units(nib.nifti1.unit_codes.code['micron'])
    header.structarr['magic'] = b'nz2\0'
    header = np.frombuffer(header.structarr.tobytes(), dtype='u1')
    opt = {
        'chunks': [len(header)],
        'dimension_separator': r'/',
        'order': 'F',
        'dtype': '|u1',
        'fill_value': None,
        'compressor': None,
    }
    omz.create_dataset('nifti', data=header, shape=shape, **opt)
    print('done.')


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


class WrappedJ2K:
    """
    A wrapper around the J2K object at any resolution level, and
    with virtual transposition of the axes into [C, H, W] order.

    The resulting object can be sliced, but each index must be a `slice`
    (dropping axes using integer indices or adding axes using `None`
    indices is forbidden).

    The point is to ensure that the zarr writer only loads chunk-sized data.
    """

    def __init__(self, j2k, level=0, channel_first=True):
        """
        Parameters
        ----------
        j2k : glymur.Jp2k
            The JPEG2000 object.
        level : int
            Resolution level to map (highest resolution = 0).
        channel_first : bool
            Return an array with shape (C, H, W) instead of (H, W, C)
            when there is a channel dimension.
        """
        self.j2k = j2k
        self.level = level
        self.channel_first = channel_first

    @property
    def shape(self):
        channel = list(self.j2k.shape[2:])
        shape = [ceildiv(s, 2**self.level) for s in self.j2k.shape[:2]]
        if self.channel_first:
            shape = channel + shape
        else:
            shape += channel
        return tuple(shape)

    @property
    def dtype(self):
        return self.j2k.dtype

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        if Ellipsis not in index:
            index += (Ellipsis,)
        if any(idx is None for idx in index):
            raise TypeError('newaxis not supported')

        # substitute ellipses
        new_index = []
        has_seen_ellipsis = False
        last_was_ellipsis = False
        nb_ellipsis = max(0, self.j2k.ndim + 1 - len(index))
        for idx in index:
            if idx is Ellipsis:
                if not has_seen_ellipsis:
                    new_index += [slice(None)] * nb_ellipsis
                elif not last_was_ellipsis:
                    raise ValueError('Multiple ellipses should be contiguous')
                has_seen_ellipsis = True
                last_was_ellipsis = True
            elif not isinstance(idx, slice):
                raise TypeError('Only slices are supported')
            elif idx.step not in (None, 1):
                raise ValueError('Striding not supported')
            else:
                last_was_ellipsis = False
                new_index += [idx]
        index = new_index

        if self.channel_first:
            *cidx, hidx, widx = index
        else:
            hidx, widx, *cidx = index
        hstart, hstop = hidx.start or 0, hidx.stop or 0
        wstart, wstop = widx.start or 0, widx.stop or 0

        # convert to level 0 indices
        hstart *= 2**self.level
        hstop *= 2**self.level
        wstart *= 2**self.level
        wstop *= 2**self.level
        hstop = min(hstop or self.j2k.shape[0], self.j2k.shape[0])
        wstop = min(wstop or self.j2k.shape[1], self.j2k.shape[1])
        area = (hstart, wstart, hstop, wstop)

        data = self.j2k.read(rlevel=self.level, area=area)
        if cidx:
            data = data[:, :, cidx[0]]
            if self.channel_first:
                data = np.transpose(data, [2, 0, 1])
        return data


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


def get_pixelsize(j2k):
    # Adobe XMP metadata
    # https://en.wikipedia.org/wiki/Extensible_Metadata_Platform
    XMP_UUID = 'BE7ACFCB97A942E89C71999491E3AFAC'
    TAG_Images = '{http://ns.adobe.com/xap/1.0/}Images'
    Tag_Desc = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description'
    Tag_PixelWidth = '{http://ns.adobe.com/xap/1.0/}PixelWidth'
    Tag_PixelHeight = '{http://ns.adobe.com/xap/1.0/}PixelHeight'

    vxw = vxh = 1.0
    for box in j2k.box:
        if getattr(box, 'uuid', None) == uuid.UUID(XMP_UUID):
            try:
                images = list(box.data.iter(TAG_Images))[0]
                desc = list(images.iter(Tag_Desc))[0]
                vxw = float(desc.attrib[Tag_PixelWidth])
                vxh = float(desc.attrib[Tag_PixelHeight])
            except Exception:
                pass
    return vxw, vxh


if __name__ == "__main__":
    app()