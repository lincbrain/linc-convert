"""Utilities for JPEG2000 files."""

# stdlib
import uuid
from dataclasses import dataclass

# externals
import numpy as np
from glymur import Jp2k

# internals
from linc_convert.utils.math import ceildiv


def get_pixelsize(j2k: Jp2k) -> tuple[float, float]:
    """Read pixelsize from the JPEG2000 file."""
    # Adobe XMP metadata
    # https://en.wikipedia.org/wiki/Extensible_Metadata_Platform
    XMP_UUID = "BE7ACFCB97A942E89C71999491E3AFAC"
    TAG_Images = "{http://ns.adobe.com/xap/1.0/}Images"
    Tag_Desc = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"
    Tag_PixelWidth = "{http://ns.adobe.com/xap/1.0/}PixelWidth"
    Tag_PixelHeight = "{http://ns.adobe.com/xap/1.0/}PixelHeight"

    vxw = vxh = 1.0
    for box in j2k.box:
        if getattr(box, "uuid", None) == uuid.UUID(XMP_UUID):
            try:
                images = list(box.data.iter(TAG_Images))[0]
                desc = list(images.iter(Tag_Desc))[0]
                vxw = float(desc.attrib[Tag_PixelWidth])
                vxh = float(desc.attrib[Tag_PixelHeight])
            except Exception:
                pass
    return vxw, vxh


@dataclass
class WrappedJ2K:
    """
    Array-like wrapper around a JPEG2000 object.

    A wrapper around the J2K object at any resolution level, and
    with virtual transposition of the axes into [C, H, W] order.

    The resulting object can be sliced, but each index must be a `slice`
    (dropping axes using integer indices or adding axes using `None`
    indices is forbidden).

    The point is to ensure that the zarr writer only loads chunk-sized data.

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

    j2k: Jp2k
    level: int = 0
    channel_first: bool = True

    @property
    def shape(self) -> tuple[int]:
        """Shape of the current level."""
        channel = list(self.j2k.shape[2:])
        shape = [ceildiv(s, 2**self.level) for s in self.j2k.shape[:2]]
        if self.channel_first:
            shape = channel + shape
        else:
            shape += channel
        return tuple(shape)

    @property
    def dtype(self) -> np.dtype:
        """Data type of the wrapped image."""
        return self.j2k.dtype

    def __getitem__(self, index: tuple[slice] | slice) -> np.ndarray:
        """Multidimensional slicing of the wrapped array."""
        if not isinstance(index, tuple):
            index = (index,)
        if Ellipsis not in index:
            index += (Ellipsis,)
        if any(idx is None for idx in index):
            raise TypeError("newaxis not supported")

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
                    raise ValueError("Multiple ellipses should be contiguous")
                has_seen_ellipsis = True
                last_was_ellipsis = True
            elif not isinstance(idx, slice):
                raise TypeError("Only slices are supported")
            elif idx.step not in (None, 1):
                raise ValueError("Striding not supported")
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
