import itertools
import re
from typing import Any, Literal

import nibabel as nib
import numpy as np
import zarr

from linc_convert.utils.math import ceildiv
from linc_convert.utils.unit import convert_unit


def make_json(oct_meta: str) -> dict:
    """
    Make json from OCT metadata.

    Expected input:
    ---------------
    Image medium: 60% TDE
    Center Wavelength: 1294.84nm
    Axial resolution: 4.9um
    Lateral resolution: 4.92um
    FOV: 3x3mm
    Voxel size: 3x3x3um
    Depth focus range: 225um
    Number of focuses: 2
    Focus #: 2
    Slice thickness: 450um.
    Number of slices: 75
    Slice #:23
    Modality: dBI
    """

    def _parse_value_unit(
        string: str, n: int = None
    ) -> tuple[float | list[float], str | Any]:
        number = r"-?(\d+\.?\d*|\d*\.?\d+)(E-?\d+)?"
        value = "x".join([number] * (n or 1))
        match = re.fullmatch(r"(?P<value>" + value + r")(?P<unit>\w*)", string)
        value, unit = match.group("value"), match.group("unit")
        value = list(map(float, value.split("x")))
        if n is None:
            value = value[0]
        return value, unit

    meta = {
        "BodyPart": "BRAIN",
        "Environment": "exvivo",
        "SampleStaining": "none",
    }

    for line in oct_meta.split("\n"):
        if ":" not in line:
            continue

        key, value = line.split(":")
        key, value = key.strip(), value.strip()

        if key == "Image medium":
            parts = value.split()
            if "TDE" in parts:
                parts[parts.index("TDE")] = "2,2' Thiodiethanol (TDE)"
            meta["SampleMedium"] = " ".join(parts)

        elif key == "Center Wavelength":
            value, unit = _parse_value_unit(value)
            meta["Wavelength"] = value
            meta["WavelengthUnit"] = unit

        elif key == "Axial resolution":
            value, unit = _parse_value_unit(value)
            meta["ResolutionAxial"] = value
            meta["ResolutionAxialUnit"] = unit

        elif key == "Lateral resolution":
            value, unit = _parse_value_unit(value)
            meta["ResolutionLateral"] = value
            meta["ResolutionLateralUnit"] = unit

        elif key == "Voxel size":
            value, unit = _parse_value_unit(value, n=3)
            meta["PixelSize"] = value
            meta["PixelSizeUnits"] = unit

        elif key == "Depth focus range":
            value, unit = _parse_value_unit(value)
            meta["DepthFocusRange"] = value
            meta["DepthFocusRangeUnit"] = unit

        elif key == "Number of focuses":
            value, unit = _parse_value_unit(value)
            meta["FocusCount"] = int(value)

        elif key == "Slice thickness":
            value, unit = _parse_value_unit(value)
            unit = convert_unit(value, unit[:-1], "u")
            meta["SliceThickness"] = value

        elif key == "Number of slices":
            value, unit = _parse_value_unit(value)
            meta["SliceCount"] = int(value)

        elif key == "Modality":
            meta["OCTModality"] = value

        else:
            continue

    return meta


def generate_pyramid(
    omz: zarr.Group,
    levels: int | None = None,
    ndim: int = 3,
    max_load: int = 512,
    mode: Literal["mean", "median"] = "median",
    no_pyramid_axis: int | str | None = None,
) -> list[list[int]]:
    """
    Generate the levels of a pyramid in an existing Zarr.

    Parameters
    ----------
    path : PathLike | str
        Path to parent Zarr
    levels : int
        Number of additional levels to generate.
        By default, stop when all dimensions are smaller than their
        corresponding chunk size.
    shard : list[int] | bool | {"auto"} | None
        Shard size.
        * If `None`, use same shard size as the input array;
        * If `False`, no dot use sharding;
        * If `True` or `"auto"`, automatically find shard size;
        * Otherwise, use provided shard size.
    ndim : int
        Number of spatial dimensions.
    max_load : int
        Maximum number of voxels to load along each dimension.
    mode : {"mean", "median"}
        Whether to use a mean or median moving window.

    Returns
    -------
    shapes : list[list[int]]
        Shapes of all levels, from finest to coarsest, including the
        existing top level.
    """
    # Read properties from base level
    shape = list(omz["0"].shape)
    chunk_size = omz["0"].chunks
    opt = {
        "dimension_separator": omz["0"]._dimension_separator,
        "order": omz["0"]._order,
        "dtype": omz["0"]._dtype,
        "fill_value": omz["0"]._fill_value,
        "compressor": omz["0"]._compressor,
        "chunks": omz["0"].chunks,
    }

    # Select windowing function
    if mode == "median":
        func = np.median
    else:
        assert mode == "mean"
        func = np.mean

    level = 0
    batch, shape = shape[:-ndim], shape[-ndim:]
    allshapes = [shape]

    while True:
        level += 1

        # Compute downsampled shape
        prev_shape, shape = shape, []
        for i, length in enumerate(prev_shape):
            if i == no_pyramid_axis:
                shape.append(length)
            else:
                shape.append(max(1, length // 2))

        # Stop if seen enough levels or level shape smaller than chunk size
        if levels is None:
            if all(x <= c for x, c in zip(shape, chunk_size[-ndim:])):
                break
        elif level > levels:
            break

        print("Compute level", level, "with shape", shape)

        allshapes.append(shape)
        omz.create_dataset(str(level), shape=batch + shape, **opt)

        # Iterate across `max_load` chunks
        # (note that these are unrelared to underlying zarr chunks)
        grid_shape = [ceildiv(n, max_load) for n in prev_shape]
        for chunk_index in itertools.product(*[range(x) for x in grid_shape]):
            print(f"chunk {chunk_index} / {tuple(grid_shape)})", end="\r")

            # Read one chunk of data at the previous resolution
            slicer = [Ellipsis] + [
                slice(i * max_load, min((i + 1) * max_load, n))
                for i, n in zip(chunk_index, prev_shape)
            ]
            fullshape = omz[str(level - 1)].shape
            dat = omz[str(level - 1)][tuple(slicer)]

            # Discard the last voxel along odd dimensions
            crop = [
                0 if y == 1 else x % 2 for x, y in zip(dat.shape[-ndim:], fullshape)
            ]
            # Don't crop the axis not down-sampling
            # cannot do if not no_pyramid_axis since it could be 0
            if no_pyramid_axis is not None:
                crop[no_pyramid_axis] = 0
            slcr = [slice(-1) if x else slice(None) for x in crop]
            dat = dat[tuple([Ellipsis, *slcr])]

            if any(n == 0 for n in dat.shape):
                # last strip had a single voxel, nothing to do
                continue

            patch_shape = dat.shape[-ndim:]

            # Reshape into patches of shape 2x2x2
            windowed_shape = [
                x for n in patch_shape for x in (max(n // 2, 1), min(n, 2))
            ]
            if no_pyramid_axis is not None:
                windowed_shape[2 * no_pyramid_axis] = patch_shape[no_pyramid_axis]
                windowed_shape[2 * no_pyramid_axis + 1] = 1

            dat = dat.reshape(batch + windowed_shape)
            # -> last `ndim`` dimensions have shape 2x2x2
            dat = dat.transpose(
                list(range(len(batch)))
                + list(range(len(batch), len(batch) + 2 * ndim, 2))
                + list(range(len(batch) + 1, len(batch) + 2 * ndim, 2))
            )
            # -> flatten patches
            smaller_shape = [max(n // 2, 1) for n in patch_shape]
            if no_pyramid_axis is not None:
                smaller_shape[no_pyramid_axis] = patch_shape[no_pyramid_axis]

            dat = dat.reshape(batch + smaller_shape + [-1])

            # Compute the median/mean of each patch
            dtype = dat.dtype
            dat = func(dat, axis=-1)
            dat = dat.astype(dtype)

            # Write output
            slicer = [Ellipsis] + [
                slice(i * max_load // 2, min((i + 1) * max_load // 2, n))
                if axis_index != no_pyramid_axis
                else slice(i * max_load, min((i + 1) * max_load, n))
                for i, axis_index, n in zip(chunk_index, range(ndim), shape)
            ]

            omz[str(level)][tuple(slicer)] = dat

    print("")

    return allshapes

    pass


def write_ome_metadata(
    omz: zarr.Group,
    axes: list[str],
    space_scale: float | list[float] = 1,
    time_scale: float = 1,
    space_unit: str = "micrometer",
    time_unit: str = "second",
    name: str = "",
    pyramid_aligns: str | int | list[str | int] = 2,
    levels: int | None = None,
    no_pool: int | None = None,
    multiscales_type: str = "",
) -> None:
    """
    Write OME metadata into Zarr.

    Parameters
    ----------
    path : str | PathLike
        Path to parent Zarr.
    axes : list[str]
        Name of each dimension, in Zarr order (t, c, z, y, x)
    space_scale : float | list[float]
        Finest-level voxel size, in Zarr order (z, y, x)
    time_scale : float
        Time scale
    space_unit : str
        Unit of spatial scale (assumed identical across dimensions)
    space_time : str
        Unit of time scale
    name : str
        Name attribute
    pyramid_aligns : float | list[float] | {"center", "edge"}
        Whether the pyramid construction aligns the edges or the centers
        of the corner voxels. If a (list of) number, assume that a moving
        window of that size was used.
    levels : int
        Number of existing levels. Default: find out automatically.
    zarr_version : {2, 3} | None
        Zarr version. If `None`, guess from existing zarr array.

    """
    # Read shape at each pyramid level
    shapes = []
    level = 0
    while True:
        if levels is not None and level > levels:
            break

        if str(level) not in omz.keys():
            levels = level
            break
        shapes += [
            omz[str(level)].shape,
        ]
        level += 1

    axis_to_type = {
        "x": "space",
        "y": "space",
        "z": "space",
        "t": "time",
        "c": "channel",
    }

    # Number of spatial (s), batch (b) and total (n) dimensions
    ndim = len(axes)
    sdim = sum(axis_to_type[axis] == "space" for axis in axes)
    bdim = ndim - sdim

    if isinstance(pyramid_aligns, (int, str)):
        pyramid_aligns = [pyramid_aligns]
    pyramid_aligns = list(pyramid_aligns)
    if len(pyramid_aligns) < sdim:
        repeat = pyramid_aligns[:1] * (sdim - len(pyramid_aligns))
        pyramid_aligns = repeat + pyramid_aligns
    pyramid_aligns = pyramid_aligns[-sdim:]

    if isinstance(space_scale, (int, float)):
        space_scale = [space_scale]
    space_scale = list(space_scale)
    if len(space_scale) < sdim:
        repeat = space_scale[:1] * (sdim - len(space_scale))
        space_scale = repeat + space_scale
    space_scale = space_scale[-sdim:]

    multiscales = [
        {
            "version": "0.4",
            "axes": [
                {
                    "name": axis,
                    "type": axis_to_type[axis],
                }
                if axis_to_type[axis] == "channel"
                else {
                    "name": axis,
                    "type": axis_to_type[axis],
                    "unit": (
                        space_unit
                        if axis_to_type[axis] == "space"
                        else time_unit
                        if axis_to_type[axis] == "time"
                        else None
                    ),
                }
                for axis in axes
            ],
            "datasets": [],
            "type": "median window " + "x".join(["2"] * sdim)
            if not multiscales_type
            else multiscales_type,
            "name": name,
        }
    ]

    shape0 = shapes[0]
    for n in range(len(shapes)):
        shape = shapes[n]
        multiscales[0]["datasets"].append({})
        level = multiscales[0]["datasets"][-1]
        level["path"] = str(n)

        scale = [1] * bdim + [
            (
                pyramid_aligns[i] ** n
                if not isinstance(pyramid_aligns[i], str)
                else (shape0[bdim + i] / shape[bdim + i])
                if pyramid_aligns[i][0].lower() == "e"
                else ((shape0[bdim + i] - 1) / (shape[bdim + i] - 1))
            )
            * space_scale[i]
            if i != no_pool
            else space_scale[i]
            for i in range(sdim)
        ]
        translation = [0] * bdim + [
            (
                pyramid_aligns[i] ** n - 1
                if not isinstance(pyramid_aligns[i], str)
                else (shape0[bdim + i] / shape[bdim + i]) - 1
                if pyramid_aligns[i][0].lower() == "e"
                else 0
            )
            * 0.5
            * space_scale[i]
            if i != no_pool
            else 0
            for i in range(sdim)
        ]

        level["coordinateTransformations"] = [
            {
                "type": "scale",
                "scale": scale,
            },
            {
                "type": "translation",
                "translation": translation,
            },
        ]

    scale = [1.0] * ndim
    if "t" in axes:
        scale[axes.index("t")] = time_scale
    multiscales[0]["coordinateTransformations"] = [{"scale": scale, "type": "scale"}]

    multiscales[0]["version"] = "0.4"
    omz.attrs["multiscales"] = multiscales


def niftizarr_write_header(
    omz: zarr.Group,
    shape: list[int],
    affine: np.ndarray,
    dtype: np.dtype | str,
    unit: Literal["micron", "mm"] | None = None,
    header: nib.Nifti1Header | nib.Nifti2Header | None = None,
    nifti_version: Literal[1, 2] = 1,
) -> None:
    """
    Write NIfTI header in a NIfTI-Zarr file.

    Parameters
    ----------
    path : PathLike | str
        Path to parent Zarr.
    affine : (4, 4) matrix
        Orientation matrix.
    shape : list[int]
        Array shape, in NIfTI order (x, y, z, t, c).
    dtype : np.dtype | str
        Data type.
    unit : {"micron", "mm"}, optional
        World unit.
    header : nib.Nifti1Header | nib.Nifti2Header, optional
        Pre-instantiated header.
    zarr_version : int, default=3
        Zarr version.
    """
    # TODO: we do not write the json zattrs, but it should be added in
    #       once the nifti-zarr package is released

    # If dimensions do not fit in a short (which is often the case), we
    # use NIfTI 2.
    if all(x < 32768 for x in shape) or nifti_version == 1:
        NiftiHeader = nib.Nifti1Header
    else:
        NiftiHeader = nib.Nifti2Header

    header = header or NiftiHeader()
    header.set_data_shape(shape)
    header.set_data_dtype(dtype)
    header.set_qform(affine)
    header.set_sform(affine)
    if unit:
        header.set_xyzt_units(nib.nifti1.unit_codes.code[unit])
    header = np.frombuffer(header.structarr.tobytes(), dtype="u1")

    metadata = {
        "chunks": [len(header)],
        "order": "F",
        "dtype": "|u1",
        "fill_value": None,
        "compressor": None,  # TODO: Subject to change compression
    }

    omz.create_dataset("nifti", data=header, shape=len(header), **metadata)

    print("done.")
