"""Module in which converters register themselves."""
from typing import Callable

known_converters = {}
known_extensions = {
    '.nii': ['nifti'],
    '.mgh': ['mgh'],
    '.mgz': ['mgh'],
    '.zarr': ['zarr'],
    '.ome.zarr': ['omezarr'],
    '.nii.zarr': ['niftizarr'],
    '.ome.tiff': ['ometiff'],
    '.ome.tif': ['ometiff'],
    '.tiff': ['tiff'],
    '.tif': ['tiff'],
    '.png': ['png'],
    '.jpeg': ['jpeg'],
    '.jpg': ['jpeg'],
    '.jp2': ['jpeg2000'],
    '.mat': ['matlab'],
}
format_to_extension = {
    'nifti': ['.nii.gz', '.nii', '.nii.bz2'],
    'mgh': ['.mgz', '.mgh'],
    'zarr': ['.zarr'],
    'omezarr': ['.ome.zarr'],
    'niftizarr': ['.nii.zarr'],
    'ometiff': ['.ome.tiff', '.ome.tif'],
    'tiff': ['.tiff', '.tif'],
    'png': ['.png'],
    'jpeg': ['.jpeg', '.jpg'],
    'jpeg2000': ['.jp2'],
    'matlab': ['.mat'],
}


def register_converter(src: str, dst: str) -> Callable:
    """Register a converter (decorator)."""

    def _decorator(func: Callable) -> Callable:
        known_converters[(src, dst)] = func
        return func

    return _decorator


def register_extension(ext: str, format: str) -> None:
    """Register a known extension."""
    known_extensions.setdefault(ext, [])
    if format not in known_extensions[ext]:
        known_extensions[ext].append(format)
    format_to_extension.setdefault(format, [])
    if ext not in format_to_extension[format]:
        format_to_extension[format].append(ext)
