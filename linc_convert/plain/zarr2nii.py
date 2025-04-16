"""Convert from Zarr to NIfTI."""
import zarr.storage
from cyclopts import App
from niizarr import zarr2nii

from linc_convert.cli import main, plain_group
from linc_convert.plain.cli import make_output_path
from linc_convert.plain.register import register_converter
from linc_convert.utils.opener import filesystem

app = App(name="zarr2nii", help_format="markdown", group=plain_group)
main.command(app)


if hasattr(zarr.storage, "FSStore"):
    FSStore = zarr.storage.FSStore
elif hasattr(zarr.storage, "FsspecStore"):
    FSStore = zarr.storage.FsspecStore
else:
    FSStore = None


@app.default
@register_converter('zarr', 'nifti')
@register_converter('omezarr', 'nifti')
@register_converter('niftizarr', 'nifti')
def convert(
    inp: str,
    out: str | None,
    *,
    level: int = 0,
) -> None:
    """
    Convert from Zarr to NIfTI.

    The input Zarr can be a NIfTI-Zarr, OME-Zarr or plain Zarr asset.

    Parameters
    ----------
    inp
        Path to input file.
    out
        Path to output file. Default: "{base}.nii.gz".
    level
        Pyramid level to extract if {OME|NIfTI}-Zarr.
    """
    # Default output name
    if not out:
        out = make_output_path(inp, "nifti")

    if inp.startswith("dandi://"):
        # Create an authentified fsspec store to pass to zarr2nii
        if not FSStore:
            raise ValueError("Cannot create a fsspec store.")
        import linc_convert.utils.dandifs  # noqa: F401  (register filesystem)
        url = filesystem(inp).s3_url(inp)
        fs = filesystem(url)
        inp = FSStore(url, fs=fs, mode="r")

    zarr2nii(inp, out, level)
