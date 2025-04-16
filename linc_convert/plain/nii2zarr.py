"""Convert from NIfTI to Zarr."""
import dataclasses
from typing import Annotated, Literal

from cyclopts import App, Parameter
from niizarr import nii2zarr

from linc_convert.cli import main, plain_group
from linc_convert.plain.cli import make_output_path
from linc_convert.plain.register import register_converter
from linc_convert.utils.opener import filesystem, open

app = App(name="nii2zarr", help_format="markdown", group=plain_group)
main.command(app)


@dataclasses.dataclass
class _Nii2ZarrConfig:
    """
    Configuration specific to NIfTI-to-Zarr conversion.

    Parameters
    ----------
    chunk
        Chunk size for spatial dimensions.
        The tuple allows different chunk sizes to be used along each dimension.
    chunk_channel
        Chunk size of the channel dimension. If 0, combine all channels
        in a single chunk.
    chunk_time
        Chunk size for the time dimension. If 0, combine all timepoints
        in a single chunk.
    shard
        Shard size for spatial dimensions.
        The tuple allows different shard sizes to be used along each dimension.
    shard_channel
        Shard size of the channel dimension. If 0, combine all channels
        in a single shard.
    shard_time
        Shard size for the time dimension. If 0, combine all timepoints
        in a single shard.
    nb_levels
        Number of pyramid levels to generate.
        If -1, make all possible levels until the level can be fit into
        one chunk.
    method
        Method used to compute the pyramid.
    label
        Is this is a label volume?  If `None`, guess from intent code.
    no_time
        If True, there is no time dimension so the 4th dimension
        (if it exists) should be interpreted as the channel dimensions.
    no_pyramid_axis
        Axis that should not be downsampled. If None, downsample
        across all three dimensions.
    fill_value
        Value to use for missing tiles
    compressor
        Compression to use
    compressor_options
        Options for the compressor.
    zarr_version
        Zarr format version.
    ome_version
        OME-Zarr version.
    """

    chunk: tuple[int] = 64
    chunk_channel: int = 1
    chunk_time: int = 1
    shard: tuple[int] | None = None
    shard_channel: int | None = None
    shard_time: int | None = None
    nb_levels: int = -1
    method: Literal['gaussian', 'laplacian'] = 'gaussian'
    label: bool | None = None
    no_time: bool = False
    no_pyramid_axis: str | int = None
    fill_value: int | float | complex | None = None
    compressor: Literal['blosc', 'zlib'] = 'blosc'
    compressor_options: dict = dataclasses.field(default_factory=(lambda: {}))
    zarr_version: Literal[2, 3] = 2
    ome_version: Literal["0.4", "0.5"] = "0.4"


Nii2ZarrConfig = Annotated[_Nii2ZarrConfig, Parameter(name="*")]


@app.default
@register_converter('zarr', 'nifti')
@register_converter('omezarr', 'nifti')
@register_converter('niftizarr', 'nifti')
def convert(
    inp: str,
    out: str | None,
    *,
    config: Nii2ZarrConfig | None = None,
    **kwargs: Annotated[dict, Parameter(show=False)],
) -> None:
    """
    Convert from NIfTI to Zarr.

    Parameters
    ----------
    inp
        Path to input file.
    out
        Path to output file. Default: "{base}.nii.zarr".

    """
    config = config or Nii2ZarrConfig()
    dataclasses.replace(config, **kwargs)

    # Default output name
    if not out:
        out = make_output_path(inp, "niftizarr")

    if inp.startswith("dandi://"):
        # Create an authentified fsspec store to pass to
        import linc_convert.utils.dandifs  # noqa: F401  (register filesystem)
        url = filesystem(inp).s3_url(inp)
        fs = filesystem(url)
        with open(fs.open(url)) as stream:
            return nii2zarr(stream, out, **dataclasses.asdict(config))

    return nii2zarr(inp, out, **dataclasses.asdict(config))
