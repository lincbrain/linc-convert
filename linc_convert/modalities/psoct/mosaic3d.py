# given a file with tileinfo, create a mosaic3d, the input needs to be either complex
# data or volume itself, support focus offset


@autoconfig
@mosaic3d.default
def mosaic3d(
    tile_info_file: str,
    *,
    dbi_output: Annotated[str, Parameter(name=["--dBI", "-d"])],
    o3d_output: Annotated[str, Parameter(name=["--O3D", "-o"])],
    r3d_output: Annotated[str, Parameter(name=["--R3D", "-r"])],
    zarr_config: ZarrConfig = None,
    general_config: GeneralConfig = None,
    nifti_config: NiftiConfig = None,
) -> None:
    # load tile info
    # load 
    pass


def mosaic3d_vol(
    tile_info_file: str,
    *,
    tile_overlap: float | Literal["auto"] = "auto",
    circular_mean: bool = False,
    general_config: GeneralConfig = None,
    nifti_config: NiftiConfig = None,
    zarr_config: ZarrConfig = None,
) -> None:
    # load tile info
    # normalize tile_overlap, if auto, find the max overlap of all tiles, if float number, use as percentile of the tile size

    # load volume
    # mosaic volume
    # save mosaic volume
    pass

