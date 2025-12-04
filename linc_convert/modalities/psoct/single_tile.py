import dask.array as da


def process_complex3d(
    complex3d: da.Array, offset: float = 100, flip_phi: bool = False
    ) -> tuple[da.Array, da.Array, da.Array]:
    offset = da.deg2rad(offset)
    if complex3d.shape[0] % 4 != 0:
        raise ValueError("First dimension size must be multiple of 4.")
    raw_tile_width = complex3d.shape[0] // 4
    complex3d = complex3d.rechunk({0: raw_tile_width})
    comp = complex3d.reshape(
        (4, raw_tile_width, complex3d.shape[1], complex3d.shape[2]))
    j1r, j1i, j2r, j2i = comp[0], comp[1], comp[2], comp[3]

    j1 = j1r + 1j * j1i
    j2 = j2r + 1j * j2i
    mag1 = da.abs(j1)
    mag2 = da.abs(j2)

    dBI3D = da.flip(10 * da.log10(mag1 ** 2 + mag2 ** 2), axis=2)
    R3D = da.flip(
        da.arctan(mag1 / mag2) / da.pi * 180,
        axis=2
    )
    if flip_phi:
        phi = da.angle(j2) - da.angle(j1)
    else:
        phi = da.angle(j1) - da.angle(j2)
    phi += offset * 2
    # wrap into [-π, π]
    phi = da.where(phi > da.pi, phi - 2 * da.pi, phi)
    phi = da.where(phi < -da.pi, phi + 2 * da.pi, phi)
    O3D = da.flip(phi / 2, axis=2)
    O3D = da.rad2deg(O3D)
    return dBI3D, R3D, O3D
