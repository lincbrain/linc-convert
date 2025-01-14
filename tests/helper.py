import numpy as np
import zarr


def _cmp_zarr_archives(path1: str, path2: str) -> bool:
    """
    Compare two Zarr archives to check if they contain the same data.

    Parameters
    ----------
    - path1 (str): Path to the first Zarr archive.
    - path2 (str): Path to the second Zarr archive.

    Returns
    -------
    - bool: True if both archives contain the same data, False otherwise.
    """
    # Open both Zarr groups
    zarr1 = zarr.open(path1, mode="r")
    zarr2 = zarr.open(path2, mode="r")

    # Compare keys (dataset structure)
    if zarr1.keys() != zarr2.keys():
        print("keys mismatch")
        return False
    if zarr1.attrs != zarr2.attrs:
        print("attrs mismatch")
        return False

    # Compare each array in both archives
    for key in zarr1.keys():
        array1 = zarr1[key][:]
        array2 = zarr2[key][:]

        np.testing.assert_allclose(array1, array2)
        if zarr1[key].attrs != zarr2[key].attrs:
            print("attrs mismatch")
            return False

    # If all checks pass
    print("The Zarr archives are identical.")
    return True
