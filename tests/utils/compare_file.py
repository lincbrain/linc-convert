import argparse
import os
import sys
from typing import Union

import numpy as np
import zarr


def load_file(path):
    """
    Load data from a file based on its extension.
    Supports .nii, .nii.gz (via nibabel), .mat (via scipy.io), and .zarr (via zarr-python).
    """
    ext = os.path.splitext(path)[1].lower()
    # Handle .nii and .nii.gz
    if ext in ['.nii', '.gz'] and (path.endswith('.nii') or path.endswith('.nii.gz')):
        try:
            import nibabel as nib
        except ImportError:
            print("Error: nibabel is not installed.")
            sys.exit(1)
        img = nib.load(path)
        data = img.get_fdata()
        return data

    # Handle .mat
    if ext == '.mat':
        try:
            from scipy.io import loadmat
        except ImportError:
            print("Error: scipy is not installed.")
            sys.exit(1)
        mat = loadmat(path)
        # Extract variable array from mat dict (ignoring metadata keys)
        keys = [k for k in mat.keys() if not k.startswith('__')]
        if len(keys) != 1:
            print(
                f"Error: Expected one variable in the MAT file, found {len(keys)}: {keys}")
            sys.exit(1)
        return mat[keys[0]]

    # Handle .zarr
    if ext == '.zarr' or '.zarr' in path:
        try:
            import zarr
        except ImportError:
            print("Error: zarr is not installed.")
            sys.exit(1)
        arr = zarr.open(path, mode='r')
        return arr[:]  # load full array into memory

    print(f"Unsupported file extension: {ext}")
    sys.exit(1)


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
    zarr1 = zarr.open(path1, mode="r")
    zarr2 = zarr.open(path2, mode="r")

    # Compare keys (dataset structure)
    if set(zarr1.keys()) != set(zarr2.keys()):
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


def assert_zarr_equal(
        store1: Union[str, zarr.storage.StoreLike],
        store2: Union[str, zarr.storage.StoreLike]) -> None:
    """
    Assert that two Zarr groups—opened from either a path or a store—have identical contents.

    Parameters
    ----------
    store1 : StoreLike
        A filesystem path (str) or a Zarr store/mapping pointing to the first group.
    store2 : StoreLike
        A filesystem path (str) or a Zarr store/mapping pointing to the second group.

    Returns
    -------
    None
    """
    zarr1 = zarr.open(store1, mode="r")
    zarr2 = zarr.open(store2, mode="r")
    diffs = []

    if dict(zarr1.attrs) != dict(zarr2.attrs):
        diffs.append(
            f"Group attrs differ:\n ‣ {dict(zarr1.attrs)}\n ‣ {dict(zarr2.attrs)}")
    if set(zarr1.keys()) != set(zarr2.keys()):
        diffs.append(
            f"Group keys differ:\n ‣ {set(zarr1.keys())}\n ‣ {set(zarr2.keys())}")

    keys = set(zarr1.keys()).intersection(set(zarr2.keys()))
    for key in keys:
        obj1 = zarr1[key]
        obj2 = zarr2[key]
        if dict(obj1.attrs) != dict(obj2.attrs):
            diffs.append(
                f"Attributes for key '{key}' differ:\n ‣ {dict(obj1.attrs)}\n ‣ {dict(obj2.attrs)}")
        if isinstance(obj1, zarr.Array) ^ isinstance(obj2, zarr.Array):
            diffs.append(f"Key '{key}' is an array in one group but not the other.")
            continue
        elif isinstance(obj1, zarr.Array):
            try:
                np.testing.assert_array_equal(obj1[:], obj2[:])
            except AssertionError as e:
                diffs.append(f"Array '{key}' differs:\n ‣ {e}")
        elif isinstance(obj1, zarr.Group):
            try:
                np.testing.assert_allclose(obj1, obj2)
            except AssertionError as e:
                diffs.append(f"Group '{key}' differs:\n ‣ {e}")
        else:
            diffs.append(f"Key '{key}' is neither an array nor a group in both stores.")

    if diffs:
        raise AssertionError("\n".join(diffs))


def main():
    parser = argparse.ArgumentParser(description='Compare arrays in two files.')
    parser.add_argument('file1', help='First file (.nii, .nii.gz, .mat, .zarr)')
    parser.add_argument('file2', help='Second file (.nii, .nii.gz, .mat, .zarr)')
    args = parser.parse_args()

    a = load_file(args.file1)
    b = load_file(args.file2)
    try:
        np.testing.assert_array_almost_equal(a, b, decimal=4)
        print("Arrays are almost equal up to 6 decimal places.")
    except AssertionError as e:
        print("Arrays differ:")
        print(e)


if __name__ == '__main__':
    main()
