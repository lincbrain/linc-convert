"""Utilities for reading HDF5 and MATLAB .mat files, and wrapping arrays."""

from _warnings import warn
from typing import Mapping, Optional

import h5py
import numpy as np
from scipy.io import loadmat


class ArrayWrapper:
    """Abstract base class for array wrappers."""

    def _get_key(self, f: Mapping) -> str:
        """Get the key to use to access the array in the file/dictionary."""
        key = self.key
        if key is None:
            if not len(f.keys()):
                raise Exception(f"{self.file} is empty")
            # Select the first non-hidden key
            for key in f.keys():
                if key[0] != "_":
                    break
            if len(f.keys()) > 1:
                warn(
                    f"More than one key in .mat file {self.file}, "
                    f'arbitrarily loading "{key}"'
                )

        if key not in f.keys():
            raise Exception(f"Key {key} not found in file {self.file}")
        return key


class H5ArrayWrapper(ArrayWrapper):
    """Wrapper for arrays stored in HDF5 files."""

    def __init__(self, file: h5py.File, key: Optional[str]) -> None:
        self.file = file
        self.key = key
        self.array = file.get(self._get_key(self.file))

    def __del__(self) -> None:
        """Close the file if it is still open."""
        if hasattr(self.file, "close"):
            self.file.close()

    def load(self) -> np.ndarray:
        """Load the array from the HDF5 file."""
        self.array = self.array[...]
        if hasattr(self.file, "close"):
            self.file.close()
        self.file = None
        return self.array

    @property
    def shape(self) -> list[int]:
        """Get shape of the array."""
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        """Get data type of the array."""
        return self.array.dtype

    def __len__(self) -> int:
        """Get length of the first dimension."""
        return len(self.array)

    def __getitem__(self, index: object) -> np.ndarray:
        """Get item by index."""
        return self.array[index]

    @property
    def ndim(self) -> int:
        """Get dimension of the array."""
        return self.array.ndim

class MatArraywrapper(ArrayWrapper):
    """Wrapper for arrays stored in old-style MATLAB .mat files."""

    def __init__(self, file: str, key: Optional[str]) -> None:
        self.file = file
        self.key = key
        self.array = None

    def __del__(self) -> None:
        """Close the file if it is still open."""
        if hasattr(self.file, "close"):
            self.file.close()

    def load(self) -> np.ndarray:
        """Load the array from the .mat file."""
        data = loadmat(self.file)
        self.array = data.get(self._get_key(data))
        self.file = None
        return self.array

    @property
    def shape(self) -> tuple[int]:
        """Get shape of the array, loading the array if necessary."""
        if self.array is None:
            self.load()
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        """Get data type of the array, loading the array if necessary."""
        if self.array is None:
            self.load()
        return self.array.dtype

    def __len__(self) -> int:
        """Get length of the first dimension, loading the array if necessary."""
        if self.array is None:
            self.load()
        return len(self.array)

    def __getitem__(self, index: object) -> np.ndarray:
        """Get item by index, loading the array if necessary."""
        if self.array is None:
            self.load()
        return self.array[index]

    @property
    def ndim(self) -> int:
        """Get dimension of the array."""
        return self.array.ndim