from _warnings import warn
from typing import Mapping, Optional

import h5py
import numpy as np
from scipy.io import loadmat


class _ArrayWrapper:
    def _get_key(self, f: Mapping) -> str:
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


class _H5ArrayWrapper(_ArrayWrapper):
    def __init__(self, file: h5py.File, key: Optional[str]) -> None:
        self.file = file
        self.key = key
        self.array = file.get(self._get_key(self.file))

    def __del__(self) -> None:
        if hasattr(self.file, "close"):
            self.file.close()

    def load(self) -> np.ndarray:
        self.array = self.array[...]
        if hasattr(self.file, "close"):
            self.file.close()
        self.file = None
        return self.array

    @property
    def shape(self) -> list[int]:
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, index: object) -> np.ndarray:
        return self.array[index]


class _MatArrayWrapper(_ArrayWrapper):
    def __init__(self, file: str, key: Optional[str]) -> None:
        self.file = file
        self.key = key
        self.array = None

    def __del__(self) -> None:
        if hasattr(self.file, "close"):
            self.file.close()

    def load(self) -> np.ndarray:
        data = loadmat(self.file)
        self.array = data.get(self._get_key(data))
        self.file = None
        return self.array

    @property
    def shape(self) -> tuple[int]:
        if self.array is None:
            self.load()
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        if self.array is None:
            self.load()
        return self.array.dtype

    def __len__(self) -> int:
        if self.array is None:
            self.load()
        return len(self.array)

    def __getitem__(self, index: object) -> np.ndarray:
        if self.array is None:
            self.load()
        return self.array[index]
