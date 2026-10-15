"""Load Kinetix `.prd` image stacks from local disk.

A `.prd` file is:
    [ header ][ frame 0 ][ gap ][ frame 1 ][ gap ] ... [ frame N-1 ] [ padding ]

Each frame is `width * height` little-endian uint16 pixels stored row-major.
There is a `gap` of padding bytes between frames but not after the last one.

The geometry parameters are read from a `kinetixMetadata.txt` file sitting
next to the `.prd` file(s):

    16      # data type, uint16
    1000    # width
    400     # height
    8272    # header bytes
    2661    # expected frames per full file
    6912    # gap bytes (between frames, not after last frame)
"""

from __future__ import annotations

import glob
import os
import re
from os import PathLike
from typing import List

import numpy as np


class PrdSetInterpreter:
    """Assemble prd files into a single 3D volume with a Dask array."""

    def __init__(
        self, prd_set_path: str | PathLike[str]
    ) -> None:
        self.prd_files = self.get_prd_files(prd_set_path)

        metadata = self.read_kinetix_metadata(prd_set_path)
        self.data_type = metadata["dataType"]
        if self.data_type == 16:
            self.data_type = "<u2"
        self.width = metadata["width"]
        self.height = metadata["height"]
        self.header_bytes = metadata["headerBytes"]
        self.expectedFramesPerFullFile = metadata["expectedFramesPerFullFile"]
        self.gap_bytes = metadata["gapBytes"]
        
        self.dtype = np.dtype(self.data_type)
        self.bytes_per_pixel = self.dtype.itemsize
        self.pixels_per_frame = self.width * self.height
        self.bytes_per_frame = self.pixels_per_frame * self.bytes_per_pixel
        self.stride_bytes = self.bytes_per_frame + self.gap_bytes

        self.frames_per_file = [self._frames_in_file(f) for f in self.prd_files]
        # first global frame index of each file (plus total at the end)
        self.file_offsets = np.concatenate(
            [[0], np.cumsum(self.frames_per_file)]
        ).astype(int)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the assembled volume: (frames_total, height, width)."""
        return (int(self.file_offsets[-1]), self.height, self.width)

    @staticmethod
    def read_kinetix_metadata(path: str) -> dict:
        """Parse a `kinetixMetadata.txt` file into its geometry parameters.

        `path` may be the metadata file itself or a directory containing it.
        """
        if os.path.isdir(path):
            path = os.path.join(path, "kinetixMetadata.txt")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"No kinetixMetadata.txt found for {path}."
            )

        metadata_keys = {
            'dataType': int,
            'width': int,
            'height': int,
            'headerBytes': int,
            'expectedFramesPerFullFile': int,
            'gapBytes': int,
        }

        meta: dict = {}
        # Assumes the kinetixMetadata.txt file lists one value per line, 
        # in the same order as `metadata_keys`. 
        with open(path) as fh:
            for key, line in zip(metadata_keys, fh):
                meta[key] = metadata_keys[key](line.strip())
        return meta

    @staticmethod
    def get_prd_files(path: str) -> List[str]:
        """Return the `.prd` files for `prd_set_path`, in acquisition order."""
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Not a directory: {path}")
    
        def acquisition_key(p: str) -> int:
            # Sort by the trailing integer so ss_stack_9 precedes ss_stack_10.
            m = re.search(r"ss_stack_(\d+)\.prd$", p)
            return int(m.group(1)) if m else float("inf")

        file_list = sorted(glob.glob(os.path.join(path, "ss_stack_*.prd")), 
            key=acquisition_key)

        if not file_list:
            raise FileNotFoundError(f"No ss_stack_*.prd files found in {path}")

        return file_list

    def _frames_in_file(self, path: str) -> int:
        """Return the number of frames in a single `.prd` file."""
        usable = os.path.getsize(path) - self.header_bytes
        if usable < self.bytes_per_frame:
            raise ValueError(f"File too small to contain one frame: {path}")

        # Floor division discards the padding at the end of the file, which assumes 
        # the padding is smaller than one stride (frame + gap).
        n_frames = (usable + self.gap_bytes) // self.stride_bytes

        if n_frames > self.expectedFramesPerFullFile:
            raise ValueError(
                f"Found {n_frames} frames, more than the expected "
                f"{self.expectedFramesPerFullFile} per full file: {path}"
            )

        return n_frames

    def read_frames(
        self, start: int, stop: int, out: np.ndarray | None = None
    ) -> np.ndarray:
        """Read global frames `[start, stop)` into a (n, height, width) array.

        Frames may span several `.prd` files. Only the requested frames are
        read from disk, so memory use is bounded by the size of `out`.
        """
        stop = min(stop, self.shape[0])
        n = stop - start
        if out is None:
            out = np.empty((n, self.height, self.width), dtype=self.dtype)
        else:
            out = out[:n]
        flat = out.reshape(n, -1)
        first = int(np.searchsorted(self.file_offsets, start, side="right")) - 1
        i = 0
        for f in range(first, len(self.prd_files)):
            if i >= n:
                break
            local = start + i - int(self.file_offsets[f])
            count = min(self.frames_per_file[f] - local, n - i)
            with open(self.prd_files[f], "rb", buffering=0) as fh:
                for k in range(count):
                    fh.seek(self.header_bytes + (local + k) * self.stride_bytes)
                    view = memoryview(flat[i + k]).cast("B")
                    if fh.readinto(view) != self.bytes_per_frame:
                        raise IOError(f"Short read in {self.prd_files[f]}")
            i += count
        return out
