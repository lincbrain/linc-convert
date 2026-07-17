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

import dask.array as da
import numpy as np
from dask import delayed


class PrdSetInterpreter:
    """Assemble prd files into a single 3D volume with a Dask array."""

    def __init__(
        self, prd_set_path: str | PathLike[str]
    ) -> None:
        self.prd_files = self.get_prd_files(prd_set_path)

        metadata = self.read_kinetix_metadata(prd_set_path)
        self.data_type = metadata["dataType"]
        if self.data_type == 16:
            self.data_type = "uint16"
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

        n_frames = (usable + self.gap_bytes) // self.stride_bytes

        if n_frames > self.expectedFramesPerFullFile:
            raise ValueError(
                f"Found {n_frames} frames, more than the expected "
                f"{self.expectedFramesPerFullFile} per full file: {path}"
            )

        return n_frames

    def _load_prd_file(self, path: str) -> np.ndarray:
        """Load one .prd file into a (n_frames, height, width) array.

        Seek to each frame's offset and stack them, skipping the inter-frame
        gaps and any trailing padding.
        """
        n_frames = self._frames_in_file(path)

        frames = np.empty((n_frames, self.pixels_per_frame), dtype="<u2")
        
        with open(path, "rb") as fh:
            for i in range(n_frames):
                fh.seek(self.header_bytes + i * self.stride_bytes)
                frames[i] = np.fromfile(fh,
                                        dtype="<u2",
                                        count=self.pixels_per_frame)

        return frames.reshape(n_frames, self.height, self.width)

    def assemble(self) -> da.Array:
        """
        Assemble all prd files into a single 3D volume with a lazy Dask array.

        Returns
        -------
        A Dask array of shape (frames_total, height, width), where
        frames_total is the sum of the frame counts across all `.prd` files.
        """
        if not self.prd_files:
            empty = np.zeros((0, self.height, self.width), dtype=self.dtype)
            return da.from_array(empty)

        delayed_chunks = [
            da.from_delayed(
                delayed(self._load_prd_file)(path),
                shape=(self._frames_in_file(path), self.height, self.width),
                dtype=self.dtype,
            )
            for path in self.prd_files
        ]

        return da.concatenate(delayed_chunks, axis=0)

