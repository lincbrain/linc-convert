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

These are PVCAM PRD files. The header starts with a 48-byte PRD header
(format version, bit depth, frame count, sensor region), and a 24-byte
record of per-frame metadata (frame number, readout time, exposure time,
begin/end-of-frame times) sits `header_bytes - 4096` bytes before each
frame. The remaining non-pixel bytes (camera metadata blocks, fill) are not
interpreted, but `PrdCapture` keeps them so the files can be rebuilt.
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
import struct
from concurrent.futures import Future, ThreadPoolExecutor
from os import PathLike
from typing import List, Optional

import numpy as np

PRD_FRAME_FIELDS = (
    "frame_number", "readout_time", "exposure_time", "bof_time", "eof_time",
    "roi_count",
)


def parse_prd_header(header: bytes) -> dict | None:
    """Decode the PVCAM PRD file header (None if the signature is absent)."""
    if header[:4] != b"PRD\0":
        return None
    (version, bit_depth, frame_count, s1, s2, sbin, p1, p2, pbin,
     metadata_size, exposure_resolution) = struct.unpack_from("<HHI6HII", header, 4)
    return dict(
        version=version, bit_depth=bit_depth, frame_count=frame_count,
        region=dict(s1=s1, s2=s2, sbin=sbin, p1=p1, p2=p2, pbin=pbin),
        metadata_size=metadata_size, exposure_resolution=exposure_resolution,
    )


class PrdCapture:
    """Keep what is needed to rebuild `.prd` files byte for byte.

    `PrdSetInterpreter.read_frames` feeds it every byte of each file, in
    order, during a sequential read: the non-pixel bytes are kept and, with
    `checksum`, each file is hashed (SHA-256) on its own thread. SHA-256 is
    sequential within a file (~1.3-1.8 GB/s per core), so up to
    `max_queued_bytes` may wait, letting consecutive files hash in parallel.
    """

    def __init__(
        self, n_files: int, checksum: bool = True, max_queued_bytes: int = 4 << 30
    ) -> None:
        self.nonpixel: list[list[bytes]] = [[] for _ in range(n_files)]
        self.sha256: list[Optional[str]] = [None] * n_files
        self._hash = [hashlib.sha256() for _ in range(n_files)] if checksum else None
        self._lane: dict[int, ThreadPoolExecutor] = {}
        self._queued: list[tuple[Future, int]] = []
        self._queued_bytes = 0
        self._max_queued_bytes = max_queued_bytes

    def feed(self, f: int, data: bytes, pixels: bool) -> None:
        """Append the next bytes of file `f`."""
        if not pixels:
            self.nonpixel[f].append(data)
        if self._hash is None:
            return
        lane = self._lane.setdefault(f, ThreadPoolExecutor(1))
        self._queued.append((lane.submit(self._hash[f].update, data), len(data)))
        self._queued_bytes += len(data)
        while self._queued_bytes > self._max_queued_bytes:
            fut, size = self._queued.pop(0)
            fut.result()
            self._queued_bytes -= size

    def close_file(self, f: int) -> None:
        """All bytes of file `f` have been fed."""
        if self._hash is None:
            return
        self._lane.pop(f).shutdown()
        self.sha256[f] = self._hash[f].hexdigest()

    def finish(self) -> None:
        """Wait for pending hashes."""
        for f in list(self._lane):
            self.close_file(f)
        for fut, _ in self._queued:
            fut.result()
        self._queued.clear()
        self._queued_bytes = 0


class PrdSetInterpreter:
    """Read a directory of `.prd` files as one (frames, height, width) volume."""

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
        self.capture: Optional[PrdCapture] = None
        self._next_frame = 0  # sequential reads feed `capture`
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
        capture = self.capture if start == self._next_frame else None
        first = int(np.searchsorted(self.file_offsets, start, side="right")) - 1
        i = 0
        for f in range(first, len(self.prd_files)):
            if i >= n:
                break
            local = start + i - int(self.file_offsets[f])
            count = min(self.frames_per_file[f] - local, n - i)
            with open(self.prd_files[f], "rb", buffering=0) as fh:
                if capture and local == 0:
                    capture.feed(f, fh.read(self.header_bytes), pixels=False)
                for k in range(count):
                    fh.seek(self.header_bytes + (local + k) * self.stride_bytes)
                    view = memoryview(flat[i + k]).cast("B")
                    if fh.readinto(view) != self.bytes_per_frame:
                        raise IOError(f"Short read in {self.prd_files[f]}")
                    if capture:
                        capture.feed(f, bytes(view), pixels=True)
                        last = local + k == self.frames_per_file[f] - 1
                        # gap after the frame, or the trailing bytes of the file
                        capture.feed(f, fh.read() if last else fh.read(self.gap_bytes),
                                     pixels=False)
                        if last:
                            capture.close_file(f)
            i += count
        self._next_frame = stop
        return out

    def frame_metadata(self, capture: PrdCapture) -> dict[str, np.ndarray]:
        """Per-frame PRD records (`PRD_FRAME_FIELDS`) from captured bytes."""
        off = self.header_bytes - 4096  # record sits this far before a frame
        if off < 24 or self.gap_bytes < off:
            return {}  # not the PVCAM layout
        rows = []
        for f, parts in enumerate(capture.nonpixel):
            # parts: header, then one gap (or the trailing bytes) per frame
            before = [parts[0]] + parts[1:self.frames_per_file[f]]
            rows += [struct.unpack_from("<6I", p, len(p) - off) for p in before]
        table = np.asarray(rows, dtype="<u4").reshape(-1, len(PRD_FRAME_FIELDS))
        return {name: table[:, i] for i, name in enumerate(PRD_FRAME_FIELDS)}

