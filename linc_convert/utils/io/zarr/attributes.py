"""
Attributes Handling for Zarr.

This file contains code from the Zarr project
https://github.com/zarr-developers/zarr-python
"""

import json
import os
import tempfile
import threading
from collections.abc import Iterator, MutableMapping
from typing import Any

from upath import UPath


class Attributes(MutableMapping[str, Any]):
    """
    In-memory, write-through (configurable) attributes mapping for Zarr v2/v3.

    Reads are served from an in-memory cache. On mutation, the cache is updated
    and, if write_through is True, the file is flushed atomically. Otherwise,
    call .flush() explicitly to persist.

    Works for both arrays and groups, as long as the parent exposes:
      - .store_path (path-like, directory)
      - .zarr_version (2 or 3)
    """

    def __init__(self, obj: Any, *, write_through: bool = True) -> None:  # noqa: ANN401
        self._obj = obj
        self._write_through = write_through
        self._lock = threading.RLock()
        self._loaded = False
        self._attrs: dict[str, Any] = {}
        # Only for v3 we need to preserve other top-level keys in zarr.json
        self._v3_other_keys: dict[str, Any] = {}
        # cache the path computation
        store = UPath(self._obj.store_path)
        if self._obj.zarr_version == 2:
            self._path = store / ".zattrs"
            self._is_v3 = False
        elif self._obj.zarr_version == 3:
            self._path = store / "zarr.json"
            self._is_v3 = True
        else:
            raise ValueError(f"Unsupported zarr_version: {self._obj.zarr_version}")

    # ---------- public helpers ----------
    def asdict(self) -> dict[str, Any]:
        """Return a snapshot of attributes as a dict."""
        with self._lock:
            self._ensure_loaded()
            return dict(self._attrs)

    def put(self, d: dict[str, Any]) -> None:
        """Overwrite all attributes with d (in-memory), then flush."""
        with self._lock:
            self._ensure_loaded()
            self._attrs = dict(d)
            if self._write_through:
                self._flush_locked()

    def flush(self) -> None:
        """Persist current in-memory attributes to disk atomically."""
        with self._lock:
            self._ensure_loaded()
            self._flush_locked()

    def refresh(self) -> None:
        """Discard in-memory cache and re-read from disk."""
        with self._lock:
            self._loaded = False
            self._attrs.clear()
            self._v3_other_keys.clear()
            self._ensure_loaded()

    # ---------- MutableMapping interface ----------
    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Get an attribute by key."""
        with self._lock:
            self._ensure_loaded()
            return self._attrs[key]

    def __setitem__(self, key: str, value: Any) -> None:  # noqa: ANN401
        """Set or update an attribute."""
        with self._lock:
            self._ensure_loaded()
            self._attrs[key] = value
            if self._write_through:
                self._flush_locked()

    def __delitem__(self, key: str) -> None:
        """Delete an attribute."""
        with self._lock:
            self._ensure_loaded()
            del self._attrs[key]
            if self._write_through:
                self._flush_locked()

    def __iter__(self) -> Iterator[str]:
        """Iterate over a snapshot of keys."""
        with self._lock:
            self._ensure_loaded()
            return iter(dict(self._attrs))  # iterate over a snapshot

    def __len__(self) -> int:
        """Return number of attributes."""
        with self._lock:
            self._ensure_loaded()
            return len(self._attrs)

    # ---------- internals ----------
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self._path.exists():
            # not present on disk yet
            self._attrs = {}
            self._v3_other_keys = {}
            self._loaded = True
            return

        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if self._is_v3:
            self._attrs = dict(data.get("attributes", {}))
            # keep all other top-level keys to preserve on write
            self._v3_other_keys = {k: v for k, v in data.items() if k != "attributes"}
        else:
            self._attrs = dict(data)
            self._v3_other_keys = {}

        self._loaded = True

    def _flush_locked(self) -> None:
        if self._is_v3:
            data = dict(self._v3_other_keys)
            data["attributes"] = self._attrs
        else:
            data = self._attrs

        _atomic_json_write(self._path, data)


def _atomic_json_write(path: UPath, data: dict[str, Any]) -> None:
    """
    Atomically write JSON to 'path' via a temp file + rename.

    Works with local and fsspec-backed paths that expose .open / .parent.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    # Use a temp file in the same directory to keep rename atomic on POSIX FS.
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=".attrs_tmp_", dir=str(parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        # UPath may wrap non-local FS; try best-effort replace
        # pathlib.Path has replace(); UPath typically forwards.
        UPath(tmp_name).replace(path)
    finally:
        # If replace failed, clean up temp file best-effort
        try:
            if UPath(tmp_name).exists():
                UPath(tmp_name).unlink()
        except Exception:
            pass
