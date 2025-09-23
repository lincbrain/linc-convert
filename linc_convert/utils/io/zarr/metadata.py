"""
Metadata handling for Zarr.

This file contains code from the Zarr project
https://github.com/zarr-developers/zarr-python
"""

import json
import os
import tempfile
from dataclasses import dataclass, field, fields, replace
from typing import Any, Literal, Self, Sequence

from upath import UPath

JSON = Any


@dataclass(frozen=True)
class Metadata:
    """Frozen, recursive, JSON-serializable metadata class."""

    def to_dict(self) -> dict[str, JSON]:
        """Convert this metadata to a JSON-serializable dict."""
        out: dict[str, JSON] = {}
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if isinstance(v, Metadata):
                out[k] = v.to_dict()
            elif isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
                out[k] = tuple(x.to_dict() if isinstance(x, Metadata) else x for x in v)
            else:
                out[k] = v
        return out

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        """Create an instance from a JSON-serializable dict."""
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class GroupMetadata(Metadata):
    """Metadata for a Zarr group, including attributes and format version."""

    attributes: dict[str, Any] = field(default_factory=dict)
    zarr_format: Literal[2, 3] = 3
    node_type: Literal["group"] = field(default="group", init=False)

    # Convenience updaters (immutably return new metadata)
    def update_attributes(self, attributes: dict[str, JSON]) -> Self:
        """Return a new GroupMetadata with updated attributes."""
        return replace(self, attributes=dict(attributes))

    # ---- I/O helpers for disk persistence ----
    @staticmethod
    def _atomic_write(path: "UPath", data: dict[str, Any]) -> None:
        """Write data to path atomically."""
        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".meta_tmp_", dir=str(parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            path.__class__(tmp).replace(path)
        finally:
            try:
                if path.__class__(tmp).exists():
                    path.__class__(tmp).unlink()
            except Exception:
                pass

    @classmethod
    def from_files(cls, root: "UPath") -> "GroupMetadata":
        """Load metadata from the specified root directory."""
        # Prefer zarr.json if present; otherwise v2 split files
        zarr_json = root / "zarr.json"
        if zarr_json.exists():
            with open(zarr_json, "r", encoding="utf-8") as f:
                d = json.load(f)
            attrs = d.get("attributes", {}) or {}
            return cls(attributes=attrs, zarr_format=3)
        # v2: .zgroup + .zattrs (attributes may be missing)
        zgroup = root / ".zgroup"
        zattrs = root / ".zattrs"
        zf = 2
        if zgroup.exists():
            with open(zgroup, "r", encoding="utf-8") as f:
                g = json.load(f)
                zf = g.get("zarr_format", 2)
        attrs = {}
        if zattrs.exists():
            with open(zattrs, "r", encoding="utf-8") as f:
                attrs = json.load(f)
        return cls(attributes=attrs, zarr_format=zf)

    def to_files(self, root: "UPath") -> None:
        """Write this metadata to the specified root directory."""
        if self.zarr_format == 3:
            path = root / "zarr.json"
            data = {}
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            data["zarr_format"] = 3
            data["node_type"] = "group"
            data["attributes"] = self.attributes
            self._atomic_write(path, data)
        else:
            # v2 writes two files
            gpath, apath = root / ".zgroup", root / ".zattrs"
            self._atomic_write(gpath, {"zarr_format": 2})
            self._atomic_write(apath, dict(self.attributes))
