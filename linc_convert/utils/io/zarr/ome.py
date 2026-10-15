"""OME-Zarr metadata version helpers (0.5 <-> 0.6).

OME-Zarr 0.6 (RFC-5) replaces multiscales ``axes`` by named
``coordinateSystems`` and makes dataset transformations explicit, typed
transformations with ``input`` (the dataset path) and ``output`` (the
"intrinsic" coordinate system). Dataset transformations are a single
``scale`` or a ``sequence`` of a scale followed by a translation.

`niizarr` writes 0.4/0.5 metadata; `ome_v05_to_v06` upgrades it.
`ome_v06_to_v05_view` gives a 0.5-shaped view for code (e.g. `niizarr`)
that only understands ``axes``.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

INTRINSIC = "intrinsic"


def _compose(
    scale: list[float], translation: Optional[list[float]],
    top_scale: Optional[list[float]], top_translation: Optional[list[float]],
) -> tuple[list[float], Optional[list[float]]]:
    """Fold a 0.5 multiscale-level transform into a dataset transform.

    x_phys = top_s * (s * x + t) + top_t
    """
    n = len(scale)
    ts = top_scale or [1.0] * n
    tt = top_translation or [0.0] * n
    t = translation or [0.0] * n
    new_scale = [float(a) * float(b) for a, b in zip(ts, scale)]
    new_trans = [float(a) * float(b) + float(c) for a, b, c in zip(ts, t, tt)]
    if not any(new_trans):
        return new_scale, None
    return new_scale, new_trans


def _split(transforms: list[dict]) -> tuple[Optional[list], Optional[list]]:
    scale = translation = None
    for t in transforms or []:
        if t.get("type") == "scale":
            scale = list(t["scale"])
        elif t.get("type") == "translation":
            translation = list(t["translation"])
        elif t.get("type") == "identity":
            pass
        else:
            raise ValueError(f"Unsupported 0.5 transformation: {t.get('type')}")
    return scale, translation


def ome_v05_to_v06(
    ome: dict[str, Any],
    *,
    method: Optional[dict[str, Any]] = None,
    downsampling_type: Optional[str] = None,
    cs_name: str = INTRINSIC,
) -> dict[str, Any]:
    """Convert the ``attributes.ome`` dict of an OME-Zarr 0.5 image to 0.6.

    Multiscale-level ``coordinateTransformations`` (scale/translation) are
    folded into each dataset's transformation, so that every dataset maps
    directly to the "intrinsic" (physical) coordinate system.
    """
    out: dict[str, Any] = {k: copy.deepcopy(v) for k, v in ome.items()
                           if k not in ("version", "multiscales")}
    out["version"] = "0.6"
    out["multiscales"] = []
    for ms in ome.get("multiscales", []):
        if "coordinateSystems" in ms:  # already 0.6
            out["multiscales"].append(copy.deepcopy(ms))
            continue
        top_s, top_t = _split(ms.get("coordinateTransformations", []))
        new: dict[str, Any] = {}
        if "name" in ms:
            new["name"] = ms["name"]
        new["coordinateSystems"] = [
            {"name": cs_name, "axes": copy.deepcopy(ms["axes"])}]
        datasets = []
        for ds in ms["datasets"]:
            s, t = _split(ds.get("coordinateTransformations", []))
            if s is None:
                s = [1.0] * len(ms["axes"])
            s, t = _compose(s, t, top_s, top_t)
            io = {"input": {"path": ds["path"]}, "output": {"name": cs_name}}
            if t is None:
                ct = {"type": "scale", "scale": s, **io}
            else:
                ct = {"type": "sequence", **io, "transformations": [
                    {"type": "scale", "scale": s},
                    {"type": "translation", "translation": t},
                ]}
            datasets.append({"path": ds["path"], "coordinateTransformations": [ct]})
        new["datasets"] = datasets
        if downsampling_type or ms.get("type"):
            new["type"] = downsampling_type or ms["type"]
        meta = method if method is not None else ms.get("metadata")
        if meta:
            new["metadata"] = copy.deepcopy(meta)
        out["multiscales"].append(new)
    return out


def ome_v06_to_v05_view(ome: dict[str, Any]) -> dict[str, Any]:
    """A 0.5-shaped copy of 0.6 metadata (``axes`` + flat scale/translation).

    Used to feed code that predates 0.6; not meant to be written to disk.
    """
    out = copy.deepcopy(ome)
    out["version"] = "0.5"
    views = []
    for ms in ome.get("multiscales", []):
        if "coordinateSystems" not in ms:
            views.append(copy.deepcopy(ms))
            continue
        cs = ms["coordinateSystems"][0]
        # multiscale-level 0.6 transforms map "intrinsic" to other systems
        # (e.g. nifti:qform); 0.5 has no equivalent, so they are dropped
        new = {k: copy.deepcopy(v) for k, v in ms.items()
               if k not in ("coordinateSystems", "datasets",
                            "coordinateTransformations")}
        new["axes"] = copy.deepcopy(cs["axes"])
        new["datasets"] = []
        for ds in ms["datasets"]:
            ct = ds["coordinateTransformations"][0]
            steps = ct["transformations"] if ct["type"] == "sequence" else [ct]
            flat = []
            for st in steps:
                flat.append({k: v for k, v in st.items()
                             if k not in ("input", "output")})
            new["datasets"].append({"path": ds["path"],
                                    "coordinateTransformations": flat})
        views.append(new)
    out["multiscales"] = views
    return out


def multiscales_v05_view(attrs: dict[str, Any]) -> Optional[list]:
    """Return 0.4/0.5-shaped ``multiscales`` from group attributes."""
    ome = attrs.get("ome", attrs)
    ms = ome.get("multiscales")
    if ms and any("coordinateSystems" in m for m in ms):
        return ome_v06_to_v05_view(ome)["multiscales"]
    return ms


def add_nifti_systems(ome: dict[str, Any], header: Any) -> dict[str, Any]:  # noqa: ANN401
    """Describe the NIfTI qform/sform spaces as OME-Zarr 0.6 coordinate systems.

    Adds "nifti:qform" / "nifti:sform" (when their code is set), each with an
    affine from the "intrinsic" system (scaled voxel grid, zarr axis order).
    Spatial output axes are the world x, y, z in the intrinsic axis order;
    other axes are passed through. Previous "nifti:*" entries are replaced.
    Same as niizarr.write_nifti_header in nifti-zarr-py#55.
    """
    ome = copy.deepcopy(ome)
    ms = ome["multiscales"][0]
    intrinsic = ms["coordinateSystems"][0]
    axes = [a["name"] for a in intrinsic["axes"]]
    level0 = ms["datasets"][0]["coordinateTransformations"][0]
    scale = level0.get("scale") or level0["transformations"][0]["scale"]
    world = {"x": 0, "y": 1, "z": 2}
    n = len(axes)

    systems = [cs for cs in ms["coordinateSystems"]
               if not cs["name"].startswith("nifti:")]
    transforms = [t for t in ms.get("coordinateTransformations", [])
                  if not t["output"]["name"].startswith("nifti:")]
    for kind, get in (("qform", header.get_qform), ("sform", header.get_sform)):
        affine, code = get(coded=True)
        if affine is None or not code:
            continue
        matrix = [[0.0] * (n + 1) for _ in range(n)]
        for r, a in enumerate(axes):
            if a not in world:
                matrix[r][r] = 1.0
                continue
            for c, b in enumerate(axes):
                if b in world:
                    matrix[r][c] = float(affine[world[a], world[b]]) / scale[c]
            matrix[r][n] = float(affine[world[a], 3])
        systems.append({**intrinsic, "name": f"nifti:{kind}"})
        transforms.append({"type": "affine", "affine": matrix,
                           "input": {"name": intrinsic["name"]},
                           "output": {"name": f"nifti:{kind}"}})
    ms["coordinateSystems"] = systems
    if transforms:
        ms["coordinateTransformations"] = transforms
    else:
        ms.pop("coordinateTransformations", None)
    return ome

