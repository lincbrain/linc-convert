import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
import zarr
from utils.compare_file import assert_zarr_equal

from linc_convert.modalities.lsm import prd
from linc_convert.utils.io.prd import PrdSetInterpreter

# Directory holding the .prd input files that the conversion reads from.
PRD_INPUT_DIR = "data/lsm_prd"


def build_prd_subset(dest: Path) -> None:
    """Fetch a small subset of a Kinetix .prd assets to build the golden test input.

    Data: https://dandiarchive.org/dandiset/001769/draft/files?location=rawdata/HG9_Z1_Y49
    """
    import requests
    from dandi.dandiapi import DandiAPIClient

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    PRD_DANDISET = "001769"
    PRD_FILE_PATH = "rawdata/HG9_Z1_Y49"
    PRD_N_FRAMES = 10  # frames pulled from the front of each file
    PRD_N_FILES = 2  # number of files to subset

    with DandiAPIClient.for_dandi_instance("dandi") as client:
        dandiset = client.get_dandiset(PRD_DANDISET, "draft")

        def content_url(name: str) -> str:
            asset = dandiset.get_asset_by_path(f"{PRD_FILE_PATH}/{name}")
            return asset.get_content_url(follow_redirects=1, strip_query=True)

        # Download the metadata file
        (dest / "kinetixMetadata.txt").write_bytes(
            requests.get(content_url("kinetixMetadata.txt")).content
        )
        meta = PrdSetInterpreter.read_kinetix_metadata(str(dest))
        if meta["dataType"] != 16:
            raise ValueError(f"Unsupported dataType: {meta['dataType']}")
        bytes_per_pixel = np.dtype("uint16").itemsize
        stride = meta["width"] * meta["height"] * bytes_per_pixel + meta["gapBytes"]
        prefix_bytes = meta["headerBytes"] + PRD_N_FRAMES * stride

        # Download the first PRD_N_FILES files and only the first PRD_N_FRAMES frames of each
        for i in range(0, PRD_N_FILES):
            resp = requests.get(
                content_url(f"ss_stack_{i}.prd"),
                headers={"Range": f"bytes=0-{prefix_bytes - 1}"},
            )
            resp.raise_for_status()
            if resp.status_code != 206:
                raise RuntimeError(
                    f"Server ignored the Range request (status {resp.status_code}); "
                    "refusing to download the file."
                )
            (dest / f"ss_stack_{i}.prd").write_bytes(resp.content)


def test_lsm_prd(tmp_path, zarr_version):
    expected_zarr = f"data/lsm_prd_zarr{zarr_version}.nii.zarr.zip"
    output = tmp_path / "prd.nii.zarr"
    prd.convert(
        PRD_INPUT_DIR,
        out=str(output),
        zarr_version=zarr_version,
    )
    assert_zarr_equal(str(output), zarr.storage.ZipStore(expected_zarr, mode="r"))


@pytest.mark.golden
def test_lsm_prd_regen_golden(tmp_path, zarr_version):
    build_prd_subset(PRD_INPUT_DIR)
    base = f"data/lsm_prd_zarr{zarr_version}.nii.zarr"
    output = tmp_path / "prd.nii.zarr"
    prd.convert(
        PRD_INPUT_DIR,
        out=str(output),
        zarr_version=zarr_version,
    )
    shutil.make_archive(base, "zip", str(output))


# --------------------------------------------------------------------------
# synthetic inputs and independent oracles
# --------------------------------------------------------------------------
HEADER, GAP, PAD = 64, 18, 10


def _write_prd_set(path: Path, frames_per_file, height, width, seed=0,
                   pattern="random"):
    """Write a synthetic Kinetix .prd set; return the full (z, y, x) volume.

    pattern="edge" mixes full-range noise with all-zero, all-max and
    constant regions (exercises empty-chunk skipping and extreme values).
    """
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True, exist_ok=True)
    (path / "kinetixMetadata.txt").write_text(
        f"16\n{width}\n{height}\n{HEADER}\n{max(frames_per_file)}\n{GAP}\n"
    )
    vols = []
    for i, n in enumerate(frames_per_file):
        vol = rng.integers(0, 2**16, (n, height, width), dtype="<u2")
        if pattern == "edge":
            vol[:, : height // 2, : width // 2] = 0
            vol[:, height // 2:, : width // 3] = 65535
            vol[: n // 2, :, width // 2:] = 1234
        vols.append(vol)
        with open(path / f"ss_stack_{i}.prd", "wb") as fh:
            fh.write(b"\xab" * HEADER)
            for k in range(n):
                fh.write(vol[k].tobytes())
                if k < n - 1:
                    fh.write(b"\xcd" * GAP)
            fh.write(b"\xef" * PAD)
    return np.concatenate(vols)


def _raw_frames(path: Path) -> np.ndarray:
    """Parse a .prd set with numpy only (independent of PrdSetInterpreter)."""
    meta = [int(x) for x in (path / "kinetixMetadata.txt").read_text().split()]
    _, width, height, header, _, gap = meta[:6]
    frame = width * height * 2
    files = sorted(path.glob("ss_stack_*.prd"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    out = []
    for f in files:
        buf = f.read_bytes()
        n = (len(buf) - header + gap) // (frame + gap)
        for k in range(n):
            o = header + k * (frame + gap)
            out.append(np.frombuffer(buf[o:o + frame], "<u2").reshape(height, width))
    return np.stack(out)


def _reference_pyramid(vol: np.ndarray, chunk: int) -> list[np.ndarray]:
    """Numpy oracle: 2x mean windows, trimmed, truncated; factor 1 on size-1 axes.

    Number of levels follows linc-convert's default: max over axes of
    ceil(log2(size / chunk)).
    """
    import math

    nlev = max(0, max(int(math.ceil(math.log2(s / chunk))) for s in vol.shape))
    levels = [vol]
    for _ in range(nlev):
        prev = levels[-1].astype(np.float64)
        f = [1 if s == 1 else 2 for s in prev.shape]
        out = [max(1, s // 2) if s > 1 else 1 for s in prev.shape]
        prev = prev[tuple(slice(0, o * k) for o, k in zip(out, f))]
        prev = prev.reshape(out[0], f[0], out[1], f[1], out[2], f[2])
        levels.append(prev.mean(axis=(1, 3, 5)).astype(vol.dtype))
    return levels


def _assert_pyramid(out: Path, expected: list[np.ndarray]) -> None:
    g = zarr.open_group(str(out), mode="r")
    keys = sorted((k for k in g.array_keys() if k.isdigit()), key=int)
    assert keys == [str(i) for i in range(len(expected))]
    for k, want in zip(keys, expected):
        np.testing.assert_array_equal(g[k][:], want, err_msg=f"level {k}")


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------
def test_prd_read_frames(tmp_path):
    vol = _write_prd_set(tmp_path / "in", [7, 7, 3], 5, 6)
    np.testing.assert_array_equal(_raw_frames(tmp_path / "in"), vol)
    reader = PrdSetInterpreter(str(tmp_path / "in"))
    assert reader.shape == vol.shape
    for a, b in [(0, 17), (2, 9), (6, 8), (13, 17), (16, 30)]:
        np.testing.assert_array_equal(reader.read_frames(a, b), vol[a:b])


# --------------------------------------------------------------------------
# lossless level 0
# --------------------------------------------------------------------------
LOSSLESS_CASES = {
    # name: convert kwargs
    "v3-blosc-zstd5": dict(chunk=(8,)),
    "v3-blosc-zstd1": dict(chunk=(8,), compressor_opt={"clevel": 1}),
    "v3-blosc-zstd9": dict(chunk=(8,), compressor_opt={"clevel": 9}),
    "v3-blosc-lz4": dict(chunk=(8,), compressor_opt={"cname": "lz4"}),
    "v3-blosc-noshuffle": dict(chunk=(8,), compressor_opt={"shuffle": "noshuffle"}),
    "v3-zstd-codec": dict(chunk=(8,), compressor="zlib"),
    "v3-uncompressed": dict(chunk=(8,), compressor="none"),
    "v3-shard": dict(chunk=(4,), shard=(8,)),
    "v3-shard-uneven": dict(chunk=(4,), shard=(8, 12, 16)),
    "v3-shard-volume": dict(chunk=(4,), max_shard_size="5TB"),
    "v3-shard-rows": dict(chunk=(4,), max_shard_size="40kB"),
    "v3-shard-volume-deep": dict(chunk=(4,), max_shard_size="5TB", slab_depth=24),
    "v3-shard-deep-zstd1": dict(chunk=(4,), shard=(8,), slab_depth=20,
                                compressor_opt={"clevel": 1}),
    "v3-shard-1worker": dict(chunk=(4,), max_shard_size="5TB", workers=1),
    "ts-v3": dict(chunk=(8,), driver="tensorstore"),
    "ts-v3-shard-deep": dict(chunk=(4,), shard=(8,), slab_depth=20,
                             driver="tensorstore"),
    "v3-ome06-shard-volume": dict(chunk=(4,), max_shard_size="5TB",
                                  ome_version="0.6"),
}


@pytest.mark.parametrize("pattern", ["random", "edge"])
@pytest.mark.parametrize("case", sorted(LOSSLESS_CASES))
def test_prd_level0_lossless(tmp_path, case, pattern):
    """Level 0 decodes bit-for-bit to the .prd frames, for every layout/codec."""
    kw = dict(LOSSLESS_CASES[case])
    _write_prd_set(tmp_path / "in", [37, 37, 11], 21, 34, pattern=pattern)
    raw = _raw_frames(tmp_path / "in")
    out = tmp_path / "out.nii.zarr"
    kw.setdefault("workers", 3)
    prd.convert(str(tmp_path / "in"), out=str(out), overwrite=True,
                verify=True, **kw)
    assert not list(out.rglob("*.partial"))
    level0 = zarr.open_array(str(out / "0"), mode="r")
    assert level0.dtype == np.dtype("uint16") and level0.shape == raw.shape
    np.testing.assert_array_equal(level0[:], raw)
    # random sub-blocks (exercise partial shard/chunk reads)
    rng = np.random.default_rng(1)
    for _ in range(10):
        z, y, x = (int(rng.integers(0, s)) for s in raw.shape)
        sl = (slice(z, z + 9), slice(y, y + 7), slice(x, x + 11))
        np.testing.assert_array_equal(level0[sl], raw[sl])
    ts = pytest.importorskip("tensorstore")
    t = ts.open({"driver": "zarr3", "kvstore": {
        "driver": "file", "path": str(out / "0")}}).result()
    np.testing.assert_array_equal(t.read().result(), raw)


def test_prd_level0_lossless_real_subset(tmp_path):
    """The golden real-data subset round-trips exactly in the large-shard layout."""
    out = tmp_path / "out.nii.zarr"
    prd.convert(PRD_INPUT_DIR, out=str(out), chunk=(32,), max_shard_size="5TB",
                compressor_opt={"clevel": 1}, verify=True)
    raw = _raw_frames(Path(PRD_INPUT_DIR))
    np.testing.assert_array_equal(zarr.open_array(str(out / "0"))[:], raw)


def test_prd_verify_detects_corruption(tmp_path):
    _write_prd_set(tmp_path / "in", [37, 11], 21, 34)
    out = tmp_path / "out.nii.zarr"
    prd.convert(str(tmp_path / "in"), out=str(out), chunk=(4,), compressor="none",
                max_shard_size="5TB")
    shard = next(p for p in (out / "0").rglob("*") if p.is_file()
                 and p.name != "zarr.json")
    buf = bytearray(shard.read_bytes())
    buf[100] ^= 0xFF  # flip one payload byte (index is at the end)
    shard.write_bytes(bytes(buf))
    with pytest.raises(RuntimeError, match="Verification failed"):
        prd._verify_level0(PrdSetInterpreter(str(tmp_path / "in")),
                           zarr.open_array(str(out / "0")))


# --------------------------------------------------------------------------
# pyramid
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "frames,height,width,cfg",
    [
        ([37, 37, 11], 21, 34, dict(chunk=(8,), zarr_version=3)),
        ([50, 13], 17, 9, dict(chunk=(4,), shard=(16,))),
        ([5, 5], 3, 40, dict(chunk=(4,))),
        ([37, 11], 21, 34, dict(chunk=(8,), driver="tensorstore")),
        ([37, 11], 21, 34, dict(chunk=(8,), shard=(16,), driver="tensorstore")),
        ([37, 37, 11], 21, 34, dict(chunk=(4,), max_shard_size="1GB")),
        ([37, 37, 11], 21, 34, dict(chunk=(4,), max_shard_size="40kB")),
        ([37, 37, 11], 21, 34, dict(chunk=(4,), max_shard_size="40kB",
                                    slab_depth=24)),
        ([37, 37, 11], 21, 34, dict(chunk=(4,), shard=(8,), slab_depth=20)),
        ([37, 37, 11], 21, 34, dict(chunk=(4,), shard=(8,), slab_depth=20,
                                    driver="tensorstore")),
    ],
)
def test_prd_pyramid_matches_oracles(tmp_path, frames, height, width, cfg):
    """All levels match a numpy oracle and the library's generate_pyramid."""
    from linc_convert.utils.io.zarr import from_config
    from linc_convert.utils.zarr_config import ZarrConfig

    vol = _write_prd_set(tmp_path / "in", frames, height, width)
    out = tmp_path / "stream.nii.zarr"
    prd.convert(str(tmp_path / "in"), out=str(out), workers=3, overwrite=True, **cfg)
    _assert_pyramid(out, _reference_pyramid(vol, cfg["chunk"][0]))

    # generic (dask) pyramid code, same layout, as a second oracle
    zcfg = {k: v for k, v in cfg.items() if k in ("chunk", "shard", "zarr_version")}
    shards = zarr.open_array(str(out / "0")).shards
    if shards:
        zcfg["shard"] = tuple(shards)
    zc = ZarrConfig(overwrite=True, **zcfg)
    g = from_config(str(tmp_path / "generic.zarr"), zc)
    g.create_array("0", shape=vol.shape, zarr_config=zc, dtype=vol.dtype)[:] = vol
    g.generate_pyramid()
    a = zarr.open_group(str(out), mode="r")
    b = zarr.open_group(str(tmp_path / "generic.zarr"), mode="r")
    levels = sorted((k for k in b.array_keys() if k.isdigit()), key=int)
    assert sorted((k for k in a.array_keys() if k.isdigit()), key=int) == levels
    for k in levels:
        np.testing.assert_array_equal(a[k][:], b[k][:], err_msg=f"level {k}")


def test_prd_append_matches_api_sink(tmp_path):
    """Append sink and zarr-python API writes decode to the same pyramid."""
    from linc_convert.utils.io.zarr import from_config
    from linc_convert.utils.io.zarr.stream_pyramid import (
        StreamingPyramidWriter,
        stream_volume,
    )
    from linc_convert.utils.zarr_config import ZarrConfig

    _write_prd_set(tmp_path / "in", [37, 37, 11], 21, 34)
    reader = PrdSetInterpreter(str(tmp_path / "in"))
    cfg = ZarrConfig(chunk=(4,), shard=(8,), overwrite=True)
    for name, append in [("a.zarr", True), ("b.zarr", False)]:
        g = from_config(str(tmp_path / name), cfg)
        w = StreamingPyramidWriter(g, reader.shape, reader.dtype, cfg,
                                   max_workers=2, append_shards=append)
        stream_volume(w, reader.read_frames)
    a = zarr.open_group(str(tmp_path / "a.zarr"))
    b = zarr.open_group(str(tmp_path / "b.zarr"))
    assert sorted(a.array_keys()) == sorted(b.array_keys())
    for k in a.array_keys():
        np.testing.assert_array_equal(a[k][:], b[k][:])


def test_prd_output_independent_of_threads_and_depth(tmp_path):
    """Compressed bytes do not depend on worker count or slab depth."""
    import hashlib

    _write_prd_set(tmp_path / "in", [37, 37, 11], 21, 34)
    digests = set()
    for i, (w, d) in enumerate([(1, None), (4, None), (3, 12), (8, 36)]):
        out = tmp_path / f"o{i}.nii.zarr"
        prd.convert(str(tmp_path / "in"), out=str(out), chunk=(4,),
                    max_shard_size="5TB", workers=w, slab_depth=d)
        h = hashlib.sha256()
        for f in sorted(p for p in out.rglob("*") if p.is_file()):
            h.update(f.relative_to(out).as_posix().encode())
            h.update(f.read_bytes())
        digests.add(h.hexdigest())
    assert len(digests) == 1


# --------------------------------------------------------------------------
# memory budget
# --------------------------------------------------------------------------
def test_prd_max_memory(tmp_path):
    from linc_convert.utils.io.zarr.stream_pyramid import workers_for_memory

    shape, unit = (26478, 1200, 1800), (256, 256, 256)
    assert workers_for_memory(64 * 2**30, shape, 2, 256, unit, 8, 8) == 8
    assert 1 <= workers_for_memory(int(5.5 * 2**30), shape, 2, 256, unit, 8, 8) < 8
    with pytest.raises(MemoryError):
        workers_for_memory(2**30, shape, 2, 256, unit, 8, 8)
    assert workers_for_memory(2**30, shape, 2, 32, (32, 32, 32), 8, 8) == 8

    vol = _write_prd_set(tmp_path / "in", [37, 11], 21, 34)
    prd.convert(str(tmp_path / "in"), out=str(tmp_path / "c.nii.zarr"),
                chunk=(8,), workers=4, max_memory=0.5, max_shard_size="5TB")
    _assert_pyramid(tmp_path / "c.nii.zarr", _reference_pyramid(vol, 8))


# --------------------------------------------------------------------------
# OME-Zarr 0.6
# --------------------------------------------------------------------------
def test_ome_v05_to_v06_folds_and_roundtrips():
    from linc_convert.utils.io.zarr.ome import ome_v05_to_v06, ome_v06_to_v05_view

    v05 = {"version": "0.5", "multiscales": [{
        "name": "x", "type": "t",
        "axes": [{"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"],
        "datasets": [
            {"path": "0", "coordinateTransformations": [
                {"type": "scale", "scale": [1, 1, 1]},
                {"type": "translation", "translation": [0, 0, 0]}]},
            {"path": "1", "coordinateTransformations": [
                {"type": "scale", "scale": [2, 2, 2]},
                {"type": "translation", "translation": [0.5, 0.5, 0.5]}]},
        ],
        "coordinateTransformations": [{"type": "scale", "scale": [3, 1, 1]}],
    }]}
    v06 = ome_v05_to_v06(v05)
    ms = v06["multiscales"][0]
    assert v06["version"] == "0.6" and "axes" not in ms
    assert ms["coordinateSystems"][0]["name"] == "intrinsic"
    d0, d1 = (d["coordinateTransformations"][0] for d in ms["datasets"])
    assert d0 == {"type": "scale", "scale": [3.0, 1.0, 1.0],
                  "input": {"path": "0"}, "output": {"name": "intrinsic"}}
    assert d1["type"] == "sequence"
    assert d1["transformations"][0]["scale"] == [6.0, 2.0, 2.0]
    assert d1["transformations"][1]["translation"] == [1.5, 0.5, 0.5]
    back = ome_v06_to_v05_view(v06)["multiscales"][0]
    assert back["axes"] == v05["multiscales"][0]["axes"]


def test_prd_ome_v06(tmp_path):
    """0.6 output validates, maps voxels like 0.5, and reads back as NIfTI."""
    import niizarr

    from linc_convert.utils.io.zarr.ome import ome_v06_to_v05_view

    omz = pytest.importorskip("ome_zarr_models.v06.image")
    vol = _write_prd_set(tmp_path / "in", [37, 37, 11], 21, 34)
    outs = {}
    for v in ("0.5", "0.6"):
        outs[v] = tmp_path / f"o{v}.nii.zarr"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prd.convert(str(tmp_path / "in"), out=str(outs[v]), chunk=(8,),
                        max_shard_size="5TB", voxel_size=[0.75, 0.75, 2.0],
                        ome_version=v)
        future = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert bool(future) == (v == "0.5")
    g5 = zarr.open_group(str(outs["0.5"]), mode="r")
    g6 = zarr.open_group(str(outs["0.6"]), mode="r")
    ome6 = dict(g6.attrs)["ome"]
    assert ome6["version"] == "0.6"
    omz.ImageAttrs.model_validate(ome6)
    # same voxel->physical mapping as the 0.5 output, level by level
    view = ome_v06_to_v05_view(ome6)["multiscales"][0]
    ms5 = dict(g5.attrs)["ome"]["multiscales"][0]
    top = ms5.get("coordinateTransformations", [{"scale": [1, 1, 1]}])[0]["scale"]
    for d5, d6 in zip(ms5["datasets"], view["datasets"]):
        t5 = {t["type"]: t for t in d5["coordinateTransformations"]}
        t6 = {t["type"]: t for t in d6["coordinateTransformations"]}
        np.testing.assert_allclose(
            t6["scale"]["scale"],
            np.multiply(top, t5["scale"]["scale"]))
        np.testing.assert_allclose(
            t6.get("translation", {}).get("translation", [0, 0, 0]),
            np.multiply(top, t5.get("translation", {}).get("translation", [0, 0, 0])))
    # identical NIfTI header bytes
    assert bytes(g5["nifti"][:]) == bytes(g6["nifti"][:])
    nii = niizarr.zarr2nii(str(outs["0.5"]))
    np.testing.assert_array_equal(np.asarray(nii.dataobj), vol.transpose(2, 1, 0))
    # 0.6: the NIfTI qform/sform spaces are coordinate systems whose affines
    # map voxel indices to the same world coordinates as the NIfTI header
    header = niizarr.zarr2nii(str(outs["0.5"])).header
    ms6 = ome6["multiscales"][0]
    axes = [a["name"] for a in ms6["coordinateSystems"][0]["axes"]]
    scale = ms6["datasets"][0]["coordinateTransformations"][0]["scale"]
    kinds = [t["output"]["name"] for t in ms6["coordinateTransformations"]]
    assert kinds == ["nifti:qform", "nifti:sform"]
    ijk = {"x": 3, "y": 5, "z": 7}
    point = [ijk[a] * s for a, s in zip(axes, scale)] + [1]
    for trf in ms6["coordinateTransformations"]:
        affine = header.get_qform() if trf["output"]["name"].endswith("qform") \
            else header.get_sform()
        world = dict(zip(axes, np.asarray(trf["affine"]) @ point))
        np.testing.assert_allclose(
            [world["x"], world["y"], world["z"]],
            (affine @ [ijk["x"], ijk["y"], ijk["z"], 1])[:3], atol=1e-6)
    # Known limitation: released niizarr (1.0.0rc8) reads multiscales[0]["axes"],
    # which 0.6 replaces by coordinateSystems; fixed in neuroscales/nifti-zarr-py#55.
    with pytest.raises(KeyError, match="axes"):
        niizarr.zarr2nii(str(outs["0.6"]))


def test_zarr_v2_write_unsupported(tmp_path):
    with pytest.raises(ValueError, match="Zarr v2"):
        prd.convert(PRD_INPUT_DIR, out=str(tmp_path / "o.nii.zarr"), zarr_version=2)
