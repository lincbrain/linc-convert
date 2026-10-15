import shutil
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

def _write_prd_set(path: Path, frames_per_file, height, width, seed=0):
    """Write a synthetic Kinetix .prd set; return the full (z, y, x) volume."""
    rng = np.random.default_rng(seed)
    header, gap, pad = 64, 18, 10
    path.mkdir(parents=True, exist_ok=True)
    (path / "kinetixMetadata.txt").write_text(
        f"16\n{width}\n{height}\n{header}\n{max(frames_per_file)}\n{gap}\n"
    )
    vols = []
    for i, n in enumerate(frames_per_file):
        vol = rng.integers(0, 2**16, (n, height, width), dtype="<u2")
        vols.append(vol)
        with open(path / f"ss_stack_{i}.prd", "wb") as fh:
            fh.write(b"\xab" * header)
            for k in range(n):
                fh.write(vol[k].tobytes())
                if k < n - 1:
                    fh.write(b"\xcd" * gap)
            fh.write(b"\xef" * pad)
    return np.concatenate(vols)


def test_prd_read_frames(tmp_path):
    vol = _write_prd_set(tmp_path / "in", [7, 7, 3], 5, 6)
    reader = PrdSetInterpreter(str(tmp_path / "in"))
    assert reader.shape == vol.shape
    np.testing.assert_array_equal(reader.assemble().compute(), vol)
    for a, b in [(0, 17), (2, 9), (6, 8), (13, 17), (16, 30)]:
        np.testing.assert_array_equal(reader.read_frames(a, b), vol[a:b])


@pytest.mark.parametrize(
    "frames,height,width,cfg",
    [
        ([37, 37, 11], 21, 34, dict(chunk=(8,), zarr_version=3)),
        ([37, 37, 11], 21, 34, dict(chunk=(8,), zarr_version=2)),
        ([50, 13], 17, 9, dict(chunk=(4,), shard=(16,), zarr_version=3)),
        ([5, 5], 3, 40, dict(chunk=(4,), zarr_version=3)),
        ([37, 11], 21, 34, dict(chunk=(8,), zarr_version=3, driver="tensorstore")),
        ([37, 11], 21, 34, dict(chunk=(8,), shard=(16,), driver="tensorstore")),
        # one shard per level (append sink), odd edges
        ([37, 37, 11], 21, 34, dict(chunk=(4,), max_shard_size="1GB")),
        # several shard rows along z, frame-wide shards
        ([37, 37, 11], 21, 34, dict(chunk=(4,), max_shard_size="40kB")),
        # deep slabs: batches spanning several chunks and shard rows
        ([37, 37, 11], 21, 34, dict(chunk=(4,), max_shard_size="40kB",
                                    slab_depth=24)),
        ([37, 37, 11], 21, 34, dict(chunk=(4,), shard=(8,), slab_depth=20)),
        ([37, 37, 11], 21, 34, dict(chunk=(4,), shard=(8,), slab_depth=20,
                                    driver="tensorstore")),
    ],
)
def test_prd_streaming_matches_legacy(tmp_path, frames, height, width, cfg):
    _write_prd_set(tmp_path / "in", frames, height, width)
    legacy_cfg = dict(cfg, overwrite=True)
    max_shard = legacy_cfg.pop("max_shard_size", None)
    depth = legacy_cfg.pop("slab_depth", None)
    stream_cfg = dict(legacy_cfg, max_shard_size=max_shard, slab_depth=depth)
    prd.convert(str(tmp_path / "in"), out=str(tmp_path / "stream.nii.zarr"),
                workers=3, **stream_cfg)
    if max_shard:
        # legacy does not stream shards: give it the same layout explicitly
        shards = zarr.open_array(str(tmp_path / "stream.nii.zarr" / "0")).shards
        legacy_cfg["shard"] = tuple(shards)
    prd.convert(str(tmp_path / "in"), out=str(tmp_path / "legacy.nii.zarr"),
                legacy=True, **legacy_cfg)
    assert_zarr_equal(str(tmp_path / "stream.nii.zarr"),
                      str(tmp_path / "legacy.nii.zarr"))


def test_prd_append_shards_readable(tmp_path):
    """Append-written shards decode identically with zarr-python and tensorstore."""
    vol = _write_prd_set(tmp_path / "in", [37, 37, 11], 21, 34)
    out = tmp_path / "out.nii.zarr"
    prd.convert(str(tmp_path / "in"), out=str(out), chunk=(4,), shard=(8, 12, 16),
                workers=2)
    assert not list(out.rglob("*.partial"))
    arr = zarr.open_array(str(out / "0"))
    np.testing.assert_array_equal(arr[:], vol)
    np.testing.assert_array_equal(arr[5:23, 3:19, 7:33], vol[5:23, 3:19, 7:33])
    try:
        import tensorstore as ts
    except ImportError:
        return
    t = ts.open({"driver": "zarr3", "kvstore": {"driver": "file",
                 "path": str(out / "0")}}).result()
    np.testing.assert_array_equal(t.read().result(), vol)


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


def test_prd_max_memory(tmp_path):
    from linc_convert.utils.io.zarr.stream_pyramid import workers_for_memory

    shape, unit = (26478, 1200, 1800), (256, 256, 256)
    assert workers_for_memory(64 * 2**30, shape, 2, 256, unit, 8, 8) == 8
    assert 1 <= workers_for_memory(int(5.5 * 2**30), shape, 2, 256, unit, 8, 8) < 8
    with pytest.raises(MemoryError):
        workers_for_memory(2**30, shape, 2, 256, unit, 8, 8)
    # append sink: 32-slice slabs fit comfortably
    assert workers_for_memory(2**30, shape, 2, 32, (32, 32, 32), 8, 8) == 8

    _write_prd_set(tmp_path / "in", [37, 11], 21, 34)
    prd.convert(str(tmp_path / "in"), out=str(tmp_path / "a.nii.zarr"),
                chunk=(8,), legacy=True)
    prd.convert(str(tmp_path / "in"), out=str(tmp_path / "b.nii.zarr"),
                chunk=(8,), workers=4, max_memory=0.5)
    prd.convert(str(tmp_path / "in"), out=str(tmp_path / "c.nii.zarr"),
                chunk=(8,), workers=4, max_memory=0.5, max_shard_size="5TB")
    assert_zarr_equal(str(tmp_path / "c.nii.zarr"), str(tmp_path / "a.nii.zarr"))
    assert_zarr_equal(str(tmp_path / "b.nii.zarr"), str(tmp_path / "a.nii.zarr"))
