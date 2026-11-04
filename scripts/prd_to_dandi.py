#!/usr/bin/env python
"""Batch Kinetix .prd scans: transfer -> convert -> upload -> verify -> delete.

Each scan directory (a directory holding `ss_stack_*.prd` files) found under
`--source` is processed in turn:

1. **transfer**: rsync the scan and its sibling files (`<scan>_info.mat`,
   `<scan>_stim_data.bin`, ... = every file named `<scan>_*`) into
   `--staging`. `--source` may be a local/mounted path or `host:/path`.
   `--no-stage` converts straight from a mounted source instead.
2. **convert**: `linc-convert lsm prd`, written into the local Dandiset under
   `rawdata/<path relative to --source>.nii.zarr`, then moved into place.
3. **upload**: `dandi upload` of the zarr and the sibling files, which are
   uploaded unchanged next to it (`rawdata/<path>_info.mat`, ...).
4. **verify**: the assets on the server have the same checksums as the
   local copies.
5. **delete** (only with `--delete`): remove the staged copy of the raw scan
   and the local zarr. The source is never touched.

Progress is kept in `<dandiset>/.prd2dandi/state/<scan>.json`, so a rerun
resumes where it stopped and skips finished scans. Scans are locked while
processed, so several instances (e.g. on different machines sharing the
Dandiset directory) can run at once. `--dry-run` only prints the plan.

Example
-------
    dandi download --download dandiset.yaml https://dandiarchive.org/dandiset/001769
    export DANDI_API_KEY=...
    python prd_to_dandi.py --source /mnt/nas/lsm --dandiset 001769 \
        --staging /data/stage --voxel-size 0.7 0.7 2.0 \
        --workers 8 --max-memory 8 --dry-run
"""

# ruff: noqa: D101, D102, D103 (a script, not part of the package API)
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

STEPS = ("transfer", "convert", "upload", "verify", "delete")
# Output size relative to the raw .prd bytes (measured ~0.39 with zstd 1;
# a margin for the pyramid and less compressible data).
OUTPUT_RATIO = 0.6


@dataclass
class Scan:
    rel: str  # path relative to --source, e.g. "SAMPLE/run1"
    src: str  # rsync/local path of the scan directory
    dandi_path: str  # e.g. "rawdata/SAMPLE/run1.nii.zarr"
    voxel_size: list[float] | None = None
    raw_bytes: int | None = None
    sidecars: list[str] = field(default_factory=list)  # e.g. "_info.mat"
    state: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "__", self.rel)


def main() -> int:
    args = parse_args()
    for tool in (args.linc_convert, args.dandi):
        if not shutil.which(tool.split()[0]):
            sys.exit(f"Not found on PATH: {tool}")
    if not os.path.isfile(os.path.join(args.dandiset, "dandiset.yaml")):
        sys.exit(f"{args.dandiset}/dandiset.yaml not found. Create it with: "
                 "dandi download --download dandiset.yaml "
                 "https://dandiarchive.org/dandiset/<ID>")
    if not (args.dry_run or args.no_upload or os.environ.get("DANDI_API_KEY")):
        sys.exit("Set DANDI_API_KEY (or use --no-upload / --dry-run).")
    if args.staging is None and not args.no_stage:
        sys.exit("Give --staging DIR, or --no-stage to read a mounted source.")
    if args.no_stage and is_remote(args.source):
        sys.exit("--no-stage needs a local/mounted --source.")

    meta = os.path.join(args.dandiset, ".prd2dandi")
    for sub in ("state", "logs", "work"):
        os.makedirs(os.path.join(meta, sub), exist_ok=True)

    scans = discover(args)
    if args.include:
        scans = [s for s in scans if re.search(args.include, s.rel)]
    for s in scans:
        s.state = load_state(meta, s)
    todo = [s for s in scans if not finished(s, args)]
    print(f"{len(scans)} scan(s) found, {len(scans) - len(todo)} already done, "
          f"{len(todo)} to process.")
    if args.limit:
        todo = todo[:args.limit]

    if args.dry_run:
        dry_run(args, todo)
        return 0

    failed = 0
    for i, s in enumerate(todo, 1):
        lock = open(os.path.join(meta, "state", s.key + ".lock"), "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"[{i}/{len(todo)}] {s.rel}: locked by another run, skipping")
            continue
        s.state = load_state(meta, s)  # may have advanced meanwhile
        print(f"[{i}/{len(todo)}] {s.rel} -> {s.dandi_path}", flush=True)
        try:
            process(args, meta, s)
        except StopRun as e:
            print(f"  STOP: {e}", flush=True)
            return 1
        except Exception as e:  # noqa: BLE001 - record and move to next scan
            failed += 1
            s.state["error"] = f"{type(e).__name__}: {e}"
            save_state(meta, s)
            print(f"  FAILED: {s.state['error']} (log: {log_path(meta, s)})",
                  flush=True)
        finally:
            lock.close()
    print(f"Done, {failed} failure(s).")
    return 1 if failed else 0


class StopRun(Exception):
    """A problem that would affect every following scan (e.g. disk full)."""


def process(args: argparse.Namespace, meta: str, s: Scan) -> None:
    st = s.state
    st.pop("error", None)
    out = os.path.join(args.dandiset, s.dandi_path)
    staged = s.src if args.no_stage else os.path.join(args.staging, s.rel)
    # sibling files, next to the scan and next to the zarr
    side = [(staged + x, out.removesuffix(".nii.zarr") + x) for x in s.sidecars]
    log = log_path(meta, s)

    def done(step: str, **extra: object) -> None:
        st[step] = dict(at=time.strftime("%Y-%m-%dT%H:%M:%S"), **extra)
        save_state(meta, s)

    if "convert" not in st and not (s.voxel_size or "_info.mat" in s.sidecars):
        raise RuntimeError("voxel size unknown: no _info.mat next to the scan; "
                           "give --voxel-size or a manifest voxel_size")

    if "transfer" not in st:
        t0 = time.time()
        if not args.no_stage:
            need = int(s.raw_bytes * (1 + OUTPUT_RATIO)) + args.min_free_gb * 2**30
            check_space(args.staging, need)
            os.makedirs(os.path.dirname(staged) or ".", exist_ok=True)
            run(rsync_cmd(args, s.src + "/", staged + "/"), log)
            for x in s.sidecars:
                run(rsync_cmd(args, s.src + x, staged + x), log)
        done("transfer", seconds=round(time.time() - t0), bytes=s.raw_bytes,
             staged=staged)

    if "convert" not in st:
        check_space(os.path.dirname(meta), int(s.raw_bytes * OUTPUT_RATIO)
                    + args.min_free_gb * 2**30)
        work = os.path.join(meta, "work", s.key)
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        tmp = os.path.join(work, os.path.basename(out))
        t0 = time.time()
        run(convert_cmd(args, s, staged, tmp), log)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out):
            shutil.rmtree(out)
        os.rename(tmp, out)
        shutil.rmtree(work, ignore_errors=True)
        for src, dst in side:
            shutil.copy2(src, dst)
        done("convert", seconds=round(time.time() - t0), zarr_bytes=du(out))

    if args.no_upload:
        return

    if "upload" not in st:
        t0 = time.time()
        paths = [out] + [dst for _, dst in side]
        cmd = [*shlex.split(args.dandi), "upload", "-i", args.dandi_instance,
               "-J", args.upload_jobs, "--validation", args.validation,
               "--allow-any-path", *paths]
        run(cmd, log, cwd=args.dandiset)
        done("upload", seconds=round(time.time() - t0))

    if "verify" not in st:
        t0 = time.time()
        checksum = verify_upload(args, s, out, [dst for _, dst in side])
        done("verify", seconds=round(time.time() - t0), zarr_checksum=checksum)

    if args.delete and "delete" not in st:
        if not args.no_stage:
            shutil.rmtree(staged, ignore_errors=True)
            for src, _ in side:
                if os.path.isfile(src):
                    os.remove(src)
        shutil.rmtree(out, ignore_errors=True)
        for _, dst in side:
            if os.path.isfile(dst):
                os.remove(dst)
        done("delete", staged=not args.no_stage)


def verify_upload(
    args: argparse.Namespace, s: Scan, out: str, files: list[str]
) -> str:
    """Check that the server has the same zarr `out` and `files`."""
    from pathlib import Path

    from dandi.dandiapi import DandiAPIClient
    from dandi.support.digests import get_digest, get_zarr_checksum

    ds_id = read_dandiset_id(args.dandiset)
    local = get_zarr_checksum(Path(out))
    with DandiAPIClient.for_dandi_instance(
            args.dandi_instance, token=os.environ.get("DANDI_API_KEY")) as client:
        ds = client.get_dandiset(ds_id, "draft")
        remote = ds.get_asset_by_path(s.dandi_path).get_digest().value
        if remote != local:
            raise RuntimeError(f"zarr checksum mismatch: local {local}, "
                               f"server {remote}")
        for path in files:
            rel = os.path.relpath(path, args.dandiset)
            r = ds.get_asset_by_path(rel).get_raw_digest()
            if r != get_digest(path, "dandi-etag"):
                raise RuntimeError(f"{rel}: checksum mismatch on the server")
    return local


# --------------------------------------------------------------------------
# Discovery


def discover(args: argparse.Namespace) -> list[Scan]:
    """Scan directories under --source (or listed in --manifest)."""
    root = args.source.rstrip("/")
    rpath = root.split(":", 1)[1] if is_remote(root) else root  # path on host
    manifest = read_manifest(args.manifest) if args.manifest else None
    if manifest is not None:
        rels = list(manifest)
    else:
        found = sh_lines(args, f"find {shlex.quote(rpath)}/ -name 'ss_stack_*.prd' "
                               "-printf '%h\\n' | sort -u")
        rels = [os.path.relpath(d, rpath) for d in found]
    if not rels:
        sys.exit(f"No ss_stack_*.prd files found under {root}")

    # Sizes of the scans and of their sibling files `<scan>_*`, in one call
    # each (over ssh for a remote source).
    quoted = " ".join(shlex.quote(f"{rpath}/{r}") for r in rels)
    sizes = {}
    for line in sh_lines(args, f"du -sbL {quoted}"):
        n, path = line.split("\t", 1)
        sizes[os.path.relpath(path, rpath)] = int(n)
    sidecars: dict[str, list[str]] = {r: [] for r in rels}
    for line in sh_lines(args, f"for d in {quoted}; do for f in \"$d\"_*; do "
                               "[ -f \"$f\" ] && stat -L -c '%s\t%n' \"$f\"; "
                               "done; done; true"):
        n, path = line.split("\t", 1)
        f = os.path.relpath(path, rpath)
        # `run1_HR_info.mat` belongs to scan `run1_HR`, not to `run1`.
        owner = max((r for r in rels if f.startswith(r + "_")), key=len)
        if f.removeprefix(owner) not in sidecars[owner]:
            sidecars[owner].append(f.removeprefix(owner))
            sizes[owner] = sizes.get(owner, 0) + int(n)

    scans = []
    for rel in rels:
        row = (manifest or {}).get(rel, {})
        dpath = row.get("dandi_path") or f"{args.prefix}/{rel}.nii.zarr"
        if not dpath.endswith(".nii.zarr"):
            dpath += ".nii.zarr"
        vs = row.get("voxel_size")
        vs = [float(v) for v in vs.split()] if vs else args.voxel_size
        if vs is not None and len(vs) != 3:
            sys.exit(f"{rel}: voxel size needs 3 values (X Y Z), got {vs}")
        src_dir = f"{root}/{rel}"
        scans.append(Scan(rel=rel, src=src_dir, dandi_path=dpath, voxel_size=vs,
                          raw_bytes=sizes.get(rel, 0), sidecars=sidecars[rel]))
    return scans


def read_manifest(path: str) -> dict[str, dict]:
    """Read a manifest of scans.

    A TSV with a `scan` column (relative to --source) and optional
    `dandi_path` and `voxel_size` ("X Y Z", microns) columns.
    """
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(
            (line for line in fh if not line.startswith("#")), delimiter="\t"))
    if rows and "scan" not in rows[0]:
        sys.exit(f"{path}: needs a 'scan' column")
    return {r["scan"].strip().strip("/"): {k: (v or "").strip() for k, v in r.items()}
            for r in rows}


# --------------------------------------------------------------------------
# Commands


def convert_cmd(args: argparse.Namespace, s: Scan, staged: str, out: str) -> list:
    cmd = []
    if args.nice is not None:
        cmd += ["nice", "-n", str(args.nice)]
    if args.cpus:
        cmd += ["taskset", "-c", args.cpus]
    cmd += [*shlex.split(args.linc_convert), "lsm", "prd", staged, "--out", out]
    if s.voxel_size:
        cmd += ["--voxel-size", *map(str, s.voxel_size)]
    if args.workers:
        cmd += ["--workers", str(args.workers)]
    if args.max_memory:
        cmd += ["--max-memory", str(args.max_memory)]
    return cmd + shlex.split(args.convert_args)


def rsync_cmd(args: argparse.Namespace, src: str, dst: str) -> list:
    cmd = ["rsync", "-a", "--copy-links", "--partial", "--info=stats1"]
    if args.bwlimit:
        cmd.append(f"--bwlimit={args.bwlimit}")
    return cmd + [src, dst]


def is_remote(path: str) -> bool:
    return re.match(r"^[^/]+:", path) is not None


def sh_lines(args: argparse.Namespace, script: str) -> list[str]:
    """Run a shell snippet next to the source (over ssh if it is remote)."""
    if is_remote(args.source):
        host = args.source.split(":", 1)[0]
        cmd = ["ssh", host, script]
    else:
        cmd = ["bash", "-c", script]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [line for line in res.stdout.splitlines() if line]


def run(cmd: list, log: str, cwd: str | None = None) -> None:
    with open(log, "a") as fh:
        fh.write(f"\n$ {shlex.join(cmd)}\n")
        fh.flush()
        res = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd)
    if res.returncode:
        raise RuntimeError(f"{cmd[0]} ... exited with {res.returncode}")


def check_space(path: str, need: int) -> None:
    free = shutil.disk_usage(path).free
    if free < need:
        raise StopRun(f"{path}: {free / 2**30:.0f} GiB free, "
                      f"{need / 2**30:.0f} GiB needed")


def du(path: str) -> int:
    return sum(os.path.getsize(os.path.join(d, f))
               for d, _, files in os.walk(path) for f in files)


# --------------------------------------------------------------------------
# State


def load_state(meta: str, s: Scan) -> dict:
    p = os.path.join(meta, "state", s.key + ".json")
    if os.path.isfile(p):
        with open(p) as fh:
            st = json.load(fh)
        if st.get("dandi_path") == s.dandi_path:
            return st.get("steps", {})
    return {}


def save_state(meta: str, s: Scan) -> None:
    p = os.path.join(meta, "state", s.key + ".json")
    with open(p + ".tmp", "w") as fh:
        json.dump(dict(scan=s.rel, source=s.src, dandi_path=s.dandi_path,
                       voxel_size=s.voxel_size, steps=s.state), fh, indent=1)
    os.replace(p + ".tmp", p)


def log_path(meta: str, s: Scan) -> str:
    return os.path.join(meta, "logs", s.key + ".log")


def finished(s: Scan, args: argparse.Namespace) -> bool:
    last = "convert" if args.no_upload else "delete" if args.delete else "verify"
    return last in s.state


def read_dandiset_id(dandiset: str) -> str:
    with open(os.path.join(dandiset, "dandiset.yaml")) as fh:
        m = re.search(r"^identifier:\s*['\"]?(?:DANDI:)?(\d+)", fh.read(), re.M)
    if not m:
        sys.exit(f"No identifier in {dandiset}/dandiset.yaml")
    return m.group(1)


def dry_run(args: argparse.Namespace, todo: list[Scan]) -> None:
    total = sum(s.raw_bytes for s in todo)
    print(f"DRY RUN: nothing is copied, converted, uploaded or deleted.\n"
          f"Dandiset {read_dandiset_id(args.dandiset)} on '{args.dandi_instance}', "
          f"local dir {args.dandiset}")
    print(f"{len(todo)} scan(s), {total / 1e12:.2f} TB raw, "
          f"~{total * 0.4 / 1e12:.2f} TB to upload")
    if not args.no_stage:
        free = shutil.disk_usage(args.staging if os.path.isdir(args.staging)
                                 else os.path.dirname(args.staging) or ".").free
        big = max((s.raw_bytes for s in todo), default=0)
        print(f"Staging {args.staging}: {free / 2**30:.0f} GiB free; largest scan "
              f"needs ~{big * (1 + OUTPUT_RATIO) / 2**30:.0f} GiB "
              f"+ {args.min_free_gb} GiB margin")
    for s in todo:
        staged = s.src if args.no_stage else os.path.join(args.staging, s.rel)
        out = os.path.join(args.dandiset, s.dandi_path)
        vs = ("--voxel-size " + " ".join(map(str, s.voxel_size)) if s.voxel_size
              else "from _info.mat" if "_info.mat" in s.sidecars else
              "MISSING (no _info.mat, no --voxel-size): conversion would fail")
        steps = [st for st in STEPS if st not in s.state
                 and not (st == "delete" and not args.delete)
                 and not (args.no_upload and st in ("upload", "verify", "delete"))]
        print(f"\n{s.rel}  ({s.raw_bytes / 2**30:.1f} GiB)  steps: {', '.join(steps)}")
        print(f"  voxel size: {vs}")
        if s.sidecars:
            print(f"  sibling files (uploaded as is): {', '.join(s.sidecars)}")
        if "transfer" in steps and not args.no_stage:
            print("  $ " + shlex.join(rsync_cmd(args, s.src + "/", staged + "/")))
        if "convert" in steps:
            print("  $ " + shlex.join(convert_cmd(args, s, staged, out)))
        if "upload" in steps:
            print(f"  $ (cd {args.dandiset}) dandi upload -i {args.dandi_instance} "
                  f"... {s.dandi_path}")
        if "delete" in steps:
            what = [out] + [out.removesuffix(".nii.zarr") + x for x in s.sidecars]
            what += [] if args.no_stage else [staged] + [staged + x for x in s.sidecars]
            print(f"  delete after verified upload: {', '.join(what)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True,
                   help="Root of the raw scans: a local/mounted path or host:/path.")
    p.add_argument("--dandiset", required=True,
                   help="Local Dandiset directory (holding dandiset.yaml).")
    p.add_argument("--staging", help="Local directory raw scans are copied to.")
    p.add_argument("--no-stage", action="store_true",
                   help="Convert directly from a mounted --source (no copy).")
    p.add_argument("--manifest",
                   help="TSV of scans to process: column 'scan' (relative to "
                        "--source), optional 'dandi_path' and 'voxel_size' "
                        "('X Y Z' microns).")
    p.add_argument("--prefix", default="rawdata",
                   help="Dandiset folder for outputs (default: rawdata).")
    p.add_argument("--include", help="Only scans whose relative path matches "
                                     "this regex.")
    p.add_argument("--limit", type=int, help="Process at most N scans.")
    p.add_argument("--voxel-size", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="Voxel size in microns (camera columns, camera rows, "
                        "frames), overriding _info.mat for every scan.")
    p.add_argument("--workers", type=int, help="Converter compression threads.")
    p.add_argument("--max-memory", type=float, help="Converter memory (GiB).")
    p.add_argument("--cpus", help="Pin the converter to these CPUs "
                                  "(taskset list, e.g. 0-7).")
    p.add_argument("--nice", type=int, default=10,
                   help="Converter niceness (default 10).")
    p.add_argument("--convert-args",
                   default="--chunk 64 --compressor-opt.clevel 1 --verify",
                   help="Extra linc-convert arguments (default: %(default)s).")
    p.add_argument("--bwlimit", help="rsync --bwlimit (e.g. 200M).")
    p.add_argument("--min-free-gb", type=int, default=50,
                   help="Free space to keep on staging/output disks.")
    p.add_argument("--dandi-instance", default="dandi")
    p.add_argument("--upload-jobs", default="4:8",
                   help="dandi upload -J (default %(default)s).")
    p.add_argument("--validation", default="require",
                   choices=["require", "skip", "ignore"])
    p.add_argument("--no-upload", action="store_true",
                   help="Stop after conversion.")
    p.add_argument("--delete", action="store_true",
                   help="After a verified upload, delete the staged raw copy "
                        "and the local zarr. The source is never deleted.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done, and do nothing.")
    p.add_argument("--linc-convert", default="linc-convert")
    p.add_argument("--dandi", default="dandi")
    args = p.parse_args()
    args.dandiset = os.path.abspath(args.dandiset)
    if args.staging:
        args.staging = os.path.abspath(args.staging)
    if args.delete and args.no_upload:
        p.error("--delete needs an upload (drop --no-upload)")
    return args


if __name__ == "__main__":
    sys.exit(main())
