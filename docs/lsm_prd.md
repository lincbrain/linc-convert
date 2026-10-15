# Kinetix `.prd` light-sheet conversion

`linc-convert lsm prd` converts a directory of Kinetix `ss_stack_*.prd` files
(plus `kinetixMetadata.txt`) into a pyramidal NIfTI-Zarr / OME-Zarr volume.

```bash
linc-convert lsm prd INPUT_DIR --out OUT.nii.zarr \
    --chunk 64 --compressor-opt.clevel 1 --verify
```

## How it works

The converter makes a single pass over the frames:

1. Frames are read in z-slabs of `--slab-depth` slices (default: one chunk).
   The next slab is prefetched while the current one is compressed.
2. Each slab is written to level 0, then downsampled (2×2×2 mean, truncated
   to the input dtype, odd trailing voxels trimmed) into every pyramid level.
3. With sharding, inner chunks are compressed by a thread pool and appended
   to their shard file as soon as a slab is complete. The shard index
   (+crc32c) is written, and `<shard>.partial` renamed, when the shard is
   complete. Shards can therefore be as large as the volume without
   read-modify-write.

Memory is bounded by a few slabs and does not depend on the volume size.
The output is bit-identical to the previous implementation, which loaded
whole files and computed the pyramid with dask (validated on the 114 GB
`AG9-AY8-178/slice178_run5` stripe against the uploaded reference).

## Options

| Option | Default | Effect |
|---|---|---|
| `--chunk N` | 128 | Inner chunk (z, y, x). Smaller chunks give faster small reads and slightly better compression, but slower bulk reads with zarr-python. |
| `--shard N` / `--max-shard-size S` | `40GB` | Shard shape, or the largest uncompressed shard (S3 allows up to `5TB`; `none` disables sharding). Shards span the whole frame in y/x and as many slices along z as fit; `40GB` gives ~13 GB shard files at the ~3× compression of this data. |
| `--compressor-opt.clevel L` | 5 | blosc-zstd level. Level 1 is smaller *and* faster on this data (see below). |
| `--workers N` | all CPUs | Compression threads. |
| `--slab-depth N` | one chunk | Slices per batch. Deeper slabs only help when compression-bound (many cores, zstd ≥ 3). |
| `--max-memory GiB` | none | Memory budget: fewer workers first, then deeper slabs if the budget allows. |
| `--ome-version` | 0.6 | OME-Zarr 0.6 (coordinate systems, with the NIfTI qform/sform as `nifti:qform` / `nifti:sform`). Writing 0.5 will be deprecated. |
| `--verify` | off | Re-read level 0 and check it is bit-identical to the `.prd` frames. |

Only Zarr v3 is written. Zarr v2 inputs can still be read by the other converters.

## Measured trade-offs

The data is the full 114.5 GB stripe `AG9-AY8-178/slice178_run5`
(26478 × 1200 × 1800, uint16). Unless noted, input is staged in RAM (tmpfs),
output goes to node-local disk, reads start with a cold page cache, and runs
use 8 CPUs. With the input on the same local disk as the output, the
chunk-32 / zstd-1 conversion wrote at 596 MB/s instead of ~970 MB/s.

### Compression level and chunk size (local disk)

| chunk / shard | zstd | level 0 ratio | total size | files | zarr-python bulk read | tensorstore bulk read | tensorstore 256² tile | tensorstore XY plane |
|---|---|---|---|---|---|---|---|---|
| 128 / 256³ | 5 | 2.89 | 44.45 GB | 4928 | 1366 MB/s | 3528 MB/s | 8.3 ms | 93 ms |
| 128 / 256³ | 1 | 2.90 | 44.22 GB | 4928 | 1454 MB/s | 3604 MB/s | 8.1 ms | 91 ms |
| 32 / 256³ | 1 | 2.96 | 42.94 GB | 4932 | 384 MB/s | 2997 MB/s | 1.8 ms | 23 ms |
| 32 / volume (≤5 TB) | 5 | 2.97 | 43.35 GB | 25 | 368 MB/s | 1486 MB/s | 3.1 ms | 49 ms |
| 32 / volume (≤5 TB) | 1 | 2.96 | 43.22 GB | 25 | 401 MB/s | 2105 MB/s | 3.0 ms | 40 ms |

- **zstd level 1 vs 5.** Level 1 is the same size or smaller. It reads as
  fast or faster (decompression speed barely depends on the level), and
  compresses about 7× faster per core (985 vs 139 MB/s per core on 128³
  chunks of this data). Level 1 is the better choice for write-once,
  read-many archives of this data.
- **Chunk 32³ vs 128³.** On local disk, 32³ is about 3% smaller and 3–5×
  faster for small reads (tiles, planes, regions) with tensorstore. On
  network storage, the extra requests outweigh this and 64³ is the better
  default (see below). Bulk reads
  with zarr-python are about 4× slower, because of per-chunk Python
  overhead.
- **Whole-volume shards vs 256³ shards.** One file per level (25 files in
  total; the largest is 38.6 GB, well under the 5 TB S3 limit) instead of
  ~4900 objects. The cost is 1.5–2× slower tensorstore reads, because each
  read goes through a larger shard index. zarr-python reads every shard
  index without caching, and its 32³ small reads were about 2× slower.
- Readers of whole-volume shards should cache shard indexes (tensorstore
  and neuroglancer do).
- The default number of pyramid levels grows as chunks shrink (levels are
  added until the coarsest fits in one chunk): 11 levels with 32³ chunks
  vs 9 with 128³ for this stripe. Levels present in both are identical.

### Chunk and shard size on network storage

These reads are from the cluster's network file system (closer to how
archives are read from S3 than local disk). The outputs were read one at a
time with zstd 1.

| chunk / shard | size | files | zarr-python bulk | tensorstore 64³ | tensorstore 256² tile | tensorstore XY plane | first read of a shard |
|---|---|---|---|---|---|---|---|
| 32 / ~1k slices | 42.97 GB | 72 | 357 MB/s | 15.6 ms | 19.8 ms | 92 ms | 18 ms |
| 32 / ~4k | 43.00 GB | 35 | 346 MB/s | 16.9 ms | 19.0 ms | 99 ms | 27 ms |
| 32 / volume | 43.22 GB | 25 | 340 MB/s | 15.8 ms | 18.7 ms | 85 ms | 75 ms |
| 64 / ~1k | 43.38 GB | 73 | 640 MB/s | 8.4 ms | 13.0 ms | 151 ms | 13 ms |
| 64 / ~4k | 43.38 GB | 33 | 642 MB/s | 9.8 ms | 14.0 ms | 164 ms | 17 ms |
| 64 / volume | 43.41 GB | 23 | 711 MB/s | 9.3 ms | 13.9 ms | 149 ms | 18 ms |
| 128 / ~1k | 44.22 GB | 75 | 727 MB/s | 15.1 ms | 27.2 ms | 336 ms | 25 ms |
| 128 / ~4k | 44.22 GB | 32 | 744 MB/s | 14.8 ms | 24.3 ms | 334 ms | 29 ms |
| 128 / volume | 44.22 GB | 21 | 734 MB/s | 27.8 ms | 43.2 ms | 476 ms | 172 ms |
| 128 / 256³ (existing uploads) | 44.22 GB | 4928 | 1207 MB/s | 25.6 ms | 38.3 ms | 361 ms | 36 ms |

- **Chunk 64 is the best balance.** It has the fastest small reads (about
  1.5–2× faster than 32 or 128), about 2× the zarr-python bulk speed of 32,
  and is within 1% of 32's size. Chunk 32 is best only for full XY planes.
- **Shard size, once large, barely affects steady reads.** It mainly sets the
  cost of the first read of a shard, which fetches the shard's index. Shards
  of a few thousand slices keep that low while holding the file count at
  ~25–75 per stripe, compared with ~4,900 for 256³ shards.
- The default `--max-shard-size 40GB` (~8,700 slices with chunk 64, ~13 GB
  per file) sits between the ~4k-slice and whole-volume rows.

### Write throughput

Chunk 32, whole-volume shard, zstd 1, 1200×1800 frames:

| Python | CPUs | slab depth | write | peak RSS |
|---|---|---|---|---|
| 3.12 | 8 | 32 | 968 MB/s (908–1029, n=3) | 1.15 GB |
| 3.12 | 8 | 64 | 1055 MB/s | 1.24 GB |
| 3.12 | 8 | 256 | 864 MB/s (n=3) | 3.5 GB |
| 3.14t | 8 | 32 | 936 MB/s (n=3) | 1.5 GB |
| 3.14t | 16 | 32 | 1082 MB/s | 1.6 GB |

These full-volume runs were I/O-bound: 3.12 and 3.14t were equal, using
about 3.5 of 8 cores, while writing about 330 MB/s of compressed output to
the node-local disk. Writing to RAM instead (13 GB subset, 16 threads)
shows the compute ceiling:

| Python | zstd 1, depth 32 | zstd 1, depth 128 | zstd 5, depth 128 |
|---|---|---|---|
| 3.12 | 708 MB/s | 741 MB/s | 588 MB/s |
| 3.14 (GIL) | 705 MB/s | 611 MB/s | 566 MB/s |
| 3.14t (free-threaded) | 1076 MB/s | **1307 MB/s** | 656 MB/s |

With small chunks, per-chunk Python work holds the GIL. The free-threaded
3.14t build removes that limit, and deeper slabs then help, because each
batch has more independent tasks. Peak RSS stayed at or below 2.4 GB. The
output is byte-identical across the three builds, the worker counts and the
slab depths. The converter logs whether the GIL is on, and warns if an
extension re-enabled it.

The reference layout (chunk 128, shard 256³, zstd 5) writes at about
760 MB/s with 16 CPUs, using 3.5 GB RSS.

The previous implementation wrote at 168–208 MB/s and used 22–23 GB RSS
with memory limits from 32 to 128 GB. It was killed (out of memory) with a
16 GB limit.

### Memory

Peak memory is about `0.5 GB + 3 × slab + 25 MB × workers`, where
`slab = slab_depth × frame_height × frame_width × 2` bytes. It is
independent of the number of frames, so it scales with each acquisition's
frame size, not its length. For 1200×1800 frames:

| layout | minimum that worked | comfortable |
|---|---|---|
| chunk 32, streamed shards (depth 32) | 1 GB (8 threads, 3.12) / 2 GB (16 threads, 3.14t) | 2–4 GB |
| chunk 128, shard 256³ (256-slice writes) | more than 2 GB (2 GB failed; 3.5 GB peak when unconstrained) | 4–8 GB |

## Keeping up with acquisition

1. **Compute the data rate R.** For example, 100 GB per 2-minute scan is
   about 830 MB/s.
2. **Use zstd level 1** (`--compressor-opt.clevel 1`). It is smaller and
   faster here.
3. **Disk.** It must sustain reading R plus writing about R / 3 (the
   compression ratio is ~3). On the test node the local disk (~350 MB/s
   writes) was the limit; a workstation NVMe SSD normally is not.
4. **CPU.** On free-threaded Python 3.14t, budget about 80 MB/s per thread
   for zstd 1 (1307 MB/s with 16 threads and `--slab-depth 128`), so
   `--workers ≈ R / 80`, leaving the remaining cores to the acquisition
   software. With the GIL (3.12 / 3.14) throughput levels off around
   700–750 MB/s however many cores are used. zstd 5 is CPU-bound at
   roughly 650 MB/s with 16 threads.
5. **Memory.** `--max-memory 4` is plenty for chunk 32, and lets the
   planner pick deeper slabs (up to 256 slices), which helps on 3.14t when
   compute-bound. When I/O-bound, extra memory does not help.
6. **Limiting the converter.** Cap it with `--workers` and
   `--max-memory`, and with the OS: `nice`, `taskset`, or a cgroup/Job
   Object limit on Windows.

## Guarantees and checks

- **Lossless level 0.** Tests compare decoded level 0 bit-for-bit against
  the raw `.prd` bytes, using an independent NumPy parser. They cover:
  - codecs: blosc zstd 1/5/9, lz4, no-shuffle, zstd, uncompressed;
  - layouts: chunks, 256³ shards, uneven shards, whole-volume shards,
    multi-row shards;
  - slab depths and worker counts; zarr-python and tensorstore readers;
  - data patterns: random, and all-zero / constant / 0 / 65535 regions.
- **Deterministic.** Compressed bytes do not depend on the worker count,
  the slab depth, or the Python build (3.12, 3.14, 3.14t).
- **Pyramid.** Checked against a NumPy implementation and against the
  library's generic `generate_pyramid`.
- **`--verify`** re-reads level 0 after conversion and fails on any
  differing voxel, including corruption.
- **OME-Zarr 0.6.** Output validates against `ome-zarr-models` (v0.6). The
  voxel-to-physical mapping matches the 0.5 metadata, and the qform/sform
  coordinate systems map voxels to the same world coordinates as the NIfTI
  header. Requires nifti-zarr ≥ 1.0.0rc9 (reads and writes 0.6).
