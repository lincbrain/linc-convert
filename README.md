# linc-convert: Data conversion tools for the LINC project

Documentation is available at: https://convert.lincbrain.org

![diagram](./docs/img/linc-convert.png)

## Kinetix `.prd` light-sheet data

`linc-convert lsm prd` streams `.prd` stacks into a pyramidal NIfTI-Zarr in
bounded memory (a few GB, independent of scan size), with lossless level 0.
For write-once, read-many archives, use
`--chunk 64 --compressor-opt.clevel 1` (shards default to frame-wide files of
up to 40 GB uncompressed, ~13 GB compressed).
See [docs/lsm_prd.md](docs/lsm_prd.md) for the measured trade-offs and how
to keep up with acquisition rates.

