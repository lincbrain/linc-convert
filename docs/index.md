# linc-convert: Data conversion tools for the LINC project

The `linc-convert` package converts dark-field microscopy, light-sheet microscopy, and polarization-sensitive optical coherence tomography (PS-OCT) files to the OME-Zarr file format.

![diagram](./img/linc-convert.png)

## Quick Links

- [Installation](./installation.md)
- [LINC data conversion code on GitHub](https://github.com/lincbrain/linc-convert)
- [LINC Homepage](https://connects.mgh.harvard.edu/)

## Basic Usage Pattern

`linc-convert <MODALITY> <PIPELINE> [ARGS] [OPTIONS]`

Examples:

```
# List pipelines for a modality
linc-convert psoct --help

# Show help for a pipeline
linc-convert psoct single_volume --help

# Run a conversion
linc-convert psoct single_volume /path/input.mat \
  --key Psi_ObsLSQ \
  --out /path/output.nii.zarr
```

## Support

For questions, bug reports, and feature requests, please file an issue on the [linc-convert](https://github.com/lincbrain/linc-convert) repository.
