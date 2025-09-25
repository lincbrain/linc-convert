# Basic Usage Guide

1. The top-level help lists all modalities (data families) and global flags.
    ```bash
    linc-convert --help
    ```
2. Each modality contains one or more pipelines (subcommands).
    ```bash
    linc-convert psoct --help
    ```
3. Show the detailed help for a pipeline, including the input, output, chunking, sharding and additional flags.
    ```bash
    linc-convert psoct single_volume --help
    ```
4. Basic conversion example
    ```bash
    linc-convert psoct single_volume IN.mat -o OUT.zarr
    ```
5. Convert to NIfTI-Zarr and provide the custom key for the MATLAB array
    ```bash
    linc-convert psoct single_volume IN.mat --key Psi_ObsLSQ -o OUT.nii.zarr
    ```
6. Convert to Zarr version 3 with sharding
    ```bash
    linc-convert psoct single_volume IN.mat -o OUT.zarr --zarr-version 3 --shard 1024
    ```
