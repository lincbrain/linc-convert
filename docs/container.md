# Container Installation Guide

This guide covers installing and using `linc-convert` with Docker and Apptainer (Singularity) containers.

## Docker

### Prerequisites

- [Docker](https://www.docker.com/get-started) installed on your system

### Installation

1. Pull the `linc-convert` image from GitHub Container Registry:
   ```bash
   docker pull ghcr.io/lincbrain/linc-convert:latest
   ```

### Usage

Run `linc-convert` commands using Docker:

```bash
docker run --rm -v /path/to/your/data:/data linc-convert:latest linc-convert --help
```

To convert files, mount your data directory and run:

```bash
docker run --rm \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  linc-convert:latest \
  linc-convert psoct single_volume /input/file.mat \
    --key Psi_ObsLSQ \
    --out /output/file.nii.zarr
```

### Building from Source

To build the Docker image from source:

1. Clone the repository:
   ```bash
   git clone https://github.com/lincbrain/linc-convert.git
   cd linc-convert
   ```

2. Build the Docker image:
   ```bash
   docker build -t linc-convert:latest .
   ```

   The build process uses a multi-stage build that:
   - Installs Poetry and all dependencies
   - Builds the package as a wheel
   - Creates a minimal runtime image with only the necessary components

3. (Optional) Tag the image with a specific version:
   ```bash
   docker tag linc-convert:latest linc-convert:v0.0.1
   ```

## Apptainer / Singularity

### Prerequisites

- [Apptainer](https://apptainer.org/) or [Singularity](https://sylabs.io/singularity/) installed on your system

### Installation

1. Pull the container image:
   ```bash
   apptainer pull docker://ghcr.io/lincbrain/linc-convert:latest
   ```

   Or with Singularity:
   ```bash
   singularity pull docker://ghcr.io/lincbrain/linc-convert:latest
   ```

   This will create a `.sif` file (e.g., `linc-convert_latest.sif`).

### Usage

Run `linc-convert` commands using Apptainer/Singularity:

```bash
apptainer exec linc-convert_latest.sif linc-convert --help
```

To convert files, bind mount your data directories:

```bash
apptainer exec \
  --bind /path/to/input:/input \
  --bind /path/to/output:/output \
  linc-convert_latest.sif \
  linc-convert psoct single_volume /input/file.mat \
    --key Psi_ObsLSQ \
    --out /output/file.nii.zarr
```

### Building from Source

If you need to build the container from source:

1. Clone the repository:
   ```bash
   git clone https://github.com/lincbrain/linc-convert.git
   cd linc-convert
   ```

2. Build with Apptainer:
   ```bash
   apptainer build linc-convert.sif docker://ghcr.io/lincbrain/linc-convert:latest
   ```

   Or build with Singularity:
   ```bash
   singularity build linc-convert.sif docker://ghcr.io/lincbrain/linc-convert:latest
   ```

## Notes

- Ensure that input and output directories are properly mounted/bound when using containers
- File paths inside the container should use the mounted paths (e.g., `/input`, `/output`)
- For HPC environments, Apptainer/Singularity is often preferred over Docker
- Container images include all optional dependencies (`[all]` extras)

