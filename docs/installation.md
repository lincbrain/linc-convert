# Installation

The `linc-convert` package is available on [PyPI](https://pypi.org/project/linc-convert/) and can be installed with pip or pipx.

## Installation with pip

1. (Optional) Create a new environment
   ```bash
   conda create -n linc-convert python=3.11
   ```
2. (Optional) Activate the environment
   ```bash
   conda activate linc-convert
   ```
3. Install the `linc-convert` package in your environment
   ```bash
   pip install "linc-convert[all]"
   ```
   If you don't want to install all dependencies, you can install the dependencies for your specific use case. Replace `all` in the above command with:
   - `df` for dark-field microscopy
   - `lsm` for light-sheet microscopy
   - `psoct` for polarization-sensitive optical coherence tomography
   - `wk` for Webknossos annotations
   - `ts` for the TensorStore backend

   To install the latest development version from GitHub instead, run:
   ```bash
   pip install "linc-convert[all] @ git+https://github.com/lincbrain/linc-convert.git@main"
   ```
4. Run the command-line interface to ensure that installation was successful
   ```bash
   linc-convert --help
   ```
5. View the list of arguments for a modality by running, for example:
   ```bash
   linc-convert psoct --help
   ```
6. View the full list of parameters by running, for example:
   ```bash
   linc-convert psoct single_volume --help
   ```

## Installation with pipx

[pipx](https://pipx.pypa.io/) installs Python applications in isolated environments, making it ideal for command-line tools like `linc-convert`.

1. Install pipx (if not already installed):
   ```bash
   python -m pip install --user pipx
   python -m pipx ensurepath
   ```

2. Install `linc-convert` with pipx:
   ```bash
   pipx install "linc-convert[all]"
   ```
   To install the latest development version from GitHub instead, run:
   ```bash
   pipx install "linc-convert[all] @ git+https://github.com/lincbrain/linc-convert.git@main"
   ```
3. Verify the installation:
   ```bash
   linc-convert --help
   ```
