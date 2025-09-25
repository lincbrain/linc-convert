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
   pip install "linc-convert[all] @ git+https://github.com/lincbrain/linc-convert.git@main"
   ```
If you don't want to install all dependencies, you can install the dependencies for your specific use case. Replace `all` in the above command with:
      - `df` for dark-field microscopy
      - `lsm` for light-sheet microscopy
      - `psoct` for polarization-sensitive optical coherence tomography
      - `wk` for Webknossos annotations
      - `ts` for the TensorStore backend
4. Run the command-line interface to ensure that installation was successful
   ```bash
   linc-convert --help
   ```
