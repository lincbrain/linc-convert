1. Create a new environment.
   ```bash
   conda create -n linc-convert python=3.11
   ```

2. Activate the environment.
   ```bash
   conda activate linc-convert
   ```

3. Install the `linc-convert` package in your environment.
   ```bash
   pip install "linc-convert[all] @ git+https://github.com/lincbrain/linc-convert.git@main"
   ```

If you don't want to install all dependencies, you can install the dependencies for your specific use case. Replace `all` in the above command with:
- `df` for Dark-field microscopy
- `lsm` for Light-sheet microscopy
- `psoct` for polarization-sensitive optical coherence tomography
- `wk` for Webknossos annotations
- `ts` for the TensorStore backend

4. Run the command-line interface to ensure that installation was successful.
   ```bash
   linc-convert --help
   ```

5. View the list of arguments for a modality by running, for example:
   ```bash
   linc-convert psoct --help
   ```

And view the full list of parameters by running:
   ```bash
   linc-convert psoct single-volume --help
   ```
