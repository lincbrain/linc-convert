1. Clone the repository and switch to the `zarr3` branch.

  ```bash
  git clone https://github.com/lincbrain/linc-convert.git
  cd linc-convert
  git checkout zarr3
  ```

2. Install the package in your environment.

  ```bash
  pip install ./
  ```

3. Run the CLI based on your workflow. For a single-volume PS-OCT conversion:

  ```bash
  linc-convert psoct single-volume <input> <output>
  ```

  View the full list of options and flags by running:

  ```bash
  linc-convert psoct single-volume --help
  ```