# PSOCT Stitching Tile Configuration Quick Reference Guide

This guide explains how to use `generate_tile_config.py` to create tile configuration files for non-Fiji stitching with `linc_convert`.

## Overview

The `generate_tile_config.py` script generates YAML configuration files that describe the layout and positions of image tiles for mosaic stitching. These configurations can be used with `linc_convert`'s `psoct mosaic` command.

## Step 1: Generate Tile Configuration

### Basic Usage

```bash
python scripts/generate_tile_config.py \
    --columns <num_columns> \
    --rows <num_rows> \
    --tile-size <size> \
    --base-dir <base_directory> \
    --naming-format <format_string> \
    --output <output_yaml> \
    --grid-type <grid_type> \
    --order <order_direction> \
    --overlap-percentage <overlap>
```

### Common Example

```bash
python scripts/generate_tile_config.py \
    --columns 9 \
    --rows 11 \
    --tile-size 1000 \
    --base-dir /path/to/data/ \
    --naming-format slice_169_tile_{tile_number:03d}_Cross.mat \
    --output stitch.yaml \
    --grid-type column-by-column \
    --order down-left \
    --overlap-percentage 0.1
```

### Key Parameters

#### Required/Common Parameters

- `--columns`: Number of columns in the grid (default: 20)
- `--rows`: Number of rows in the grid (default: 28)
- `--tile-size`: Size of tiles in both x and y directions (pixels)
  - Or use `--tile-size-x` and `--tile-size-y` for different dimensions
- `--base-dir`: Base directory where tile files are located
- `--naming-format`: Format string for tile filenames
  - Use `{tile_number}` or `{tile_number:03d}` for zero-padded numbers
  - Example: `tile_{tile_number:04d}.nii` → `tile_0001.nii`
- `--output`: Output YAML file path (default: `tile_config.yaml`)

#### Grid Configuration

- `--grid-type`: Grid pattern type
  - `row-by-row`: Images arranged line by line
  - `column-by-column`: Images arranged column by column
  - `snake-by-rows`: Snake pattern across rows
  - `snake-by-columns`: Snake pattern down columns (default)

- `--order`: Order direction (depends on grid type)
  
  **For `row-by-row` and `snake-by-rows`:**
  - `right-down`: Start top-left, go right (default for row-by-row)
  - `left-down`: Start top-right, go left
  - `right-up`: Start bottom-left, go right
  - `left-up`: Start bottom-right, go left
  
  **For `column-by-column` and `snake-by-columns`:**
  - `down-right`: Start top-left, go down (default for snake-by-columns)
  - `down-left`: Start top-right, go down
  - `up-right`: Start bottom-left, go up
  - `up-left`: Start bottom-right, go up

- `--overlap-percentage`: Overlap as fraction (0.0-1.0)
  - Example: `0.1` = 10% overlap
  - Affects tile spacing in coordinate calculation
  - Stored in YAML metadata for stitching algorithms

### Grid Type Examples

#### Column-by-Column (Sequential Columns)
```bash
--grid-type column-by-column --order down-right
```
Tiles are processed column by column, going down each column before moving to the next.

#### Snake-by-Columns (Alternating Column Direction)
```bash
--grid-type snake-by-columns --order down-right
```
Tiles snake down columns: column 0 goes top-to-bottom, column 1 goes bottom-to-top, etc.

#### Row-by-Row (Sequential Rows)
```bash
--grid-type row-by-row --order right-down
```
Tiles are processed row by row, going left-to-right across each row.

#### Snake-by-Rows (Alternating Row Direction)
```bash
--grid-type snake-by-rows --order right-down
```
Tiles snake across rows: row 0 goes left-to-right, row 1 goes right-to-left, etc.

## Step 2: Use Configuration with linc_convert

After generating the tile configuration YAML, use it with `linc_convert`'s `psoct mosaic` command:

```bash
linc-convert psoct mosaic <tile_config.yaml> \
    --out <output.zarr> \
    --tile-overlap <overlap> \
    --voxel-size <x> --voxel-size <y> --voxel-size <z> \
    [additional options]
```

### Complete Example

```bash
# Step 1: Generate tile configuration
python scripts/generate_tile_config.py \
    --columns 9 \
    --rows 11 \
    --tile-size 1000 \
    --base-dir /path/to/data/ \
    --naming-format slice_169_tile_{tile_number:03d}_Cross.mat \
    --output stitch.yaml \
    --grid-type column-by-column \
    --order down-left \
    --overlap-percentage 0.1

# Step 2: Create mosaic
psoct mosaic stitch.yaml \
    --out aip.zarr \
    --zarr_version 3 \
    --shard 1024 \
    --chunk 256 \
    --driver tensorstore \
    --tile-overlap 0.1 \
    --overwrite \
    --voxel-size 5 --voxel-size 5 --voxel-size 5
```

### Common linc_convert Options

- `--out`: Output Zarr file path
- `--tile-overlap`: Overlap fraction (should match `--overlap-percentage` from step 1)
- `--voxel-size`: Voxel size in each dimension (specify three times for x, y, z)
- `--zarr_version`: Zarr format version (default: 2)
- `--shard`: Shard size for Zarr v3
- `--chunk`: Chunk size
- `--driver`: Storage driver (`tensorstore`, `zarr`, etc.)
- `--overwrite`: Overwrite existing output

## Determining Grid Type and Order

To determine the correct `--grid-type` and `--order`:

1. **Identify the scanning pattern:**
   - How were the tiles acquired? Column by column? Row by row?
   - Does the pattern alternate direction (snake pattern)?

2. **Check tile numbering:**
   - Look at your tile filenames to see the numbering sequence
   - The first few tiles indicate the starting position and direction

3. **Common patterns:**
   - **Microscopy column scans**: Usually `column-by-column` with `down-right` or `down-left`
   - **Snake pattern scans**: `snake-by-columns` or `snake-by-rows`
   - **Sequential row scans**: `row-by-row` with `right-down` or `left-down`

## Tips

1. **Overlap consistency**: Use the same overlap percentage in both `generate_tile_config.py` and `psoct mosaic` commands.

2. **Naming format**: The `--naming-format` should match your actual tile filenames. Use:
   - `{tile_number}` for simple numbering: `tile_1.mat`
   - `{tile_number:03d}` for zero-padded: `tile_001.mat`
   - `{tile_number:04d}` for 4-digit padding: `tile_0001.mat`

3. **Base directory**: If tile paths in `--naming-format` are already absolute, you can set `--base-dir` to empty string or `.`.

4. **Verification**: The script prints the first and last few tiles. Verify these match your expected layout before proceeding to stitching.

## Troubleshooting

- **Wrong tile order**: Try different `--order` options or `--grid-type`
- **Incorrect spacing**: Check `--overlap-percentage` matches your actual tile overlap
- **File not found**: Verify `--base-dir` and `--naming-format` correctly construct file paths
- **Coordinate issues**: Ensure `--tile-size` matches your actual tile dimensions

## See Also

- Run `python scripts/generate_tile_config.py --help` for full parameter list
- Run `psoct mosaic --help` for linc_convert mosaic options

