# PSOCT Stitching Tile Configuration Quick Reference Guide

This guide explains how to use `generate_tile_config` to create tile configuration files for non-Fiji stitching with `linc_convert`.

## Overview

The `generate_tile_config` command generates YAML configuration files that describe the layout and positions of image tiles for mosaic stitching. These configurations can be used with `linc_convert`'s `psoct mosaic` command.

## Grid Layout and Terminology

### Grid Layout Example (3 columns × 2 rows)

```
    ┌─────────┬─────────┬─────────┐
    │  Tile 1 │  Tile 2 │  Tile 3 │  ← Row 0 (top row)
    │  (0,0)  │ (200,0) │ (400,0) │
    │         │         │         │
    │         │         │         │
    │         │         │         │
    ├─────────┼─────────┼─────────┤
    │  Tile 4 │  Tile 5 │  Tile 6 │  ← Row 1 (bottom row)
    │ (0,300) │(200,300)│(400,300)│
    │         │         │         │
    │         │         │         │
    │         │         │         │
    └─────────┴─────────┴─────────┘
      Col 0    Col 1    Col 2
```

### Terminology

- **Columns**: Number of tiles horizontally (x-direction). In the example above, there are 3 columns.
- **Rows**: Number of tiles vertically (y-direction). In the example above, there are 2 rows.
- **Tile Size**: Dimensions of each individual tile in pixels.
  - `tile_size_x`: Width of each tile (horizontal dimension)
  - `tile_size_y`: Height of each tile (vertical dimension)
  - For the mosaic above, `tile_size_x=200`, `tile_size_y=300`, `num_columns=3`, `num_rows=2`.

### Grid Patterns (tile numbering for 3×2 grid)

The diagrams below show how tiles are numbered for each grid type and order combination. Numbers indicate the sequence in which tiles are processed.

#### Row-by-row

```
  right-down:    left-down:    right-up:      left-up:
  ┌───┬───┬───┐  ┌───┬───┬───┐  ┌───┬───┬───┐  ┌───┬───┬───┐
  │ 1 │ 2 │ 3 │  │ 3 │ 2 │ 1 │  │ 4 │ 5 │ 6 │  │ 6 │ 5 │ 4 │
  ├───┼───┼───┤  ├───┼───┼───┤  ├───┼───┼───┤  ├───┼───┼───┤
  │ 4 │ 5 │ 6 │  │ 6 │ 5 │ 4 │  │ 1 │ 2 │ 3 │  │ 3 │ 2 │ 1 │
  └───┴───┴───┘  └───┴───┴───┘  └───┴───┴───┘  └───┴───┴───┘
```

#### Column-by-column

```
  down-right:    down-left:     up-right:      up-left:
  ┌───┬───┬───┐  ┌───┬───┬───┐  ┌───┬───┬───┐  ┌───┬───┬───┐
  │ 1 │ 3 │ 5 │  │ 5 │ 3 │ 1 │  │ 2 │ 4 │ 6 │  │ 6 │ 4 │ 2 │
  ├───┼───┼───┤  ├───┼───┼───┤  ├───┼───┼───┤  ├───┼───┼───┤
  │ 2 │ 4 │ 6 │  │ 6 │ 4 │ 2 │  │ 1 │ 3 │ 5 │  │ 5 │ 3 │ 1 │
  └───┴───┴───┘  └───┴───┴───┘  └───┴───┴───┘  └───┴───┴───┘
```

#### Snake-by-rows

```
  right-down:    left-down:     right-up:      left-up:
  ┌───┬───┬───┐  ┌───┬───┬───┐  ┌───┬───┬───┐  ┌───┬───┬───┐
  │ 1 │ 2 │ 3 │  │ 3 │ 2 │ 1 │  │ 6 │ 5 │ 4 │  │ 4 │ 5 │ 6 │
  ├───┼───┼───┤  ├───┼───┼───┤  ├───┼───┼───┤  ├───┼───┼───┤
  │ 6 │ 5 │ 4 │  │ 4 │ 5 │ 6 │  │ 1 │ 2 │ 3 │  │ 3 │ 2 │ 1 │
  └───┴───┴───┘  └───┴───┴───┘  └───┴───┴───┘  └───┴───┴───┘
```

#### Snake-by-columns

```
  down-right:    down-left:     up-right:      up-left:
  ┌───┬───┬───┐  ┌───┬───┬───┐  ┌───┬───┬───┐  ┌───┬───┬───┐
  │ 1 │ 4 │ 5 │  │ 5 │ 4 │ 1 │  │ 2 │ 3 │ 6 │  │ 6 │ 3 │ 2 │
  ├───┼───┼───┤  ├───┼───┼───┤  ├───┼───┼───┤  ├───┼───┼───┤
  │ 2 │ 3 │ 6 │  │ 6 │ 3 │ 2 │  │ 1 │ 4 │ 5 │  │ 5 │ 4 │ 1 │
  └───┴───┴───┘  └───┴───┴───┘  └───┴───┴───┘  └───┴───┴───┘
```

## Step 1: Generate Tile Configuration

### Basic Usage

```bash
linc-convert psoct generate_tile_config \
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
linc-convert psoct generate_tile_config \
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
  - Or use `--tile-size-x` (default: 200) and `--tile-size-y` (default: 300) for different dimensions
- `--base-dir`: Base directory where tile files are located (default: `./`)
- `--naming-format`: Format string for tile filenames (default: `mosaic_001_tile_{tile_number:04d}_aip.nii`)
  - Use `{tile_number}` or `{tile_number:03d}` for zero-padded numbers
  - Example: `tile_{tile_number:04d}.nii` → `tile_0001.nii`
- `--output` or `-o`: Output YAML file path (default: `tile_config.yaml`)

#### Grid Configuration

- `--grid-type`: Grid pattern type
  - `row-by-row`: Images arranged line by line
  - `column-by-column`: Images arranged column by column
  - `snake-by-rows`: Snake pattern across rows
  - `snake-by-columns`: Snake pattern down columns (default)

- `--order`: Order direction (depends on grid type)
  
  **For `row-by-row` and `snake-by-rows`:**
  - `right-down`: Start top-left, go right (default when order not specified)
  - `left-down`: Start top-right, go left
  - `right-up`: Start bottom-left, go right
  - `left-up`: Start bottom-right, go left
  
  **For `column-by-column` and `snake-by-columns`:**
  - `down-right`: Start top-left, go down (default when order not specified)
  - `down-left`: Start top-right, go down
  - `up-right`: Start bottom-left, go up
  - `up-left`: Start bottom-right, go up

- `--overlap-percentage`: Overlap as fraction (0.0-1.0, default: 0.0)
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
linc-convert psoct generate_tile_config \
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
linc-convert psoct mosaic stitch.yaml \
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
   - Compare your tile numbering with the diagrams above to find the matching pattern

3. **Common patterns:**
   - **Microscopy column scans**: Usually `column-by-column` with `down-right` or `down-left`
   - **Snake pattern scans**: `snake-by-columns` or `snake-by-rows` (see diagrams above)
   - **Sequential row scans**: `row-by-row` with `right-down` or `left-down`

**Tip**: Use the grid pattern diagrams above to visually match your tile numbering sequence. The numbers in the diagrams show the order in which tiles are processed, which should match your tile filenames.

## Tips

1. **Overlap consistency**: Use the same overlap percentage in both `generate_tile_config` and `psoct mosaic` commands.

2. **Naming format**: The `--naming-format` should match your actual tile filenames. Use:
   - `{tile_number}` for simple numbering: `tile_1.mat`
   - `{tile_number:03d}` for zero-padded: `tile_001.mat`
   - `{tile_number:04d}` for 4-digit padding: `tile_0001.mat`

3. **Base directory**: If tile paths in `--naming-format` are already absolute, you can set `--base-dir` to empty string or `.`.

4. **Verification**: The command prints the first and last few tiles. Verify these match your expected layout before proceeding to stitching.

## Troubleshooting

- **Wrong tile order**: Try different `--order` options or `--grid-type`
- **Incorrect spacing**: Check `--overlap-percentage` matches your actual tile overlap
- **File not found**: Verify `--base-dir` and `--naming-format` correctly construct file paths
- **Coordinate issues**: Ensure `--tile-size` matches your actual tile dimensions

## See Also

- Run `linc-convert psoct generate_tile_config --help` for full parameter list
- Run `linc-convert psoct mosaic --help` for linc_convert mosaic options

