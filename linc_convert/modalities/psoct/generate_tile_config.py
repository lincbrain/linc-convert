"""
Generate tile configuration YAML file for PSOCT mosaic tiles.

Grid Layout Example (3 columns × 2 rows):
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

Terminology:
- **Columns**: Number of tiles horizontally (x-direction). In the example above, there are 3 columns.
- **Rows**: Number of tiles vertically (y-direction). In the example above, there are 2 rows.
- **Tile Size**: Dimensions of each individual tile in pixels.
  - `tile_size_x`: Width of each tile (horizontal dimension)
  - `tile_size_y`: Height of each tile (vertical dimension)
  - For the mosaic above, tile_size_x=200, tile_size_y=300, num_columns=3, num_rows=2.

Grid Patterns (tile numbering for 3×2 grid):

Row-by-row:
  right-down:    left-down:    right-up:      left-up:
  ┌───┬───┬───┐  ┌───┬───┬───┐  ┌───┬───┬───┐  ┌───┬───┬───┐
  │ 1 │ 2 │ 3 │  │ 3 │ 2 │ 1 │  │ 4 │ 5 │ 6 │  │ 6 │ 5 │ 4 │
  ├───┼───┼───┤  ├───┼───┼───┤  ├───┼───┼───┤  ├───┼───┼───┤
  │ 4 │ 5 │ 6 │  │ 6 │ 5 │ 4 │  │ 1 │ 2 │ 3 │  │ 3 │ 2 │ 1 │
  └───┴───┴───┘  └───┴───┴───┘  └───┴───┴───┘  └───┴───┴───┘

Column-by-column:
  down-right:    down-left:     up-right:      up-left:
  ┌───┬───┬───┐  ┌───┬───┬───┐  ┌───┬───┬───┐  ┌───┬───┬───┐
  │ 1 │ 3 │ 5 │  │ 5 │ 3 │ 1 │  │ 2 │ 4 │ 6 │  │ 6 │ 4 │ 2 │
  ├───┼───┼───┤  ├───┼───┼───┤  ├───┼───┼───┤  ├───┼───┼───┤
  │ 2 │ 4 │ 6 │  │ 6 │ 4 │ 2 │  │ 1 │ 3 │ 5 │  │ 5 │ 3 │ 1 │
  └───┴───┴───┘  └───┴───┴───┘  └───┴───┴───┘  └───┴───┴───┘

Snake-by-rows:
  right-down:    left-down:     right-up:      left-up:
  ┌───┬───┬───┐  ┌───┬───┬───┐  ┌───┬───┬───┐  ┌───┬───┬───┐
  │ 1 │ 2 │ 3 │  │ 3 │ 2 │ 1 │  │ 6 │ 5 │ 4 │  │ 4 │ 5 │ 6 │
  ├───┼───┼───┤  ├───┼───┼───┤  ├───┼───┼───┤  ├───┼───┼───┤
  │ 6 │ 5 │ 4 │  │ 4 │ 5 │ 6 │  │ 1 │ 2 │ 3 │  │ 3 │ 2 │ 1 │
  └───┴───┴───┘  └───┴───┴───┘  └───┴───┴───┘  └───┴───┴───┘

Snake-by-columns:
  down-right:    down-left:     up-right:      up-left:
  ┌───┬───┬───┐  ┌───┬───┬───┐  ┌───┬───┬───┐  ┌───┬───┬───┐
  │ 1 │ 4 │ 5 │  │ 5 │ 4 │ 1 │  │ 2 │ 3 │ 6 │  │ 6 │ 3 │ 2 │
  ├───┼───┼───┤  ├───┼───┼───┤  ├───┼───┼───┤  ├───┼───┼───┤
  │ 2 │ 3 │ 6 │  │ 6 │ 3 │ 2 │  │ 1 │ 4 │ 5 │  │ 5 │ 4 │ 1 │
  └───┴───┴───┘  └───┴───┴───┘  └───┴───┴───┘  └───┴───┴───┘
"""

from typing import Annotated, List, Optional

import cyclopts
import yaml
from cyclopts import Parameter

from linc_convert.modalities.psoct.cli import psoct

generate_tile_config_cmd = cyclopts.App(name="generate_tile_config", help_format="markdown")
psoct.command(generate_tile_config_cmd)


def _calculate_spacing(tile_size: int, overlap: float) -> float:
    """
    Calculate tile spacing accounting for overlap.

    Parameters
    ----------
    tile_size : int
        Size of tile in pixels
    overlap : float
        Overlap percentage (0.0-1.0)

    Returns
    -------
    float
        Spacing between tile centers
    """
    return tile_size * (1.0 - overlap)


def _format_filepath(naming_format: str, tile_number: int) -> str:
    """
    Format filepath using naming format and tile number.

    Parameters
    ----------
    naming_format : str
        Format string for tile filenames
    tile_number : int
        Tile number

    Returns
    -------
    str
        Formatted filepath
    """
    try:
        return naming_format.format(tile_number=tile_number)
    except (KeyError, ValueError):
        return naming_format.replace("{tile_number}", str(tile_number))


def _create_tile_entry(tile_number: int, x: float, y: float, naming_format: str) -> dict:
    """
    Create a tile dictionary entry.

    Parameters
    ----------
    tile_number : int
        Tile number
    x : float
        X coordinate
    y : float
        Y coordinate
    naming_format : str
        Format string for tile filenames

    Returns
    -------
    dict
        Tile dictionary entry
    """
    return {
        "filepath": _format_filepath(naming_format, tile_number),
        "tile_number": tile_number,
        "x": float(x),
        "y": float(y)
    }


def _generate_row_by_row(
    num_columns: int,
    num_rows: int,
    tile_size_x: int,
    tile_size_y: int,
    overlap_x: float,
    overlap_y: float,
    order: str,
    naming_format: str
) -> List[dict]:
    """
    Generate tiles in row-by-row pattern.

    Parameters
    ----------
    num_columns : int
        Number of columns
    num_rows : int
        Number of rows
    tile_size_x : int
        Tile size in x direction
    tile_size_y : int
        Tile size in y direction
    overlap_x : float
        Overlap percentage in x direction
    overlap_y : float
        Overlap percentage in y direction
    order : str
        Order direction (right-down, left-down, right-up, left-up)
    naming_format : str
        Format string for tile filenames

    Returns
    -------
    List[dict]
        List of tile dictionaries
    """
    tiles = []
    tile_number = 1

    spacing_x = _calculate_spacing(tile_size_x, overlap_x)
    spacing_y = _calculate_spacing(tile_size_y, overlap_y)

    # Determine row and column iteration order based on order parameter
    if order == "right-down":
        row_range = range(num_rows)  # Top to bottom
        col_range = range(num_columns)  # Left to right
    elif order == "left-down":
        row_range = range(num_rows)  # Top to bottom
        col_range = range(num_columns - 1, -1, -1)  # Right to left
    elif order == "right-up":
        row_range = range(num_rows - 1, -1, -1)  # Bottom to top
        col_range = range(num_columns)  # Left to right
    elif order == "left-up":
        row_range = range(num_rows - 1, -1, -1)  # Bottom to top
        col_range = range(num_columns - 1, -1, -1)  # Right to left
    else:
        raise ValueError(f"Invalid order for row-by-row: {order}")

    for row_idx in row_range:
        y = row_idx * spacing_y
        for col_idx in col_range:
            x = col_idx * spacing_x
            tiles.append(_create_tile_entry(tile_number, x, y, naming_format))
            tile_number += 1

    return tiles


def _generate_column_by_column(
    num_columns: int,
    num_rows: int,
    tile_size_x: int,
    tile_size_y: int,
    overlap_x: float,
    overlap_y: float,
    order: str,
    naming_format: str
) -> List[dict]:
    """
    Generate tiles in column-by-column pattern.

    Parameters
    ----------
    num_columns : int
        Number of columns
    num_rows : int
        Number of rows
    tile_size_x : int
        Tile size in x direction
    tile_size_y : int
        Tile size in y direction
    overlap_x : float
        Overlap percentage in x direction
    overlap_y : float
        Overlap percentage in y direction
    order : str
        Order direction (down-right, down-left, up-right, up-left)
    naming_format : str
        Format string for tile filenames

    Returns
    -------
    List[dict]
        List of tile dictionaries
    """
    tiles = []
    tile_number = 1

    spacing_x = _calculate_spacing(tile_size_x, overlap_x)
    spacing_y = _calculate_spacing(tile_size_y, overlap_y)

    # Determine row and column iteration order based on order parameter
    if order == "down-right":
        col_range = range(num_columns)  # Left to right
        row_range = range(num_rows)  # Top to bottom
    elif order == "down-left":
        col_range = range(num_columns - 1, -1, -1)  # Right to left
        row_range = range(num_rows)  # Top to bottom
    elif order == "up-right":
        col_range = range(num_columns)  # Left to right
        row_range = range(num_rows - 1, -1, -1)  # Bottom to top
    elif order == "up-left":
        col_range = range(num_columns - 1, -1, -1)  # Right to left
        row_range = range(num_rows - 1, -1, -1)  # Bottom to top
    else:
        raise ValueError(f"Invalid order for column-by-column: {order}")

    for col_idx in col_range:
        x = col_idx * spacing_x
        for row_idx in row_range:
            y = row_idx * spacing_y
            tiles.append(_create_tile_entry(tile_number, x, y, naming_format))
            tile_number += 1

    return tiles


def _generate_snake_by_rows(
    num_columns: int,
    num_rows: int,
    tile_size_x: int,
    tile_size_y: int,
    overlap_x: float,
    overlap_y: float,
    order: str,
    naming_format: str
) -> List[dict]:
    """
    Generate tiles in snake-by-rows pattern.

    Parameters
    ----------
    num_columns : int
        Number of columns
    num_rows : int
        Number of rows
    tile_size_x : int
        Tile size in x direction
    tile_size_y : int
        Tile size in y direction
    overlap_x : float
        Overlap percentage in x direction
    overlap_y : float
        Overlap percentage in y direction
    order : str
        Order direction (right-down, left-down, right-up, left-up)
    naming_format : str
        Format string for tile filenames

    Returns
    -------
    List[dict]
        List of tile dictionaries
    """
    tiles = []
    tile_number = 1

    spacing_x = _calculate_spacing(tile_size_x, overlap_x)
    spacing_y = _calculate_spacing(tile_size_y, overlap_y)

    # Determine row iteration order and starting direction based on order parameter
    if order == "right-down":
        row_range = range(num_rows)  # Top to bottom
        first_row_goes_left = True  # First row (top) goes left-to-right
    elif order == "left-down":
        row_range = range(num_rows)  # Top to bottom
        first_row_goes_left = False  # First row (top) goes right-to-left
    elif order == "right-up":
        row_range = range(num_rows - 1, -1, -1)  # Bottom to top
        first_row_goes_left = True  # First row (bottom) goes left-to-right
    elif order == "left-up":
        row_range = range(num_rows - 1, -1, -1)  # Bottom to top
        first_row_goes_left = False  # First row (bottom) goes right-to-left
    else:
        raise ValueError(f"Invalid order for snake-by-rows: {order}")

    for row_position, row_idx in enumerate(row_range):
        y = row_idx * spacing_y
        # Alternate direction for each row based on position in sequence
        if (row_position % 2 == 0) == first_row_goes_left:
            col_range = range(num_columns)  # Left to right
        else:
            col_range = range(num_columns - 1, -1, -1)  # Right to left

        for col_idx in col_range:
            x = col_idx * spacing_x
            tiles.append(_create_tile_entry(tile_number, x, y, naming_format))
            tile_number += 1

    return tiles


def _generate_snake_by_columns(
    num_columns: int,
    num_rows: int,
    tile_size_x: int,
    tile_size_y: int,
    overlap_x: float,
    overlap_y: float,
    order: str,
    naming_format: str
) -> List[dict]:
    """
    Generate tiles in snake-by-columns pattern.

    Parameters
    ----------
    num_columns : int
        Number of columns
    num_rows : int
        Number of rows
    tile_size_x : int
        Tile size in x direction
    tile_size_y : int
        Tile size in y direction
    overlap_x : float
        Overlap percentage in x direction
    overlap_y : float
        Overlap percentage in y direction
    order : str
        Order direction (down-right, down-left, up-right, up-left)
    naming_format : str
        Format string for tile filenames

    Returns
    -------
    List[dict]
        List of tile dictionaries
    """
    tiles = []
    tile_number = 1

    spacing_x = _calculate_spacing(tile_size_x, overlap_x)
    spacing_y = _calculate_spacing(tile_size_y, overlap_y)

    # Determine column iteration order and starting direction based on order parameter
    if order == "down-right":
        col_range = range(num_columns)  # Left to right
        first_col_goes_down = True  # First column (left) goes top-to-bottom
    elif order == "down-left":
        col_range = range(num_columns - 1, -1, -1)  # Right to left
        first_col_goes_down = True  # First column (right) goes top-to-bottom
    elif order == "up-right":
        col_range = range(num_columns)  # Left to right
        first_col_goes_down = False  # First column (left) goes bottom-to-top
    elif order == "up-left":
        col_range = range(num_columns - 1, -1, -1)  # Right to left
        first_col_goes_down = False  # First column (right) goes bottom-to-top
    else:
        raise ValueError(f"Invalid order for snake-by-columns: {order}")

    for col_position, col_idx in enumerate(col_range):
        x = col_idx * spacing_x
        # Alternate direction for each column based on position in sequence
        if (col_position % 2 == 0) == first_col_goes_down:
            row_range = range(num_rows)  # Top to bottom
        else:
            row_range = range(num_rows - 1, -1, -1)  # Bottom to top

        for row_idx in row_range:
            y = row_idx * spacing_y
            tiles.append(_create_tile_entry(tile_number, x, y, naming_format))
            tile_number += 1

    return tiles


@generate_tile_config_cmd.default
def generate_tile_config(
    columns: Annotated[int, Parameter(name=["--columns"])] = 20,
    rows: Annotated[int, Parameter(name=["--rows"])] = 28,
    tile_size_x: Annotated[int, Parameter(name=["--tile-size-x"])] = 200,
    tile_size_y: Annotated[int, Parameter(name=["--tile-size-y"])] = 300,
    tile_size: Annotated[Optional[int], Parameter(name=["--tile-size"])] = None,
    base_dir: Annotated[str, Parameter(name=["--base-dir"])] = "./",
    naming_format: Annotated[str, Parameter(name=["--naming-format"])] = "mosaic_001_tile_{tile_number:04d}_aip.nii",
    out: Annotated[str, Parameter(name=["--output", "-o"])] = "tile_config.yaml",
    grid_type: Annotated[str, Parameter(name=["--grid-type"])] = "snake-by-columns",
    order: Annotated[Optional[str], Parameter(name=["--order"])] = None,
    overlap_percentage: Annotated[float, Parameter(name=["--overlap-percentage"])] = 0.0,
) -> None:
    """
    Generate tile configuration YAML file for Grid/Collection Stitching.

    This command generates a YAML configuration file that specifies the layout
    and positioning of tiles in a mosaic grid. It supports multiple grid patterns
    and ordering schemes.

    Parameters
    ----------
    columns : int
        Number of columns (default: 20)
    rows : int
        Number of rows (default: 28)
    tile_size_x : int
        Tile size in x direction (pixels, default: 200)
    tile_size_y : int
        Tile size in y direction (pixels, default: 300)
    tile_size : int, optional
        Tile size in both x and y directions (pixels). If specified, overrides
        both --tile-size-x and --tile-size-y
    base_dir : str
        Base directory path for the tiles (default: ./)
    naming_format : str
        Format string for tile filenames. Use {tile_number} or {tile_number:04d}
        for tile number substitution (default: mosaic_001_tile_{tile_number:04d}_aip.nii)
    out : str
        Output YAML file path (default: tile_config.yaml)
    grid_type : str
        Grid type. Must be one of: "row-by-row", "column-by-column", "snake-by-rows",
        "snake-by-columns" (default: snake-by-columns)

        Grid Types:
        - row-by-row: Images arranged in a grid, one line after the other
        - column-by-column: Images arranged in a grid, one column after the other
        - snake-by-rows: Images arranged in a grid, snaking across rows
        - snake-by-columns: Images arranged in a grid, snaking down columns
    order : str, optional
        Order direction. If not specified, defaults to "right-down" for row-by-row/snake-by-rows,
        or "down-right" for column-by-column/snake-by-columns.

        For row-by-row and snake-by-rows:
        - right-down: Start top-left, go right
        - left-down: Start top-right, go left
        - right-up: Start bottom-left, go right
        - left-up: Start bottom-right, go left

        For column-by-column and snake-by-columns:
        - down-right: Start top-left, go down
        - down-left: Start top-right, go down
        - up-right: Start bottom-left, go up
        - up-left: Start bottom-right, go up
    overlap_percentage : float
        Overlap percentage (0.0-1.0) that affects tile spacing (default: 0.0)

    Examples
    --------
    Generate a 20x28 grid with default settings:
    ```bash
    linc-convert psoct generate_tile_config
    ```

    Generate a 10x10 grid with snake-by-rows pattern:
    ```bash
    linc-convert psoct generate_tile_config --columns 10 --rows 10 --grid-type snake-by-rows
    ```

    Generate with custom tile size and overlap:
    ```bash
    linc-convert psoct generate_tile_config --tile-size 512 --overlap-percentage 0.1
    ```
    """
    # Handle --tile-size as a convenience option that sets both x and y
    if tile_size is not None:
        tile_size_x = tile_size
        tile_size_y = tile_size

    # Set default order based on grid_type if not specified
    if order is None:
        if grid_type in ["row-by-row", "snake-by-rows"]:
            order = "right-down"
        else:  # column-by-column or snake-by-columns
            order = "down-right"

    # Validate grid_type and order
    valid_grid_types = ["row-by-row", "column-by-column", "snake-by-rows", "snake-by-columns"]
    if grid_type not in valid_grid_types:
        raise ValueError(f"Invalid grid_type: {grid_type}. Must be one of {valid_grid_types}")

    row_based = grid_type in ["row-by-row", "snake-by-rows"]
    valid_orders = (
        ["right-down", "left-down", "right-up", "left-up"] if row_based
        else ["down-right", "down-left", "up-right", "up-left"]
    )
    if order not in valid_orders:
        raise ValueError(f"Invalid order for {grid_type}: {order}. Must be one of {valid_orders}")

    # Validate overlap_percentage
    if not 0.0 <= overlap_percentage <= 1.0:
        raise ValueError(f"overlap_percentage must be between 0.0 and 1.0, got {overlap_percentage}")

    # Generate tiles based on grid type
    overlap = overlap_percentage
    generator_map = {
        "row-by-row": _generate_row_by_row,
        "column-by-column": _generate_column_by_column,
        "snake-by-rows": _generate_snake_by_rows,
        "snake-by-columns": _generate_snake_by_columns,
    }
    tiles = generator_map[grid_type](
        columns, rows, tile_size_x, tile_size_y,
        overlap, overlap, order, naming_format
    )

    # Build metadata
    metadata = {
        "base_dir": base_dir,
        "scan_resolution": [0.01, 0.01]
    }

    # Add overlap to metadata if specified
    if overlap_percentage > 0.0:
        metadata["tile_overlap"] = overlap_percentage

    config = {
        "metadata": metadata,
        "tiles": tiles
    }

    # Write to YAML file
    with open(out, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)

    print(f"Generated tile configuration with {len(tiles)} tiles")
    print(f"Grid type: {grid_type}, Order: {order}")
    if overlap_percentage > 0.0:
        print(f"Overlap: {overlap_percentage * 100:.1f}%")
    print(f"Configuration written to: {out}")
    print(f"\nFirst few tiles:")
    for tile in tiles[:5]:
        print(f"  Tile {tile['tile_number']}: x={tile['x']}, y={tile['y']}, file={tile['filepath']}")
    print(f"\nLast few tiles:")
    for tile in tiles[-5:]:
        print(f"  Tile {tile['tile_number']}: x={tile['x']}, y={tile['y']}, file={tile['filepath']}")

