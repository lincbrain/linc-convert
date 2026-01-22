#!/usr/bin/env python3
"""
Generate tile configuration YAML file for PSOCT mosaic tiles.

Supports multiple grid/collection stitching patterns:
- Grid: row-by-row (Right & Down, Left & Down, Right & Up, Left & Up)
- Grid: column-by-column (Down & Right, Down & Left, Up & Right, Up & Left)
- Grid: snake by rows (Right & Down, Left & Down, Right & Up, Left & Up)
- Grid: snake by columns (Down & Right, Down & Left, Up & Right, Up & Left)
"""

import yaml
from typing import List, Tuple, Optional


def _calculate_spacing(tile_size: int, overlap: float) -> float:
    """
    Calculate tile spacing accounting for overlap.
    
    Args:
        tile_size: Size of tile in pixels
        overlap: Overlap percentage (0.0-1.0)
    
    Returns:
        Spacing between tile centers
    """
    return tile_size * (1.0 - overlap)


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
    
    Args:
        num_columns: Number of columns
        num_rows: Number of rows
        tile_size_x: Tile size in x direction
        tile_size_y: Tile size in y direction
        overlap_x: Overlap percentage in x direction
        overlap_y: Overlap percentage in y direction
        order: Order direction (right-down, left-down, right-up, left-up)
        naming_format: Format string for tile filenames
    
    Returns:
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
            
            try:
                filepath = naming_format.format(tile_number=tile_number)
            except (KeyError, ValueError):
                filepath = naming_format.replace("{tile_number}", str(tile_number))
            
            tiles.append({
                "filepath": filepath,
                "tile_number": tile_number,
                "x": float(x),
                "y": float(y)
            })
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
    
    Args:
        num_columns: Number of columns
        num_rows: Number of rows
        tile_size_x: Tile size in x direction
        tile_size_y: Tile size in y direction
        overlap_x: Overlap percentage in x direction
        overlap_y: Overlap percentage in y direction
        order: Order direction (down-right, down-left, up-right, up-left)
        naming_format: Format string for tile filenames
    
    Returns:
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
            
            try:
                filepath = naming_format.format(tile_number=tile_number)
            except (KeyError, ValueError):
                filepath = naming_format.replace("{tile_number}", str(tile_number))
            
            tiles.append({
                "filepath": filepath,
                "tile_number": tile_number,
                "x": float(x),
                "y": float(y)
            })
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
    
    Args:
        num_columns: Number of columns
        num_rows: Number of rows
        tile_size_x: Tile size in x direction
        tile_size_y: Tile size in y direction
        overlap_x: Overlap percentage in x direction
        overlap_y: Overlap percentage in y direction
        order: Order direction (right-down, left-down, right-up, left-up)
        naming_format: Format string for tile filenames
    
    Returns:
        List of tile dictionaries
    """
    tiles = []
    tile_number = 1
    
    spacing_x = _calculate_spacing(tile_size_x, overlap_x)
    spacing_y = _calculate_spacing(tile_size_y, overlap_y)
    
    # Determine row iteration order based on order parameter
    if order == "right-down":
        row_range = range(num_rows)  # Top to bottom
        start_left = True
    elif order == "left-down":
        row_range = range(num_rows)  # Top to bottom
        start_left = False
    elif order == "right-up":
        row_range = range(num_rows - 1, -1, -1)  # Bottom to top
        start_left = True
    elif order == "left-up":
        row_range = range(num_rows - 1, -1, -1)  # Bottom to top
        start_left = False
    else:
        raise ValueError(f"Invalid order for snake-by-rows: {order}")
    
    for row_idx in row_range:
        y = row_idx * spacing_y
        # Alternate direction for each row
        if (row_idx % 2 == 0) == start_left:
            col_range = range(num_columns)  # Left to right
        else:
            col_range = range(num_columns - 1, -1, -1)  # Right to left
        
        for col_idx in col_range:
            x = col_idx * spacing_x
            
            try:
                filepath = naming_format.format(tile_number=tile_number)
            except (KeyError, ValueError):
                filepath = naming_format.replace("{tile_number}", str(tile_number))
            
            tiles.append({
                "filepath": filepath,
                "tile_number": tile_number,
                "x": float(x),
                "y": float(y)
            })
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
    
    Args:
        num_columns: Number of columns
        num_rows: Number of rows
        tile_size_x: Tile size in x direction
        tile_size_y: Tile size in y direction
        overlap_x: Overlap percentage in x direction
        overlap_y: Overlap percentage in y direction
        order: Order direction (down-right, down-left, up-right, up-left)
        naming_format: Format string for tile filenames
    
    Returns:
        List of tile dictionaries
    """
    tiles = []
    tile_number = 1
    
    spacing_x = _calculate_spacing(tile_size_x, overlap_x)
    spacing_y = _calculate_spacing(tile_size_y, overlap_y)
    
    # Determine column iteration order and starting direction based on order parameter
    if order == "down-right":
        col_range = range(num_columns)  # Left to right
        start_bottom = True  # Even columns start from bottom
    elif order == "down-left":
        col_range = range(num_columns - 1, -1, -1)  # Right to left
        start_bottom = True  # Even columns start from bottom
    elif order == "up-right":
        col_range = range(num_columns)  # Left to right
        start_bottom = False  # Even columns start from top
    elif order == "up-left":
        col_range = range(num_columns - 1, -1, -1)  # Right to left
        start_bottom = False  # Even columns start from top
    else:
        raise ValueError(f"Invalid order for snake-by-columns: {order}")
    
    for col_idx in col_range:
        x = col_idx * spacing_x
        # Alternate direction for each column
        # Original behavior: even columns go bottom-to-top, odd columns go top-to-bottom
        if (col_idx % 2 == 0) == start_bottom:
            row_range = range(num_rows - 1, -1, -1)  # Bottom to top
        else:
            row_range = range(num_rows)  # Top to bottom
        
        for row_idx in row_range:
            y = row_idx * spacing_y
            
            try:
                filepath = naming_format.format(tile_number=tile_number)
            except (KeyError, ValueError):
                filepath = naming_format.replace("{tile_number}", str(tile_number))
            
            tiles.append({
                "filepath": filepath,
                "tile_number": tile_number,
                "x": float(x),
                "y": float(y)
            })
            tile_number += 1
    
    return tiles


def generate_tile_config(
    num_columns: int = 20,
    num_rows: int = 28,
    tile_size_x: int = 280,
    tile_size_y: int = 280,
    base_dir: str = "/autofs/space/zircon_007/users/psoct-pipeline/sub-I80",
    naming_format: str = "mosaic_001_tile_{tile_number:04d}_mip.nii",
    output_file: str = "tile_config.yaml",
    grid_type: str = "snake-by-columns",
    order: str = "down-right",
    overlap_percentage: float = 0.0
):
    """
    Generate tile configuration YAML file.
    
    Args:
        num_columns: Number of columns (default: 20)
        num_rows: Number of rows (default: 28)
        tile_size_x: Size of each tile in x direction (pixels, default: 280)
        tile_size_y: Size of each tile in y direction (pixels, default: 280)
        base_dir: Base directory path for the tiles
        naming_format: Format string for tile filenames. Use {tile_number} or {tile_number:04d} 
                      for tile number substitution (default: "mosaic_001_tile_{tile_number:04d}_mip.nii")
        output_file: Output YAML file path
        grid_type: Grid type - one of: "row-by-row", "column-by-column", "snake-by-rows", 
                  "snake-by-columns" (default: "snake-by-columns")
        order: Order direction. For row-by-row and snake-by-rows: "right-down", "left-down", 
              "right-up", "left-up". For column-by-column and snake-by-columns: "down-right", 
              "down-left", "up-right", "up-left" (default: "down-right")
        overlap_percentage: Overlap percentage (0.0-1.0) that affects tile spacing 
                           (default: 0.0, i.e., no overlap)
    """
    # Validate grid_type
    valid_grid_types = ["row-by-row", "column-by-column", "snake-by-rows", "snake-by-columns"]
    if grid_type not in valid_grid_types:
        raise ValueError(f"Invalid grid_type: {grid_type}. Must be one of {valid_grid_types}")
    
    # Validate order based on grid_type
    if grid_type in ["row-by-row", "snake-by-rows"]:
        valid_orders = ["right-down", "left-down", "right-up", "left-up"]
        if order not in valid_orders:
            raise ValueError(f"Invalid order for {grid_type}: {order}. Must be one of {valid_orders}")
    else:  # column-by-column or snake-by-columns
        valid_orders = ["down-right", "down-left", "up-right", "up-left"]
        if order not in valid_orders:
            raise ValueError(f"Invalid order for {grid_type}: {order}. Must be one of {valid_orders}")
    
    # Validate overlap_percentage
    if not 0.0 <= overlap_percentage <= 1.0:
        raise ValueError(f"overlap_percentage must be between 0.0 and 1.0, got {overlap_percentage}")
    
    # Generate tiles based on grid type
    if grid_type == "row-by-row":
        tiles = _generate_row_by_row(
            num_columns, num_rows, tile_size_x, tile_size_y,
            overlap_percentage, overlap_percentage, order, naming_format
        )
    elif grid_type == "column-by-column":
        tiles = _generate_column_by_column(
            num_columns, num_rows, tile_size_x, tile_size_y,
            overlap_percentage, overlap_percentage, order, naming_format
        )
    elif grid_type == "snake-by-rows":
        tiles = _generate_snake_by_rows(
            num_columns, num_rows, tile_size_x, tile_size_y,
            overlap_percentage, overlap_percentage, order, naming_format
        )
    else:  # snake-by-columns
        tiles = _generate_snake_by_columns(
            num_columns, num_rows, tile_size_x, tile_size_y,
            overlap_percentage, overlap_percentage, order, naming_format
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
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)
    
    print(f"Generated tile configuration with {len(tiles)} tiles")
    print(f"Grid type: {grid_type}, Order: {order}")
    if overlap_percentage > 0.0:
        print(f"Overlap: {overlap_percentage * 100:.1f}%")
    print(f"Configuration written to: {output_file}")
    print(f"\nFirst few tiles:")
    for tile in tiles[:5]:
        print(f"  Tile {tile['tile_number']}: x={tile['x']}, y={tile['y']}, file={tile['filepath']}")
    print(f"\nLast few tiles:")
    for tile in tiles[-5:]:
        print(f"  Tile {tile['tile_number']}: x={tile['x']}, y={tile['y']}, file={tile['filepath']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate tile configuration YAML file for Grid/Collection Stitching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Grid Types:
  row-by-row        - Images arranged in a grid, one line after the other
  column-by-column  - Images arranged in a grid, one column after the other
  snake-by-rows     - Images arranged in a grid, snaking across rows
  snake-by-columns  - Images arranged in a grid, snaking down columns (default)

Order Directions:
  For row-by-row and snake-by-rows:
    right-down  - Start top-left, go right (default for row-by-row)
    left-down   - Start top-right, go left
    right-up    - Start bottom-left, go right
    left-up     - Start bottom-right, go left
  
  For column-by-column and snake-by-columns:
    down-right  - Start top-left, go down (default for snake-by-columns)
    down-left   - Start top-right, go down
    up-right    - Start bottom-left, go up
    up-left     - Start bottom-right, go up
        """
    )
    parser.add_argument("--columns", type=int, default=20, help="Number of columns (default: 20)")
    parser.add_argument("--rows", type=int, default=28, help="Number of rows (default: 28)")
    parser.add_argument("--tile-size-x", type=int, default=280, 
                       help="Tile size in x direction (pixels, default: 280)")
    parser.add_argument("--tile-size-y", type=int, default=280,
                       help="Tile size in y direction (pixels, default: 280)")
    parser.add_argument("--tile-size", type=int, default=None,
                       help="Tile size in both x and y directions (pixels, overrides --tile-size-x and --tile-size-y)")
    parser.add_argument("--base-dir", type=str, 
                       default="/autofs/space/zircon_007/users/psoct-pipeline/sub-I80",
                       help="Base directory path")
    parser.add_argument("--naming-format", type=str, 
                       default="mosaic_001_tile_{tile_number:04d}_mip.nii",
                       help="Format string for tile filenames. Use {tile_number} or {tile_number:04d} "
                            "for tile number substitution (default: mosaic_001_tile_{tile_number:04d}_mip.nii)")
    parser.add_argument("--output", type=str, default="tile_config.yaml",
                       help="Output YAML file path (default: tile_config.yaml)")
    parser.add_argument("--grid-type", type=str, default="snake-by-columns",
                       choices=["row-by-row", "column-by-column", "snake-by-rows", "snake-by-columns"],
                       help="Grid type (default: snake-by-columns)")
    parser.add_argument("--order", type=str, default=None,
                       help="Order direction. Defaults to 'right-down' for row-by-row/snake-by-rows, "
                            "'down-right' for column-by-column/snake-by-columns")
    parser.add_argument("--overlap-percentage", type=float, default=0.0,
                       help="Overlap percentage (0.0-1.0) that affects tile spacing (default: 0.0)")
    
    args = parser.parse_args()
    
    # Handle --tile-size as a convenience option that sets both x and y
    tile_size_x = args.tile_size if args.tile_size is not None else args.tile_size_x
    tile_size_y = args.tile_size if args.tile_size is not None else args.tile_size_y
    
    # Set default order based on grid_type if not specified
    if args.order is None:
        if args.grid_type in ["row-by-row", "snake-by-rows"]:
            args.order = "right-down"
        else:  # column-by-column or snake-by-columns
            args.order = "down-right"
    
    # Validate overlap_percentage
    if not 0.0 <= args.overlap_percentage <= 1.0:
        parser.error("--overlap-percentage must be between 0.0 and 1.0")
    
    generate_tile_config(
        num_columns=args.columns,
        num_rows=args.rows,
        tile_size_x=tile_size_x,
        tile_size_y=tile_size_y,
        base_dir=args.base_dir,
        naming_format=args.naming_format,
        output_file=args.output,
        grid_type=args.grid_type,
        order=args.order,
        overlap_percentage=args.overlap_percentage
    )
