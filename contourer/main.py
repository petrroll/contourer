import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.spatial import Delaunay


def load_point_cloud(filepath: Path) -> np.ndarray:
    """Load point cloud data from file.
    
    File format: Either X Y Z (3 columns) or ID X Y Z (4 columns), space-separated.
    Returns (N, 3) array with X, Y, Z columns.
    """
    # First, detect number of columns
    with open(filepath) as f:
        first_line = f.readline().strip()
    num_cols = len(first_line.split())
    
    if num_cols == 3:
        # X Y Z format
        data = np.loadtxt(filepath, usecols=(0, 1, 2))
    elif num_cols >= 4:
        # ID X Y Z format (ignore ID)
        data = np.loadtxt(filepath, usecols=(1, 2, 3))
    else:
        raise ValueError(f"Expected 3 or 4 columns, got {num_cols}")
    
    return data


def compute_triangle_edge_lengths(points_xy: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Compute edge lengths for each triangle.
    
    Returns (M, 3) array with lengths of edges for each triangle.
    """
    p0 = points_xy[triangles[:, 0]]
    p1 = points_xy[triangles[:, 1]]
    p2 = points_xy[triangles[:, 2]]
    
    edge0 = np.linalg.norm(p1 - p0, axis=1)
    edge1 = np.linalg.norm(p2 - p1, axis=1)
    edge2 = np.linalg.norm(p0 - p2, axis=1)
    
    return np.column_stack([edge0, edge1, edge2])


def create_triangulation_with_filter(
    points: np.ndarray, 
    max_distance: Optional[float] = None
) -> tuple[mtri.Triangulation, np.ndarray]:
    """Create Delaunay triangulation and filter long edges.
    
    Args:
        points: (N, 3) array with X, Y, Z coordinates
        max_distance: Maximum allowed edge length. If None, uses 3× median.
    
    Returns:
        Triangulation object and boolean mask for invalid triangles.
    """
    x, y = points[:, 0], points[:, 1]
    
    # Perform Delaunay triangulation on X-Y plane
    delaunay = Delaunay(points[:, :2])
    triangles = delaunay.simplices
    
    # Compute edge lengths
    edge_lengths = compute_triangle_edge_lengths(points[:, :2], triangles)
    
    # Determine max_distance threshold
    if max_distance is None:
        median_edge = np.median(edge_lengths)
        max_distance = 3.0 * median_edge
        print(f"Auto max-distance: {max_distance:.3f} (3× median edge: {median_edge:.3f})")
    
    # Create mask for triangles with any edge exceeding threshold
    max_edges_per_triangle = np.max(edge_lengths, axis=1)
    mask = max_edges_per_triangle > max_distance
    
    print(f"Filtered {np.sum(mask)} of {len(triangles)} triangles (edge > {max_distance:.3f})")
    
    # Create matplotlib triangulation with mask
    triangulation = mtri.Triangulation(x, y, triangles)
    triangulation.set_mask(mask)
    
    return triangulation, mask


def print_z_statistics(z_values: np.ndarray) -> dict:
    """Print and return Z value statistics."""
    stats = {
        'min': np.min(z_values),
        'max': np.max(z_values),
        'p10': np.percentile(z_values, 10),
        'p20': np.percentile(z_values, 20),
        'p50': np.percentile(z_values, 50),
        'p80': np.percentile(z_values, 80),
        'p90': np.percentile(z_values, 90),
    }
    
    print("\nZ Statistics:")
    print(f"  Min:  {stats['min']:.3f}")
    print(f"  Max:  {stats['max']:.3f}")
    print(f"  P20:  {stats['p20']:.3f}")
    print(f"  P50:  {stats['p50']:.3f}")
    print(f"  P80:  {stats['p80']:.3f}")
    
    return stats


def generate_auto_levels(z_stats: dict, num_levels: int = 50) -> list[float]:
    """Generate evenly-spaced contour levels between min and max."""
    levels = np.linspace(z_stats['min'], z_stats['max'], num_levels)
    return levels.tolist()


def align_to_interval(value: float, interval: float, mode: str = 'floor') -> float:
    """Align a value to the nearest interval boundary.
    
    Args:
        value: The value to align
        interval: The interval to align to
        mode: 'floor' for rounding down, 'ceil' for rounding up
    
    Returns:
        Aligned value
    """
    if mode == 'floor':
        return np.floor(value / interval) * interval
    else:
        return np.ceil(value / interval) * interval


def generate_interval_levels(
    z_stats: dict, 
    minor_interval: float,
    major_interval: Optional[float] = None
) -> tuple[list[float], list[float]]:
    """Generate contour levels based on interval granularity.
    
    Args:
        z_stats: Dictionary with 'min' and 'max' Z values
        minor_interval: Interval for minor contour lines (e.g., 0.2)
        major_interval: Interval for major contour lines (e.g., 1.0)
                       If None, defaults to 5× minor_interval
    
    Returns:
        Tuple of (all_levels, major_levels)
        - all_levels: All contour levels (minor + major)
        - major_levels: Only the major contour levels (for visualization)
    """
    if major_interval is None:
        major_interval = minor_interval * 5
    
    # Align start to minor interval (round down)
    z_min = z_stats['min']
    z_max = z_stats['max']
    
    start = align_to_interval(z_min, minor_interval, 'floor')
    end = align_to_interval(z_max, minor_interval, 'ceil')
    
    print(f"Level generation: z_min={z_min:.3f} -> aligned start={start:.3f}")
    print(f"                  z_max={z_max:.3f} -> aligned end={end:.3f}")
    
    # Generate all levels at minor interval
    # Use round to avoid floating point precision issues
    num_steps = int(round((end - start) / minor_interval)) + 1
    all_levels = [round(start + i * minor_interval, 10) for i in range(num_steps)]
    
    # Identify major levels (those that align with major_interval)
    major_levels = [
        level for level in all_levels 
        if abs(level % major_interval) < 1e-9 or abs(level % major_interval - major_interval) < 1e-9
    ]
    
    print(f"Generated {len(all_levels)} levels (minor interval: {minor_interval})")
    print(f"Major levels ({len(major_levels)}, interval: {major_interval}): {[f'{l:.2f}' for l in major_levels[:10]]}{'...' if len(major_levels) > 10 else ''}")
    
    return all_levels, major_levels


def extract_contour_paths(
    triangulation: mtri.Triangulation,
    z_values: np.ndarray,
    levels: list[float]
) -> dict[float, list[list[tuple[float, float]]]]:
    """Extract contour line paths at specified Z levels.
    
    Returns dict mapping Z level to list of contour segments.
    Each segment is a list of (x, y) coordinate tuples.
    """
    fig, ax = plt.subplots()
    contour_set = ax.tricontour(triangulation, z_values, levels=levels)
    plt.close(fig)
    
    contours = {}
    for level_idx, level in enumerate(levels):
        segments = []
        # Use allsegs for newer matplotlib API
        if hasattr(contour_set, 'allsegs'):
            level_segs = contour_set.allsegs[level_idx]
            for seg in level_segs:
                segment = [(float(v[0]), float(v[1])) for v in seg]
                if len(segment) > 1:
                    segments.append(segment)
        else:
            # Fallback for older matplotlib
            collection = contour_set.collections[level_idx]
            for path in collection.get_paths():
                vertices = path.vertices
                segment = [(float(v[0]), float(v[1])) for v in vertices]
                if len(segment) > 1:
                    segments.append(segment)
        
        contours[float(level)] = segments
    
    return contours


def create_visualization(
    triangulation: mtri.Triangulation,
    z_values: np.ndarray,
    levels: list[float],
    output_path: Path,
    major_levels: Optional[list[float]] = None,
    show_points: bool = False
) -> None:
    """Create top-down contour map visualization.
    
    Args:
        triangulation: The triangulation mesh
        z_values: Z values for each point
        levels: All contour levels to draw
        output_path: Path to save the image
        major_levels: Optional list of major levels to draw with thicker lines
        show_points: Whether to show original data points on the map
    """
    # Center coordinates around origin for display
    x_centered = triangulation.x - np.mean(triangulation.x)
    y_centered = triangulation.y - np.mean(triangulation.y)
    
    # Create new triangulation with centered coordinates
    centered_tri = mtri.Triangulation(x_centered, y_centered, triangulation.triangles)
    centered_tri.set_mask(triangulation.mask)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot filled contours
    contourf = ax.tricontourf(centered_tri, z_values, levels=levels, cmap='terrain', alpha=0.7)
    
    # Plot contour lines with major/minor differentiation
    if major_levels is not None and len(major_levels) > 0:
        # Minor contour lines (all levels, thin)
        minor_only = [l for l in levels if l not in major_levels]
        if minor_only:
            ax.tricontour(centered_tri, z_values, levels=minor_only, colors='black', linewidths=0.3, alpha=0.6)
        
        # Major contour lines (thicker, more prominent)
        cs_major = ax.tricontour(centered_tri, z_values, levels=major_levels, colors='black', linewidths=1.2)
        
        # Add labels to major contour lines
        ax.clabel(cs_major, inline=True, fontsize=8, fmt='%.1f')
    else:
        # No major/minor distinction - draw all lines the same
        ax.tricontour(centered_tri, z_values, levels=levels, colors='black', linewidths=0.5)
    
    # Plot original point coordinates if requested
    if show_points:
        ax.scatter(x_centered, y_centered, c=z_values, cmap='terrain', 
                   s=10, edgecolors='black', linewidths=0.3, alpha=0.8, zorder=5)
    
    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax, label='Elevation (Z)')
    
    ax.set_xlabel('X (centered)')
    ax.set_ylabel('Y (centered)')
    ax.set_title('Terrain Contour Map')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_path}")


def export_contours_txt(
    contours: dict[float, list[list[tuple[float, float]]]], 
    output_path: Path
) -> None:
    """Export contour lines to text file.
    
    Format: z_value: x1,y1; x2,y2; ...
    """
    with open(output_path, 'w') as f:
        for z_level in sorted(contours.keys()):
            segments = contours[z_level]
            for segment in segments:
                coords_str = "; ".join(f"{x:.3f},{y:.3f}" for x, y in segment)
                f.write(f"{z_level:.3f}: {coords_str}\n")
    
    print(f"Contour lines exported to: {output_path}")


def export_contours_geojson(
    contours: dict[float, list[list[tuple[float, float]]]], 
    output_path: Path
) -> None:
    """Export contour lines to GeoJSON format."""
    features = []
    
    for z_level in sorted(contours.keys()):
        segments = contours[z_level]
        for segment in segments:
            feature = {
                "type": "Feature",
                "properties": {
                    "elevation": z_level
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[x, y] for x, y in segment]
                }
            }
            features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"Contour lines exported to GeoJSON: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate contour lines from point cloud terrain data"
    )
    parser.add_argument(
        "file_path", 
        type=Path, 
        help="Path to point cloud file (ID X Y Z format)"
    )
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        help="Z values for contour levels (auto-generated if omitted)"
    )
    parser.add_argument(
        "--minor-interval",
        type=float,
        help="Interval for minor contour lines (e.g., 0.2). Overrides --levels."
    )
    parser.add_argument(
        "--major-interval",
        type=float,
        help="Interval for major contour lines (e.g., 1.0). Only affects visualization. Defaults to 5× minor-interval."
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Max triangle edge length filter (default: 1.5× median)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/out/contour_lines.txt"),
        help="Output file path (default: contour_lines.txt)"
    )
    parser.add_argument(
        "--format",
        choices=["txt", "geojson"],
        default="txt",
        help="Export format (default: txt)"
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="Save visualization to PDF file (e.g., map.pdf)"
    )
    parser.add_argument(
        "--show-points",
        action="store_true",
        help="Show original data points on the visualization"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.file_path.exists():
        print(f"Error: File '{args.file_path}' not found")
        return 1
    
    print(f"Loading point cloud: {args.file_path}")
    
    # Load point cloud data
    points = load_point_cloud(args.file_path)
    print(f"Loaded {len(points)} points")
    
    # Print Z statistics
    z_stats = print_z_statistics(points[:, 2])
    
    # Create triangulation with edge filtering
    print("\nCreating triangulation...")
    triangulation, mask = create_triangulation_with_filter(points, args.max_distance)
    
    # Determine contour levels
    major_levels = None  # Track major levels for visualization
    
    if args.minor_interval:
        # Use interval-based level generation
        print(f"\nUsing interval-based levels (minor: {args.minor_interval}, major: {args.major_interval or 'auto'})")
        levels, major_levels = generate_interval_levels(
            z_stats, 
            args.minor_interval, 
            args.major_interval
        )
    elif args.levels:
        levels = args.levels
        print(f"\nUsing custom levels: {levels}")
    else:
        levels = generate_auto_levels(z_stats)
        print(f"\nAuto-generated levels: {[f'{l:.2f}' for l in levels]}")
    
    # Extract contour paths
    print("\nGenerating contour lines...")
    contours = extract_contour_paths(triangulation, points[:, 2], levels)
    
    total_segments = sum(len(segs) for segs in contours.values())
    print(f"Generated {total_segments} contour segments across {len(levels)} levels")
    
    # Export contours
    if args.format == "geojson":
        output_path = args.output.with_suffix('.geojson') if args.output.suffix == '.txt' else args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_contours_geojson(contours, output_path)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        export_contours_txt(contours, args.output)
    
    # Create visualization if requested
    if args.plot:
        print("\nCreating visualization...")
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        create_visualization(triangulation, points[:, 2], levels, args.plot, major_levels, args.show_points)
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
