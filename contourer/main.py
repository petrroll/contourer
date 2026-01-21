import argparse
import json
from pathlib import Path
from typing import Optional

import ezdxf
from ezdxf import units
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid threading issues with Flask
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot filled contours using original coordinates
    contourf = ax.tricontourf(triangulation, z_values, levels=levels, cmap='terrain', alpha=0.7)
    
    # Plot contour lines with major/minor differentiation
    if major_levels is not None and len(major_levels) > 0:
        # Minor contour lines (all levels, thin)
        minor_only = [l for l in levels if l not in major_levels]
        if minor_only:
            ax.tricontour(triangulation, z_values, levels=minor_only, colors='black', linewidths=0.3, alpha=0.6)
        
        # Major contour lines (thicker, more prominent)
        cs_major = ax.tricontour(triangulation, z_values, levels=major_levels, colors='black', linewidths=1.2)
        
        # Add labels to major contour lines
        ax.clabel(cs_major, inline=True, fontsize=8, fmt='%.1f')
    else:
        # No major/minor distinction - draw all lines the same
        ax.tricontour(triangulation, z_values, levels=levels, colors='black', linewidths=0.5)
    
    # Plot original point coordinates if requested
    if show_points:
        ax.scatter(triangulation.x, triangulation.y, c=z_values, cmap='terrain', 
                   s=10, edgecolors='black', linewidths=0.3, alpha=0.8, zorder=5)
    
    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax, label='Elevation (Z)')
    
    # Use full coordinate values without scientific notation offset
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style='plain', axis='both')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Terrain Contour Map')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_path}")


def get_output_paths(input_file: Path) -> dict[str, Path]:
    """Generate output file paths based on input file.
    
    Output files are placed in the same directory as the input file.
    
    Args:
        input_file: Path to the input data file
        
    Returns:
        Dictionary with keys 'vrs', 'geojson', 'dxf', 'pdf' and their output paths
    """
    input_stem = input_file.stem
    output_dir = input_file.parent
    base_path = output_dir / f"{input_stem}_contour"
    
    return {
        'vrs': base_path.with_suffix('.vrs'),
        'geojson': base_path.with_suffix('.geojson'),
        'dxf': base_path.with_suffix('.dxf'),
        'pdf': base_path.with_suffix('.pdf'),
    }


def export_contours_txt(
    contours: dict[float, list[list[tuple[float, float]]]], 
    output_path: Path
) -> None:
    """Export contour lines to text file.
    
    Format:
        z: elevation
        x, y
        x, y
        ...
        
        z: elevation
        x, y
        ...
    
    Each contour segment starts with 'z:' line followed by coordinate lines.
    Segments are separated by empty lines.
    """
    with open(output_path, 'w') as f:
        first_segment = True
        for z_level in sorted(contours.keys()):
            segments = contours[z_level]
            for segment in segments:
                if not first_segment:
                    f.write("\n")  # Empty line between segments
                first_segment = False
                
                f.write(f"z: {z_level:.3f}\n")
                for x, y in segment:
                    f.write(f"{x:.3f}, {y:.3f}\n")
    
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


def export_contours_dxf(
    contours: dict[float, list[list[tuple[float, float]]]], 
    output_path: Path,
    major_levels: Optional[list[float]] = None,
    use_3d: bool = True
) -> None:
    """Export contour lines to DXF format for AutoCAD import.
    
    Creates a DXF file with contour lines organized into layers:
    - Each elevation level gets its own layer named 'CONTOUR_<elevation>'
    - Major contour levels are placed on layers named 'CONTOUR_MAJOR_<elevation>'
    - Contour lines are drawn as polylines with Z coordinate at the elevation
    
    Args:
        contours: Dict mapping Z level to list of contour line segments
        output_path: Path to save the DXF file
        major_levels: Optional list of major contour levels (drawn with thicker lines)
        use_3d: If True, create 3D polylines with Z at elevation; if False, 2D polylines
    """
    # Create a new DXF document (using R2010 for wide compatibility)
    doc = ezdxf.new('R2010')
    doc.units = units.M  # Set units to meters
    
    # Get the modelspace where we'll add entities
    msp = doc.modelspace()
    
    # Create a set for quick major level lookup
    major_set = set(major_levels) if major_levels else set()
    
    # Define colors for contour lines
    # AutoCAD color indices: 7=white/black, 1=red, 2=yellow, 3=green, 4=cyan, 5=blue
    MINOR_COLOR = 8  # Gray
    MAJOR_COLOR = 7  # White/Black (standard)
    
    # Track created layers to avoid duplicates
    created_layers = set()
    
    for z_level in sorted(contours.keys()):
        segments = contours[z_level]
        
        if not segments:
            continue
            
        is_major = z_level in major_set
        
        # Create layer name (sanitize for DXF compatibility)
        # DXF layer names can't contain certain characters
        z_str = f"{z_level:.2f}".replace('.', '_').replace('-', 'N')
        if is_major:
            layer_name = f"CONTOUR_MAJOR_{z_str}"
            color = MAJOR_COLOR
            lineweight = 35  # 0.35mm for major contours
        else:
            layer_name = f"CONTOUR_{z_str}"
            color = MINOR_COLOR
            lineweight = 18  # 0.18mm for minor contours
        
        # Create layer if it doesn't exist
        if layer_name not in created_layers:
            doc.layers.add(
                layer_name,
                color=color,
                lineweight=lineweight,
            )
            created_layers.add(layer_name)
        
        # Add contour segments as polylines
        for segment in segments:
            if len(segment) < 2:
                continue
                
            if use_3d:
                # Create 3D polyline with Z coordinate at elevation
                points_3d = [(x, y, z_level) for x, y in segment]
                msp.add_polyline3d(
                    points_3d,
                    dxfattribs={'layer': layer_name}
                )
            else:
                # Create 2D polyline (LWPolyline)
                points_2d = [(x, y) for x, y in segment]
                msp.add_lwpolyline(
                    points_2d,
                    dxfattribs={
                        'layer': layer_name,
                        'elevation': z_level,  # Store elevation as attribute
                    }
                )
    
    # Save the DXF file
    doc.saveas(output_path)
    
    total_segments = sum(len(segs) for segs in contours.values())
    print(f"Contour lines exported to DXF: {output_path}")
    print(f"  - {len(created_layers)} layers created")
    print(f"  - {total_segments} polylines exported")
    if major_levels:
        print(f"  - {len(major_set)} major contour levels")


def run_export(
    file_path: Path,
    points: np.ndarray,
    z_stats: dict,
    triangulation: mtri.Triangulation,
    minor_interval: Optional[float] = None,
    major_interval: Optional[float] = None,
    num_levels: int = 30,
    show_points: bool = False,
    formats: Optional[set[str]] = None,
) -> dict[str, Path]:
    """Run the full export workflow for contour generation.
    
    This is the shared export logic used by both CLI and web interface.
    
    Args:
        file_path: Input file path (used to derive output paths)
        points: Point cloud data (N, 3) array
        z_stats: Z statistics dictionary from print_z_statistics
        triangulation: Pre-computed triangulation
        minor_interval: Interval for minor contour lines
        major_interval: Interval for major contour lines
        num_levels: Number of auto-generated levels (if no interval specified)
        show_points: Whether to show points on PDF visualization
        formats: Set of formats to export ('vrs', 'geojson', 'dxf', 'pdf'), or None for all
    
    Returns:
        Dictionary mapping format names to their output paths
    """
    # Determine which formats to export
    all_formats = {'pdf', 'vrs', 'geojson', 'dxf'}
    selected_formats = formats if formats else all_formats
    
    # Generate levels
    major_levels = None
    if minor_interval:
        print(f"Using interval-based levels (minor: {minor_interval}, major: {major_interval or 'auto'})")
        levels, major_levels = generate_interval_levels(
            z_stats, minor_interval, major_interval
        )
    else:
        levels = generate_auto_levels(z_stats, num_levels)
    
    # Extract contour paths
    print("Generating contour lines...")
    contours = extract_contour_paths(triangulation, points[:, 2], levels)
    
    total_segments = sum(len(segs) for segs in contours.values())
    print(f"Generated {total_segments} contour segments across {len(levels)} levels")
    
    # Get output paths
    output_paths = get_output_paths(file_path)
    
    print(f"Exporting formats: {', '.join(sorted(selected_formats))}")
    
    # Export contours in selected formats
    if 'vrs' in selected_formats:
        export_contours_txt(contours, output_paths['vrs'])
    if 'geojson' in selected_formats:
        export_contours_geojson(contours, output_paths['geojson'])
    if 'dxf' in selected_formats:
        export_contours_dxf(contours, output_paths['dxf'], major_levels)
    
    # Create visualization PDF
    if 'pdf' in selected_formats:
        print("Creating visualization...")
        create_visualization(triangulation, points[:, 2], levels, output_paths['pdf'], major_levels, show_points)
    
    # Return only the paths that were actually exported
    return {fmt: output_paths[fmt] for fmt in selected_formats}


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
        "--show-points",
        action="store_true",
        help="Show original data points on the visualization"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch interactive browser-based viewer"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for web server (default: 5000)"
    )
    parser.add_argument(
        "--formats",
        type=str,
        default=None,
        help="Comma-separated list of export formats: pdf,vrs,geojson,dxf (default: all)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.file_path.exists():
        print(f"Error: File '{args.file_path}' not found")
        return 1
    
    # Launch web viewer if requested
    if args.web:
        from .web import run_web_server
        run_web_server(
            args.file_path,
            port=args.port,
            minor_interval=args.minor_interval,
            major_interval=args.major_interval,
            max_distance=args.max_distance,
            show_points=args.show_points,
        )
        return 0
    
    # Get output paths (same directory as input file)
    output_paths = get_output_paths(args.file_path)
    
    print(f"Loading point cloud: {args.file_path}")
    
    # Load point cloud data
    points = load_point_cloud(args.file_path)
    print(f"Loaded {len(points)} points")
    
    # Print Z statistics
    z_stats = print_z_statistics(points[:, 2])
    
    # Create triangulation with edge filtering
    print("\nCreating triangulation...")
    triangulation, mask = create_triangulation_with_filter(points, args.max_distance)
    
    # Determine which formats to export
    all_formats = {'pdf', 'vrs', 'geojson', 'dxf'}
    if args.formats:
        selected_formats = set(f.strip().lower() for f in args.formats.split(','))
        invalid_formats = selected_formats - all_formats
        if invalid_formats:
            print(f"Warning: Unknown format(s) ignored: {', '.join(invalid_formats)}")
        selected_formats = selected_formats & all_formats
        if not selected_formats:
            print("Error: No valid formats specified")
            return 1
    else:
        selected_formats = all_formats
    
    # Run the export workflow
    run_export(
        file_path=args.file_path,
        points=points,
        z_stats=z_stats,
        triangulation=triangulation,
        minor_interval=args.minor_interval,
        major_interval=args.major_interval,
        show_points=args.show_points,
        formats=selected_formats,
    )
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
