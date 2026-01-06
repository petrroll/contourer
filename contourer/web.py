"""Browser-based interactive contour map viewer."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from flask import Flask, render_template, jsonify, request

from .main import (
    load_point_cloud,
    print_z_statistics,
    create_triangulation_with_filter,
    extract_contour_paths,
    generate_auto_levels,
    generate_interval_levels,
    export_contours_txt,
    export_contours_geojson,
    create_visualization,
)


def create_app(file_path: Path) -> Flask:
    """Create Flask app with the given data file."""
    app = Flask(__name__, template_folder='templates', static_folder='static')
    
    # Store file path and cached data
    app.config['FILE_PATH'] = file_path
    app.config['CACHE'] = {}
    
    def get_cached_data():
        """Load and cache point cloud data."""
        cache = app.config['CACHE']
        if 'points' not in cache:
            print(f"Loading point cloud: {file_path}")
            points = load_point_cloud(file_path)
            print(f"Loaded {len(points)} points")
            
            z_stats = print_z_statistics(points[:, 2])
            
            cache['points'] = points
            cache['z_stats'] = z_stats
        return cache['points'], cache['z_stats']
    
    def get_triangulation(max_distance: Optional[float] = None):
        """Get or create triangulation with given max_distance."""
        cache = app.config['CACHE']
        cache_key = f'triangulation_{max_distance}'
        
        if cache_key not in cache:
            points, _ = get_cached_data()
            print(f"Creating triangulation (max_distance={max_distance})...")
            triangulation, mask = create_triangulation_with_filter(points, max_distance)
            cache[cache_key] = triangulation
        
        return cache[cache_key]
    
    @app.route('/')
    def index():
        """Render the main map view."""
        points, z_stats = get_cached_data()
        return render_template('map.html', 
                               filename=file_path.name,
                               z_min=z_stats['min'],
                               z_max=z_stats['max'],
                               num_points=len(points))
    
    @app.route('/api/bounds')
    def get_bounds():
        """Get the bounding box of the data."""
        points, z_stats = get_cached_data()
        x, y = points[:, 0], points[:, 1]
        return jsonify({
            'x_min': float(np.min(x)),
            'x_max': float(np.max(x)),
            'y_min': float(np.min(y)),
            'y_max': float(np.max(y)),
            'z_min': z_stats['min'],
            'z_max': z_stats['max'],
        })
    
    @app.route('/api/points')
    def get_points():
        """Get all points as GeoJSON."""
        points, _ = get_cached_data()
        
        features = []
        for i, (x, y, z) in enumerate(points):
            features.append({
                "type": "Feature",
                "properties": {"elevation": float(z), "id": i},
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(x), float(y)]
                }
            })
        
        return jsonify({
            "type": "FeatureCollection",
            "features": features
        })
    
    @app.route('/api/contours')
    def get_contours():
        """Generate and return contour lines as GeoJSON."""
        points, z_stats = get_cached_data()
        
        # Get parameters from request
        minor_interval = request.args.get('minor_interval', type=float)
        major_interval = request.args.get('major_interval', type=float)
        max_distance = request.args.get('max_distance', type=float)
        num_levels = request.args.get('num_levels', default=30, type=int)
        
        # Get triangulation
        triangulation = get_triangulation(max_distance)
        
        # Generate levels
        if minor_interval:
            levels, major_levels = generate_interval_levels(
                z_stats, minor_interval, major_interval
            )
        else:
            levels = generate_auto_levels(z_stats, num_levels)
            major_levels = []
        
        # Extract contours
        contours = extract_contour_paths(triangulation, points[:, 2], levels)
        
        # Convert to GeoJSON with major/minor classification
        features = []
        major_set = set(major_levels) if major_levels else set()
        
        for z_level in sorted(contours.keys()):
            segments = contours[z_level]
            is_major = z_level in major_set or (
                major_interval and abs(z_level % major_interval) < 1e-6
            )
            
            for segment in segments:
                features.append({
                    "type": "Feature",
                    "properties": {
                        "elevation": z_level,
                        "is_major": is_major,
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[x, y] for x, y in segment]
                    }
                })
        
        return jsonify({
            "type": "FeatureCollection",
            "features": features,
            "meta": {
                "num_levels": len(levels),
                "num_segments": len(features),
                "levels": levels[:20] if len(levels) > 20 else levels,
                "major_levels": major_levels[:10] if len(major_levels) > 10 else major_levels,
            }
        })
    
    @app.route('/api/export')
    def export_files():
        """Export contour lines and map to ./data/out folder, just like CLI."""
        points, z_stats = get_cached_data()
        
        # Get parameters from request
        minor_interval = request.args.get('minor_interval', type=float)
        major_interval = request.args.get('major_interval', type=float)
        max_distance = request.args.get('max_distance', type=float)
        num_levels = request.args.get('num_levels', default=30, type=int)
        show_points = request.args.get('show_points', default='false').lower() == 'true'
        
        # Get triangulation
        triangulation = get_triangulation(max_distance)
        
        # Generate levels
        major_levels = None
        if minor_interval:
            levels, major_levels = generate_interval_levels(
                z_stats, minor_interval, major_interval
            )
        else:
            levels = generate_auto_levels(z_stats, num_levels)
        
        # Extract contours
        contours = extract_contour_paths(triangulation, points[:, 2], levels)
        
        # Derive output paths from input filename (same as CLI)
        input_stem = file_path.stem
        output_dir = Path("./data/out")
        output_dir.mkdir(parents=True, exist_ok=True)
        contour_output_path = output_dir / f"{input_stem}_contour.txt"
        map_output_path = output_dir / f"{input_stem}_map.pdf"
        
        # Export contours in both formats
        export_contours_txt(contours, contour_output_path)
        geojson_path = contour_output_path.with_suffix('.geojson')
        export_contours_geojson(contours, geojson_path)
        
        # Create visualization PDF using existing function
        create_visualization(
            triangulation, 
            points[:, 2], 
            levels, 
            map_output_path, 
            major_levels, 
            show_points
        )
        
        return jsonify({
            "success": True,
            "files": {
                "txt": str(contour_output_path),
                "geojson": str(geojson_path),
                "pdf": str(map_output_path)
            }
        })
    
    return app


def run_web_server(file_path: Path, host: str = '127.0.0.1', port: int = 5000):
    """Start the web server for interactive viewing."""
    app = create_app(file_path)
    
    print(f"\n🗺️  Contour Viewer starting...")
    print(f"📂 Data file: {file_path}")
    print(f"🌐 Open in browser: http://{host}:{port}")
    print(f"\nPress Ctrl+C to stop the server.\n")
    
    app.run(host=host, port=port, debug=False)
