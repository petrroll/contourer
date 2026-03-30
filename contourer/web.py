"""Browser-based interactive contour map viewer."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from flask import Flask, render_template, jsonify, request

from .main import (
    axis_filters_to_expressions,
    load_point_cloud,
    describe_axis_filters,
    filter_points_by_axis,
    normalize_axis_filters,
    parse_axis_filters,
    print_load_summary,
    print_z_statistics,
    create_triangulation_with_filter,
    extract_contour_paths,
    generate_auto_levels,
    generate_interval_levels,
    run_export,
)


def create_app(
    file_path: Path,
    initial_minor_interval: Optional[float] = None,
    initial_major_interval: Optional[float] = None,
    initial_max_distance: Optional[float] = None,
    initial_axis_filters: Optional[str] = None,
    initial_show_points: bool = False,
) -> Flask:
    """Create Flask app with the given data file and optional initial settings."""
    app = Flask(__name__, template_folder='templates', static_folder='static')
    
    # Store file path and cached data
    app.config['FILE_PATH'] = file_path
    app.config['CACHE'] = {}
    
    # Store initial settings from CLI
    app.config['INITIAL_SETTINGS'] = {
        'minor_interval': initial_minor_interval,
        'major_interval': initial_major_interval,
        'max_distance': initial_max_distance,
        'axis_filters': initial_axis_filters,
        'show_points': initial_show_points,
    }
    
    def get_cached_data():
        """Load and cache raw point cloud data."""
        cache = app.config['CACHE']
        if 'raw_points' not in cache:
            print(f"Loading point cloud: {file_path}")
            load_result = load_point_cloud(file_path)
            points = load_result.points
            print_load_summary(load_result.summary)

            cache['raw_points'] = points
            cache['load_summary'] = load_result.summary

        return cache['raw_points'], cache['load_summary']

    def get_filtered_data(axis_filters_text: Optional[str] = None):
        """Get cached points and statistics for the current axis filters."""
        raw_points, load_summary = get_cached_data()
        axis_filters = parse_axis_filters(axis_filters_text)
        axis_filters_key = normalize_axis_filters(axis_filters) or ''
        cache = app.config['CACHE']
        cache_key = ('filtered', axis_filters_key)

        if cache_key not in cache:
            points = filter_points_by_axis(raw_points, axis_filters)
            filtered_out_points = len(raw_points) - len(points)
            z_stats = print_z_statistics(points[:, 2])
            selection = {
                'axis_filters_text': axis_filters_key,
                'axis_filters': axis_filters_to_expressions(axis_filters),
                'active_points': len(points),
                'filtered_out_points': filtered_out_points,
                'description': describe_axis_filters(axis_filters) if axis_filters_key else None,
            }

            if axis_filters_key:
                print(f"Applying axis filters in web viewer: {selection['description']}")
                print(f"Kept {selection['active_points']} of {len(raw_points)} loaded points after axis filtering")

            cache[cache_key] = {
                'points': points,
                'z_stats': z_stats,
                'selection': selection,
            }

        filtered = cache[cache_key]
        return filtered['points'], filtered['z_stats'], load_summary, filtered['selection']

    def get_requested_axis_filters() -> Optional[str]:
        return request.args.get(
            'axis_filters',
            default=app.config['INITIAL_SETTINGS']['axis_filters'],
            type=str,
        )
    
    def get_triangulation(max_distance: Optional[float] = None, axis_filters_text: Optional[str] = None):
        """Get or create triangulation with given max_distance."""
        cache = app.config['CACHE']
        axis_filters = parse_axis_filters(axis_filters_text)
        axis_filters_key = normalize_axis_filters(axis_filters) or ''
        cache_key = ('triangulation', axis_filters_key, max_distance)
        
        if cache_key not in cache:
            points, _, _, _ = get_filtered_data(axis_filters_key)
            print(f"Creating triangulation (max_distance={max_distance}, axis_filters={axis_filters_key or 'none'})...")
            triangulation, mask = create_triangulation_with_filter(points, max_distance)
            cache[cache_key] = triangulation
        
        return cache[cache_key]
    
    @app.route('/')
    def index():
        """Render the main map view."""
        initial = app.config['INITIAL_SETTINGS']
        points, z_stats, load_summary, selection = get_filtered_data(initial['axis_filters'])
        initial_axis_filters = parse_axis_filters(initial['axis_filters'])
        initial_axis_filter_x, initial_axis_filter_y, initial_axis_filter_z = axis_filters_to_expressions(initial_axis_filters)
        return render_template('map.html', 
                               filename=file_path.name,
                               z_min=z_stats['min'],
                               z_max=z_stats['max'],
                               initial_minor_interval=initial['minor_interval'],
                               initial_major_interval=initial['major_interval'],
                               initial_max_distance=initial['max_distance'],
                               initial_axis_filter_x=initial_axis_filter_x,
                               initial_axis_filter_y=initial_axis_filter_y,
                               initial_axis_filter_z=initial_axis_filter_z,
                               initial_show_points=initial['show_points'],
                               active_points=selection['active_points'],
                               filtered_out_points=selection['filtered_out_points'],
                               load_summary=load_summary)
    
    @app.route('/api/bounds')
    def get_bounds():
        """Get the bounding box of the data."""
        try:
            points, z_stats, _, selection = get_filtered_data(get_requested_axis_filters())
        except ValueError as error:
            return jsonify({'error': str(error)}), 400

        x, y = points[:, 0], points[:, 1]
        return jsonify({
            'x_min': float(np.min(x)),
            'x_max': float(np.max(x)),
            'y_min': float(np.min(y)),
            'y_max': float(np.max(y)),
            'z_min': z_stats['min'],
            'z_max': z_stats['max'],
            'active_points': selection['active_points'],
            'filtered_out_points': selection['filtered_out_points'],
        })
    
    @app.route('/api/points')
    def get_points():
        """Get all points as GeoJSON."""
        try:
            points, _, _, _ = get_filtered_data(get_requested_axis_filters())
        except ValueError as error:
            return jsonify({'error': str(error)}), 400
        
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
        try:
            axis_filters_text = get_requested_axis_filters()
            points, z_stats, _, selection = get_filtered_data(axis_filters_text)
        except ValueError as error:
            return jsonify({'error': str(error)}), 400
        
        # Get parameters from request
        minor_interval = request.args.get('minor_interval', type=float)
        major_interval = request.args.get('major_interval', type=float)
        max_distance = request.args.get('max_distance', type=float)
        num_levels = request.args.get('num_levels', default=30, type=int)
        
        # Get triangulation
        triangulation = get_triangulation(max_distance, axis_filters_text)
        
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
                "active_points": selection['active_points'],
                "filtered_out_points": selection['filtered_out_points'],
                "z_min": z_stats['min'],
                "z_max": z_stats['max'],
            }
        })
    
    @app.route('/api/mesh')
    def get_mesh():
        """Get the triangulated mesh data for 3D visualization."""
        try:
            axis_filters_text = get_requested_axis_filters()
            points, z_stats, _, _ = get_filtered_data(axis_filters_text)
        except ValueError as error:
            return jsonify({'error': str(error)}), 400
        
        # Get parameters from request
        max_distance = request.args.get('max_distance', type=float)
        
        # Get triangulation
        triangulation = get_triangulation(max_distance, axis_filters_text)
        
        # Get vertices and triangles
        x = triangulation.x.tolist()
        y = triangulation.y.tolist()
        z = points[:, 2].tolist()
        
        # Get triangles (indices into vertices), excluding masked ones
        triangles = triangulation.get_masked_triangles().tolist()
        
        return jsonify({
            'vertices': {
                'x': x,
                'y': y,
                'z': z
            },
            'triangles': triangles,
            'z_min': z_stats['min'],
            'z_max': z_stats['max'],
        })
    
    @app.route('/api/export')
    def export_files():
        """Export contour lines and map, using the same logic as CLI."""
        try:
            axis_filters_text = get_requested_axis_filters()
            points, z_stats, _, _ = get_filtered_data(axis_filters_text)
        except ValueError as error:
            return jsonify({'error': str(error)}), 400
        
        # Get parameters from request
        minor_interval = request.args.get('minor_interval', type=float)
        major_interval = request.args.get('major_interval', type=float)
        max_distance = request.args.get('max_distance', type=float)
        num_levels = request.args.get('num_levels', default=30, type=int)
        show_points = request.args.get('show_points', default='false').lower() == 'true'
        
        # Get triangulation
        triangulation = get_triangulation(max_distance, axis_filters_text)
        
        # Run the shared export workflow
        exported_paths = run_export(
            file_path=file_path,
            points=points,
            z_stats=z_stats,
            triangulation=triangulation,
            minor_interval=minor_interval,
            major_interval=major_interval,
            num_levels=num_levels,
            show_points=show_points,
        )
        
        return jsonify({
            "success": True,
            "files": {fmt: str(path) for fmt, path in exported_paths.items()}
        })
    
    return app


def run_web_server(
    file_path: Path,
    host: str = '127.0.0.1',
    port: int = 5000,
    minor_interval: Optional[float] = None,
    major_interval: Optional[float] = None,
    max_distance: Optional[float] = None,
    axis_filters: Optional[str] = None,
    show_points: bool = False,
):
    """Start the web server for interactive viewing."""
    app = create_app(
        file_path,
        initial_minor_interval=minor_interval,
        initial_major_interval=major_interval,
        initial_max_distance=max_distance,
        initial_axis_filters=axis_filters,
        initial_show_points=show_points,
    )
    
    print(f"\nContour Viewer starting...")
    print(f"Data file: {file_path}")
    print(f"Open in browser: http://{host}:{port}")
    print(f"\nPress Ctrl+C to stop the server.\n")
    
    app.run(host=host, port=port, debug=False)
