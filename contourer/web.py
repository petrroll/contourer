"""Browser-based interactive contour map viewer."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from flask import Flask, render_template, jsonify, request

from .main import (
    axis_filters_to_expressions,
    compute_axis_filter_mask,
    load_point_cloud,
    describe_axis_filters,
    filter_point_label_data,
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
    initial_show_point_labels: bool = False,
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
        'show_point_labels': initial_show_point_labels,
    }

    def serialize_label_catalog(point_labels) -> list[dict[str, object]]:
        return [
            {
                'id': label_id,
                'label': label,
            }
            for label_id, label in enumerate(point_labels.catalog)
        ]
    
    def get_cached_data(include_labels: bool = False):
        """Load and cache raw point cloud data, with labels only when needed."""
        cache = app.config['CACHE']
        if 'raw_points' not in cache:
            print(f"Loading point cloud: {file_path}")
            load_result = load_point_cloud(file_path, parse_labels=include_labels)
            points = load_result.points
            print_load_summary(load_result.summary)

            cache['raw_points'] = points
            cache['load_summary'] = load_result.summary
            if load_result.point_labels is not None:
                cache['raw_point_labels'] = load_result.point_labels
                cache['load_summary_with_labels'] = load_result.summary

        if include_labels and 'raw_point_labels' not in cache:
            load_result = load_point_cloud(file_path, parse_labels=True)
            if load_result.point_labels is not None:
                cache['raw_point_labels'] = load_result.point_labels
                cache['load_summary_with_labels'] = load_result.summary

        point_labels = cache.get('raw_point_labels') if include_labels else None
        load_summary = cache.get('load_summary_with_labels', cache['load_summary']) if include_labels else cache['load_summary']
        return cache['raw_points'], point_labels, load_summary

    def get_filtered_data(axis_filters_text: Optional[str] = None, include_labels: bool = False):
        """Get cached points and statistics for the current axis filters."""
        raw_points, raw_point_labels, load_summary = get_cached_data(include_labels=include_labels)
        axis_filters = parse_axis_filters(axis_filters_text)
        axis_filters_key = normalize_axis_filters(axis_filters) or ''
        cache = app.config['CACHE']
        cache_key = ('filtered', axis_filters_key)

        if cache_key not in cache:
            if axis_filters_key:
                mask = compute_axis_filter_mask(raw_points, axis_filters)
                points = raw_points[mask]
            else:
                mask = None
                points = raw_points

            filtered_out_points = len(raw_points) - len(points)
            z_stats = print_z_statistics(points[:, 2])
            selection = {
                'axis_filters_text': axis_filters_key,
                'axis_filters': axis_filters_to_expressions(axis_filters),
                'active_points': len(points),
                'filtered_out_points': filtered_out_points,
                'labeled_points': 0,
                'unique_labels': 0,
                'description': describe_axis_filters(axis_filters) if axis_filters_key else None,
            }

            if axis_filters_key:
                print(f"Applying axis filters in web viewer: {selection['description']}")
                print(f"Kept {selection['active_points']} of {len(raw_points)} loaded points after axis filtering")

            cache[cache_key] = {
                'points': points,
                'mask': mask,
                'z_stats': z_stats,
                'selection': selection,
            }

        filtered = cache[cache_key]
        point_labels = None
        selection = dict(filtered['selection'])

        if include_labels:
            if raw_point_labels is None:
                raise RuntimeError('Point labels were requested but not loaded.')

            label_cache_key = ('filtered-point-labels', axis_filters_key)
            if label_cache_key not in cache:
                if filtered['mask'] is None:
                    cache[label_cache_key] = raw_point_labels
                else:
                    cache[label_cache_key] = filter_point_label_data(raw_point_labels, filtered['mask'])

            point_labels = cache[label_cache_key]
            selection['labeled_points'] = point_labels.labeled_points
            selection['unique_labels'] = point_labels.unique_labels

        return filtered['points'], point_labels, filtered['z_stats'], load_summary, selection

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
            points, _, _, _, _ = get_filtered_data(axis_filters_key)
            print(f"Creating triangulation (max_distance={max_distance}, axis_filters={axis_filters_key or 'none'})...")
            triangulation, mask = create_triangulation_with_filter(points, max_distance)
            cache[cache_key] = triangulation
        
        return cache[cache_key]
    
    @app.route('/')
    def index():
        """Render the main map view."""
        initial = app.config['INITIAL_SETTINGS']
        points, _, z_stats, load_summary, selection = get_filtered_data(
            initial['axis_filters'],
            include_labels=initial['show_point_labels'],
        )
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
                               initial_show_point_labels=initial['show_point_labels'],
                               active_points=selection['active_points'],
                               filtered_out_points=selection['filtered_out_points'],
                               load_summary=load_summary)
    
    @app.route('/api/bounds')
    def get_bounds():
        """Get the bounding box of the data."""
        try:
            points, _, z_stats, _, selection = get_filtered_data(get_requested_axis_filters())
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
            'labeled_points': selection['labeled_points'],
            'unique_labels': selection['unique_labels'],
        })
    
    @app.route('/api/points')
    def get_points():
        """Get all points as GeoJSON."""
        include_labels = request.args.get('include_labels', default='false').lower() == 'true'

        try:
            points, point_labels, _, _, selection = get_filtered_data(
                get_requested_axis_filters(),
                include_labels=include_labels,
            )
        except ValueError as error:
            return jsonify({'error': str(error)}), 400
        
        features = []
        for i, (x, y, z) in enumerate(points):
            properties = {
                "elevation": float(z),
                "id": i,
            }

            if include_labels:
                if point_labels is None:
                    raise RuntimeError('Point labels were requested but are unavailable.')

                label_id = int(point_labels.label_ids[i])
                label = point_labels.catalog[label_id] if label_id >= 0 else None
                properties['label_id'] = label_id if label_id >= 0 else None
                properties['label'] = label

            features.append({
                "type": "Feature",
                "properties": properties,
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(x), float(y)]
                }
            })

        response = {
            "type": "FeatureCollection",
            "features": features,
        }
        if include_labels:
            if point_labels is None:
                raise RuntimeError('Point labels were requested but are unavailable.')

            response['meta'] = {
                'labeled_points': selection['labeled_points'],
                'unique_labels': selection['unique_labels'],
                'label_catalog': serialize_label_catalog(point_labels),
            }

        return jsonify(response)
    
    @app.route('/api/contours')
    def get_contours():
        """Generate and return contour lines as GeoJSON."""
        try:
            axis_filters_text = get_requested_axis_filters()
            points, _, z_stats, _, selection = get_filtered_data(axis_filters_text)
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
                "labeled_points": selection['labeled_points'],
                "unique_labels": selection['unique_labels'],
                "z_min": z_stats['min'],
                "z_max": z_stats['max'],
            }
        })
    
    @app.route('/api/mesh')
    def get_mesh():
        """Get the triangulated mesh data for 3D visualization."""
        try:
            axis_filters_text = get_requested_axis_filters()
            points, _, z_stats, _, _ = get_filtered_data(axis_filters_text)
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
        show_point_labels = request.args.get('show_point_labels', default='false').lower() == 'true'

        try:
            axis_filters_text = get_requested_axis_filters()
            points, point_labels, z_stats, _, _ = get_filtered_data(
                axis_filters_text,
                include_labels=show_point_labels,
            )
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
            point_labels=point_labels,
            z_stats=z_stats,
            triangulation=triangulation,
            minor_interval=minor_interval,
            major_interval=major_interval,
            num_levels=num_levels,
            show_points=show_points,
            show_point_labels=show_point_labels,
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
    show_point_labels: bool = False,
):
    """Start the web server for interactive viewing."""
    app = create_app(
        file_path,
        initial_minor_interval=minor_interval,
        initial_major_interval=major_interval,
        initial_max_distance=max_distance,
        initial_axis_filters=axis_filters,
        initial_show_points=show_points,
        initial_show_point_labels=show_point_labels,
    )
    
    print(f"\nContour Viewer starting...")
    print(f"Data file: {file_path}")
    print(f"Open in browser: http://{host}:{port}")
    print(f"\nPress Ctrl+C to stop the server.\n")
    
    app.run(host=host, port=port, debug=False)
