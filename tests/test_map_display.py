import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pyproj import Transformer

from contourer.main import build_display_transformer, main, validate_map_display_settings
from contourer.web import create_app


def build_projected_points() -> tuple[np.ndarray, np.ndarray]:
    geographic_points = np.array(
        [
            [14.420760, 50.088040, 210.0],
            [14.421120, 50.088140, 211.0],
            [14.420950, 50.087780, 212.0],
            [14.421280, 50.087980, 213.0],
        ],
        dtype=float,
    )
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    projected_x, projected_y = transformer.transform(
        geographic_points[:, 0],
        geographic_points[:, 1],
    )
    projected_points = np.column_stack([projected_x, projected_y, geographic_points[:, 2]])
    return geographic_points, projected_points


def build_swapped_projected_points() -> tuple[np.ndarray, np.ndarray]:
    geographic_points, projected_points = build_projected_points()
    swapped_points = np.column_stack([
        projected_points[:, 1],
        projected_points[:, 0],
        geographic_points[:, 2],
    ])
    return geographic_points, swapped_points


def write_point_file(file_path: Path, points: np.ndarray) -> None:
    rows = [
        f"{point_index} {point[0]:.6f} {point[1]:.6f} {point[2]:.3f}"
        for point_index, point in enumerate(points, start=1)
    ]
    file_path.write_text('\n'.join(rows), encoding='utf-8')


class MapDisplaySettingsTests(unittest.TestCase):
    def test_validate_map_display_settings_requires_source_crs_for_osm(self):
        with self.assertRaises(ValueError):
            validate_map_display_settings('osm', None)

    def test_main_passes_map_display_settings_to_web_server(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / 'points.txt'
            file_path.write_text('0 0 0\n1 0 1\n0 1 2\n', encoding='utf-8')

            with patch('contourer.web.run_web_server') as run_web_server:
                with patch.object(
                    sys,
                    'argv',
                    [
                        'contourer',
                        str(file_path),
                        '--web',
                        '--basemap',
                        'osm',
                        '--source-crs',
                        'EPSG:4326',
                    ],
                ):
                    result = main()

        self.assertEqual(result, 0)
        self.assertEqual(run_web_server.call_count, 1)
        self.assertEqual(run_web_server.call_args.kwargs['basemap'], 'osm')
        self.assertEqual(run_web_server.call_args.kwargs['source_crs'], 'EPSG:4326')

    def test_main_rejects_osm_without_source_crs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / 'points.txt'
            file_path.write_text('0 0 0\n1 0 1\n0 1 2\n', encoding='utf-8')

            with patch.object(
                sys,
                'argv',
                ['contourer', str(file_path), '--web', '--basemap', 'osm'],
            ):
                with self.assertRaises(SystemExit) as exit_context:
                    main()

        self.assertEqual(exit_context.exception.code, 2)


class MapDisplayApiTests(unittest.TestCase):
    def test_bounds_and_points_are_reprojected_for_osm_display(self):
        geographic_points, projected_points = build_projected_points()

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / 'projected-points.txt'
            write_point_file(file_path, projected_points)

            app = create_app(file_path)
            client = app.test_client()

            bounds_response = client.get(
                '/api/bounds',
                query_string={'basemap': 'osm', 'source_crs': 'EPSG:3857'},
            )
            points_response = client.get(
                '/api/points',
                query_string={'basemap': 'osm', 'source_crs': 'EPSG:3857'},
            )

        self.assertEqual(bounds_response.status_code, 200)
        bounds_payload = bounds_response.get_json()
        self.assertEqual(bounds_payload['coordinate_mode'], 'geographic')
        self.assertEqual(bounds_payload['display_crs'], 'EPSG:4326')
        self.assertAlmostEqual(bounds_payload['x_min'], float(np.min(geographic_points[:, 0])), places=5)
        self.assertAlmostEqual(bounds_payload['x_max'], float(np.max(geographic_points[:, 0])), places=5)
        self.assertAlmostEqual(bounds_payload['y_min'], float(np.min(geographic_points[:, 1])), places=5)
        self.assertAlmostEqual(bounds_payload['y_max'], float(np.max(geographic_points[:, 1])), places=5)

        self.assertEqual(points_response.status_code, 200)
        points_payload = points_response.get_json()
        first_feature = points_payload['features'][0]
        self.assertAlmostEqual(first_feature['geometry']['coordinates'][0], geographic_points[0, 0], places=5)
        self.assertAlmostEqual(first_feature['geometry']['coordinates'][1], geographic_points[0, 1], places=5)
        self.assertAlmostEqual(first_feature['properties']['source_x'], projected_points[0, 0], places=3)
        self.assertAlmostEqual(first_feature['properties']['source_y'], projected_points[0, 1], places=3)

    def test_contours_report_geographic_display_metadata(self):
        geographic_points, projected_points = build_projected_points()

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / 'projected-contours.txt'
            write_point_file(file_path, projected_points)

            app = create_app(file_path)
            client = app.test_client()
            response = client.get(
                '/api/contours',
                query_string={'basemap': 'osm', 'source_crs': 'EPSG:3857'},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['meta']['coordinate_mode'], 'geographic')
        self.assertEqual(payload['meta']['display_crs'], 'EPSG:4326')
        self.assertGreater(len(payload['features']), 0)

        min_lon = float(np.min(geographic_points[:, 0]))
        max_lon = float(np.max(geographic_points[:, 0]))
        min_lat = float(np.min(geographic_points[:, 1]))
        max_lat = float(np.max(geographic_points[:, 1]))
        first_coordinate = payload['features'][0]['geometry']['coordinates'][0]
        self.assertGreaterEqual(first_coordinate[0], min_lon - 0.001)
        self.assertLessEqual(first_coordinate[0], max_lon + 0.001)
        self.assertGreaterEqual(first_coordinate[1], min_lat - 0.001)
        self.assertLessEqual(first_coordinate[1], max_lat + 0.001)

    def test_contours_support_gzip_responses(self):
        _, projected_points = build_projected_points()

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / 'gzip-contours.txt'
            write_point_file(file_path, projected_points)

            app = create_app(file_path)
            client = app.test_client()
            response = client.get(
                '/api/contours',
                headers={'Accept-Encoding': 'gzip'},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get('Content-Encoding'), 'gzip')
        payload = json.loads(gzip.decompress(response.data))
        self.assertEqual(payload['type'], 'FeatureCollection')
        self.assertGreater(payload['meta']['num_segments'], 0)

    def test_contours_reuse_display_transformer_with_osm(self):
        _, projected_points = build_projected_points()

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / 'transformer-cache-contours.txt'
            write_point_file(file_path, projected_points)

            app = create_app(file_path)
            client = app.test_client()
            with patch('contourer.web.build_display_transformer', wraps=build_display_transformer) as build_transformer:
                response = client.get(
                    '/api/contours',
                    query_string={'basemap': 'osm', 'source_crs': 'EPSG:3857'},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(build_transformer.call_count, 1)

    def test_manual_yx_axis_order_maps_swapped_projected_points(self):
        geographic_points, swapped_points = build_swapped_projected_points()

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / 'swapped-yx-points.txt'
            write_point_file(file_path, swapped_points)

            app = create_app(file_path)
            client = app.test_client()
            response = client.get(
                '/api/points',
                query_string={
                    'basemap': 'osm',
                    'source_crs': 'EPSG:3857',
                    'source_axis_order': 'yx',
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        first_feature = payload['features'][0]
        self.assertAlmostEqual(first_feature['geometry']['coordinates'][0], geographic_points[0, 0], places=4)
        self.assertAlmostEqual(first_feature['geometry']['coordinates'][1], geographic_points[0, 1], places=4)

    def test_local_mode_keeps_source_coordinates(self):
        _, projected_points = build_projected_points()

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / 'local-points.txt'
            write_point_file(file_path, projected_points)

            app = create_app(file_path)
            client = app.test_client()
            response = client.get('/api/bounds')

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['coordinate_mode'], 'local')
        self.assertAlmostEqual(payload['x_min'], float(np.min(projected_points[:, 0])), places=3)
        self.assertAlmostEqual(payload['y_min'], float(np.min(projected_points[:, 1])), places=3)

    def test_osm_display_rejects_invalid_source_crs(self):
        _, projected_points = build_projected_points()

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / 'invalid-map-settings.txt'
            write_point_file(file_path, projected_points)

            app = create_app(file_path)
            client = app.test_client()
            response = client.get(
                '/api/bounds',
                query_string={'basemap': 'osm', 'source_crs': 'EPSG:not-a-real-code'},
            )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn('Invalid projection', payload['error'])


if __name__ == '__main__':
    unittest.main()