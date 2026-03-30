import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from contourer.main import (
    build_point_label_data,
    filter_point_data_by_axis,
    load_point_cloud,
    parse_axis_filters,
)
from contourer.web import create_app


class PointLabelLoadingTests(unittest.TestCase):
    def test_load_point_cloud_skips_labels_when_not_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / 'numeric-only-load.txt'
            file_path.write_bytes(
                (
                    b"1 10 20 30 gate\n"
                    b"2 11 21 31 gate\n"
                    b"3 12 22 32 lamp post\n"
                )
            )

            result = load_point_cloud(file_path)

        np.testing.assert_allclose(
            result.points,
            np.array(
                [
                    [10.0, 20.0, 30.0],
                    [11.0, 21.0, 31.0],
                    [12.0, 22.0, 32.0],
                ]
            ),
        )
        self.assertIsNone(result.point_labels)
        self.assertEqual(result.summary.labeled_points, 0)
        self.assertEqual(result.summary.unique_labels, 0)

    def test_load_point_cloud_collects_labels_and_catalog(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / 'labeled-points.txt'
            file_path.write_bytes(
                (
                    b"1 10 20 30 gate\n"
                    b"2 11 21 31 gate\n"
                    b"3 12 22 32 lamp post\n"
                    b"4 13,5 23,5 33,5 gate\n"
                )
            )

            result = load_point_cloud(file_path, parse_labels=True)

        point_labels = result.point_labels
        assert point_labels is not None

        np.testing.assert_allclose(
            result.points,
            np.array(
                [
                    [10.0, 20.0, 30.0],
                    [11.0, 21.0, 31.0],
                    [12.0, 22.0, 32.0],
                    [13.5, 23.5, 33.5],
                ]
            ),
        )
        self.assertEqual(result.summary.labeled_points, 4)
        self.assertEqual(result.summary.unique_labels, 2)
        self.assertEqual(point_labels.catalog, ('gate', 'lamp post'))
        self.assertEqual(point_labels.label_ids.tolist(), [0, 0, 1, 0])

    def test_filter_point_data_by_axis_compacts_label_catalog(self):
        points = np.array(
            [
                [-1.0, 0.0, 10.0],
                [1.0, 0.0, 11.0],
                [2.0, 0.0, 12.0],
                [3.0, 0.0, 13.0],
            ]
        )
        point_labels = build_point_label_data(['gate', 'lamp post', 'lamp post', None])

        filtered_points, filtered_labels = filter_point_data_by_axis(
            points,
            point_labels,
            parse_axis_filters('>0.5,,'),
        )

        assert filtered_labels is not None

        np.testing.assert_allclose(
            filtered_points,
            np.array(
                [
                    [1.0, 0.0, 11.0],
                    [2.0, 0.0, 12.0],
                    [3.0, 0.0, 13.0],
                ]
            ),
        )
        self.assertEqual(filtered_labels.catalog, ('lamp post',))
        self.assertEqual(filtered_labels.label_ids.tolist(), [0, 0, -1])


class PointLabelApiTests(unittest.TestCase):
    def test_points_api_exposes_label_catalog_and_compacted_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / 'api-points.txt'
            file_path.write_text(
                '\n'.join(
                    [
                        '1 0 0 10 gate',
                        '2 1 1 11 lamp post',
                        '3 2 2 12 lamp post',
                        '4 3 3 13',
                        '5 4 4 14',
                    ]
                ),
                encoding='utf-8',
            )

            app = create_app(file_path)
            client = app.test_client()
            response = client.get(
                '/api/points',
                query_string={'axis_filters': '>0.5,,', 'include_labels': 'true'},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(
            payload['meta']['label_catalog'],
            [{'id': 0, 'label': 'lamp post'}],
        )
        self.assertEqual(payload['meta']['labeled_points'], 2)
        self.assertEqual(payload['meta']['unique_labels'], 1)
        self.assertEqual(len(payload['features']), 4)
        self.assertEqual(payload['features'][0]['properties']['label'], 'lamp post')
        self.assertEqual(payload['features'][0]['properties']['label_id'], 0)
        self.assertEqual(payload['features'][1]['properties']['label'], 'lamp post')
        self.assertEqual(payload['features'][1]['properties']['label_id'], 0)
        self.assertIsNone(payload['features'][2]['properties']['label'])
        self.assertIsNone(payload['features'][2]['properties']['label_id'])

    def test_points_api_loads_labels_only_when_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / 'lazy-api-points.txt'
            file_path.write_text(
                '\n'.join(
                    [
                        '1 0 0 10 gate',
                        '2 1 1 11 lamp post',
                        '3 2 2 12',
                    ]
                ),
                encoding='utf-8',
            )

            load_calls = []

            def tracked_load_point_cloud(*args, **kwargs):
                load_calls.append(kwargs.get('parse_labels', False))
                return load_point_cloud(*args, **kwargs)

            with patch('contourer.web.load_point_cloud', side_effect=tracked_load_point_cloud):
                app = create_app(file_path)
                client = app.test_client()

                unlabeled_response = client.get('/api/points')
                labeled_response = client.get('/api/points', query_string={'include_labels': 'true'})

        self.assertEqual(unlabeled_response.status_code, 200)
        unlabeled_payload = unlabeled_response.get_json()
        self.assertEqual(load_calls, [False, True])
        self.assertNotIn('meta', unlabeled_payload)
        self.assertNotIn('label', unlabeled_payload['features'][0]['properties'])
        self.assertNotIn('label_id', unlabeled_payload['features'][0]['properties'])

        self.assertEqual(labeled_response.status_code, 200)
        labeled_payload = labeled_response.get_json()
        self.assertEqual(
            labeled_payload['meta']['label_catalog'],
            [{'id': 0, 'label': 'gate'}, {'id': 1, 'label': 'lamp post'}],
        )
        self.assertEqual(labeled_payload['features'][0]['properties']['label'], 'gate')
        self.assertEqual(labeled_payload['features'][1]['properties']['label'], 'lamp post')


if __name__ == '__main__':
    unittest.main()