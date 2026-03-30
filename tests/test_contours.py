import unittest

import matplotlib.tri as mtri
import numpy as np

from contourer.main import extract_contour_paths


class ContourExtractionTests(unittest.TestCase):
    def test_extract_contour_paths_returns_expected_segment_for_single_triangle(self):
        triangulation = mtri.Triangulation(
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([[0, 1, 2]]),
        )
        z_values = np.array([0.0, 1.0, 1.0])

        contours = extract_contour_paths(triangulation, z_values, [0.5])

        self.assertEqual(list(contours.keys()), [0.5])
        self.assertEqual(len(contours[0.5]), 1)

        segment = np.asarray(contours[0.5][0])
        expected = np.array([[0.5, 0.0], [0.0, 0.5]])

        if not np.allclose(segment, expected):
            np.testing.assert_allclose(segment, expected[::-1])

    def test_extract_contour_paths_returns_empty_segments_outside_data_range(self):
        triangulation = mtri.Triangulation(
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([[0, 1, 2]]),
        )
        z_values = np.array([0.0, 1.0, 1.0])

        contours = extract_contour_paths(triangulation, z_values, [-0.5, 1.5])

        self.assertEqual(contours[-0.5], [])
        self.assertEqual(contours[1.5], [])


if __name__ == '__main__':
    unittest.main()