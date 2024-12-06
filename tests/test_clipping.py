import pytest
import numpy as np
from kernel.clipping_mod import calculate_segment_plane_intersection


class TestClipping:
    def test_segment_plane_intersection(self):
        """Test segment-plane intersection calculation"""
        # Test case 1: Segment crosses XY plane (Z=0)
        plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # XY plane
        start = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        end = np.array([1.0, 1.0, -1.0], dtype=np.float64)
        intersection, t = calculate_segment_plane_intersection(plane_normal, start, end)
        np.testing.assert_array_almost_equal(intersection, np.array([1.0, 1.0, 0.0], dtype=np.float64))
        assert t == 0.5

        # Test case 2: Segment parallel to plane
        plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        start = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        end = np.array([1.0, 0.0, 1.0], dtype=np.float64)
        intersection, t = calculate_segment_plane_intersection(plane_normal, start, end)
        assert intersection is None
        assert t is None

        # Test case 3: Segment doesn't intersect plane
        plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        start = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        end = np.array([0.0, 0.0, 2.0], dtype=np.float64)
        intersection, t = calculate_segment_plane_intersection(plane_normal, start, end)
        assert intersection is None
        assert t is None

        # Test case 4: Segment intersects at endpoint
        plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        start = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        end = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        intersection, t = calculate_segment_plane_intersection(plane_normal, start, end)
        np.testing.assert_array_almost_equal(intersection, np.array([0.0, 0.0, 0.0], dtype=np.float64))
        assert t == 0.0

        # Test case 5: Diagonal intersection
        plane_normal = np.array([1.0, 1.0, 1.0], dtype=np.float64) / np.sqrt(3)  # 45-degree plane
        start = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
        end = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        intersection, t = calculate_segment_plane_intersection(plane_normal, start, end)
        np.testing.assert_array_almost_equal(intersection, np.array([0.0, 0.0, 0.0], dtype=np.float64))
        assert abs(t - 0.5) < 1e-6
