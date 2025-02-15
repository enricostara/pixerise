import pytest
import numpy as np
from kernel.clipping_mod import calculate_bounding_sphere


class TestBoundingSphere:
    def test_single_point(self):
        """Test bounding sphere for a single point"""
        vertices = np.array([[1.0, 2.0, 3.0]])
        center, radius = calculate_bounding_sphere(vertices)

        np.testing.assert_array_almost_equal(center, np.array([1.0, 2.0, 3.0]))
        assert radius == 0.0

    def test_two_points(self):
        """Test bounding sphere for two points"""
        vertices = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        center, radius = calculate_bounding_sphere(vertices)

        np.testing.assert_array_almost_equal(center, np.array([1.0, 0.0, 0.0]))
        assert abs(radius - 1.0) < 1e-10

    def test_cube_vertices(self):
        """Test bounding sphere for vertices of a unit cube"""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],  # Origin
                [1.0, 0.0, 0.0],  # Right
                [0.0, 1.0, 0.0],  # Up
                [0.0, 0.0, 1.0],  # Forward
                [1.0, 1.0, 0.0],  # Right-Up
                [1.0, 0.0, 1.0],  # Right-Forward
                [0.0, 1.0, 1.0],  # Up-Forward
                [1.0, 1.0, 1.0],  # Right-Up-Forward
            ]
        )
        center, radius = calculate_bounding_sphere(vertices)

        # Center should be at (0.5, 0.5, 0.5)
        np.testing.assert_array_almost_equal(center, np.array([0.5, 0.5, 0.5]))
        # Radius should be sqrt(3)/2 ≈ 0.866 (distance from center to any corner)
        assert abs(radius - np.sqrt(3) / 2) < 1e-10

    def test_tetrahedron(self):
        """Test bounding sphere for a regular tetrahedron"""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, np.sqrt(0.75), 0.0],
                [0.5, np.sqrt(0.75) / 3, np.sqrt(2 / 3)],
            ]
        )
        center, radius = calculate_bounding_sphere(vertices)

        # Check that all vertices are within the sphere
        for vertex in vertices:
            distance = np.sqrt(np.sum((vertex - center) ** 2))
            assert distance <= radius + 1e-10

        # Check that the sphere is reasonably tight
        # The radius should not be much larger than the distance to the furthest vertex
        max_distance = max(np.sqrt(np.sum((v - center) ** 2)) for v in vertices)
        assert abs(radius - max_distance) < 1e-10

    def test_empty_array(self):
        """Test that empty array raises ValueError"""
        vertices = np.array([]).reshape(0, 3)
        with pytest.raises(IndexError):
            calculate_bounding_sphere(vertices)

    def test_sphere_contains_all_points(self):
        """Test that all points are contained within the bounding sphere"""
        # Generate 100 random points
        np.random.seed(42)  # For reproducibility
        vertices = np.random.rand(100, 3) * 10 - 5  # Points between -5 and 5

        center, radius = calculate_bounding_sphere(vertices)

        # Check that all points are within radius distance from center
        for vertex in vertices:
            distance = np.sqrt(np.sum((vertex - center) ** 2))
            assert distance <= radius + 1e-10  # Allow small numerical error

    def test_sphere_tightness(self):
        """Test that the bounding sphere is reasonably tight"""
        # Create points on a sphere of radius 1
        phi = np.linspace(0, 2 * np.pi, 20)
        theta = np.linspace(0, np.pi, 10)
        vertices = []
        for p in phi:
            for t in theta:
                x = np.sin(t) * np.cos(p)
                y = np.sin(t) * np.sin(p)
                z = np.cos(t)
                vertices.append([x, y, z])
        vertices = np.array(vertices)

        center, radius = calculate_bounding_sphere(vertices)

        # Center should be close to origin
        np.testing.assert_array_almost_equal(center, np.zeros(3), decimal=2)
        # Radius should be close to 1
        assert abs(radius - 1.0) < 0.1
