import pytest
import numpy as np
from kernel.clipping_mod import clip_triangle


class TestClipping:
    def test_clip_triangle(self):
        """Test triangle clipping against a plane"""
        plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # XY plane
        
        # Test case 1: Triangle completely above plane
        vertices = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0]
        ], dtype=np.float64)
        triangles, num_triangles = clip_triangle(plane_normal, vertices)
        assert num_triangles == 1
        np.testing.assert_array_almost_equal(triangles[0], vertices)

        # Test case 2: Triangle completely below plane
        vertices = np.array([
            [0.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0]
        ], dtype=np.float64)
        triangles, num_triangles = clip_triangle(plane_normal, vertices)
        assert num_triangles == 0

        # Test case 3: Triangle with one vertex above plane
        vertices = np.array([
            [0.0, 0.0, 1.0],  # above
            [1.0, 0.0, -1.0], # below
            [-1.0, 0.0, -1.0] # below
        ], dtype=np.float64)
        triangles, num_triangles = clip_triangle(plane_normal, vertices)
        assert num_triangles == 1
        # Verify the clipped triangle has correct z coordinates
        z_coords = triangles[0, :, 2]  # get all z coordinates
        assert np.sum(z_coords > 0) == 1  # one vertex above plane
        assert np.sum(np.abs(z_coords) < 1e-6) == 2  # two vertices on plane

        # Test case 4: Triangle with two vertices above plane
        vertices = np.array([
            [0.0, 0.0, 1.0],   # above
            [1.0, 0.0, 1.0],   # above
            [0.0, 1.0, -1.0]   # below
        ], dtype=np.float64)
        triangles, num_triangles = clip_triangle(plane_normal, vertices)
        assert num_triangles == 2
        # Verify both triangles have correct z coordinates
        for i in range(2):
            z_coords = triangles[i, :, 2]
            assert np.sum(z_coords > 0) >= 1  # at least one vertex above plane
            assert np.sum(np.abs(z_coords) < 1e-6) >= 1  # at least one vertex on plane

        # Test case 5: Triangle with vertex exactly on plane
        vertices = np.array([
            [0.0, 0.0, 0.0],  # on plane
            [1.0, 0.0, 1.0],  # above
            [0.0, 1.0, -1.0]  # below
        ], dtype=np.float64)
        triangles, num_triangles = clip_triangle(plane_normal, vertices)
        assert num_triangles == 1
        # Check that the resulting triangle has:
        # - one vertex from original triangle (the one above plane)
        # - one vertex from original triangle (the one on plane)
        # - one vertex at z=0 (intersection point)
        vertices_on_plane = 0
        vertices_above_plane = 0
        for i in range(3):
            if abs(triangles[0][i][2]) < 1e-6:
                vertices_on_plane += 1
            elif triangles[0][i][2] > 0:
                vertices_above_plane += 1
        assert vertices_on_plane == 2
        assert vertices_above_plane == 1
