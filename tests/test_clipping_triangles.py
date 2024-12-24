import pytest
import numpy as np
from kernel.clipping_mod import clip_triangle


class TestClipping:
    def test_clip_triangle(self):
        """Test triangle clipping against a plane"""
        plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # XY plane
        plane_d = 0.0  # Distance from the origin
        
        # Test case 1: Triangle completely above plane
        vertices = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0]
        ], dtype=np.float64)
        triangles, num_triangles = clip_triangle(vertices, plane_normal, plane_d)
        assert num_triangles == 1
        np.testing.assert_array_almost_equal(triangles[0], vertices)
        
        # Test case 2: Triangle completely below plane
        vertices = np.array([
            [0.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0]
        ], dtype=np.float64)
        triangles, num_triangles = clip_triangle(vertices, plane_normal, plane_d)
        assert num_triangles == 0

        # Test case 3: Triangle with one vertex above plane
        vertices = np.array([
            [0.0, 0.0, 1.0],  # above
            [1.0, 0.0, -1.0], # below
            [-1.0, 0.0, -1.0] # below
        ], dtype=np.float64)
        triangles, num_triangles = clip_triangle(vertices, plane_normal, plane_d)
        assert num_triangles == 1
        # Verify the clipped triangle has correct z coordinates
        z_coords = triangles[0, :, 2]  # get all z coordinates
        assert np.all(z_coords >= plane_d)

        # Test case 4: Triangle with two vertices above plane
        vertices = np.array([
            [0.0, 0.0, 1.0],  # above
            [1.0, 0.0, 1.0],  # above
            [0.0, 1.0, -1.0]  # below
        ], dtype=np.float64)
        triangles, num_triangles = clip_triangle(vertices, plane_normal, plane_d)
        assert num_triangles == 2
        # Verify both triangles have correct z coordinates
        for i in range(2):
            assert np.all(triangles[i][:, 2] >= plane_d)

        # Test case 5: Triangle exactly on plane
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float64)
        triangles, num_triangles = clip_triangle(vertices, plane_normal, plane_d)
        assert num_triangles == 1
        np.testing.assert_array_almost_equal(triangles[0], vertices)

        # Test case 6: Triangle with vertex exactly on plane
        vertices = np.array([
            [0.0, 0.0, 0.0],  # on plane
            [1.0, 0.0, 1.0],  # above
            [0.0, 1.0, -1.0]  # below
        ], dtype=np.float64)
        triangles, num_triangles = clip_triangle(vertices, plane_normal, plane_d)
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

        # Test case 7: Triangle with plane_d less than 0
        plane_d = -1.0  # Distance from the origin
        vertices = np.array([
            [0.0, 0.0, 1.0],  # above
            [1.0, 0.0, 1.0],  # above
            [0.0, 1.0, -1.0]   # below
        ], dtype=np.float64)
        triangles, num_triangles = clip_triangle(vertices, plane_normal, plane_d)
        assert num_triangles == 1
        # Verify the clipped triangle has correct z coordinates
        z_coords = triangles[0, :, 2]  # get all z coordinates
        assert np.all(z_coords >= plane_d)

        # Test case 8: Verify clockwise order is maintained after clipping
        plane_d = 0.0
        # Create a clockwise triangle that will be clipped
        vertices = np.array([
            [0.0, 0.0, 1.0],   # above (vertex 0)
            [0.0, 1.0, -0.5],  # below (vertex 1)
            [1.0, 0.0, -0.5]   # below (vertex 2)
        ], dtype=np.float64)
        
        def is_clockwise(triangle, normal):
            """Helper function to check if triangle vertices are in clockwise order
            when viewed from the direction of the normal vector"""
            # Project triangle onto the plane perpendicular to the normal
            # For XY plane (normal = [0,0,1]), we just need to look at x,y coordinates
            edge1 = triangle[1, :2] - triangle[0, :2]  # Only look at x,y components
            edge2 = triangle[2, :2] - triangle[0, :2]
            # Calculate 2D cross product (positive means counter-clockwise)
            cross_2d = edge1[0] * edge2[1] - edge1[1] * edge2[0]
            return cross_2d < 0  # Negative means clockwise when viewed from above
            
        # Verify input triangle is clockwise
        assert is_clockwise(vertices, plane_normal)
        
        # Clip the triangle
        triangles, num_triangles = clip_triangle(vertices, plane_normal, plane_d)
        assert num_triangles == 1
        
        # Verify the clipped triangle maintains clockwise order
        assert is_clockwise(triangles[0], plane_normal)
        
        # Test another case with two vertices above
        vertices = np.array([
            [0.0, 0.0, 1.0],   # above
            [0.0, 1.0, 1.0],   # above
            [1.0, 0.0, -0.5]   # below
        ], dtype=np.float64)
        
        # Verify input triangle is clockwise
        assert is_clockwise(vertices, plane_normal)
        
        # Clip the triangle
        triangles, num_triangles = clip_triangle(vertices, plane_normal, plane_d)
        assert num_triangles == 2
        
        # Verify both output triangles maintain clockwise order
        assert is_clockwise(triangles[0], plane_normal)
        assert is_clockwise(triangles[1], plane_normal)
