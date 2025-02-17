import numpy as np
from kernel.clipping_mod import clip_triangle, clip_triangle_and_normals


class TestClipping:
    def test_clip_triangle(self):
        """Test triangle clipping against a plane"""
        plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # XY plane
        plane_d = 0.0  # Distance from the origin

        # Test case 1: Triangle completely above plane
        vertices = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64
        )
        triangles, num_triangles = clip_triangle(vertices, plane_normal, plane_d)
        assert num_triangles == 1
        np.testing.assert_array_almost_equal(triangles[0], vertices)

        # Test case 2: Triangle completely below plane
        vertices = np.array(
            [[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]], dtype=np.float64
        )
        triangles, num_triangles = clip_triangle(vertices, plane_normal, plane_d)
        assert num_triangles == 0

        # Test case 3: Triangle with one vertex above plane
        vertices = np.array(
            [
                [0.0, 0.0, 1.0],  # above
                [1.0, 0.0, -1.0],  # below
                [-1.0, 0.0, -1.0],  # below
            ],
            dtype=np.float64,
        )
        triangles, num_triangles = clip_triangle(vertices, plane_normal, plane_d)
        assert num_triangles == 1
        # Verify the clipped triangle has correct z coordinates
        z_coords = triangles[0, :, 2]  # get all z coordinates
        assert np.all(z_coords >= plane_d)

        # Test case 4: Triangle with two vertices above plane
        vertices = np.array(
            [
                [0.0, 0.0, 1.0],  # above
                [1.0, 0.0, 1.0],  # above
                [0.0, 1.0, -1.0],  # below
            ],
            dtype=np.float64,
        )
        triangles, num_triangles = clip_triangle(vertices, plane_normal, plane_d)
        assert num_triangles == 2
        # Verify both triangles have correct z coordinates
        for i in range(2):
            assert np.all(triangles[i][:, 2] >= plane_d)

        # Test case 5: Triangle exactly on plane
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        )
        triangles, num_triangles = clip_triangle(vertices, plane_normal, plane_d)
        assert num_triangles == 1
        np.testing.assert_array_almost_equal(triangles[0], vertices)

        # Test case 6: Triangle with vertex exactly on plane
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],  # on plane
                [1.0, 0.0, 1.0],  # above
                [0.0, 1.0, -1.0],  # below
            ],
            dtype=np.float64,
        )
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
        vertices = np.array(
            [
                [0.0, 0.0, 1.0],  # above
                [1.0, 0.0, 1.0],  # above
                [0.0, 1.0, -1.0],  # below
            ],
            dtype=np.float64,
        )
        triangles, num_triangles = clip_triangle(vertices, plane_normal, plane_d)
        assert num_triangles == 1
        # Verify the clipped triangle has correct z coordinates
        z_coords = triangles[0, :, 2]  # get all z coordinates
        assert np.all(z_coords >= plane_d)

        # Test case 8: Verify clockwise order is maintained after clipping
        plane_d = 0.0
        # Create a clockwise triangle that will be clipped
        vertices = np.array(
            [
                [0.0, 0.0, 1.0],  # above (vertex 0)
                [0.0, 1.0, -0.5],  # below (vertex 1)
                [1.0, 0.0, -0.5],  # below (vertex 2)
            ],
            dtype=np.float64,
        )

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
        vertices = np.array(
            [
                [0.0, 0.0, 1.0],  # above
                [0.0, 1.0, 1.0],  # above
                [1.0, 0.0, -0.5],  # below
            ],
            dtype=np.float64,
        )

        # Verify input triangle is clockwise
        assert is_clockwise(vertices, plane_normal)

        # Clip the triangle
        triangles, num_triangles = clip_triangle(vertices, plane_normal, plane_d)
        assert num_triangles == 2

        # Verify both output triangles maintain clockwise order
        assert is_clockwise(triangles[0], plane_normal)
        assert is_clockwise(triangles[1], plane_normal)

        # Test case 9: Triangle clipped into two with potential winding order issues
        vertices = np.array(
            [
                [0.0, -1.0, 1.0],  # above
                [0.0, 1.0, 1.0],  # above
                [2.0, 0.0, -1.0],  # below
            ],
            dtype=np.float64,
        )

        # Verify input triangle is clockwise
        assert is_clockwise(vertices, plane_normal)

        # Clip the triangle
        triangles, num_triangles = clip_triangle(vertices, plane_normal, plane_d)
        assert num_triangles == 2

        # Verify both output triangles maintain clockwise order
        assert is_clockwise(triangles[0], plane_normal)
        assert is_clockwise(triangles[1], plane_normal)

        # Test case 10: Two vertices exactly on plane
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],  # on plane
                [0.5, 0.5, 1.0],  # above
                [1.0, 0.0, 0.0],  # on plane
            ],
            dtype=np.float64,
        )

        # Verify input triangle is clockwise
        assert is_clockwise(vertices, plane_normal)

        # Clip the triangle
        triangles, num_triangles = clip_triangle(vertices, plane_normal, plane_d)
        assert num_triangles == 1
        assert is_clockwise(triangles[0], plane_normal)

        # Test case 11: Non-axis-aligned clipping plane
        plane_normal = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        plane_normal = plane_normal / np.linalg.norm(plane_normal)  # normalize
        plane_d = 0.0

        vertices = np.array(
            [
                [1.0, 1.0, 1.0],  # above
                [-1.0, -1.0, 1.0],  # below
                [-1.0, 1.0, -1.0],  # below
            ],
            dtype=np.float64,
        )

        # Verify input triangle is clockwise when viewed along plane normal
        assert is_clockwise(vertices, plane_normal)

        # Clip the triangle
        triangles, num_triangles = clip_triangle(vertices, plane_normal, plane_d)
        assert num_triangles == 1
        assert is_clockwise(triangles[0], plane_normal)

        # Test case 12: Almost parallel to clipping plane
        plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        plane_d = 0.0

        vertices = np.array(
            [
                [0.0, 0.0, 0.001],  # slightly above
                [0.0, 1.0, 0.0],  # exactly on
                [1.0, 0.0, -0.001],  # slightly below
            ],
            dtype=np.float64,
        )

        # Verify input triangle is clockwise
        assert is_clockwise(vertices, plane_normal)

        # Clip the triangle
        triangles, num_triangles = clip_triangle(vertices, plane_normal, plane_d)
        assert num_triangles == 1
        assert is_clockwise(triangles[0], plane_normal)

    def test_clip_triangle_and_normals(self):
        """Test triangle clipping with normal interpolation"""
        plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # XY plane
        plane_d = 0.0  # Distance from the origin

        # Test case 1: Triangle completely above plane
        vertices = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64
        )
        vertex_normals = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        )
        triangles, normals, num_triangles = clip_triangle_and_normals(
            vertices, vertex_normals, plane_normal, plane_d
        )
        assert num_triangles == 1
        np.testing.assert_array_almost_equal(triangles[0], vertices)
        np.testing.assert_array_almost_equal(normals[0], vertex_normals)

        # Test case 2: Triangle completely below plane
        vertices = np.array(
            [[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]], dtype=np.float64
        )
        vertex_normals = np.array(
            [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        )
        triangles, normals, num_triangles = clip_triangle_and_normals(
            vertices, vertex_normals, plane_normal, plane_d
        )
        assert num_triangles == 0

        # Test case 3: Triangle with one vertex above plane
        vertices = np.array(
            [
                [0.0, 0.0, 1.0],  # above
                [1.0, 0.0, -1.0],  # below
                [-1.0, 0.0, -1.0],  # below
            ],
            dtype=np.float64,
        )
        vertex_normals = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
        )
        triangles, normals, num_triangles = clip_triangle_and_normals(
            vertices, vertex_normals, plane_normal, plane_d
        )
        assert num_triangles == 1
        # Verify the clipped triangle has correct z coordinates and interpolated normals
        z_coords = triangles[0, :, 2]  # get all z coordinates
        assert np.all(z_coords >= plane_d)
        # Verify normal interpolation at intersection points
        for i in range(3):
            assert np.linalg.norm(normals[0][i]) > 0.99  # Normals should be normalized

        # Test case 4: Triangle with two vertices above plane
        vertices = np.array(
            [
                [0.0, 0.0, 1.0],  # above
                [1.0, 0.0, 1.0],  # above
                [0.0, 1.0, -1.0],  # below
            ],
            dtype=np.float64,
        )
        vertex_normals = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        )
        triangles, normals, num_triangles = clip_triangle_and_normals(
            vertices, vertex_normals, plane_normal, plane_d
        )
        assert num_triangles == 2
        # Verify both triangles have correct z coordinates and normals
        for i in range(2):
            assert np.all(triangles[i][:, 2] >= plane_d)
            for j in range(3):
                assert (
                    np.linalg.norm(normals[i][j]) > 0.99
                )  # Normals should be normalized

        # Test case 5: Triangle exactly on plane
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        )
        vertex_normals = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        )
        triangles, normals, num_triangles = clip_triangle_and_normals(
            vertices, vertex_normals, plane_normal, plane_d
        )
        assert num_triangles == 1
        np.testing.assert_array_almost_equal(triangles[0], vertices)
        np.testing.assert_array_almost_equal(normals[0], vertex_normals)

        # Test case 6: Triangle with vertex exactly on plane
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],  # on plane
                [1.0, 0.0, 1.0],  # above
                [0.0, 1.0, -1.0],  # below
            ],
            dtype=np.float64,
        )
        vertex_normals = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        )
        triangles, normals, num_triangles = clip_triangle_and_normals(
            vertices, vertex_normals, plane_normal, plane_d
        )
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
            assert np.linalg.norm(normals[0][i]) > 0.99  # Normals should be normalized
        assert vertices_on_plane == 2
        assert vertices_above_plane == 1
