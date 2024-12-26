import pytest
import numpy as np
from kernel.culling_mod import cull_back_faces


class TestCullBackFaces:
    def test_single_front_facing_triangle(self):
        """Test culling with a single triangle facing the camera.
        Triangle is visible when dot(normal, view_vector) < 0"""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float32)
        indices = np.array([[0, 2, 1]], dtype=np.int32)  # CCW winding for front face
        camera_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        visible, normals = cull_back_faces(vertices, indices, camera_pos)
        
        assert len(visible) == 1
        assert visible[0] == True
        assert len(normals) == 1
        # Normal should point away from camera (negative z) when visible
        np.testing.assert_array_less(normals[0][2], 0.0)

    def test_single_back_facing_triangle(self):
        """Test culling with a single triangle facing away from camera.
        Triangle is culled when dot(normal, view_vector) > 0"""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)  # CW winding for back face
        camera_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        visible, normals = cull_back_faces(vertices, indices, camera_pos)
        
        assert len(visible) == 1
        assert visible[0] == False
        assert len(normals) == 0  # No normals returned for culled triangles

    def test_cube_front_view(self):
        """Test culling with a cube viewed from the front.
        Front faces have normals pointing away from camera."""
        vertices = np.array([
            # Front face vertices
            [-1, -1,  1],
            [ 1, -1,  1],
            [ 1,  1,  1],
            [-1,  1,  1],
            # Back face vertices
            [-1, -1, -1],
            [ 1, -1, -1],
            [ 1,  1, -1],
            [-1,  1, -1]
        ], dtype=np.float32)
        
        indices = np.array([
            # Front face (should be visible)
            [0, 2, 1], [0, 3, 2],  # CCW winding when viewed from front
            # Back face (should be culled)
            [4, 5, 6], [4, 6, 7],  # CCW winding when viewed from back
            # Left face
            [0, 4, 7], [0, 7, 3],
            # Right face
            [1, 6, 5], [1, 2, 6],
            # Top face
            [3, 7, 6], [3, 6, 2],
            # Bottom face
            [0, 1, 5], [0, 5, 4]
        ], dtype=np.int32)
        
        camera_pos = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        
        visible, normals = cull_back_faces(vertices, indices, camera_pos)
        
        # From front view, only front face triangles should be visible
        assert sum(visible) == 2  # Only front face triangles
        assert len(normals) == 2
        # Front face normals should point away from camera (negative z)
        for normal in normals:
            np.testing.assert_array_less(normal[2], 0.0)

    def test_empty_mesh(self):
        """Test culling with empty mesh"""
        vertices = np.array([], dtype=np.float32).reshape(0, 3)
        indices = np.array([], dtype=np.int32).reshape(0, 3)
        camera_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        visible, normals = cull_back_faces(vertices, indices, camera_pos)
        
        assert len(visible) == 0
        assert len(normals) == 0

    def test_degenerate_triangle(self):
        """Test culling with a degenerate triangle (zero area)"""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)
        camera_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        visible, normals = cull_back_faces(vertices, indices, camera_pos)
        
        assert len(visible) == 1
        assert not visible[0]  # Degenerate triangles should be culled
        assert len(normals) == 0

    def test_camera_inside_mesh(self):
        """Test culling with camera inside a cube mesh.
        Inside faces have normals pointing towards camera."""
        vertices = np.array([
            [-1, -1,  1],
            [ 1, -1,  1],
            [ 1,  1,  1],
            [-1,  1,  1],
            [-1, -1, -1],
            [ 1, -1, -1],
            [ 1,  1, -1],
            [-1,  1, -1]
        ], dtype=np.float32)
        
        indices = np.array([
            [0, 1, 2], [0, 2, 3],  # Front
            [4, 6, 5], [4, 7, 6],  # Back
            [0, 3, 7], [0, 7, 4],  # Left
            [1, 5, 6], [1, 6, 2],  # Right
            [3, 2, 6], [3, 6, 7],  # Top
            [0, 4, 5], [0, 5, 1]   # Bottom
        ], dtype=np.int32)
        
        camera_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Camera at center
        
        visible, normals = cull_back_faces(vertices, indices, camera_pos)
        
        # From inside, all faces should be visible (normals point away from faces)
        assert sum(visible) == 12
        assert len(normals) == 12

    def test_consistent_winding(self):
        """Test that culling is consistent with triangle winding order.
        CCW winding should be visible when viewed from front."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float32)
        
        # Test both winding orders
        indices_ccw = np.array([[0, 2, 1]], dtype=np.int32)  # Counter-clockwise (front)
        indices_cw = np.array([[0, 1, 2]], dtype=np.int32)   # Clockwise (back)
        
        camera_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        visible_ccw, normals_ccw = cull_back_faces(vertices, indices_ccw, camera_pos)
        visible_cw, normals_cw = cull_back_faces(vertices, indices_cw, camera_pos)
        
        # CCW should be visible, CW should be culled when viewed from front
        assert visible_ccw[0] == True
        assert visible_cw[0] == False
        # CCW normal should point away from camera
        assert len(normals_ccw) == 1
        np.testing.assert_array_less(normals_ccw[0][2], 0.0)
