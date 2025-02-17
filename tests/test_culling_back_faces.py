import pytest
import numpy as np
from kernel.culling_mod import cull_back_faces


class TestCullingBackFaces:
    @pytest.fixture
    def setup_vertices(self):
        """Setup test vertices in camera space"""
        # Basic triangle vertices in camera space
        vertices = np.array(
            [
                [-1.0, -1.0, 2.0],  # v0
                [1.0, -1.0, 2.0],  # v1
                [0.0, 1.0, 2.0],  # v2
                [0.0, 1.0, 3.0],  # v3 (for additional triangles)
            ],
            dtype=np.float32,
        )
        return vertices

    def verify_normal_properties(self, normal):
        """Helper method to verify normal vector properties"""
        # Check if normal is normalized (length ≈ 1)
        np.testing.assert_allclose(np.linalg.norm(normal), 1.0, rtol=1e-6)

    def test_back_facing_triangle(self, setup_vertices):
        """Test that a back-facing triangle is correctly culled"""
        vertices = setup_vertices
        # Clockwise winding order for back face
        indices = np.array([[0, 2, 1]], dtype=np.int32)
        visible_indices, normals = cull_back_faces(vertices, indices)

        # Should have no visible triangles
        assert len(visible_indices) == 0
        assert len(normals) == 0

    def test_multiple_triangles(self, setup_vertices):
        """Test culling of multiple triangles with different orientations"""
        vertices = setup_vertices
        indices = np.array(
            [
                [0, 1, 2],  # front-facing
                [1, 2, 3],  # side triangle
                [2, 1, 0],  # back-facing
            ],
            dtype=np.int32,
        )
        visible_indices, normals = cull_back_faces(vertices, indices)

        # First triangle should be visible
        assert any(
            np.array_equal(visible_indices[i], indices[0])
            for i in range(len(visible_indices))
        )
        # Back-facing triangle should not be in visible indices
        assert not any(
            np.array_equal(visible_indices[i], indices[2])
            for i in range(len(visible_indices))
        )

        # Verify normal properties for all visible triangles
        for normal in normals:
            self.verify_normal_properties(normal)

    def test_degenerate_triangle(self):
        """Test handling of degenerate (zero-area) triangles"""
        vertices = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        indices = np.array([[0, 1, 2]], dtype=np.int32)
        visible_indices, normals = cull_back_faces(vertices, indices)

        # Degenerate triangle should not be visible
        assert len(visible_indices) == 0
        assert len(normals) == 0

    def test_empty_input(self):
        """Test handling of empty input arrays"""
        vertices = np.array([], dtype=np.float32).reshape(0, 3)
        indices = np.array([], dtype=np.int32).reshape(0, 3)
        visible_indices, normals = cull_back_faces(vertices, indices)

        assert len(visible_indices) == 0
        assert len(normals) == 0

    def test_large_coordinates(self):
        """Test triangles with large coordinate values"""
        vertices = np.array(
            [[-1e6, -1e6, 2e6], [1e6, -1e6, 2e6], [0.0, 1e6, 2e6]], dtype=np.float32
        )
        indices = np.array([[0, 1, 2]], dtype=np.int32)
        visible_indices, normals = cull_back_faces(vertices, indices)

        # Front-facing triangle should be visible
        assert len(visible_indices) == 1
        np.testing.assert_array_equal(visible_indices[0], indices[0])
        if len(normals) > 0:
            self.verify_normal_properties(normals[0])
