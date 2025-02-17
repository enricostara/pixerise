import pytest
import numpy as np
from src.kernel.transforming_mod import transform_vertex_normal


class TestTransformVertexNormal:
    @pytest.fixture
    def setup_rotations(self):
        """Setup common rotation angles for tests"""
        return {
            "no_rotation": np.array([0.0, 0.0, 0.0]),
            "x_90deg": np.array([np.pi / 2, 0.0, 0.0]),
            "y_90deg": np.array([0.0, np.pi / 2, 0.0]),
            "z_90deg": np.array([0.0, 0.0, np.pi / 2]),
        }

    def test_identity_transform(self, setup_rotations):
        """Test normal transformation with no rotations applied"""
        normal = np.array([0.0, 0.0, 1.0])
        result = transform_vertex_normal(
            normal=normal,
            rotation=setup_rotations["no_rotation"],
            camera_rotation=setup_rotations["no_rotation"],
            has_camera=False,
        )
        np.testing.assert_array_almost_equal(result, normal)

    def test_rotation_x(self, setup_rotations):
        """Test rotation around X axis"""
        normal = np.array([0.0, 1.0, 0.0])  # Unit normal pointing along Y
        result = transform_vertex_normal(
            normal=normal,
            rotation=setup_rotations["x_90deg"],
            camera_rotation=setup_rotations["no_rotation"],
            has_camera=False,
        )
        # After 90-degree rotation around X, Y should become Z
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_rotation_y(self, setup_rotations):
        """Test rotation around Y axis"""
        normal = np.array([0.0, 0.0, 1.0])  # Unit normal pointing along Z
        result = transform_vertex_normal(
            normal=normal,
            rotation=setup_rotations["y_90deg"],
            camera_rotation=setup_rotations["no_rotation"],
            has_camera=False,
        )
        # After 90-degree rotation around Y, Z should become -X
        expected = np.array([-1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_rotation_z(self, setup_rotations):
        """Test rotation around Z axis"""
        normal = np.array([1.0, 0.0, 0.0])  # Unit normal pointing along X
        result = transform_vertex_normal(
            normal=normal,
            rotation=setup_rotations["z_90deg"],
            camera_rotation=setup_rotations["no_rotation"],
            has_camera=False,
        )
        # After 90-degree rotation around Z, X should become Y
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_combined_rotation(self, setup_rotations):
        """Test combination of rotations"""
        normal = np.array([1.0, 0.0, 0.0])  # Unit normal pointing along X
        # Apply rotations in Y * X * Z order
        result = transform_vertex_normal(
            normal=normal,
            rotation=np.array(
                [np.pi / 4, np.pi / 4, np.pi / 4]
            ),  # 45 degrees each axis
            camera_rotation=setup_rotations["no_rotation"],
            has_camera=False,
        )
        # Result should still be a unit vector
        np.testing.assert_almost_equal(np.linalg.norm(result), 1.0)

    def test_camera_rotation(self, setup_rotations):
        """Test transformation with camera rotation"""
        normal = np.array([0.0, 0.0, 1.0])  # Unit normal pointing along Z
        result = transform_vertex_normal(
            normal=normal,
            rotation=setup_rotations["no_rotation"],
            camera_rotation=setup_rotations["y_90deg"],  # Camera rotated 90° around Y
            has_camera=True,
        )
        # Normal should be transformed relative to camera
        # With corrected Y rotation matrix, Z becomes negative X
        expected = np.array([-1.0, 0.0, 0.0])  # Points along negative camera X axis
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_normalization(self):
        """Test that output normal is always normalized"""
        # Test with a non-unit normal input
        normal = np.array([2.0, 2.0, 2.0])
        rotation = np.array([np.pi / 6, np.pi / 4, np.pi / 3])
        camera_rotation = np.array([0.0, np.pi / 2, 0.0])

        result = transform_vertex_normal(
            normal=normal,
            rotation=rotation,
            camera_rotation=camera_rotation,
            has_camera=True,
        )
        # Result should be a unit vector regardless of input length
        np.testing.assert_almost_equal(np.linalg.norm(result), 1.0)

    def test_camera_disabled(self, setup_rotations):
        """Test that camera transformations are ignored when has_camera is False"""
        normal = np.array([0.0, 0.0, 1.0])
        result = transform_vertex_normal(
            normal=normal,
            rotation=setup_rotations["no_rotation"],
            camera_rotation=setup_rotations["y_90deg"],  # Should be ignored
            has_camera=False,
        )
        # Normal should remain unchanged since model rotation is identity and camera is disabled
        np.testing.assert_array_almost_equal(result, normal)
