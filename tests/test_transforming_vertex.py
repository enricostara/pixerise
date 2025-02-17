import pytest
import numpy as np
from pixerise import Canvas, ViewPort, Renderer
from kernel.transforming_mod import transform_vertex


class TestTransformVertex:
    @pytest.fixture
    def setup(self):
        """Setup test environment"""
        canvas = Canvas((100, 100))
        viewport = ViewPort((100, 100), 1, canvas)
        renderer = Renderer(canvas, viewport)
        return canvas, viewport, renderer

    def test_identity_transform(self, setup):
        """Test vertex transformation with no transformations applied"""
        canvas, viewport, renderer = setup
        vertex = np.array([1.0, 2.0, 3.0])
        transform = {}  # Empty transform should act as identity

        result = transform_vertex(
            vertex,
            transform.get("translation", np.zeros(3)),
            transform.get("rotation", np.zeros(3)),
            transform.get("scale", np.ones(3)),
            np.zeros(3),
            np.zeros(3),
            False,
        )
        np.testing.assert_array_almost_equal(result, vertex)

    def test_translation(self, setup):
        """Test translation transformation"""
        canvas, viewport, renderer = setup
        vertex = np.array([1.0, 2.0, 3.0])
        transform = {"translation": np.array([10.0, 20.0, 30.0])}

        expected = np.array([11.0, 22.0, 33.0])
        result = transform_vertex(
            vertex,
            transform.get("translation", np.zeros(3)),
            transform.get("rotation", np.zeros(3)),
            transform.get("scale", np.ones(3)),
            np.zeros(3),
            np.zeros(3),
            False,
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_scale(self, setup):
        """Test scale transformation"""
        canvas, viewport, renderer = setup
        vertex = np.array([1.0, 2.0, 3.0])
        transform = {"scale": np.array([2.0, 3.0, 4.0])}

        expected = np.array([2.0, 6.0, 12.0])
        result = transform_vertex(
            vertex,
            transform.get("translation", np.zeros(3)),
            transform.get("rotation", np.zeros(3)),
            transform.get("scale", np.array([2.0, 3.0, 4.0])),
            np.zeros(3),
            np.zeros(3),
            False,
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotation_x(self, setup):
        """Test rotation around X axis"""
        canvas, viewport, renderer = setup
        vertex = np.array([1.0, 1.0, 0.0])
        transform = {
            "rotation": np.array([np.pi / 2, 0.0, 0.0])  # 90 degrees around X
        }

        # After 90-degree rotation around X, y should become z and z should become -y
        expected = np.array([1.0, 0.0, 1.0])
        result = transform_vertex(
            vertex,
            transform.get("translation", np.zeros(3)),
            transform.get("rotation", np.array([np.pi / 2, 0.0, 0.0])),
            transform.get("scale", np.ones(3)),
            np.zeros(3),
            np.zeros(3),
            False,
        )
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_rotation_y(self, setup):
        """Test rotation around Y axis"""
        canvas, viewport, renderer = setup
        vertex = np.array([1.0, 1.0, 0.0])
        transform = {
            "rotation": np.array([0.0, np.pi / 2, 0.0])  # 90 degrees around Y
        }

        # After 90-degree rotation around Y, x should become -z and z should become x
        expected = np.array([0.0, 1.0, -1.0])
        result = transform_vertex(
            vertex,
            transform.get("translation", np.zeros(3)),
            transform.get("rotation", np.array([0.0, np.pi / 2, 0.0])),
            transform.get("scale", np.ones(3)),
            np.zeros(3),
            np.zeros(3),
            False,
        )
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_rotation_z(self, setup):
        """Test rotation around Z axis"""
        canvas, viewport, renderer = setup
        vertex = np.array([1.0, 0.0, 1.0])
        transform = {
            "rotation": np.array([0.0, 0.0, np.pi / 2])  # 90 degrees around Z
        }

        # After 90-degree rotation around Z, x should become -y and y should become x
        expected = np.array([0.0, 1.0, 1.0])
        result = transform_vertex(
            vertex,
            transform.get("translation", np.zeros(3)),
            transform.get("rotation", np.array([0.0, 0.0, np.pi / 2])),
            transform.get("scale", np.ones(3)),
            np.zeros(3),
            np.zeros(3),
            False,
        )
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_combined_transform(self, setup):
        """Test combination of translation, rotation, and scale"""
        canvas, viewport, renderer = setup
        vertex = np.array([1.0, 0.0, 0.0])
        transform = {
            "translation": np.array([1.0, 1.0, 1.0]),
            "rotation": np.array([0.0, 0.0, np.pi / 2]),  # 90 degrees around Z
            "scale": np.array([2.0, 2.0, 2.0]),
        }

        # Scale by 2, rotate 90° around Z (x becomes -y, y becomes x), then translate by 1
        expected = np.array([1.0, 3.0, 1.0])
        result = transform_vertex(
            vertex,
            transform.get("translation", np.zeros(3)),
            transform.get("rotation", np.array([0.0, 0.0, np.pi / 2])),
            transform.get("scale", np.array([2.0, 2.0, 2.0])),
            np.zeros(3),
            np.zeros(3),
            False,
        )
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_camera_transform(self, setup):
        """Test transformation with camera"""
        canvas, viewport, renderer = setup
        vertex = np.array([0.0, 0.0, 1.0])
        transform = {"translation": np.array([1.0, 0.0, 0.0])}

        # Add camera looking down -Z axis from 10 units away
        camera_transform = {
            "translation": np.array([0.0, 0.0, -10.0]),
            "rotation": np.array([0.0, 0.0, 0.0]),
        }

        # Vertex should be transformed relative to camera
        result = transform_vertex(
            vertex,
            transform.get("translation", np.zeros(3)),
            transform.get("rotation", np.zeros(3)),
            transform.get("scale", np.ones(3)),
            camera_transform["translation"],
            camera_transform["rotation"],
            True,
        )
        expected = np.array([1.0, 0.0, 11.0])  # Vertex is now 11 units away from camera
        np.testing.assert_array_almost_equal(result, expected)
