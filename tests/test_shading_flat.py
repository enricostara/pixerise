import numpy as np
from kernel.shading_mod import triangle_flat_shading


class TestTriangleFlatShading:
    def test_basic_shading(self):
        """Test basic shading with light directly above"""
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        color = np.array([255, 0, 0], dtype=np.float32)  # Pure red

        shaded_color = triangle_flat_shading(normal, light_dir, color)

        # Full illumination (normal aligned with light)
        assert shaded_color[0] > 240  # Red should be near max
        assert shaded_color[1] == 0  # No green
        assert shaded_color[2] == 0  # No blue

    def test_grazing_angle(self):
        """Test shading with light at grazing angle"""
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        light_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Light from side
        color = np.array([255, 255, 255], dtype=np.float32)  # White

        shaded_color = triangle_flat_shading(normal, light_dir, color)

        # Only ambient light (perpendicular light)
        assert np.all(shaded_color == np.array([25, 25, 25]))  # ~0.1 * 255

    def test_back_lighting(self):
        """Test shading with light from behind"""
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        light_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # Light from behind
        color = np.array([255, 255, 255], dtype=np.float32)  # White
        ambient = 0.1

        shaded_color = triangle_flat_shading(normal, light_dir, color, ambient)

        # Only ambient light (light from behind)
        expected = np.array([25, 25, 25])  # ambient * 255
        np.testing.assert_array_almost_equal(shaded_color, expected)

    def test_custom_ambient(self):
        """Test shading with custom ambient light"""
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        light_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # Light from behind
        color = np.array([255, 255, 255], dtype=np.float32)  # White
        ambient = 0.5  # Higher ambient

        shaded_color = triangle_flat_shading(normal, light_dir, color, ambient)

        # Higher ambient light only
        expected = np.array([127, 127, 127])  # ~0.5 * 255
        np.testing.assert_array_almost_equal(shaded_color, expected, decimal=0)

    def test_non_normalized_vectors(self):
        """Test shading with non-normalized vectors"""
        normal = np.array([0.0, 0.0, 2.0], dtype=np.float32)  # Non-normalized
        light_dir = np.array([0.0, 0.0, 3.0], dtype=np.float32)  # Non-normalized
        color = np.array([255, 0, 0], dtype=np.float32)  # Red

        shaded_color = triangle_flat_shading(normal, light_dir, color)

        # Should still work with non-normalized vectors
        assert shaded_color[0] > 240  # Red should be near max
        assert shaded_color[1] == 0  # No green
        assert shaded_color[2] == 0  # No blue

    def test_zero_color(self):
        """Test shading with zero color"""
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        color = np.array([0, 0, 0], dtype=np.float32)  # Black

        shaded_color = triangle_flat_shading(normal, light_dir, color)

        # Should be black regardless of lighting
        assert np.all(shaded_color == 0)

    def test_different_material_colors(self):
        """Test shading with different material colors"""
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        material_colors = np.array(
            [
                [255, 0, 0],  # Red
                [0, 255, 0],  # Green
                [0, 0, 255],  # Blue
            ],
            dtype=np.float32,
        )

        for mat_color in material_colors:
            color = triangle_flat_shading(normal, light_dir, mat_color)
            # Full illumination for directly facing light
            np.testing.assert_array_almost_equal(color, mat_color)

    def test_edge_cases(self):
        """Test edge cases like zero vectors"""
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        material_color = np.array([255, 255, 255], dtype=np.float32)
        ambient = 0.1

        # Test zero normal
        normal = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        color = triangle_flat_shading(normal, light_dir, material_color, ambient)
        expected = np.array([25, 25, 25])  # ambient * 255
        np.testing.assert_array_almost_equal(color, expected)

        # Test zero light direction
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        light_dir = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        color = triangle_flat_shading(normal, light_dir, material_color, ambient)
        expected = np.array([25, 25, 25])  # ambient * 255
        np.testing.assert_array_almost_equal(color, expected)
