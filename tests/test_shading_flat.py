import pytest
import numpy as np
from kernel.shading_mod import triangle_flat_shading


class TestShading:
    def test_compute_flat_shading(self):
        """Test flat shading computation with various lighting scenarios"""
        # Test case 1: Single triangle facing the light directly
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float64)
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # Normal pointing towards Z
        light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # Light from Z direction
        material_color = np.array([255, 0, 0], dtype=np.uint8)  # Red material
        ambient = 0.1

        color = triangle_flat_shading(vertices, normal, light_dir, material_color, ambient)
        assert color.shape == (3,)  # RGB color
        # Maximum illumination (ambient + full diffuse) for directly facing light
        expected_color = np.array(material_color * (ambient + (1.0 - ambient) * 1.0), dtype=np.uint8)
        np.testing.assert_array_equal(color, expected_color)

        # Test case 2: Triangle at 45 degrees to light
        normal = np.array([0.707, 0.0, 0.707], dtype=np.float64)  # 45 degrees to Z
        color = triangle_flat_shading(vertices, normal, light_dir, material_color, ambient)
        # Illumination should be cos(45°) ≈ 0.707 for 45-degree angle
        expected_color = np.array(material_color * (ambient + (1.0 - ambient) * 0.707), dtype=np.uint8)
        np.testing.assert_array_equal(color, expected_color)

        # Test case 3: Triangle facing away from light
        normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)  # Away from light
        color = triangle_flat_shading(vertices, normal, light_dir, material_color, ambient)
        # Only ambient light when facing away
        expected_color = np.array(material_color * ambient, dtype=np.uint8)
        np.testing.assert_array_equal(color, expected_color)

        # Test case 4: Different material colors
        material_colors = np.array([
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255]     # Blue
        ], dtype=np.uint8)
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        for mat_color in material_colors:
            color = triangle_flat_shading(vertices, normal, light_dir, mat_color, ambient)
            expected_color = np.array(mat_color * (ambient + (1.0 - ambient) * 1.0), dtype=np.uint8)
            np.testing.assert_array_equal(color, expected_color)

        # Test case 5: Different ambient values
        ambient = 0.5  # Higher ambient light
        color = triangle_flat_shading(vertices, normal, light_dir, material_color, ambient)
        expected_color = np.array(material_color * (ambient + (1.0 - ambient) * 1.0), dtype=np.uint8)
        np.testing.assert_array_equal(color, expected_color)

        # Test case 6: Zero ambient light
        ambient = 0.0
        normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)  # Away from light
        color = triangle_flat_shading(vertices, normal, light_dir, material_color, ambient)
        # Should be completely dark when facing away with no ambient
        expected_color = np.zeros(3, dtype=np.uint8)
        np.testing.assert_array_equal(color, expected_color)

    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float64)
        normal = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # Zero normal
        light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        material_color = np.array([255, 255, 255], dtype=np.uint8)
        ambient = 0.1

        # Test case 1: Zero-length normal
        color = triangle_flat_shading(vertices, normal, light_dir, material_color, ambient)
        # Should handle zero normal gracefully (treat as unlit)
        expected_color = np.array(material_color * ambient, dtype=np.uint8)
        np.testing.assert_array_equal(color, expected_color)

        # Test case 2: Zero-length light direction
        light_dir = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        color = triangle_flat_shading(vertices, normal, light_dir, material_color, ambient)
        # Should handle zero light direction gracefully (treat as unlit)
        expected_color = np.array(material_color * ambient, dtype=np.uint8)
        np.testing.assert_array_equal(color, expected_color)
