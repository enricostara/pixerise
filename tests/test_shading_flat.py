import pytest
import numpy as np
from kernel.shading_mod import compute_flat_shading


class TestShading:
    def test_compute_flat_shading(self):
        """Test flat shading computation with various lighting scenarios"""
        # Test case 1: Single triangle facing the light directly
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float64)
        indices = np.array([[0, 1, 2]], dtype=np.int32)
        normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)  # Normal pointing towards Z
        light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # Light from Z direction
        material_color = np.array([255, 0, 0], dtype=np.uint8)  # Red material
        ambient = 0.1

        colors = compute_flat_shading(vertices, indices, normals, light_dir, material_color, ambient)
        assert colors.shape == (1, 3)  # One triangle, RGB color
        # Maximum illumination (ambient + full diffuse) for directly facing light
        expected_color = np.array(material_color * (ambient + (1.0 - ambient) * 1.0), dtype=np.uint8)
        np.testing.assert_array_equal(colors[0], expected_color)

        # Test case 2: Triangle at 45 degrees to light
        normals = np.array([[0.707, 0.0, 0.707]], dtype=np.float64)  # 45 degrees to Z
        colors = compute_flat_shading(vertices, indices, normals, light_dir, material_color, ambient)
        # Illumination should be cos(45°) ≈ 0.707 for 45-degree angle
        expected_color = np.array(material_color * (ambient + (1.0 - ambient) * 0.707), dtype=np.uint8)
        np.testing.assert_array_equal(colors[0], expected_color)

        # Test case 3: Triangle facing away from light
        normals = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)  # Away from light
        colors = compute_flat_shading(vertices, indices, normals, light_dir, material_color, ambient)
        # Only ambient light when facing away
        expected_color = np.array(material_color * ambient, dtype=np.uint8)
        np.testing.assert_array_equal(colors[0], expected_color)

        # Test case 4: Multiple triangles with different orientations
        indices = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int32)
        normals = np.array([
            [0.0, 0.0, 1.0],   # Facing light
            [0.0, 0.0, -1.0]   # Away from light
        ], dtype=np.float64)
        colors = compute_flat_shading(vertices, indices, normals, light_dir, material_color, ambient)
        assert colors.shape == (2, 3)  # Two triangles, RGB colors
        expected_colors = np.array([
            material_color * (ambient + (1.0 - ambient) * 1.0),  # Fully lit
            material_color * ambient  # Only ambient
        ], dtype=np.uint8)
        np.testing.assert_array_equal(colors, expected_colors)

        # Test case 5: Different material colors
        material_colors = np.array([
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255]     # Blue
        ], dtype=np.uint8)
        indices = np.array([[0, 1, 2]], dtype=np.int32)
        normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        for color in material_colors:
            colors = compute_flat_shading(vertices, indices, normals, light_dir, color, ambient)
            expected_color = np.array(color * (ambient + (1.0 - ambient) * 1.0), dtype=np.uint8)
            np.testing.assert_array_equal(colors[0], expected_color)

        # Test case 6: Different ambient values
        ambient = 0.5  # Higher ambient light
        colors = compute_flat_shading(vertices, indices, normals, light_dir, material_color, ambient)
        expected_color = np.array(material_color * (ambient + (1.0 - ambient) * 1.0), dtype=np.uint8)
        np.testing.assert_array_equal(colors[0], expected_color)

        # Test case 7: Zero ambient light
        ambient = 0.0
        normals = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)  # Away from light
        colors = compute_flat_shading(vertices, indices, normals, light_dir, material_color, ambient)
        # Should be completely dark when facing away with no ambient
        expected_color = np.zeros(3, dtype=np.uint8)
        np.testing.assert_array_equal(colors[0], expected_color)

    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        indices = np.array([[0, 0, 0]], dtype=np.int32)
        normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        material_color = np.array([255, 255, 255], dtype=np.uint8)
        ambient = 0.1

        # Test case 1: Zero-length normal
        normals = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        colors = compute_flat_shading(vertices, indices, normals, light_dir, material_color, ambient)
        # Should handle zero normal gracefully (treat as unlit)
        expected_color = np.array(material_color * ambient, dtype=np.uint8)
        np.testing.assert_array_equal(colors[0], expected_color)

        # Test case 2: Zero-length light direction
        light_dir = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        colors = compute_flat_shading(vertices, indices, normals, light_dir, material_color, ambient)
        # Should handle zero light direction gracefully (treat as unlit)
        expected_color = np.array(material_color * ambient, dtype=np.uint8)
        np.testing.assert_array_equal(colors[0], expected_color)

        # Test case 3: Empty input arrays
        vertices = np.array([], dtype=np.float64).reshape(0, 3)
        indices = np.array([], dtype=np.int32).reshape(0, 3)
        normals = np.array([], dtype=np.float64).reshape(0, 3)
        light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        colors = compute_flat_shading(vertices, indices, normals, light_dir, material_color, ambient)
        assert colors.shape == (0, 3)  # Should return empty array with correct shape
