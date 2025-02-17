import numpy as np
from kernel.shading_mod import triangle_gouraud_shading


class TestTriangleGouraudShading:
    def test_basic_shading(self):
        """Test basic shading with light directly above"""
        vertex_normals = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        vertex_intensities = triangle_gouraud_shading(vertex_normals, light_dir)

        # Full illumination (normals aligned with light)
        assert np.allclose(
            vertex_intensities, [1.0, 1.0, 1.0]
        )  # 0.1 ambient + 0.9 * 1.0 diffuse

    def test_varying_normals(self):
        """Test shading with different normals at each vertex"""
        vertex_normals = np.array(
            [
                [0.0, 0.0, 1.0],  # Facing up
                [1.0, 0.0, 0.0],  # Facing right
                [0.0, 1.0, 0.0],  # Facing forward
            ],
            dtype=np.float32,
        )
        light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Light from above

        vertex_intensities = triangle_gouraud_shading(vertex_normals, light_dir)

        # Different intensities based on normal orientation
        assert vertex_intensities[0] > 0.8  # First vertex faces light
        assert vertex_intensities[1] < 0.2  # Second vertex perpendicular to light
        assert vertex_intensities[2] < 0.2  # Third vertex perpendicular to light

    def test_back_lighting(self):
        """Test shading with light from behind"""
        vertex_normals = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        light_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # Light from behind
        ambient = 0.1

        vertex_intensities = triangle_gouraud_shading(
            vertex_normals, light_dir, ambient
        )

        # Only ambient light (light from behind)
        assert np.allclose(vertex_intensities, [0.1, 0.1, 0.1])

    def test_custom_ambient(self):
        """Test shading with custom ambient light"""
        vertex_normals = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        light_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # Light from behind
        ambient = 0.5  # Higher ambient

        vertex_intensities = triangle_gouraud_shading(
            vertex_normals, light_dir, ambient
        )

        # Higher ambient light only
        assert np.allclose(vertex_intensities, [0.5, 0.5, 0.5])

    def test_angled_normals(self):
        """Test shading with normals at 45 degrees"""
        s = np.sqrt(2.0) / 2.0  # sin/cos of 45 degrees
        vertex_normals = np.array(
            [
                [s, 0.0, s],  # 45 degrees to Z in XZ plane
                [0.0, s, s],  # 45 degrees to Z in YZ plane
                [-s, 0.0, s],  # -45 degrees to Z in XZ plane
            ],
            dtype=np.float32,
        )
        light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Light from above

        vertex_intensities = triangle_gouraud_shading(vertex_normals, light_dir)

        # Each normal is at 45 degrees to light, so intensity should be cos(45°) = 1/√2
        expected = 0.1 + 0.9 * s  # ambient + (1-ambient) * cos(45)
        assert np.allclose(
            vertex_intensities, [expected, expected, expected], rtol=1e-5
        )

    def test_mixed_lighting(self):
        """Test shading with varying normals and light direction"""
        s = np.sqrt(2.0) / 2.0  # sin/cos of 45 degrees
        vertex_normals = np.array(
            [
                [0.0, 0.0, 1.0],  # Straight up
                [s, 0.0, s],  # 45 degrees in XZ
                [1.0, 0.0, 0.0],  # Sideways
            ],
            dtype=np.float32,
        )
        light_dir = np.array([s, 0.0, s], dtype=np.float32)  # 45 degree light

        vertex_intensities = triangle_gouraud_shading(vertex_normals, light_dir)

        # Check relative intensities
        assert vertex_intensities[0] > 0.6  # Partial illumination
        assert (
            vertex_intensities[1] > 0.8
        )  # Most illuminated (normal aligned with light)
        assert (
            vertex_intensities[2] > 0.3
        )  # Least illuminated (normal at angle to light)
