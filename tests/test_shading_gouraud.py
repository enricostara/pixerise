import pytest
import numpy as np
from kernel.shading_mod import triangle_gouraud_shading


class TestTriangleGouraudShading:
    def test_basic_shading(self):
        """Test basic Gouraud shading with light directly above"""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float32)
        vertex_normals = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        color = np.array([255, 0, 0], dtype=np.float32)  # Pure red
        
        intensities = triangle_gouraud_shading(vertices, vertex_normals, light_dir, color)
        
        # All vertices should have full illumination (normals aligned with light)
        assert np.all(intensities == 1.0)

    def test_varying_normals(self):
        """Test Gouraud shading with different normals per vertex"""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float32)
        vertex_normals = np.array([
            [0.0, 0.0, 1.0],   # Facing up
            [1.0, 0.0, 0.0],   # Facing right
            [0.0, 1.0, 0.0]    # Facing forward
        ], dtype=np.float32)
        light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Light from above
        color = np.array([255, 255, 255], dtype=np.float32)
        
        intensities = triangle_gouraud_shading(vertices, vertex_normals, light_dir, color)
        
        # First vertex should be fully lit, others should only have ambient
        assert intensities[0] == 1.0
        assert np.all(intensities[1:] == 0.1)  # Only ambient light

    def test_back_lighting(self):
        """Test Gouraud shading with light from behind"""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float32)
        vertex_normals = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        light_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # Light from behind
        color = np.array([255, 255, 255], dtype=np.float32)
        ambient = 0.1
        
        intensities = triangle_gouraud_shading(vertices, vertex_normals, light_dir, color, ambient)
        
        # Only ambient light (light from behind)
        assert np.all(intensities == ambient)

    def test_custom_ambient(self):
        """Test Gouraud shading with custom ambient light"""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float32)
        vertex_normals = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        light_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # Light from behind
        color = np.array([255, 255, 255], dtype=np.float32)
        ambient = 0.5  # Higher ambient
        
        intensities = triangle_gouraud_shading(vertices, vertex_normals, light_dir, color, ambient)
        
        # Only ambient light, but higher value
        assert np.all(intensities == ambient)

    def test_angled_normals(self):
        """Test Gouraud shading with angled normals"""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float32)
        # 45-degree angle normals
        s = np.sqrt(2.0) / 2.0  # sin/cos of 45 degrees
        vertex_normals = np.array([
            [s, 0.0, s],
            [s, 0.0, s],
            [s, 0.0, s]
        ], dtype=np.float32)
        light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        color = np.array([255, 0, 0], dtype=np.float32)
        
        intensities = triangle_gouraud_shading(vertices, vertex_normals, light_dir, color)
        
        # Light at 45 degrees should give ~0.707 intensity
        expected = 0.1 + 0.9 * s  # ambient + (1-ambient) * cos(45)
        np.testing.assert_almost_equal(intensities, expected, decimal=3)

    def test_mixed_lighting(self):
        """Test Gouraud shading with mixed lighting conditions"""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float32)
        s = np.sqrt(2.0) / 2.0  # For 45-degree angle
        vertex_normals = np.array([
            [0.0, 0.0, 1.0],    # Facing light
            [s, 0.0, s],        # 45 degrees
            [1.0, 0.0, 0.0]     # Perpendicular
        ], dtype=np.float32)
        light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        color = np.array([255, 255, 255], dtype=np.float32)
        ambient = 0.1
        
        intensities = triangle_gouraud_shading(vertices, vertex_normals, light_dir, color)
        
        # Check varying intensities
        assert intensities[0] == 1.0  # Full intensity
        np.testing.assert_almost_equal(intensities[1], 0.1 + 0.9 * s, decimal=3)  # 45-degree angle
        assert intensities[2] == ambient  # Only ambient
