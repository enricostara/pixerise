import unittest
import numpy as np

from pixerise.canvas import Canvas
from pixerise.rasterizer import Rasterizer
from pixerise.viewport import ViewPort


class TestShadedTriangleDrawing(unittest.TestCase):
    def setUp(self):
        self.width = 100
        self.height = 100
        self.canvas = Canvas((self.width, self.height))
        self.viewport = ViewPort((self.width, self.height), 1, self.canvas)
        self.scene = {}
        self.rasterizer = Rasterizer(self.canvas, self.viewport, self.scene, background_color=(0, 0, 0))
        self.color = (255, 0, 0)  # Red color for visibility

    def tearDown(self):
        self.canvas = None
        self.viewport = None
        self.rasterizer = None

    def test_basic_shaded_triangle(self):
        """Test drawing a simple shaded triangle in the center of the canvas."""
        self.canvas.grid.fill(0)  # Set background to black
        self.rasterizer.draw_shaded_triangle(
            (0, 20), (-20, -20), (20, -20),
            self.color,
            1.0, 0.5, 0.0  # Varying intensities
        )
        # Check if pixels are set in the expected triangle area
        self.assertTrue(np.any(self.canvas.grid != 0))
        # Check if we have varying intensities (not all pixels same value)
        red_channel = self.canvas.grid[..., 0]
        nonzero_pixels = red_channel[red_channel != 0]
        self.assertTrue(len(np.unique(nonzero_pixels)) > 1)

    def test_uniform_intensity(self):
        """Test triangle with uniform intensity across all vertices."""
        self.canvas.grid.fill(0)
        self.rasterizer.draw_shaded_triangle(
            (0, 20), (-20, -20), (20, -20),
            self.color,
            0.5, 0.5, 0.5  # Uniform intensity
        )
        # Check if all non-zero pixels have the same value
        red_channel = self.canvas.grid[..., 0]
        nonzero_pixels = red_channel[red_channel != 0]
        self.assertTrue(len(np.unique(nonzero_pixels)) == 1)
        self.assertTrue(np.all(nonzero_pixels == int(255 * 0.5)))

    def test_zero_intensity(self):
        """Test triangle with zero intensity at all vertices."""
        self.canvas.grid.fill(0)
        self.rasterizer.draw_shaded_triangle(
            (0, 20), (-20, -20), (20, -20),
            self.color,
            0.0, 0.0, 0.0
        )
        # Should not modify any pixels
        self.assertTrue(np.all(self.canvas.grid == 0))

    def test_degenerate_line(self):
        """Test shaded triangle that collapses to a line."""
        self.canvas.grid.fill(0)
        self.rasterizer.draw_shaded_triangle(
            (0, 0), (10, 10), (20, 20),
            self.color,
            1.0, 0.5, 0.0
        )
        # Should still draw something (the line)
        self.assertTrue(np.any(self.canvas.grid != 0))

    def test_degenerate_point(self):
        """Test shaded triangle where all points are the same."""
        self.canvas.grid.fill(0)
        point = (0, 0)
        intensity = 0.5
        self.rasterizer.draw_shaded_triangle(
            point, point, point,
            self.color,
            intensity, intensity, intensity
        )
        # Should draw exactly one pixel with correct intensity
        nonzero_count = np.count_nonzero(self.canvas.grid)
        self.assertEqual(nonzero_count, 1)  # One pixel, 3 color channels

    def test_fully_outside_canvas(self):
        """Test triangle completely outside the canvas bounds."""
        self.canvas.grid.fill(0)
        self.rasterizer.draw_shaded_triangle(
            (self.width + 10, 0),
            (self.width + 20, 0),
            (self.width + 15, 10),
            self.color,
            1.0, 1.0, 1.0
        )
        # Should not modify any pixels
        self.assertTrue(np.all(self.canvas.grid == 0))

    def test_intensity_interpolation(self):
        """Test proper intensity interpolation across the triangle."""
        self.canvas.grid.fill(0)
        # Draw triangle with intensity gradient from top to bottom
        self.rasterizer.draw_shaded_triangle(
            (0, 20), (-20, -20), (20, -20),
            self.color,
            1.0, 0.0, 0.0  # Full intensity at top, zero at bottom corners
        )
        red_channel = self.canvas.grid[..., 0]
        nonzero_pixels = red_channel[red_channel != 0]
        
        # Should have multiple intensity levels
        unique_intensities = np.unique(nonzero_pixels)
        self.assertTrue(len(unique_intensities) > 10)
        
        # Should include both high and low intensity values
        self.assertTrue(np.max(nonzero_pixels) > 200)  # High intensity near 1.0
        self.assertTrue(np.min(nonzero_pixels) < 50)   # Low intensity near 0.0


if __name__ == '__main__':
    unittest.main()
