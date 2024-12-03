import unittest
import numpy as np
from pixerise import Canvas, ViewPort, Renderer


class TestTriangleDrawing(unittest.TestCase):
    def setUp(self):
        self.width = 100
        self.height = 100
        self.canvas = Canvas((self.width, self.height))
        self.viewport = ViewPort((self.width, self.height), 1, self.canvas)
        self.scene = {}
        self.renderer = Renderer(self.canvas, self.viewport, self.scene)
        self.color = (255, 0, 0)  # Red color for visibility

    def tearDown(self):
        self.canvas = None
        self.viewport = None
        self.renderer = None

    def test_basic_triangle(self):
        """Test drawing a simple triangle in the center of the canvas."""
        self.canvas.grid.fill(0)  # Set background to black
        self.renderer.draw_triangle(
            (0, 20), (-20, -20), (20, -20),
            self.color
        )
        # Check if pixels are set in the expected triangle area
        self.assertTrue(np.any(self.canvas.grid != 0))

    def test_degenerate_line(self):
        """Test triangle that collapses to a line (all points collinear)."""
        self.canvas.grid.fill(0)  # Set background to black
        self.renderer.draw_triangle(
            (0, 0), (10, 10), (20, 20),
            self.color
        )
        # Should still draw something (the line)
        self.assertTrue(np.any(self.canvas.grid != 0))

    def test_degenerate_point(self):
        """Test triangle where all points are the same (collapses to a point)."""
        point = (0, 0)
        self.renderer.draw_triangle(
            point, point, point,
            self.color
        )
        # Should draw at least one pixel
        self.assertTrue(np.any(self.canvas.grid != 0))

    def test_partially_outside_canvas(self):
        """Test triangle that is partially outside the canvas bounds."""
        self.canvas.grid.fill(0)  # Set background to black
        self.renderer.draw_triangle(
            (0, 0), (self.width + 10, 10), (10, self.height + 10),
            self.color
        )
        # Should draw the visible portion
        self.assertTrue(np.any(self.canvas.grid != 0))

    def test_fully_outside_canvas(self):
        """Test triangle that is completely outside the canvas bounds."""
        self.canvas.grid.fill(0)  # Set background to black
        self.renderer.draw_triangle(
            (self.width + 10, 0),
            (self.width + 20, 0),
            (self.width + 15, 10),
            self.color
        )
        # Should not draw anything
        self.assertTrue(np.all(self.canvas.grid == 0))

    def test_flat_top_triangle(self):
        """Test triangle with a flat top edge."""
        self.canvas.grid.fill(0)  # Set background to black
        self.renderer.draw_triangle(
            (-20, 20), (20, 20), (0, -20),
            self.color
        )
        # Check if pixels are set in the expected triangle area
        self.assertTrue(np.any(self.canvas.grid != 0))

    def test_flat_bottom_triangle(self):
        """Test triangle with a flat bottom edge."""
        self.canvas.grid.fill(0)  # Set background to black
        self.renderer.draw_triangle(
            (0, 20), (-20, -20), (20, -20),
            self.color
        )
        # Check if pixels are set in the expected triangle area
        self.assertTrue(np.any(self.canvas.grid != 0))

    def test_flat_side_triangle(self):
        """Test triangle with a vertical edge."""
        self.canvas.grid.fill(0)  # Set background to black
        self.renderer.draw_triangle(
            (0, 20), (0, -20), (20, 0),
            self.color
        )
        # Check if pixels are set in the expected triangle area
        self.assertTrue(np.any(self.canvas.grid != 0))

    def test_very_thin_triangle(self):
        """Test very thin triangle (nearly degenerate)."""
        self.canvas.grid.fill(0)  # Set background to black
        self.renderer.draw_triangle(
            (0, 20), (1, -20), (2, 20),
            self.color
        )
        # Should still draw something
        self.assertTrue(np.any(self.canvas.grid != 0))

    def test_very_small_triangle(self):
        """Test very small triangle (few pixels)."""
        self.canvas.grid.fill(0)  # Set background to black
        self.renderer.draw_triangle(
            (0, 0), (1, 1), (0, 1),
            self.color
        )
        # Should draw at least one pixel
        self.assertTrue(np.any(self.canvas.grid != 0))

    def test_color_values(self):
        """Test different color values including edge cases."""
        # Test with maximum color values
        self.canvas.grid.fill(0)  # Set background to black
        self.renderer.draw_triangle(
            (0, 10), (-10, -10), (10, -10),
            (255, 255, 255)
        )
        self.assertTrue(np.any(self.canvas.grid == 255))

        # Clear canvas with black background
        self.canvas.grid.fill(0)  # Set background to black

        # Test with minimum color values
        self.renderer.draw_triangle(
            (0, 10), (-10, -10), (10, -10),
            (0, 0, 0)
        )
        # Should not change canvas from black background
        self.assertTrue(np.all(self.canvas.grid == 0))


if __name__ == '__main__':
    unittest.main()
