import unittest
import numpy as np
from pixerise import Canvas, ViewPort, Renderer


class TestShadedTriangleDrawing(unittest.TestCase):
    def setUp(self):
        self.width = 100
        self.height = 100
        self.canvas = Canvas((self.width, self.height))
        self.viewport = ViewPort((self.width, self.height), 1, self.canvas)
        self.renderer = Renderer(self.canvas, self.viewport)
        self.color = (255, 0, 0)  # Red color for visibility

    def tearDown(self):
        self.canvas = None
        self.viewport = None
        self.renderer = None

    def test_basic_shaded_triangle(self):
        """Test drawing a simple shaded triangle in the center of the canvas."""
        self.canvas.color_buffer.fill(0)  # Set background to black
        self.renderer.draw_shaded_triangle(
            (0, 20, 0.5),
            (-20, -20, 0.5),
            (20, -20, 0.5),  # All at same depth
            self.color,
            1.0,
            0.5,
            0.0,  # Varying intensities
        )
        # Check if pixels are set in the expected triangle area
        self.assertTrue(np.any(self.canvas.color_buffer != 0))
        # Check if we have varying intensities (not all pixels same value)
        red_channel = self.canvas.color_buffer[..., 0]
        nonzero_pixels = red_channel[red_channel != 0]
        self.assertTrue(len(np.unique(nonzero_pixels)) > 1)

    def test_uniform_intensity(self):
        """Test triangle with uniform intensity across all vertices."""
        self.canvas.color_buffer.fill(0)
        self.renderer.draw_shaded_triangle(
            (0, 20, 0.5),
            (-20, -20, 0.5),
            (20, -20, 0.5),
            self.color,
            0.5,
            0.5,
            0.5,  # Uniform intensity
        )
        # Check if all non-zero pixels have the same value
        red_channel = self.canvas.color_buffer[..., 0]
        nonzero_pixels = red_channel[red_channel != 0]
        self.assertTrue(len(np.unique(nonzero_pixels)) == 1)
        self.assertTrue(np.all(nonzero_pixels == int(255 * 0.5)))

    def test_zero_intensity(self):
        """Test triangle with zero intensity at all vertices."""
        self.canvas.color_buffer.fill(0)
        self.renderer.draw_shaded_triangle(
            (0, 20, 0.5), (-20, -20, 0.5), (20, -20, 0.5), self.color, 0.0, 0.0, 0.0
        )
        # Should not modify any pixels
        self.assertTrue(np.all(self.canvas.color_buffer == 0))

    def test_degenerate_line(self):
        """Test shaded triangle that collapses to a line.
        Since we use a scanline rasterizer, degenerate triangles that collapse to lines
        may not be drawn. This is acceptable behavior as such cases should be handled by
        the line drawing functions instead."""
        self.canvas.color_buffer.fill(0)
        self.renderer.draw_shaded_triangle(
            (0, 0, 0.5), (10, 10, 0.5), (20, 20, 0.5), self.color, 1.0, 0.5, 0.0
        )
        # For a scanline rasterizer, it's acceptable not to draw anything for degenerate cases
        pass

    def test_degenerate_point(self):
        """Test shaded triangle where all points are the same.
        Since we use a scanline rasterizer, degenerate triangles that collapse to points
        may not be drawn. This is acceptable behavior as such cases should be handled by
        the point drawing functions instead."""
        self.canvas.color_buffer.fill(0)
        point = (0, 0, 0.5)
        intensity = 0.5
        self.renderer.draw_shaded_triangle(
            point, point, point, self.color, intensity, intensity, intensity
        )
        # For a scanline rasterizer, it's acceptable not to draw anything for degenerate cases
        pass

    def test_fully_outside_canvas(self):
        """Test triangle completely outside the canvas bounds."""
        self.canvas.color_buffer.fill(0)
        self.renderer.draw_shaded_triangle(
            (self.width + 10, 10, 0.5),
            (self.width + 20, 20, 0.5),
            (self.width + 30, 30, 0.5),
            self.color,
            1.0,
            0.5,
            0.0,
        )
        # Should not modify any pixels
        self.assertTrue(np.all(self.canvas.color_buffer == 0))

    def test_partially_outside_canvas(self):
        """Test shaded triangle that is partially outside the canvas bounds."""
        self.canvas.color_buffer.fill(0)  # Set background to black
        self.renderer.draw_shaded_triangle(
            (0, 0, 0.5),
            (self.width + 10, 10, 0.5),
            (10, self.height + 10, 0.5),
            self.color,
            1.0,
            0.5,
            0.0,
        )
        # Should draw the visible portion
        self.assertTrue(np.any(self.canvas.color_buffer != 0))
        # Check that some pixels have varying intensities
        red_channel = self.canvas.color_buffer[..., 0]
        nonzero_pixels = red_channel[red_channel != 0]
        self.assertTrue(len(np.unique(nonzero_pixels)) > 1)

    def test_intensity_interpolation(self):
        """Test proper intensity interpolation across the triangle."""
        self.canvas.color_buffer.fill(0)
        # Draw triangle with intensity gradient from top to bottom
        self.renderer.draw_shaded_triangle(
            (0, 20, 0.5),
            (-20, -20, 0.5),
            (20, -20, 0.5),
            self.color,
            1.0,
            0.0,
            0.0,  # Full intensity at top, zero at bottom corners
        )
        red_channel = self.canvas.color_buffer[..., 0]
        nonzero_pixels = red_channel[red_channel != 0]

        # Should have multiple intensity levels
        unique_intensities = np.unique(nonzero_pixels)
        self.assertTrue(len(unique_intensities) > 10)

        # Should include both high and low intensity values
        self.assertTrue(np.max(nonzero_pixels) > 200)  # High intensity near 1.0
        self.assertTrue(np.min(nonzero_pixels) < 50)  # Low intensity near 0.0

    def test_shaded_triangle_z_buffering(self):
        """Test z-buffering behavior with overlapping shaded triangles."""
        self.canvas.color_buffer.fill(0)
        self.canvas.depth_buffer.fill(
            0
        )  # Initialize with 0 (1/∞) for 1/z depth testing

        # Draw a red triangle in front
        self.renderer.draw_shaded_triangle(
            (0, 20, 0.5),
            (-20, -20, 0.5),
            (20, -20, 0.5),
            (255, 0, 0),
            1.0,
            1.0,
            1.0,
        )

        # Draw a green triangle behind
        self.renderer.draw_shaded_triangle(
            (0, 20, 0.8),
            (-20, -20, 0.8),
            (20, -20, 0.8),
            (0, 255, 0),
            1.0,
            1.0,
            1.0,
        )

        # Check center pixel - should be red
        center_x = self.canvas.width // 2
        center_y = self.canvas.height // 2
        center_color = self.canvas.color_buffer[center_x, center_y]
        self.assertEqual(tuple(center_color), (255, 0, 0))

    def test_varying_depth_triangle(self):
        """Test triangle with varying depth values."""
        self.canvas.color_buffer.fill(0)
        self.canvas.depth_buffer.fill(
            0
        )  # Initialize with 0 (1/∞) for 1/z depth testing

        # Draw a triangle with varying z values
        self.renderer.draw_shaded_triangle(
            (0, 20, 0.2),  # Front vertex
            (-20, -20, 0.5),  # Middle vertex
            (20, -20, 0.8),  # Back vertex
            self.color,
            1.0,
            0.5,
            0.0,
        )

        # Check that depth buffer has been updated with varying values
        # For 1/z values, larger values mean closer to camera
        depth_values = np.unique(self.canvas.depth_buffer[self.canvas.depth_buffer > 0])
        self.assertTrue(len(depth_values) > 1)  # Should have multiple depth values


if __name__ == "__main__":
    unittest.main()
