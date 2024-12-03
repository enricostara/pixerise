import pytest
import numpy as np
from pixerise import Canvas, ViewPort, Rasterizer


class TestRasterizer:
    @pytest.fixture
    def setup(self):
        """Setup test environment with a small canvas for easier verification"""
        canvas = Canvas((10, 10))  # Small 10x10 canvas for testing
        viewport = ViewPort((10, 10), 1, canvas)
        scene = {}
        rasterizer = Rasterizer(canvas, viewport, scene)
        return rasterizer, canvas

    def verify_pixel(self, canvas, x, y, expected_color):
        """Helper method to verify pixel color at given coordinates"""
        # Account for canvas center offset and convert color to numpy array
        center_x, center_y = canvas._center
        actual_color = canvas.grid[center_x + x, center_y - y]  # Column-major order for pygame compatibility
        np.testing.assert_array_equal(actual_color, np.array(expected_color, dtype=np.uint8))

    def test_horizontal_line(self, setup):
        """Test drawing a horizontal line"""
        rasterizer, canvas = setup
        color = (255, 0, 0)
        rasterizer.draw_line((-2, 0), (2, 0), color)
        
        # Verify pixels along the horizontal line
        for x in range(-2, 3):
            self.verify_pixel(canvas, x, 0, color)

    def test_vertical_line(self, setup):
        """Test drawing a vertical line"""
        rasterizer, canvas = setup
        color = (0, 255, 0)
        rasterizer.draw_line((0, -2), (0, 2), color)
        
        # Verify pixels along the vertical line
        for y in range(-2, 3):
            self.verify_pixel(canvas, 0, y, color)

    def test_diagonal_line(self, setup):
        """Test drawing a diagonal line"""
        rasterizer, canvas = setup
        color = (0, 0, 255)
        rasterizer.draw_line((-2, -2), (2, 2), color)
        
        # Verify pixels along the diagonal line
        for i in range(-2, 3):
            self.verify_pixel(canvas, i, i, color)

    def test_single_point_line(self, setup):
        """Test drawing a line where start and end points are the same"""
        rasterizer, canvas = setup
        color = (255, 255, 0)
        point = (1, 1)
        rasterizer.draw_line(point, point, color)
        
        # Verify only the single point is drawn
        self.verify_pixel(canvas, 1, 1, color)
        # Verify adjacent pixels are not drawn
        center_x, center_y = canvas._center
        assert not np.array_equal(
            canvas.grid[center_x + 2, center_y - 2],  # Column-major order
            np.array(color, dtype=np.uint8)
        )

    def test_steep_line(self, setup):
        """Test drawing a steep line (|dy| > |dx|)"""
        rasterizer, canvas = setup
        color = (255, 0, 255)
        rasterizer.draw_line((1, -2), (2, 2), color)
        
        # Verify key points along the steep line
        expected_points = [(1, -2), (1, -1), (1, 0), (2, 1), (2, 2)]
        for x, y in expected_points:
            self.verify_pixel(canvas, x, y, color)

    def test_negative_slope_line(self, setup):
        """Test drawing a line with negative slope"""
        rasterizer, canvas = setup
        color = (0, 255, 255)
        rasterizer.draw_line((-2, 2), (2, -2), color)
        
        # Verify key points along the negative slope line
        for i in range(-2, 3):
            self.verify_pixel(canvas, i, -i, color)

    def test_out_of_bounds_line(self, setup):
        """Test drawing a line that extends beyond canvas boundaries"""
        rasterizer, canvas = setup
        color = (128, 128, 128)
        # Draw line from far left to far right
        rasterizer.draw_line((-20, 0), (20, 0), color)
        
        # Only pixels within canvas bounds should be drawn
        for x in range(-4, 5):  # Canvas is 10x10 centered at (0,0)
            self.verify_pixel(canvas, x, 0, color)

    def test_zero_length_line(self, setup):
        """Test drawing a zero-length line (point)"""
        rasterizer, canvas = setup
        color = (200, 200, 200)
        rasterizer.draw_line((0, 0), (0, 0), color)
        
        # Verify only center point is drawn
        self.verify_pixel(canvas, 0, 0, color)
        # Verify adjacent pixels are not drawn
        center_x, center_y = canvas._center
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            assert not np.array_equal(
                canvas.grid[center_x + dx, center_y - dy],  # Column-major order
                np.array(color, dtype=np.uint8)
            )
