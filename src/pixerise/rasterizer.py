import numpy as np
from numba import jit
from pixerise.canvas import Canvas, _draw_pixel
from pixerise.viewport import ViewPort

@jit(nopython=True)
def _draw_line(x0: int, y0: int, x1: int, y1: int, 
               canvas_grid: np.ndarray, center_x: int, center_y: int,
               color_r: int, color_g: int, color_b: int,
               canvas_width: int, canvas_height: int) -> None:
    """JIT-compiled line drawing algorithm using Bresenham's algorithm."""
    # Calculate absolute differences and direction of movement
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    
    # Determine direction to step in x and y
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    
    # Initial error term
    err = dx - dy
    
    while True:
        # Draw point if within bounds (check transformed coordinates)
        px = center_x + x0
        py = center_y - y0  # Flip y coordinate
        if 0 <= px < canvas_width and 0 <= py < canvas_height:
            _draw_pixel(canvas_grid, x0, y0, center_x, center_y,
                       color_r, color_g, color_b, canvas_width, canvas_height)
        
        # Check if we've reached the end point
        if x0 == x1 and y0 == y1:
            break
            
        # Update error term and coordinates
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


@jit(nopython=True)
def _draw_triangle(x0: int, y0: int, x1: int, y1: int, x2: int, y2: int,
                  canvas_grid: np.ndarray, center_x: int, center_y: int,
                  color_r: int, color_g: int, color_b: int,
                  canvas_width: int, canvas_height: int) -> None:
    """JIT-compiled triangle filling algorithm using scanline approach."""
    # Early exit if color is black (0,0,0)
    if color_r == 0 and color_g == 0 and color_b == 0:
        return

    # Convert to screen coordinates for bounds checking
    sx0, sy0 = center_x + x0, center_y - y0
    sx1, sy1 = center_x + x1, center_y - y1
    sx2, sy2 = center_x + x2, center_y - y2

    # Early exit if triangle is completely outside the canvas
    min_x = min(sx0, sx1, sx2)
    max_x = max(sx0, sx1, sx2)
    min_y = min(sy0, sy1, sy2)
    max_y = max(sy0, sy1, sy2)
    
    if (max_x < 0 or min_x >= canvas_width or
        max_y < 0 or min_y >= canvas_height):
        return

    # Sort vertices by y-coordinate
    if y1 < y0:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    if y2 < y0:
        x0, x2 = x2, x0
        y0, y2 = y2, y0
    if y2 < y1:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    # Calculate slopes
    if y1 - y0 != 0:
        slope1 = (x1 - x0) / (y1 - y0)
    else:
        slope1 = 0

    if y2 - y0 != 0:
        slope2 = (x2 - x0) / (y2 - y0)
    else:
        slope2 = 0

    if y2 - y1 != 0:
        slope3 = (x2 - x1) / (y2 - y1)
    else:
        slope3 = 0

    # Fill the upper triangle
    if y1 - y0 > 0:
        for y in range(y0, y1):
            x_start = int(x0 + (y - y0) * slope1)
            x_end = int(x0 + (y - y0) * slope2)
            
            if x_start > x_end:
                x_start, x_end = x_end, x_start
                
            for x in range(x_start, x_end + 1):
                _draw_pixel(canvas_grid, x, y, center_x, center_y,
                          color_r, color_g, color_b, canvas_width, canvas_height)

    # Fill the lower triangle
    if y2 - y1 > 0:
        for y in range(y1, y2 + 1):
            x_start = int(x1 + (y - y1) * slope3)
            x_end = int(x0 + (y - y0) * slope2)
            
            if x_start > x_end:
                x_start, x_end = x_end, x_start
                
            for x in range(x_start, x_end + 1):
                _draw_pixel(canvas_grid, x, y, center_x, center_y,
                          color_r, color_g, color_b, canvas_width, canvas_height)


class Rasterizer:
    def __init__(self, canvas: Canvas, viewport: ViewPort, scene: dict, background_color=(32, 32, 32)):
        self._canvas = canvas
        self._viewport = viewport
        self._scene = scene
        self._background_color = np.array(background_color, dtype=int)

    def draw_line(self, start: (float, float), end: (float, float), color: (int, int, int)):
        """Draw a line using Bresenham's algorithm for better performance."""
        _draw_line(
            int(start[0]), int(start[1]),
            int(end[0]), int(end[1]),
            self._canvas.grid, 
            self._canvas._center[0], self._canvas._center[1],
            color[0], color[1], color[2],
            self._canvas.width, self._canvas.height
        )

    def draw_triangle(self, p1: (float, float), p2: (float, float), p3: (float, float), color: (int, int, int)):
        """Draw a filled triangle defined by three points."""
        _draw_triangle(
            int(p1[0]), int(p1[1]),
            int(p2[0]), int(p2[1]),
            int(p3[0]), int(p3[1]),
            self._canvas.grid,
            self._canvas._center[0], self._canvas._center[1],
            color[0], color[1], color[2],
            self._canvas.width, self._canvas.height
        )