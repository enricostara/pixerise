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
    """JIT-compiled triangle filling algorithm using fixed-point arithmetic."""
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

    # Initialize edge traversal variables for the first two edges
    # Edge 1: y0 to y1
    dx1 = x1 - x0
    dy1 = y1 - y0
    x_left = x0 << 16  # Fixed-point x-coordinate (16.16)
    step_left = (dx1 << 16) // dy1 if dy1 != 0 else 0

    # Edge 2: y0 to y2
    dx2 = x2 - x0
    dy2 = y2 - y0
    x_right = x0 << 16  # Fixed-point x-coordinate (16.16)
    step_right = (dx2 << 16) // dy2 if dy2 != 0 else 0

    # Fill the upper triangle
    if y1 - y0 > 0:
        for y in range(y0, y1):
            start_x = x_left >> 16
            end_x = x_right >> 16
            
            if start_x > end_x:
                start_x, end_x = end_x, start_x
                
            for x in range(start_x, end_x + 1):
                _draw_pixel(canvas_grid, x, y, center_x, center_y,
                          color_r, color_g, color_b, canvas_width, canvas_height)
            
            x_left += step_left
            x_right += step_right

    # Edge 3: y1 to y2
    dx3 = x2 - x1
    dy3 = y2 - y1
    x_left = x1 << 16
    step_left = (dx3 << 16) // dy3 if dy3 != 0 else 0

    # Fill the lower triangle
    if y2 - y1 > 0:
        for y in range(y1, y2 + 1):
            start_x = x_left >> 16
            end_x = x_right >> 16
            
            if start_x > end_x:
                start_x, end_x = end_x, start_x
                
            for x in range(start_x, end_x + 1):
                _draw_pixel(canvas_grid, x, y, center_x, center_y,
                          color_r, color_g, color_b, canvas_width, canvas_height)
            
            x_left += step_left
            x_right += step_right


@jit(nopython=True)
def _draw_shaded_triangle(x0: int, y0: int, x1: int, y1: int, x2: int, y2: int,
                         canvas_grid: np.ndarray, center_x: int, center_y: int,
                         color_r: int, color_g: int, color_b: int,
                         i0: float, i1: float, i2: float,
                         canvas_width: int, canvas_height: int) -> None:
    """Draw a shaded triangle using scan-line algorithm with linear interpolation for intensities."""
    # Early exit if all intensities are zero or color is black
    if max(i0, i1, i2) <= 0.001 or (color_r == 0 and color_g == 0 and color_b == 0):
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

    # Sort vertices by y-coordinate, keeping intensities matched with vertices
    if y1 < y0:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        i0, i1 = i1, i0
    if y2 < y0:
        x0, x2 = x2, x0
        y0, y2 = y2, y0
        i0, i2 = i2, i0
    if y2 < y1:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        i1, i2 = i2, i1

    # Initialize edge traversal variables for the first two edges
    # Edge 1: y0 to y1
    dx1 = x1 - x0
    dy1 = y1 - y0
    x_left = x0 << 16  # Fixed-point x-coordinate (16.16)
    step_left = (dx1 << 16) // max(1, dy1)  # Ensure non-zero denominator
    i_left = i0  # Start with intensity at top vertex
    i_step_left = (i1 - i0) / max(1, dy1)  # Ensure non-zero denominator

    # Edge 2: y0 to y2
    dx2 = x2 - x0
    dy2 = y2 - y0
    x_right = x0 << 16
    step_right = (dx2 << 16) // max(1, dy2)  # Ensure non-zero denominator
    i_right = i0
    i_step_right = (i2 - i0) / max(1, dy2)  # Ensure non-zero denominator

    # Fill the upper triangle (including zero-height case)
    for y in range(y0, max(y0 + 1, y1)):  # Always draw at least one scanline
        start_x = x_left >> 16
        end_x = x_right >> 16
        
        # Ensure left is actually on the left
        if start_x > end_x:
            start_x, end_x = end_x, start_x
            i_curr, i_end = i_right, i_left
        else:
            i_curr, i_end = i_left, i_right
        
        # Draw the scanline with interpolated intensities
        i_step = (i_end - i_curr) / max(1, end_x - start_x + 1)  # Include start point
        for x in range(start_x, end_x + 1):
            if i_curr > 0.001:
                r = int(color_r * i_curr)
                g = int(color_g * i_curr)
                b = int(color_b * i_curr)
                _draw_pixel(canvas_grid, x, y, center_x, center_y, r, g, b, canvas_width, canvas_height)
            i_curr += i_step
        
        if y1 > y0:  # Only update edges if actually moving
            x_left += step_left
            x_right += step_right
            i_left += i_step_left
            i_right += i_step_right

    # Edge 3: y1 to y2
    dx3 = x2 - x1
    dy3 = y2 - y1
    x_left = x1 << 16
    step_left = (dx3 << 16) // max(1, dy3)  # Ensure non-zero denominator
    i_left = i1
    i_step_left = (i2 - i1) / max(1, dy3)  # Ensure non-zero denominator

    # Fill the lower triangle (including zero-height case)
    for y in range(y1, max(y1 + 1, y2 + 1)):  # Always draw at least one scanline
        start_x = x_left >> 16
        end_x = x_right >> 16
        
        # Ensure left is actually on the left
        if start_x > end_x:
            start_x, end_x = end_x, start_x
            i_curr, i_end = i_right, i_left
        else:
            i_curr, i_end = i_left, i_right
        
        # Draw the scanline with interpolated intensities
        i_step = (i_end - i_curr) / max(1, end_x - start_x + 1)  # Include start point
        for x in range(start_x, end_x + 1):
            if i_curr > 0.001:
                r = int(color_r * i_curr)
                g = int(color_g * i_curr)
                b = int(color_b * i_curr)
                _draw_pixel(canvas_grid, x, y, center_x, center_y, r, g, b, canvas_width, canvas_height)
            i_curr += i_step
        
        if y2 > y1:  # Only update edges if actually moving
            x_left += step_left
            x_right += step_right
            i_left += i_step_left
            i_right += i_step_right


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

    def draw_triangle(self, p1: (float, float), p2: (float, float), p3: (float, float), color: (int, int, int), fill: bool = True):
        """Draw a triangle defined by three points. If fill is True, the triangle will be filled,
        otherwise only the outline will be drawn."""
        if fill:
            _draw_triangle(
                int(p1[0]), int(p1[1]), 
                int(p2[0]), int(p2[1]), 
                int(p3[0]), int(p3[1]), 
                self._canvas.grid,
                self._canvas._center[0], self._canvas._center[1],
                color[0], color[1], color[2],
                self._canvas.width, self._canvas.height
            )
        else:
            # Draw the three edges of the triangle
            self.draw_line(p1, p2, color)
            self.draw_line(p2, p3, color)
            self.draw_line(p3, p1, color)

    def draw_shaded_triangle(self, p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float],
                           color: tuple[int, int, int],
                           intensity1: float, intensity2: float, intensity3: float) -> None:
        """
        Draw a triangle with interpolated intensities for a single color.
        
        Args:
            p1, p2, p3: Vertex positions as (x, y) tuples
            color: RGB color as (r, g, b) tuple
            intensity1, intensity2, intensity3: Intensity values (0.0 to 1.0) for each vertex
        """
        # Clamp intensities to valid range
        i1 = max(0.0, min(1.0, intensity1))
        i2 = max(0.0, min(1.0, intensity2))
        i3 = max(0.0, min(1.0, intensity3))
        
        _draw_shaded_triangle(
            int(p1[0]), int(p1[1]), 
            int(p2[0]), int(p2[1]), 
            int(p3[0]), int(p3[1]), 
            self._canvas.grid, self._canvas._center[0], self._canvas._center[1],
            color[0], color[1], color[2],
            i1, i2, i3,
            self._canvas.width, self._canvas.height
        )

    def project_vertex(self, vertex, position=None):
        # Apply translation if position is provided
        if position is not None:
            vertex = vertex + position
            
        x, y, z = vertex
        d = self._viewport._plane_distance / z
        return self._viewport.viewport_to_canvas(x * d, y * d)

    def render(self, scene: dict):
        """Render a scene containing models and their instances."""
        models = scene['models']
        
        # Render each instance
        for instance in scene['instances']:
            model_name = instance['model']
            position = instance.get('position', np.array([0, 0, 0], dtype=float))
            
            # Get the model data
            model = models[model_name]
            vertices = model['vertices']
            triangles = model['triangles']
            
            # Project vertices with instance position
            projected_vertices = [self.project_vertex(vertex, position) for vertex in vertices]
            
            # Draw each triangle
            for triangle in triangles:
                v1, v2, v3 = triangle
                self.draw_triangle(
                    projected_vertices[v1],
                    projected_vertices[v2],
                    projected_vertices[v3],
                    (255, 255, 255),  # White color for now
                    fill=False)