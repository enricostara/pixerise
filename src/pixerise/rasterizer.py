import numpy as np
from pixerise.canvas import Canvas
from pixerise.viewport import ViewPort

class Rasterizer:
    def __init__(self, canvas: Canvas, viewport: ViewPort, scene: dict, background_color=(32, 32, 32)):
        self._canvas = canvas
        self._viewport = viewport
        self._scene = scene
        self._background_color = np.array(background_color, dtype=int)

    def draw_line(self, start: (float, float), end: (float, float), color: (int, int, int)):
        start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)
        direction = end - start
        length = np.linalg.norm(direction)
        if length == 0:
            return
        direction = direction / length  # Normalize
        for t in np.linspace(0, length, int(length) + 1):
            point = start + t * direction
            x, y = int(point[0]), int(point[1])
            # Check boundaries
            if 0 <= x < self._canvas.size[0] and 0 <= y < self._canvas.size[1]:
                self._canvas.draw_unchecked_point(x, y, color)

    def draw_line_bresenham(self, start: (float, float), end: (float, float), color: (int, int, int)):
        """Draw a line using Bresenham's algorithm for better performance."""
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(end[0]), int(end[1])
        
        # Calculate absolute differences and direction of movement
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        # Determine direction to step in x and y
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        # Initial error term
        err = dx - dy
        
        while True:
            # Draw current pixel if within bounds
            if 0 <= x0 < self._canvas.size[0] and 0 <= y0 < self._canvas.size[1]:
                self._canvas.draw_unchecked_point(x0, y0, color)
            
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