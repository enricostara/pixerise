import numpy as np
from numba import jit

@jit(nopython=True)
def _draw_pixel(grid: np.ndarray, x: int, y: int, center_x: int, center_y: int, 
                color_r: int, color_g: int, color_b: int, width: int, height: int) -> None:
    """JIT-compiled pixel drawing function."""
    px = center_x + x
    py = center_y - y  # Flip y coordinate
    if 0 <= px < width and 0 <= py < height:
        grid[py, px, 0] = color_r  # Changed order to [py, px] for correct row-major indexing
        grid[py, px, 1] = color_g
        grid[py, px, 2] = color_b

class Canvas:
    def __init__(self, size: (int, int) = (800, 600)):
        self.size = size
        self.width = size[0]
        self.height = size[1]
        self.grid = np.ones((self.height, self.width, 3), dtype=np.uint8) * 32  # Changed order to (height, width)
        self.half_width = self.width // 2
        self.half_height = self.height // 2
        self._center = (self.half_width, self.half_height)

    def draw_point(self, x, y, color) -> None:
        _draw_pixel(self.grid, x, y, self._center[0], self._center[1], 
                   color[0], color[1], color[2], self.width, self.height)
