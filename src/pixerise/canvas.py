import numpy as np


class Canvas:
    def __init__(self, size: (int, int) = (800, 600)):
        self.size = size
        self.width = size[0]
        self.height = size[1]
        self.grid = np.ones((self.width, self.height, 3), dtype=np.uint8) * 32
        self.half_width = self.width // 2
        self.half_height = self.height // 2
        self._center = (self.half_width, self.half_height)

    def set_pixel(self, x, y, color) -> None:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return
        self.grid[x, y] = color

    def draw_point(self, x, y, color) -> None:
        self.set_pixel(self._center[0] + x, self._center[1] - y, color)

    def draw_unchecked_point(self, x, y, color) -> None:
        self.grid[self._center[0] + x, self._center[1] - y] = color
