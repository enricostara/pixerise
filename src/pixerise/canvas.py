import numpy as np


class Canvas:
    def __init__(self, size=(800, 600)):
        self.size = size
        self.grid = np.ones((size[0], size[1], 3), dtype=np.uint8) * 32
        self._center = (size[0] // 2, size[1] // 2)

    def set_pixel(self, x, y, color):
        if x < 0 or x >= self.size[0] or y < 0 or y >= self.size[1]:
            return
        self.grid[x, y] = color

    def draw_point(self, x, y, color):
        self.set_pixel(self._center[0] + x, self._center[1] + y, color)

    def draw_unchecked_point(self, x, y, color):
        self.grid[self._center[0] + x, self._center[1] + y] = color
