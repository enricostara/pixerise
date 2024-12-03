import numpy as np
from numba import jit
import pygame
from pixerise.kernel import draw_pixel

class Canvas:
    def __init__(self, size: (int, int) = (800, 600)):
        self.size = size
        self.width = size[0]
        self.height = size[1]
        self.grid = np.ones((self.width, self.height, 3), dtype=np.uint8) * 32  # Back to column-major order (width, height)
        self.half_width = self.width // 2
        self.half_height = self.height // 2
        self._center = (self.half_width, self.half_height)

    def draw_point(self, x, y, color) -> None:
        draw_pixel(self.grid, x, y, self._center[0], self._center[1], 
                   color[0], color[1], color[2], self.width, self.height)
