import numpy as np
import pygame
from pygame import SCALED


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


class Frame:
    def __init__(self, canvas: Canvas):
        self._canvas = canvas
        self.screen = pygame.display.set_mode(canvas.size, SCALED | pygame.DOUBLEBUF | pygame.FULLSCREEN)
        self.clock = pygame.time.Clock()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._canvas.set_pixel(*event.pos, (255, 0, 0))
                    print(self.clock.get_fps())

            self.clock.tick(60)
            surf = pygame.surfarray.make_surface(self._canvas.grid)
            self.screen.blit(surf, (0, 0))
            pygame.display.update()


if __name__ == '__main__':
    canvas = Canvas((320, 200))
    frame = Frame(canvas)
    print('start')
    canvas.draw_point(0, 0, (255, 0, 0))
    canvas.draw_unchecked_point(-50, -50, (255, 0, 0))
    frame.run()
