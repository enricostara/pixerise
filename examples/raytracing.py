import pygame

from pixerise.canvas import Canvas


class RayTracing:
    def __init__(self, canvas: Canvas):
        self._canvas = canvas
        self.screen = pygame.display.set_mode(canvas.size, pygame.SCALED | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()

    def run(self):
        surf = pygame.surfarray.make_surface(self._canvas.grid)
        self.screen.blit(surf, (0, 0))
        pygame.display.update()
        while True:
            for event in pygame.event.get():
                if (event.type == pygame.QUIT or
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return

            self.clock.tick(60)
            pygame.display.set_caption(str(self.clock.get_fps()))


if __name__ == '__main__':
    image = Canvas((320, 200))
    rayTracing = RayTracing(image)
    image.draw_point(0, 0, (255, 0, 0))
    image.draw_unchecked_point(-50, -50, (255, 0, 0))
    rayTracing.run()
