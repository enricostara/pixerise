import pygame

from pixerise.canvas import Canvas
from pixerise.raytracer import RayTracer
from pixerise.viewport import ViewPort


def display(image: Canvas):
    screen = pygame.display.set_mode(image.size, pygame.SCALED)
    clock = pygame.time.Clock()
    surf = pygame.surfarray.make_surface(image.grid)
    screen.blit(surf, (0, 0))
    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if (event.type == pygame.QUIT or
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return
        clock.tick(60)
        pygame.display.set_caption(str(clock.get_fps()))


if __name__ == '__main__':
    canvas = Canvas((320, 200))
    scene = {
        'spheres': [
            {
                'center': (0, -1, 3),
                'radius': 1,
                'color': (255, 0, 0),
            },
            {
                'center': (2, 0, 4),
                'radius': 1,
                'color': (0, 0, 255)
            }
        ]
    }
    view_port = ViewPort((1, 1), 1, canvas)
    ray_tracer = RayTracer(canvas, view_port, scene)
    ray_tracer.render((0, 0, 0))
    display(canvas)
