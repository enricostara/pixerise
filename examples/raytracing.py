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
                'color': (255, 0, 0),  # red
                'specular': 100
            },
            {
                'center': (2, 0, 4),
                'radius': 1,
                'color': (0, 0, 255),  # blue
                'specular': -1
            },
            {
                'center': (-2, 0, 4),
                'radius': 1,
                'color': (0, 255, 0),  # green
                'specular': 10
            },
            {
                'center': (0, -10000.85, 0),
                'radius': 10000,
                'color': (255, 255, 0),  # yellow
                'specular': 100
            }
        ],
        'lights': [
            {
                'type': 'ambient',
                'intensity': 0.2
            },
            {
                'type': 'point',
                'position': (2, 1, 0),
                'intensity': 0.2
            },
            {
                'type': 'directional',
                'direction': (1, 4, 4),
                'intensity': 0.2
            }
        ]
    }
    view_port = ViewPort((1.6, .9), 1, canvas)
    ray_tracer = RayTracer(canvas, view_port, scene)
    ray_tracer.render((0, 0, 0))
    display(canvas)
