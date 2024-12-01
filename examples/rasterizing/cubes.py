import pygame
import numpy as np

from pixerise.canvas import Canvas
from pixerise.rasterizer import Rasterizer
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


def main():
    # Initialize canvas and viewport
    width, height = 800, 600
    canvas = Canvas((width, height))
    viewport = ViewPort((1.6, 1.2), 1, canvas)

    # Define scene with models and instances
    scene = {
        'models': {
            'cube': {
                'vertices': np.array([
                    (-1, -1, -1),  # 0: front bottom left
                    (1, -1, -1),   # 1: front bottom right
                    (1, 1, -1),    # 2: front top right
                    (-1, 1, -1),   # 3: front top left
                    (-1, -1, 1),   # 4: back bottom left
                    (1, -1, 1),    # 5: back bottom right
                    (1, 1, 1),     # 6: back top right
                    (-1, 1, 1),    # 7: back top left
                ], dtype=float),
                'triangles': np.array([
                    # Front face
                    (0, 1, 2), (0, 2, 3),
                    # Back face
                    (4, 6, 5), (4, 7, 6),
                    # Right face
                    (1, 5, 6), (1, 6, 2),
                    # Left face
                    (4, 0, 3), (4, 3, 7),
                    # Top face
                    (3, 2, 6), (3, 6, 7),
                    # Bottom face
                    (4, 5, 1), (4, 1, 0)
                ], dtype=int)
            }
        },
        'instances': [
            {
                'model': 'cube',
                'position': np.array([-2, 0, 7], dtype=float)  # 2 units left, 7 units forward
            },
            {
                'model': 'cube',
                'position': np.array([2, 1, 8], dtype=float)   # 2 units right, 1 unit up, 8 units forward
            }
        ]
    }
    
    # Create rasterizer
    rasterizer = Rasterizer(canvas, viewport, scene)
    rasterizer.render(scene)
    
    # Display the result
    display(canvas)


if __name__ == "__main__":
    pygame.init()
    main()
    pygame.quit()
