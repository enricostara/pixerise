import pygame
import numpy as np

from pixerise import Canvas, ViewPort, Rasterizer


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
        'camera': {
            'transform': {
                'translation': np.array([0, 5, 1], dtype=float),  # Moved higher (y from 1 to 2.5)
                'rotation': np.array([np.pi/4, 0, 0], dtype=float)  # More downward tilt (from pi/12 to pi/4)
            }
        },
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
                'transform': {
                    'translation': np.array([-2, 0, 7], dtype=float),  # 2 units left, 7 units forward
                    'rotation': np.array([0, 0, 0], dtype=float),      # rotation angles in radians (x, y, z)
                    'scale': np.array([1, 1, 1], dtype=float)         # uniform scale
                }
            },
            {
                'model': 'cube',
                'transform': {
                    'translation': np.array([2, 1, 8], dtype=float),   # 2 units right, 1 up, 8 forward
                    'rotation': np.array([0, np.pi/4, 0], dtype=float), # rotated 45 degrees around Y axis
                    'scale': np.array([0.5, 0.5, 0.5], dtype=float)    # half the size
                }
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
