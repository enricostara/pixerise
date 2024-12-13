import pygame
import numpy as np

from pixerise import Canvas, ViewPort, Renderer


def display(image: Canvas):
    screen = pygame.display.set_mode(image.size, pygame.SCALED)
    clock = pygame.time.Clock()
    surf = pygame.surfarray.make_surface(image.color_buffer)
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
    viewport = ViewPort((width, height), 1, canvas)
    scene = {}
    
    # Create renderer
    renderer = Renderer(canvas, viewport, scene)
    
    # Draw a colorful triangle pattern
    center_x, center_y = (0, 0)
    radius = 300
    num_triangles = 16
    
    # Draw rotating triangles with different colors
    for i in range(num_triangles):
        # Calculate three points for each triangle
        angle1 = (2 * np.pi * i) / num_triangles
        angle2 = (2 * np.pi * (i + 0.15)) / num_triangles
        angle3 = (2 * np.pi * (i + 0.85)) / num_triangles
        
        # First point (outer)
        x1 = center_x + radius * np.cos(angle1)
        y1 = center_y + radius * np.sin(angle1)
        
        # Second point (inner)
        x2 = center_x + (radius * 0.5) * np.cos(angle2)
        y2 = center_y + (radius * 0.5) * np.sin(angle2)
        
        # Third point (inner)
        x3 = center_x + (radius * 0.5) * np.cos(angle3)
        y3 = center_y + (radius * 0.5) * np.sin(angle3)
        
        # Create rainbow-like colors
        hue = (i / num_triangles) * 360
        # Convert HSV to RGB (assuming hue in range 0-360)
        h = hue / 60
        c = 255  # Maximum value
        x = int(c * (1 - abs(h % 2 - 1)))
        
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
            
        # Draw the triangle
        renderer.draw_triangle(
            (x1, y1),
            (x2, y2),
            (x3, y3),
            (r, g, b)
        )
    
    # Display the result
    display(canvas)


if __name__ == "__main__":
    pygame.init()
    main()
    pygame.quit()
