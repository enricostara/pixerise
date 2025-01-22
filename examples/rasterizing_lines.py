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
    
    # Create renderer
    renderer = Renderer(canvas, viewport)
    
    # Draw some example lines
    # Draw a colorful star pattern
    center_x, center_y = (0, 0)
    radius = 300
    num_points = 200
    
    for i in range(num_points):
        angle = (2 * np.pi * i) / num_points
        # Calculate points relative to center (0,0)
        end_x = radius * np.cos(angle)
        end_y = radius * np.sin(angle)
        
        # Convert to screen coordinates
        screen_x = center_x + end_x
        screen_y = center_y + end_y
        
        # Create rainbow-like colors
        hue = (i / num_points) * 360
        # Convert HSV to RGB (simplified conversion)
        h = hue / 60
        x = int(255 * (1 - abs(h % 2 - 1)))
        if h < 1: color = (255, x, 0)
        elif h < 2: color = (x, 255, 0)
        elif h < 3: color = (0, 255, x)
        elif h < 4: color = (0, x, 255)
        elif h < 5: color = (x, 0, 255)
        else: color = (255, 0, x)
        
        # Draw line from center to point
        renderer.draw_line((center_x, center_y, 0), (screen_x, screen_y, 0), color)
    
    # Display the result
    display(canvas)


if __name__ == "__main__":
    pygame.init()
    main()
    pygame.quit()
