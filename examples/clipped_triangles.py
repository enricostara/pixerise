import pygame
import numpy as np

from pixerise import Canvas, ViewPort, Renderer
from kernel.clipping_mod import clip_triangle


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


def draw_clipped_triangle(renderer, vertices, color, plane_normals):
    """Draw a triangle after clipping against multiple planes."""
    # Convert 2D vertices to 3D (z=0)
    vertices_3d = np.array([
        [vertices[0][0], vertices[0][1], 0],
        [vertices[1][0], vertices[1][1], 0],
        [vertices[2][0], vertices[2][1], 0]
    ])
    
    # Start with the original triangle
    current_triangles = [vertices_3d]
    
    # Clip against each plane
    for normal in plane_normals:
        next_triangles = []
        for tri in current_triangles:
            clipped, num_triangles = clip_triangle(tri, normal)
            for i in range(num_triangles):
                next_triangles.append(clipped[i])
        current_triangles = next_triangles
        
        if not current_triangles:  # Triangle completely clipped away
            return
    
    # Draw all resulting triangles
    for tri in current_triangles:
        renderer.draw_triangle(
            (tri[0][0], tri[0][1]),  # Convert back to 2D
            (tri[1][0], tri[1][1]),
            (tri[2][0], tri[2][1]),
            color
        )


def main():
    # Initialize canvas and viewport
    width, height = 800, 600
    canvas = Canvas((width, height))
    viewport = ViewPort((width, height), 1, canvas)
    scene = {}
    
    # Create renderer
    renderer = Renderer(canvas, viewport, scene)
    
    # Define clipping planes 
    angle = np.pi/2.3  # 30 degrees
    
    # Create normalized plane normals
    plane_normals = [
        np.array([np.cos(angle), np.sin(angle), 0]),  # Left plane
        np.array([-np.cos(angle), np.sin(angle), 0])  # Right plane
    ]
    
    # Draw white lines to mark the clipping planes
    center_x, center_y = (0, 0)
    radius = 300
    line_length = radius * 1.5  # Make lines longer than the pattern
    for i, normal in enumerate(plane_normals):
        # Calculate two points for each line
        # The line will be perpendicular to the normal vector
        # We can get this by using the normal's y component as x and negative x component as y
        dx, dy = -normal[1], normal[0]  # Direction vector for the line (negated from before)
        if i == 1:  # Second line should go in opposite direction for clockwise order
            dx, dy = -dx, -dy
        renderer.draw_line(
            (0, 0),  # Start from origin
            (dx * line_length, dy * line_length),  # Extend in one direction
            (255, 255, 255)  # White color
        )
    
    # Draw a colorful triangle pattern
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
            
        # Draw the clipped triangle
        vertices = [(x1, y1), (x2, y2), (x3, y3)]
        draw_clipped_triangle(renderer, vertices, (r, g, b), plane_normals)
    
    # Display the result
    display(canvas)


if __name__ == "__main__":
    pygame.init()
    main()
    pygame.quit()
