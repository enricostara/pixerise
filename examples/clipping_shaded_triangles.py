import pygame
import numpy as np

from pixerise import Canvas, ViewPort, Renderer
from kernel.clipping_mod import clip_triangle

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

def draw_clipped_triangle(renderer, vertices, base_color, plane_normals):
    """Draw a triangle after clipping against multiple planes."""
    # Convert 2D vertices to 3D (z=0)
    vertices_3d = np.array([
        [vertices[0][0], vertices[0][1], 0],
        [vertices[1][0], vertices[1][1], 0],
        [vertices[2][0], vertices[2][1], 0]
    ], dtype=np.float32)
    
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
    
    # Draw all resulting triangles with gradient intensities
    for tri in current_triangles:
        # Calculate varying intensities based on vertex positions
        # This creates a radial gradient effect
        dist1 = np.sqrt(tri[0][0]**2 + tri[0][1]**2) / 300  # Normalize by radius
        dist2 = np.sqrt(tri[1][0]**2 + tri[1][1]**2) / 300
        dist3 = np.sqrt(tri[2][0]**2 + tri[2][1]**2) / 300
        
        # Create smooth intensity variations
        intensity1 = 0.3 + 0.7 * (1.0 - dist1)
        intensity2 = 0.3 + 0.7 * (1.0 - dist2)
        intensity3 = 0.3 + 0.7 * (1.0 - dist3)
        
        renderer.draw_shaded_triangle(
            (tri[0][0], tri[0][1], tri[0][2]),
            (tri[1][0], tri[1][1], tri[1][2]),
            (tri[2][0], tri[2][1], tri[2][2]),
            base_color,
            intensity1, intensity2, intensity3
        )

def main():
    # Initialize canvas and viewport
    width, height = 800, 600
    canvas = Canvas((width, height))
    viewport = ViewPort((width, height), 1, canvas)
    
    # Create renderer
    renderer = Renderer(canvas, viewport)
    
    # Define clipping planes 
    angle = np.pi/2.25
    
    # Create normalized plane normals
    plane_normals = [
        np.array([np.cos(angle), np.sin(angle), 0], dtype=np.float32),  # Left plane
        np.array([-np.cos(angle), np.sin(angle), 0], dtype=np.float32)  # Right plane
    ]
    
    # Draw white lines to mark the clipping planes
    line_length = 450
    for i, normal in enumerate(plane_normals):
        dx, dy = -normal[1], normal[0]
        if i == 1:
            dx, dy = -dx, -dy
        renderer.draw_line(
            (0, 0, 0),
            (dx * line_length, dy * line_length, 0),
            (255, 255, 255)
        )
    
    # Draw a pattern of colorful triangles
    num_triangles = 16  # Increased number of triangles
    center_x, center_y = (0, 0)
    radius = 300
    
    # Draw rotating triangles with vibrant colors
    for i in range(num_triangles):
        # Calculate angles with overlap for interesting patterns
        angle1 = (2 * np.pi * i) / num_triangles
        angle2 = (2 * np.pi * (i + 0.15)) / num_triangles
        angle3 = (2 * np.pi * (i + 0.85)) / num_triangles
        
        # Create triangle vertices with varying distances
        x1 = center_x + radius * np.cos(angle1)
        y1 = center_y + radius * np.sin(angle1)
        
        x2 = center_x + (radius * 0.6) * np.cos(angle2)
        y2 = center_y + (radius * 0.6) * np.sin(angle2)
        
        x3 = center_x + (radius * 0.6) * np.cos(angle3)
        y3 = center_y + (radius * 0.6) * np.sin(angle3)
        
        # Create vibrant rainbow colors
        hue = (i / num_triangles) * 360
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
        
        # Draw the shaded and clipped triangle
        vertices = [(x1, y1), (x2, y2), (x3, y3)]
        draw_clipped_triangle(renderer, vertices, (r, g, b), plane_normals)
    
    # Display the result
    display(canvas)

if __name__ == "__main__":
    pygame.init()
    main()
    pygame.quit()
