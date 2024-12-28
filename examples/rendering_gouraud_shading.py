import pygame
import numpy as np

from pixerise import ShadingMode, Canvas, ViewPort, Renderer


def display(image: Canvas, scene, renderer):
    screen = pygame.display.set_mode(image.size, pygame.SCALED)
    clock = pygame.time.Clock()
    
    # Initialize mouse state
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    
    def update_display():
        renderer.render(scene, shading_mode=ShadingMode.GOURAUD)
        surf = pygame.surfarray.make_surface(image.color_buffer)
        screen.blit(surf, (0, 0))
        pygame.display.update()
    
    # Initial render
    update_display()
    
    # Movement speed
    move_speed = 0.1

    # Track if any movement occurred
    movement_occurred = False
    
    while True:
        for event in pygame.event.get():
            if (event.type == pygame.QUIT or
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return
            
            # Handle mouse movement
            if event.type == pygame.MOUSEMOTION:
                dx, dy = event.rel
                # Convert mouse movement to rotation (scale down the movement)
                rot_y = -dx * 0.002  # Horizontal mouse movement controls Y rotation
                rot_x = -dy * 0.002  # Vertical mouse movement controls X rotation
                
                # Update camera rotation
                current_rot = scene['camera']['transform']['rotation']
                current_rot[0] += rot_x  # Remove clipping for vertical rotation
                current_rot[1] = (current_rot[1] + rot_y) % (2 * np.pi)  # Allow full horizontal rotation
                scene['camera']['transform']['rotation'] = current_rot
                
                # Update display
                movement_occurred = True
        
        # Continuous movement
        keys = pygame.key.get_pressed()
        camera_trans = scene['camera']['transform']['translation']
        camera_rot = scene['camera']['transform']['rotation']
        
        # Calculate forward direction based on current rotation
        forward = np.array([
            -np.sin(camera_rot[1]) * np.cos(camera_rot[0]),
            np.sin(camera_rot[0]),
            -np.cos(camera_rot[1]) * np.cos(camera_rot[0])
        ])
        
        # Calculate right vector
        right = np.array([
            np.cos(camera_rot[1]) * np.cos(camera_rot[0]),
            0,
            -np.sin(camera_rot[1]) * np.cos(camera_rot[0])
        ])
        
        # Move forward with W key
        if keys[pygame.K_w]:
            camera_trans -= forward * move_speed
            movement_occurred = True
        
        # Move backward with S key
        if keys[pygame.K_s]:
            camera_trans += forward * move_speed
            movement_occurred = True
        
        # Move left with A key
        if keys[pygame.K_a]:
            camera_trans -= right * move_speed
            movement_occurred = True
        
        # Move right with D key
        if keys[pygame.K_d]:
            camera_trans += right * move_speed
            movement_occurred = True

        # Arrow key controls for rotation
        if keys[pygame.K_UP]:
            camera_rot[0] -= move_speed * 0.2
            movement_occurred = True
        if keys[pygame.K_DOWN]:
            camera_rot[0] += move_speed * 0.2
            movement_occurred = True
        if keys[pygame.K_LEFT]:
            camera_rot[1] += move_speed * 0.2
            movement_occurred = True
        if keys[pygame.K_RIGHT]:
            camera_rot[1] -= move_speed * 0.2
            movement_occurred = True

        scene['camera']['transform']['rotation'] = camera_rot

        # Update display only if movement occurred
        if movement_occurred:
            update_display()
        
        clock.tick(60)
        pygame.display.set_caption(str(clock.get_fps())[0:2])


def main():
    # Initialize canvas and viewport
    width, height = 800, 600
    canvas = Canvas((width, height))
    viewport = ViewPort((1.6, 1.2), 1, canvas)

    # Define scene with models and instances
    scene = {
        'camera': {
            'transform': {
                'translation': np.array([0, 1, 1], dtype=float),
                'rotation': np.array([0, 0, 0], dtype=float)
            }
        },
        'lights': {
            'directional': {
                'direction': np.array([-1, -1, -1], dtype=float),  # Light coming from top-left-front
                'intensity': 0.7,
                'ambient': 0.3  # Slightly increased ambient for better Gouraud visualization
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
            },
            'sphere': {
                'vertices': [],
                'triangles': []
            }
        },
        'instances': [
            {
                'model': 'cube',
                'transform': {
                    'translation': np.array([-2, 0, 7], dtype=float),
                    'rotation': np.array([np.pi/6, np.pi/4, 0], dtype=float),  # Rotated for better shading visibility
                    'scale': np.array([1, 1, 1], dtype=float)
                },
                'color': (200, 100, 100)  # Reddish color to show shading variations
            },
            {
                'model': 'sphere',  # Changed from cube to sphere
                'transform': {
                    'translation': np.array([2, 1, 8], dtype=float),
                    'rotation': np.array([0, 0, 0], dtype=float),  # Sphere rotation less important
                    'scale': np.array([0.8, 0.8, 0.8], dtype=float)
                },
                'color': (100, 200, 100)  # Greenish color
            }
        ]
    }
    
    # Generate sphere vertices and triangles
    radius = 1.0
    subdivisions = 16  # subdivision count for smooth shading
    
    # Generate vertices
    vertices = []
    for i in range(subdivisions + 1):
        lat = np.pi * (-0.5 + float(i) / subdivisions)
        for j in range(subdivisions + 1):
            lon = 2 * np.pi * float(j) / subdivisions
            x = np.cos(lat) * np.cos(lon)
            y = np.sin(lat)
            z = np.cos(lat) * np.sin(lon)
            vertices.append([x * radius, y * radius, z * radius])
    
    # Generate triangles
    triangles = []
    for i in range(subdivisions):
        for j in range(subdivisions):
            first = i * (subdivisions + 1) + j
            second = first + subdivisions + 1
            triangles.extend([
                [first, second, first + 1],
                [second, second + 1, first + 1]
            ])
    
    # Convert to numpy arrays and assign to sphere model
    scene['models']['sphere']['vertices'] = np.array(vertices, dtype=np.float32)
    scene['models']['sphere']['triangles'] = np.array(triangles, dtype=np.int32)
    
    # Create renderer
    renderer = Renderer(canvas, viewport, scene)
    
    # Display the result with the scene and renderer
    display(canvas, scene, renderer)


if __name__ == "__main__":
    pygame.init()
    main()
    pygame.quit()
