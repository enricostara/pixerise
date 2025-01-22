import pygame
import numpy as np

from pixerise import Canvas, ViewPort, Renderer


def display(image: Canvas, scene, renderer):
    screen = pygame.display.set_mode(image.size, pygame.SCALED)
    clock = pygame.time.Clock()
    
    # Initialize mouse state
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    
    def update_display():
        renderer.render(scene)
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
                'translation': np.array([0, 1, 1], dtype=float),  # Moved higher (y from 1 to 2.5)
                'rotation': np.array([0, 0, 0], dtype=float)  # More downward tilt (from pi/12 to pi/4)
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
                },
                'color': (255, 50, 50)  # Red color for the left cube
            },
            {
                'model': 'cube',
                'transform': {
                    'translation': np.array([2, 1, 8], dtype=float),   # 2 units right, 1 up, 8 forward
                    'rotation': np.array([0, np.pi/4, 0], dtype=float), # rotated 45 degrees around Y axis
                    'scale': np.array([0.5, 0.5, 0.5], dtype=float)    # half the size
                },
                'color': (0, 255, 0)  # Green color for the right cube
            }
        ]
    }
    
    # Create renderer
    renderer = Renderer(canvas, viewport)
    
    # Display the result with the scene and renderer
    display(canvas, scene, renderer)


if __name__ == "__main__":
    pygame.init()
    main()
    pygame.quit()
