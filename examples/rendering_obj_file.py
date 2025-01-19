import pygame
import numpy as np
from pathlib import Path

from pixerise import ShadingMode, Canvas, ViewPort, Renderer


def load_obj_file(file_path):
    """Load vertices and triangles from an OBJ file."""
    vertices = []
    triangles = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):  # Vertex
                # Split line and convert coordinates to float
                _, x, y, z = line.split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith('f '):  # Face
                # Split line and get vertex indices (OBJ indices start at 1)
                # Handle both v and v/vt/vn formats
                face = []
                parts = line.split()[1:]  # Skip the 'f' at the start
                for part in parts:
                    # Get the vertex index (before any '/' character)
                    vertex_idx = int(part.split('/')[0]) - 1  # Subtract 1 for 0-based indexing
                    face.append(vertex_idx)
                # Convert polygons to triangles (assuming convex polygons)
                for i in range(1, len(face)-1):
                    triangles.append([face[0], face[i+1], face[i]])  # Invert vertex index order
    
    return np.array(vertices, dtype=float), np.array(triangles, dtype=int)


def display(image: Canvas, scene, renderer):
    screen = pygame.display.set_mode(image.size, pygame.SCALED)
    clock = pygame.time.Clock()
    
    # Initialize mouse state
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    
    def update_display():
        renderer.render(scene, shading_mode=ShadingMode.WIREFRAME)
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
        # print(scene['camera']['transform'])
        pygame.display.set_caption(str(clock.get_fps())[0:2])


def main():
    # Initialize canvas and viewport
    width, height = 800, 600
    canvas = Canvas((width, height))
    viewport = ViewPort((1.6, 1.2), 1, canvas)

    # Load Tank model
    obj_path = Path(__file__).parent / 'tank.obj'
    vertices, triangles = load_obj_file(obj_path)
    
    # Calculate model scale to fit viewport
    max_coord = np.max(np.abs(vertices))
    # scale_factor = 2.0 / max_coord  # Scale to fit in a 2x2x2 box
    scale_factor = 0.1

    # Define scene with Tank model
    scene = {
        'camera': {
            'transform': {
                'translation': np.array([4.4,  1.25, -3.8], dtype=float),  # Position camera above and back
                'rotation': np.array([0.2, 5.45, 0], dtype=float)  # Slight downward tilt
            }
        },
        'lights': {
            'directional': {
                'direction': np.array([-1, -1, -1], dtype=float),  # Light coming from top-left-front
                'intensity': 0.7,
                'ambient': 0.2
            }
        },
        'models': {
            'tank': {
                'vertices': vertices,
                'triangles': triangles
            }
        },
        'instances': [
            {
                'model': 'tank',
                'transform': {
                    'translation': np.array([0, 0, 0], dtype=float),
                    'rotation': np.array([0, np.pi, 0], dtype=float),  # Rotate to face camera
                    'scale': np.array([scale_factor, scale_factor, scale_factor], dtype=float)
                },
                'color': (180, 180, 180)  # Gray color for the tank
            }
        ]
    }
    
    # Create renderer
    renderer = Renderer(canvas, viewport, scene)
    
    # Display the result with the scene and renderer
    display(canvas, scene, renderer)


if __name__ == "__main__":
    pygame.init()
    main()
    pygame.quit()
