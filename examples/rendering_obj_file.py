import pygame
import numpy as np
from pathlib import Path

from pixerise import ShadingMode, Canvas, ViewPort, Renderer


def load_obj_file(file_path):
    """Load vertices, vertex normals, and faces from an OBJ file, organizing them into groups.
    
    The function supports:
    - Vertex positions (v)
    - Vertex normals (vn)
    - Face definitions (f)
    - Groups (g) and objects (o)
    
    Returns:
        dict: A dictionary containing model data with groups
    """
    vertices = []
    vertex_normals = []
    groups = {}
    current_group = 'default'
    groups[current_group] = {
        'vertices': [],
        'triangles': [],
        'vertex_normals': []
    }
    
    vertex_map = {}  # Maps (pos_idx, normal_idx) to new vertex index
    
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if not parts:
                continue
                
            if parts[0] == 'v':  # Vertex position
                vertices.append([float(x) for x in parts[1:4]])
            
            elif parts[0] == 'vn':  # Vertex normal
                vertex_normals.append([float(x) for x in parts[1:4]])
            
            elif parts[0] in ('g', 'o'):  # Group or object
                current_group = ' '.join(parts[1:]) or 'default'
                if current_group not in groups:
                    groups[current_group] = {
                        'vertices': [],
                        'triangles': [],
                        'vertex_normals': []
                    }
                    vertex_map = {}  # Reset vertex map for new group
            
            elif parts[0] == 'f':  # Face
                # Get vertex data for each face vertex
                face_vertices = []
                
                for vert in parts[1:]:
                    # Parse vertex indices
                    v_data = vert.split('/')
                    v_idx = int(v_data[0]) - 1  # OBJ is 1-based
                    vn_idx = int(v_data[2]) - 1 if len(v_data) > 2 and v_data[2] else -1
                    
                    # Create unique vertex if needed
                    vert_key = (v_idx, vn_idx)
                    if vert_key not in vertex_map:
                        new_idx = len(groups[current_group]['vertices'])
                        vertex_map[vert_key] = new_idx
                        groups[current_group]['vertices'].append(vertices[v_idx])
                        if vn_idx >= 0:
                            groups[current_group]['vertex_normals'].append(vertex_normals[vn_idx])
                    
                    face_vertices.append(vertex_map[vert_key])
                
                # Triangulate face
                for i in range(1, len(face_vertices) - 1):
                    groups[current_group]['triangles'].append([
                        face_vertices[0],
                        face_vertices[i + 1],
                        face_vertices[i]
                    ])
    
    # Convert lists to numpy arrays and clean up empty groups
    result = {'groups': {}}
    for group_name, group_data in groups.items():
        if len(group_data['triangles']) > 0:  # Only keep groups with geometry
            result['groups'][group_name] = {
                'vertices': np.array(group_data['vertices'], dtype=np.float32),
                'triangles': np.array(group_data['triangles'], dtype=np.int32),
                'vertex_normals': np.array(group_data['vertex_normals'], dtype=np.float32) if group_data['vertex_normals'] else []
            }
            print(group_name, len(group_data['triangles']), len(groups))
    
    return result


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
    model_data = load_obj_file(obj_path)
    
    # Calculate model scale to fit viewport
    # Find max coordinate across all groups
    max_coord = max(
        np.max(np.abs(group_data['vertices']))
        for group_data in model_data['groups'].values()
    )
    scale_factor = 0.1

    # Create scene with camera and directional light
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
            'tank': model_data
        },
        'instances': [
            {
                'model': 'tank',
                'name': 'tank',
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
    renderer = Renderer(canvas, viewport)
    
    # Display the result with the scene and renderer
    display(canvas, scene, renderer)


if __name__ == "__main__":
    pygame.init()
    main()
    pygame.quit()
