import pygame
import numpy as np
from pathlib import Path

from pixerise import ShadingMode, Canvas, ViewPort, Renderer
from scene import Scene, Model


def load_obj_file(file_path):
    """Load vertices, vertex normals, and faces from an OBJ file, organizing them into groups.

    The function supports:
    - Vertex positions (v)
    - Vertex normals (vn)
    - Face definitions (f)
    - Groups (g) and objects (o)

    Returns:
        Model: A Model object containing the loaded geometry, centered at origin
    """
    vertices = []
    vertex_normals = []
    groups = {}
    current_group = "default"
    groups[current_group] = {"vertices": [], "triangles": [], "vertex_normals": []}

    vertex_map = {}  # Maps (pos_idx, normal_idx) to new vertex index

    with open(file_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == "v":  # Vertex position
                vertices.append([float(x) for x in parts[1:4]])

            elif parts[0] == "vn":  # Vertex normal
                vertex_normals.append([float(x) for x in parts[1:4]])

            elif parts[0] in ("g", "o"):  # Group or object
                current_group = " ".join(parts[1:]) or "default"
                if current_group not in groups:
                    groups[current_group] = {
                        "vertices": [],
                        "triangles": [],
                        "vertex_normals": [],
                    }
                    vertex_map = {}  # Reset vertex map for new group

            elif parts[0] == "f":  # Face
                # Get vertex data for each face vertex
                face_vertices = []

                for vert in parts[1:]:
                    # Parse vertex indices
                    v_data = vert.split("/")
                    v_idx = int(v_data[0]) - 1  # OBJ is 1-based
                    vn_idx = int(v_data[2]) - 1 if len(v_data) > 2 and v_data[2] else -1

                    # Create unique vertex if needed
                    vert_key = (v_idx, vn_idx)
                    if vert_key not in vertex_map:
                        new_idx = len(groups[current_group]["vertices"])
                        vertex_map[vert_key] = new_idx
                        groups[current_group]["vertices"].append(vertices[v_idx])
                        if vn_idx >= 0:
                            groups[current_group]["vertex_normals"].append(
                                vertex_normals[vn_idx]
                            )

                    face_vertices.append(vertex_map[vert_key])

                # Triangulate face
                for i in range(1, len(face_vertices) - 1):
                    groups[current_group]["triangles"].append(
                        [face_vertices[0], face_vertices[i + 1], face_vertices[i]]
                    )

    # Create Model object and add groups
    model: Model = Model()

    # Calculate overall bounding box to center the model
    all_vertices = []
    for group_data in groups.values():
        if len(group_data["vertices"]) > 0:
            all_vertices.extend(group_data["vertices"])

    if all_vertices:
        # Convert to numpy array for efficient computation
        vertices_array = np.array(all_vertices, dtype=np.float32)

        # Calculate bounding box
        min_coords = np.min(vertices_array, axis=0)
        max_coords = np.max(vertices_array, axis=0)

        # Calculate center offset
        center_offset = (min_coords + max_coords) / 2

        # Add centered groups to model
        for group_name, group_data in groups.items():
            if len(group_data["triangles"]) > 0:  # Only keep groups with geometry
                # Center vertices by subtracting center_offset
                centered_vertices = (
                    np.array(group_data["vertices"], dtype=np.float32) - center_offset
                )

                model.add_group(
                    group_name,
                    centered_vertices,
                    np.array(group_data["triangles"], dtype=np.int32),
                    -np.array(group_data["vertex_normals"], dtype=np.float32)
                    if group_data["vertex_normals"]
                    else None,
                )

    return model


def display(canvas: Canvas, scene: Scene, renderer: Renderer):
    screen = pygame.display.set_mode(canvas.size, pygame.SCALED)
    clock = pygame.time.Clock()

    # Initialize mouse state
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    mouse_locked = True

    # Initialize shading mode
    shading_modes = [
        ShadingMode.WIREFRAME,
        ShadingMode.FLAT,
        ShadingMode.GOURAUD,
        ShadingMode.FLAT,
    ]
    current_mode_index = 0

    # Track selected groups
    selected_groups = []

    def update_display():
        renderer.render(scene, shading_mode=shading_modes[current_mode_index])
        surf = pygame.surfarray.make_surface(canvas.color_buffer)
        screen.blit(surf, (0, 0))
        pygame.display.update()

    # Initial render
    update_display()

    # Movement speed
    move_speed = 0.1
    wheel_speed = 0.1  # Speed for mouse wheel movement
    wheel_momentum = 0.9  # How much wheel velocity is retained (0-1)
    rotation_speed = 0.007  # Speed of tank rotation

    # Track if any movement occurred
    movement_occurred = False

    # Initialize wheel velocity
    wheel_velocity = 0.0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                return

            # Handle shading mode switching with spacebar
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                current_mode_index = (current_mode_index + 1) % len(shading_modes)
                movement_occurred = True

            # Handle mouse toggle with backspace
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_BACKSPACE:
                mouse_locked = not mouse_locked
                pygame.mouse.set_visible(not mouse_locked)
                pygame.event.set_grab(mouse_locked)

            # Handle mouse clicks when not locked
            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and not mouse_locked
            ):  # Left click
                # Get mouse position
                mouse_pos = pygame.mouse.get_pos()

                # Cast ray and get hit
                hit = renderer.cast_ray(mouse_pos[0], mouse_pos[1], scene)

                # Update selection based on hit
                if hit:
                    instance_name, group_name = hit
                    print(f"Selected group: {group_name} in instance {instance_name}")
                    instance = scene._instances[instance_name]
                    
                    # Toggle selection - if already selected, deselect it
                    if hit in selected_groups:
                        selected_groups.remove(hit)
                        instance.set_group_color(group_name, None)  # Remove group-specific color
                    else:
                        selected_groups.append(hit)
                        # Set color for the specific group
                        instance.set_group_color(group_name, np.array([255, 0, 0], dtype=np.int32))
                else:
                    # Reset all selected groups' colors when clicking empty space
                    for inst_name, grp_name in selected_groups:
                        instance = scene._instances[inst_name]
                        instance.set_group_color(grp_name, None)  # Remove group-specific color
                    selected_groups.clear()

                movement_occurred = True

            # Only handle mouse movement when locked
            elif event.type == pygame.MOUSEMOTION and mouse_locked:
                dx, dy = event.rel
                # Convert mouse movement to rotation (scale down the movement)
                rot_y = -dx * 0.002  # Horizontal mouse movement controls Y rotation
                rot_x = -dy * 0.002  # Vertical mouse movement controls X rotation

                # Update camera rotation
                camera_rot = np.array(scene._camera.rotation)
                camera_rot[0] += rot_x  # Remove clipping for vertical rotation
                camera_rot[1] = (camera_rot[1] + rot_y) % (
                    2 * np.pi
                )  # Allow full horizontal rotation
                scene._camera.rotation = camera_rot
                movement_occurred = True

            # Handle mouse wheel for forward/backward movement
            elif event.type == pygame.MOUSEWHEEL and mouse_locked:
                # Add to current wheel velocity
                wheel_velocity += event.y * wheel_speed

        # Continuous movement
        keys = pygame.key.get_pressed()

        # Apply wheel momentum for smooth movement
        if (
            mouse_locked and abs(wheel_velocity) > 0.0001
        ):  # Small threshold to stop tiny movements
            # Calculate forward vector based on camera rotation
            camera_rot = np.array(scene._camera.rotation)
            forward = np.array([np.sin(camera_rot[1]), 0, np.cos(camera_rot[1])])
            # Move based on current wheel velocity
            scene._camera.translation += forward * wheel_velocity
            # Apply momentum (gradually reduce velocity)
            wheel_velocity *= wheel_momentum
            movement_occurred = True
        else:
            wheel_velocity = 0.0  # Reset to exactly zero when very small

        if mouse_locked and any(
            [
                keys[key]
                for key in [
                    pygame.K_w,
                    pygame.K_s,
                    pygame.K_a,
                    pygame.K_d,
                    pygame.K_q,
                    pygame.K_e,
                ]
            ]
        ):
            movement_occurred = True
            camera_pos = np.array(scene._camera.translation)
            camera_rot = np.array(scene._camera.rotation)

            # Calculate forward and right vectors
            forward = np.array([np.sin(camera_rot[1]), 0, np.cos(camera_rot[1])])
            right = np.array([np.cos(camera_rot[1]), 0, -np.sin(camera_rot[1])])

            # Apply movement based on keys
            if keys[pygame.K_w]:
                camera_pos += forward * move_speed
            if keys[pygame.K_s]:
                camera_pos -= forward * move_speed
            if keys[pygame.K_a]:
                camera_pos -= right * move_speed
            if keys[pygame.K_d]:
                camera_pos += right * move_speed
            if keys[pygame.K_q]:  # Move down
                camera_pos[1] -= move_speed
            if keys[pygame.K_e]:  # Move up
                camera_pos[1] += move_speed

            scene._camera.translation = camera_pos

        # Rotate the tank model
        tank_instance = scene._instances["tank"]
        tank_rot = tank_instance.rotation
        tank_rot[1] = (tank_rot[1] + rotation_speed) % (
            2 * np.pi
        )  # Rotate around Y axis
        tank_instance.rotation = tank_rot
        movement_occurred = True

        # Update display only if movement occurred
        if movement_occurred:
            update_display()
            movement_occurred = False

        clock.tick(60)
        pygame.display.set_caption(
            f"{shading_modes[current_mode_index].value} - {str(clock.get_fps())[0:2]} fps - press SPACE to toggle shading mode"
        )


def main():
    # Initialize canvas and viewport
    width, height = 800, 600
    canvas = Canvas((width, height))
    viewport = ViewPort((1.6, 1.2), 1, canvas)

    # Load Tank model
    obj_path = Path(__file__).parent / "tank.obj"
    model = load_obj_file(obj_path)

    scale_factor = 0.1

    # Create scene with camera and directional light
    scene_dict = {
        "camera": {
            "transform": {
                "translation": np.array(
                    [0, 1, -5], dtype=float
                ),  # Position camera above and back
                "rotation": np.array([0.2, 0, 0], dtype=float),  # Slight downward tilt
            }
        },
        "lights": {
            "directional": {
                "direction": np.array(
                    [-1, 1, -1], dtype=float
                ),  # Light coming from top-left-front
                "intensity": 0.8,  # Increased intensity for better visibility
                "ambient": 0.3,  # Increased ambient for better shadow detail
            }
        },
        "instances": [
            {
                "model": "tank",
                "name": "tank",
                "transform": {
                    "translation": np.array([0, 0, 0], dtype=float),
                    "rotation": np.array(
                        [0, np.pi, 0], dtype=float
                    ),  # Rotate to face camera
                    "scale": np.array(
                        [scale_factor, scale_factor, scale_factor], dtype=float
                    ),
                },
                "color": [200, 200, 200],  # Gray color as integer RGB values
            }
        ],
    }

    # Create renderer
    renderer = Renderer(canvas, viewport)
    scene = Scene.from_dict(scene_dict)
    scene.add_model("tank", model)
    # Display the result with the scene and renderer
    display(canvas, scene, renderer)


if __name__ == "__main__":
    pygame.init()
    main()
    pygame.quit()
