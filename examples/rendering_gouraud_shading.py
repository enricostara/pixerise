import pygame
import numpy as np

from pixerise.pixerise import ShadingMode, Canvas, ViewPort, Renderer, Scene


def display(canvas: Canvas, scene: Scene, renderer: Renderer):
    screen = pygame.display.set_mode(canvas.size, pygame.SCALED)
    clock = pygame.time.Clock()

    # Initialize mouse state
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    def update_display():
        renderer.render(scene, shading_mode=ShadingMode.GOURAUD)
        surf = pygame.surfarray.make_surface(canvas.color_buffer)
        screen.blit(surf, (0, 0))
        pygame.display.update()

    # Initial render
    update_display()

    # Movement speed
    move_speed = 0.1
    rotation_speed = 0.01  # Speed of rotation
    wheel_speed = 0.1  # Speed for mouse wheel movement
    wheel_momentum = 0.95  # How much wheel velocity is retained (0-1)

    # Track if any movement occurred
    movement_occurred = False

    # Initialize rotation angles for objects
    cube_rotation = 0.0
    sphere_rotation = 0.0

    # Initialize wheel velocity
    wheel_velocity = 0.0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                return

            # Handle mouse movement
            if event.type == pygame.MOUSEMOTION:
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
            elif event.type == pygame.MOUSEWHEEL:
                # Add to current wheel velocity
                wheel_velocity += event.y * wheel_speed

        # Continuous movement
        keys = pygame.key.get_pressed()

        # Apply wheel momentum for smooth movement
        if abs(wheel_velocity) > 0.0001:  # Small threshold to stop tiny movements
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

        if any(
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

            # Update camera position based on key presses
            if keys[pygame.K_w]:
                camera_pos += forward * move_speed
            if keys[pygame.K_s]:
                camera_pos -= forward * move_speed
            if keys[pygame.K_a]:
                camera_pos -= right * move_speed
            if keys[pygame.K_d]:
                camera_pos += right * move_speed
            if keys[pygame.K_q]:
                camera_pos[1] -= move_speed  # Move down
            if keys[pygame.K_e]:
                camera_pos[1] += move_speed  # Move up

            scene._camera.translation = camera_pos

        # Update object rotations
        cube_rotation += rotation_speed
        sphere_rotation += rotation_speed * 0.7  # Slower rotation for spheres

        # Update instance rotations
        cube = scene.get_instance("left")
        middle_sphere = scene.get_instance("middle")
        right_sphere = scene.get_instance("right")

        if cube and middle_sphere and right_sphere:
            # Cube: complex rotation around multiple axes
            cube.rotation = np.array(
                [
                    np.pi / 6 + cube_rotation / 2,
                    np.pi / 4 + cube_rotation,
                    cube_rotation / 3,
                ],
                dtype=float,
            )

            # Middle sphere (flat): gentle wobble
            middle_sphere.rotation = np.array(
                [np.sin(sphere_rotation / 2) * 0.2, sphere_rotation, 0], dtype=float
            )

            # Right sphere (smooth): slow steady rotation
            right_sphere.rotation = np.array([0, sphere_rotation * 0.5, 0], dtype=float)

            movement_occurred = True

        # Update display only if movement occurred
        if movement_occurred:
            update_display()

        clock.tick(120)
        pygame.display.set_caption(str(clock.get_fps())[0:3])


def main():
    # Initialize canvas and viewport
    width, height = 800, 600
    canvas = Canvas((width, height))
    viewport = ViewPort((1.6, 1.2), 1, canvas)

    # Define scene with models and instances
    scene = {
        "camera": {
            "transform": {
                "translation": np.array([0, 1, 1], dtype=float),
                "rotation": np.array([0, 0, 0], dtype=float),
            }
        },
        "lights": {
            "directional": {
                "direction": np.array(
                    [-1, 1, -1], dtype=float
                ),  # Light coming from top-left-front
                "intensity": 0.7,
                "ambient": 0.3,  # Slightly increased ambient for better Gouraud visualization
            }
        },
        "models": {
            "cube": {
                "vertices": np.array(
                    [
                        (-1, -1, -1),  # 0: front bottom left
                        (1, -1, -1),  # 1: front bottom right
                        (1, 1, -1),  # 2: front top right
                        (-1, 1, -1),  # 3: front top left
                        (-1, -1, 1),  # 4: back bottom left
                        (1, -1, 1),  # 5: back bottom right
                        (1, 1, 1),  # 6: back top right
                        (-1, 1, 1),  # 7: back top left
                    ],
                    dtype=float,
                ),
                "triangles": np.array(
                    [
                        # Front face
                        (0, 1, 2),
                        (0, 2, 3),
                        # Back face
                        (4, 6, 5),
                        (4, 7, 6),
                        # Right face
                        (1, 5, 6),
                        (1, 6, 2),
                        # Left face
                        (4, 0, 3),
                        (4, 3, 7),
                        # Top face
                        (3, 2, 6),
                        (3, 6, 7),
                        # Bottom face
                        (4, 5, 1),
                        (4, 1, 0),
                    ],
                    dtype=int,
                ),
            },
            "sphere": {"vertices": [], "triangles": [], "vertex_normals": []},
            "sphere_flat": {"vertices": [], "triangles": [], "vertex_normals": []},
        },
        "instances": [
            {
                "model": "cube",
                "name": "left",
                "transform": {
                    "translation": np.array([-2, 1, 6], dtype=float),
                    "rotation": np.array(
                        [np.pi / 6, np.pi / 4, 0], dtype=float
                    ),  # Rotated for better shading visibility
                    "scale": np.array([0.6, 0.6, 0.6], dtype=float),
                },
                "color": (220, 60, 60),  # Reddish color to show shading variations
            },
            {
                "model": "sphere_flat",  # Changed from cube to sphere
                "name": "middle",
                "transform": {
                    "translation": np.array([0, 1, 6], dtype=float),
                    "rotation": np.array(
                        [0, 0, 0], dtype=float
                    ),  # Sphere rotation less important
                    "scale": np.array([0.8, 0.8, 0.8], dtype=float),
                },
                "color": (60, 220, 60),  # Greenish color
            },
            {
                "model": "sphere",  # Changed from cube to sphere
                "name": "right",
                "transform": {
                    "translation": np.array([2, 1, 6], dtype=float),
                    "rotation": np.array(
                        [0, 0, 0], dtype=float
                    ),  # Sphere rotation less important
                    "scale": np.array([0.8, 0.8, 0.8], dtype=float),
                },
                "color": (60, 60, 220),  # Blueish color
            },
        ],
    }

    # Generate sphere vertices and triangles using icosphere
    radius = 1.0
    subdivisions = 2  # Each subdivision multiplies triangle count by 4

    # Generate initial icosahedron
    t = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio

    # Initial 12 vertices of icosahedron
    vertices = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ],
        dtype=float,
    )

    # Normalize vertices to create unit sphere
    vertices /= np.sqrt(1 + t * t)
    vertices *= radius

    # Initial 20 triangles of icosahedron - clockwise winding order
    triangles = [
        [0, 5, 11],
        [0, 1, 5],
        [0, 7, 1],
        [0, 10, 7],
        [0, 11, 10],
        [1, 9, 5],
        [5, 4, 11],
        [11, 2, 10],
        [10, 6, 7],
        [7, 8, 1],
        [3, 4, 9],
        [3, 2, 4],
        [3, 6, 2],
        [3, 8, 6],
        [3, 9, 8],
        [4, 5, 9],
        [2, 11, 4],
        [6, 10, 2],
        [8, 7, 6],
        [9, 1, 8],
    ]

    # Subdivision function
    vertex_cache = {}

    def get_middle_point(p1, p2, vertices, radius):
        # Generate vertex key
        key = tuple(sorted([p1, p2]))
        if key in vertex_cache:
            return vertex_cache[key]

        # Calculate middle point
        point1 = np.array(vertices[p1])
        point2 = np.array(vertices[p2])
        middle = (point1 + point2) / 2.0

        # Normalize to sphere surface
        length = np.sqrt(np.sum(middle**2))
        middle = middle / length * radius

        # Add vertex and return index
        vertices.append(middle.tolist())
        index = len(vertices) - 1
        vertex_cache[key] = index
        return index

    # Perform subdivision
    for _ in range(subdivisions):
        new_triangles = []
        vertices = vertices.tolist()  # Convert to list for easier appending

        for tri in triangles:
            v1, v2, v3 = tri
            # Get midpoints
            a = get_middle_point(v1, v2, vertices, radius)
            b = get_middle_point(v2, v3, vertices, radius)
            c = get_middle_point(v3, v1, vertices, radius)

            # Create 4 triangles with clockwise winding order
            new_triangles.extend([[v1, c, a], [v2, a, b], [v3, b, c], [a, c, b]])

        triangles = new_triangles
        vertices = np.array(vertices)

    # Convert final lists to numpy arrays
    scene["models"]["sphere"]["vertices"] = np.array(vertices, dtype=np.float32)
    scene["models"]["sphere"]["triangles"] = np.array(triangles, dtype=np.int32)
    scene["models"]["sphere"]["vertex_normals"] = (
        -vertices / radius
    )  # For unit sphere, normalized vertices = normals

    # Convert final lists to numpy arrays
    scene["models"]["sphere_flat"]["vertices"] = np.array(vertices, dtype=np.float32)
    scene["models"]["sphere_flat"]["triangles"] = np.array(triangles, dtype=np.int32)

    # Create renderer
    renderer = Renderer(canvas, viewport)

    # Display the result with the scene and renderer
    display(canvas, Scene.from_dict(scene), renderer)


if __name__ == "__main__":
    pygame.init()
    main()
    pygame.quit()
