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
        renderer.render(scene, shading_mode=ShadingMode.FLAT)
        surf = pygame.surfarray.make_surface(canvas.color_buffer)
        screen.blit(surf, (0, 0))
        pygame.display.update()

    # Initial render
    update_display()

    # Movement speed
    move_speed = 0.1
    rotation_speed = 0.01  # Speed of cube rotation
    wheel_speed = 0.1  # Speed for mouse wheel movement
    wheel_momentum = 0.95  # How much wheel velocity is retained (0-1)

    # Track if any movement occurred
    movement_occurred = False

    # Initialize rotation angles for cubes
    left_cube_rotation = 0.0
    right_cube_rotation = 0.0

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

        # Update cube rotations
        left_cube_rotation += rotation_speed
        right_cube_rotation += rotation_speed * 1.5  # Right cube rotates faster

        # Update cube instance rotations
        left_instance = scene.get_instance("left")
        right_instance = scene.get_instance("right")

        if left_instance and right_instance:
            # Left cube: rotate around Y axis
            left_instance.rotation = np.array([0, left_cube_rotation, 0], dtype=float)

            # Right cube: rotate around both Y and X axes
            right_instance.rotation = np.array(
                [right_cube_rotation, np.pi / 4 + right_cube_rotation, 0], dtype=float
            )

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
                "translation": np.array(
                    [0, 1, 1], dtype=float
                ),  # Moved higher (y from 1 to 2.5)
                "rotation": np.array(
                    [0, 0, 0], dtype=float
                ),  # More downward tilt (from pi/12 to pi/4)
            }
        },
        "lights": {
            "directional": {
                "direction": np.array(
                    [-1, 1, -1], dtype=float
                ),  # Light coming from top-left-front
                "intensity": 0.7,
                "ambient": 0.2,  # Add ambient light intensity
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
            }
        },
        "instances": [
            {
                "model": "cube",
                "name": "left",
                "transform": {
                    "translation": np.array(
                        [-2, 0, 7], dtype=float
                    ),  # 2 units left, 7 units forward
                    "rotation": np.array(
                        [0, 0, 0], dtype=float
                    ),  # rotation angles in radians (x, y, z)
                    "scale": np.array([1, 1, 1], dtype=float),  # uniform scale
                },
                "color": (255, 50, 50),  # Red color for the left cube
            },
            {
                "model": "cube",
                "name": "right",
                "transform": {
                    "translation": np.array(
                        [2, 1, 8], dtype=float
                    ),  # 2 units right, 1 up, 8 forward
                    "rotation": np.array(
                        [0, np.pi / 4, 0], dtype=float
                    ),  # rotated 45 degrees around Y axis
                    "scale": np.array([0.5, 0.5, 0.5], dtype=float),  # half the size
                },
                "color": (0, 255, 0),  # Green color for the right cube
            },
        ],
    }

    # Create renderer
    renderer = Renderer(canvas, viewport)

    # Display the result with the scene and renderer
    display(canvas, Scene.from_dict(scene), renderer)


if __name__ == "__main__":
    pygame.init()
    main()
    pygame.quit()
