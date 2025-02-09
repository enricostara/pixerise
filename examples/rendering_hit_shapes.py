import pygame
import numpy as np

from pixerise import Canvas, ViewPort, Renderer, ShadingMode
from scene import Scene, Model, Instance


def create_cube_model() -> Model:
    """Create a simple cube model with unit size."""
    vertices = np.array([
        [-0.5, -0.5, -0.5],  # 0
        [0.5, -0.5, -0.5],   # 1
        [0.5, 0.5, -0.5],    # 2
        [-0.5, 0.5, -0.5],   # 3
        [-0.5, -0.5, 0.5],   # 4
        [0.5, -0.5, 0.5],    # 5
        [0.5, 0.5, 0.5],     # 6
        [-0.5, 0.5, 0.5],    # 7
    ], dtype=np.float32)

    triangles = np.array([
        # Front face
        [0, 1, 2], [0, 2, 3],
        # Back face
        [5, 4, 7], [5, 7, 6],
        # Left face
        [4, 0, 3], [4, 3, 7],
        # Right face
        [1, 5, 6], [1, 6, 2],
        # Top face
        [3, 2, 6], [3, 6, 7],
        # Bottom face
        [4, 5, 1], [4, 1, 0]
    ], dtype=np.int32)

    model = Model()
    model.add_group("cube", vertices, triangles)
    return model


def display(canvas: Canvas, scene: Scene, renderer: Renderer):
    screen = pygame.display.set_mode(canvas.size, pygame.SCALED)
    clock = pygame.time.Clock()
    
    # Initialize mouse state
    pygame.mouse.set_visible(True)  # Show cursor for clicking
    pygame.event.set_grab(False)    # Don't grab mouse for free movement
    
    def update_display():
        renderer.render(scene)
        surf = pygame.surfarray.make_surface(canvas.color_buffer)
        screen.blit(surf, (0, 0))
        pygame.display.update()
    
    # Initial render
    update_display()
    
    # Movement speed
    rotation_speed = 0.00
    
    # Track selected cube
    selected_cube = None
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return
            
            # Handle mouse clicks
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
                # Get hit cube
                mouse_pos = pygame.mouse.get_pos()
                print(mouse_pos)
                hit = renderer.cast_ray(mouse_pos[0], mouse_pos[1], scene)
                
                # Reset previous selection
                if selected_cube:
                    scene.instances[selected_cube].set_color(255, 255, 255) 
                
                # Update new selection
                if hit:
                    instance_name, _ = hit
                    print(f"Selected cube: {instance_name}")
                    scene.instances[instance_name].set_color(255, 0, 0)
                    selected_cube = instance_name
                else:
                    selected_cube = None
        
        # Rotate all cubes
        for instance in scene.instances.values():
            instance.rotation[1] += rotation_speed
        
        # Update display
        update_display()
        clock.tick(60)


def main():
    # Create canvas and viewport
    canvas = Canvas((800, 600))
    viewport = ViewPort((3.0, 2.25), 1, canvas)  # Adjusted viewport size for better perspective
    renderer = Renderer(canvas, viewport)
    
    # Create scene
    scene = Scene()
    
    # Create cube model
    cube_model = create_cube_model()
    scene.add_model("cube", cube_model)
    
    # Create grid of cubes
    grid_size = 3
    spacing = 1.5  # Reduced spacing between cubes
    
    for i in range(grid_size):
        for j in range(grid_size):
            x = (i - grid_size/2 + 0.5) * spacing
            z = (j - grid_size/2 + 0.5) * spacing
            
            instance = Instance(model="cube")
            instance.set_translation(x, 0, z)
            instance.set_scale(1, 1, 1)
            # instance.set_scale(0.8, 0.8, 0.8)
            instance.set_color(255, 255, 255)
            
            scene.add_instance(f"cube_{i}_{j}", instance)
    
    # Set up camera
    scene.camera.translation = np.array([0, 1, -6], dtype=np.float32)  # Adjusted camera position
    scene.camera.rotation = np.array([0.15, 0, 0], dtype=np.float32)  # Slight downward tilt
    
    # Start display
    display(canvas, scene, renderer)


if __name__ == "__main__":
    pygame.init()
    main()
    pygame.quit()
