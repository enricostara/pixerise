# Pixerise

A high-performance 3D software renderer implemented in Python, optimized with NumPy and Numba JIT compilation.

![Tank Model Rendering Demo](media/tank.gif)

## Overview

Pixerise is a Python 3D rendering engine that focuses on CPU-based rendering, making it ideal for educational purposes, embedded systems, and applications where GPU acceleration is not available or desired.

## Features

### Core Rendering
- Multiple shading modes (Wireframe, Flat, Gouraud)
- View frustum culling with bounding spheres
- Backface culling for performance optimization
- Directional lighting with ambient and diffuse components
- Efficient batch processing of vertices and normals
- Ray casting for precise 3D object selection

### Performance
- NumPy-accelerated array operations for fast geometry processing
- JIT-compiled core rendering functions using Numba
- Optimized batch transformations of vertices and normals
- Efficient memory layout with contiguous arrays
- Early culling of invisible geometry

### Integration
- Agnostic rendering buffer system compatible with any display library
- No direct dependencies on specific media or rendering libraries
- Clean separation between rendering and display logic
- Example integrations with popular libraries (e.g., Pygame)

### Scene Management
- Complete scene graph system
- Support for model instancing
- Hierarchical transformations
- Flexible camera controls
- Material and lighting properties

## Installation

### 1. Install PDM (Python Dependency Manager)

PDM is required to manage project dependencies. Here's how to install it on different platforms:

#### Windows
```bash
powershell -ExecutionPolicy ByPass -c "irm https://pdm-project.org/install-pdm.py | py -"
```

#### Linux/Mac
```bash
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
```

#### All (using pip)
```bash
pip install pdm
```

### 2. Install Pixerise
Clone the repository and install dependencies:
```bash
git clone https://github.com/enricostara/pixerise.git
cd pixerise
pdm install
```

## Quick Start

```python
import pygame
from pixerise import Canvas, ViewPort, Renderer
from scene import Scene

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Pixerise Quick Start")

# Initialize rendering components
canvas = Canvas((800, 600))
viewport = ViewPort((1.6, 1.2), 1, canvas)
renderer = Renderer(canvas, viewport)

# Define scene structure
scene_dict = {
    'camera': {
            'transform': {
                'translation': [0, 0, -3],  
                'rotation': [0, 0, 0] 
            }
        },
    "models": {
        "triangle": {
            "vertices": [
                [0, 1, 0],               # top vertex
                [-0.866, -0.5, 0],       # bottom left vertex
                [0.866, -0.5, 0]         # bottom right vertex
            ],
            "triangles": [[0,1,2]]
        }
    },
    "instances": [
        {
            "model": "triangle",
            "name": "a_triangle",
            "color": [0, 255, 0],
            'transform': {
                'translation': [0, 0, 0],
                'rotation': [0, 0, 0],
                'scale': [1, 1, 1]
            }
        }
    ]
}

# Create scene from dictionary
scene = Scene.from_dict(scene_dict)

# Main loop
running = True
clock = pygame.time.Clock()
while running:
    clock.tick(60)  # Limit to 60 FPS
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
    
    # Update triangle rotation
    scene.get_instance("a_triangle").rotation -= [0, 0, .01]
    
    # Render the scene
    renderer.render(scene)
    
    # Display the rendered image
    surf = pygame.surfarray.make_surface(canvas.color_buffer)
    screen.blit(surf, (0, 0))
    pygame.display.update()

pygame.quit()

```

For detailed API documentation, see [src/README.md](src/README.md).

## Examples

The `examples` directory contains several demonstrations:

![Ray Casting Selection Demo](media/ray-casting.gif)

- `rendering_wireframe.py`: Basic wireframe rendering with interactive camera
- `rendering_flat_shading.py`: Flat shading with directional lighting
- `rendering_gouraud_shading.py`: Smooth shading using vertex normals
- `rendering_obj_file.py`: Loading and rendering 3D models from an OBJ file with interactive controls
  - Left click: Select and highlight objects using ray casting
  - Right click: Toggle group visibility

Run the tank example using:
```bash
pdm run python examples/rendering_obj_file.py
```

Each example demonstrates different features of the engine and includes interactive controls:
- WASD: Move camera position
- Mouse: Look around
- Mouse wheel: Move forward/backward
- Q/E: Move up/down
- Space: Toggle between shading modes (where available)
- Backspace: Toggle mouse grab mode (to click and highlight objects using ray casting)
- Esc: Quit the example

## Who Should Use Pixerise?

### Ideal For:
- Educational projects learning 3D graphics fundamentals
- Embedded systems without GPU access
- Cross-platform applications requiring consistent rendering
- Custom 3D visualization tools
- Projects requiring full control over the rendering pipeline

### Not Recommended For:
- Applications requiring real-time GPU acceleration
- Complex 3D applications needing advanced graphics features

## Architecture

```mermaid
graph TD
    subgraph Scene Management
        Scene[Scene Container]
        Model[Model<br/>Reusable Geometry]
        Instance[Instance<br/>Model Occurrences]
        Camera[Camera<br/>Viewpoint]
        Light[DirectionalLight<br/>Scene Lighting]
        
        Scene --> Model
        Scene --> Instance
        Scene --> Camera
        Scene --> Light
        Model --> Instance
    end

    subgraph Core Components
        Canvas[Canvas<br/>2D Drawing Surface]
        ViewPort[ViewPort<br/>View Frustum]
        Renderer[Renderer<br/>Main Pipeline]
        
        Canvas --> Renderer
        ViewPort --> Renderer
    end

    subgraph Renderer Pipeline
        Shading[Shading Modes]
        Culling[Face Culling]
        Lighting[Lighting System]
        Transform[Coordinate<br/>Transforms]
        
        Renderer --> Shading
        Renderer --> Culling
        Renderer --> Lighting
        Renderer --> Transform
    end

    subgraph Optimization
        NumPy[NumPy Arrays<br/>Memory Layout]
        Numba[Numba JIT<br/>@njit + cache]
        
        NumPy --> Renderer
        Numba --> Renderer
    end

    Scene --> Renderer
```

## Development

### Running Tests
```bash
pdm run pytest
```

### Contributing
We welcome contributions! Here's how you can help:

1. Open an issue first to discuss your proposed changes
2. Fork the repository
3. Create your feature branch (`git checkout -b feature/amazing-feature`)
4. Commit your changes (`git commit -m 'feat: add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

This way we can ensure your contribution aligns with the project's goals and avoid duplicate efforts.

## License

[MIT License](LICENSE)

## Acknowledgments

Special thanks to:
- [Gabriel Gambetta](https://github.com/ggambetta) and his amazing book [Computer Graphics from Scratch](https://gabrielgambetta.com/computer-graphics-from-scratch), which inspired many of the rendering techniques used in this project
- [Windsurf](https://codeium.com/windsurf), the excellent agentic IDE that made this project feasible in a few months by working after dinner
- The NumPy and Numba teams for their awesome libraries
