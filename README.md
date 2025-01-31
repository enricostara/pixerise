# Pixerise

A high-performance 3D software renderer implemented in Python, optimized with NumPy and Numba JIT compilation.

## Overview

Pixerise is a pure Python 3D rendering engine that focuses on CPU-based rendering, making it ideal for educational purposes, embedded systems, and applications where GPU acceleration is not available or desired.

## Features

### Core Rendering
- Multiple shading modes (Wireframe, Flat, Gouraud)
- View frustum culling with bounding spheres
- Backface culling for performance optimization
- Directional lighting with ambient and diffuse components
- Efficient batch processing of vertices and normals

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

```bash
# Install using PDM (recommended)
pdm install
```

## Quick Start

```python
from pixerise import Canvas, ViewPort, Renderer, Scene, Model
import numpy as np

# Create a simple triangle scene
vertices = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=np.float32)
triangles = np.array([[0,1,2]], dtype=np.int32)

# Initialize rendering components
canvas = Canvas((800, 600))
viewport = ViewPort((1.6, 1.2), 1, canvas)
renderer = Renderer(canvas, viewport)

# Set up scene
scene = Scene()
model = Model()
model.add_group("default", vertices, triangles)
scene.add_model("triangle", model)

# Render the scene
renderer.render(scene)
```

## Examples

The `examples` directory contains several demonstrations:

- `rendering_wireframe.py`: Basic wireframe rendering with interactive camera
- `rendering_flat_shading.py`: Flat shading with directional lighting
- `rendering_gouraud_shading.py`: Smooth shading using vertex normals
- `rendering_obj_file.py`: Loading and rendering 3D models from an OBJ file with interactive controls

Run any example using:
```bash
pdm run python examples/rendering_obj_file.py
```

Each example demonstrates different features of the engine and includes interactive controls:
- WASD: Move camera position
- Mouse: Look around
- Mouse wheel: Move forward/backward
- Q/E: Move up/down
- Space: Toggle between shading modes (where available)
- Esc: Exit

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

## Development

### Running Tests
```bash
pdm run pytest
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

[MIT License](LICENSE)

## Acknowledgments

Special thanks to:
- [Gabriel Gambetta](https://github.com/ggambetta) and his excellent book [Computer Graphics from Scratch](https://gabrielgambetta.com/computer-graphics-from-scratch), which inspired many of the rendering techniques used in this project
- [Windsurf](https://github.com/codeium/windsurf), the amazing agentic IDE that made this project feasible
- The NumPy and Numba teams for their excellent libraries


## What's Next?

Future enhancements may include:
- [ ] More shading models
- [ ] Texturing support
- [ ] Scene serialization
- [ ] Additional example scenes