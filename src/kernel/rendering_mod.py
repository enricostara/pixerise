"""
JIT-compiled kernel functions for the Pixerise rasterizer.
These functions are optimized using Numba's JIT compilation for better performance.
"""

import numpy as np
from numba import njit
from typing import Tuple

from .rasterizing_mod import (draw_triangle, draw_flat_triangle, draw_shaded_triangle)
from .shading_mod import triangle_flat_shading, triangle_gouraud_shading
from .transforming_mod import project_vertex


@njit(cache=True)
def project_and_draw_triangle(vertices: np.ndarray,
                                   vertex_normals: np.ndarray,
                                   shading_mode: str,
                                   canvas_width: int, canvas_height: int,
                                   viewport_width: float, viewport_height: float,
                                   canvas_buffer: np.ndarray, depth_buffer: np.ndarray,
                                   center_x: int, center_y: int,
                                   color: np.ndarray,
                                   light_dir: np.ndarray,
                                   ambient: float,
                                   has_vertex_normals: bool) -> None:
    """
    JIT-compiled function to project and draw a triangle with various shading modes.
    This is a low-level kernel function that handles vertex projection and triangle rendering.
    
    Args:
        vertices: Array of shape (3, 3) containing triangle vertices
        vertex_normals: Array of shape (3, 3) containing vertex normals
        shading_mode: String indicating shading mode ('wireframe', 'flat', or 'gouraud')
        canvas_width, canvas_height: Canvas dimensions
        viewport_width, viewport_height: Viewport dimensions
        canvas_buffer: RGB color buffer of shape (width, height, 3)
        depth_buffer: Depth buffer of shape (width, height)
        center_x, center_y: Canvas center coordinates
        color: RGB color array of shape (3,)
        light_dir: Directional light vector of shape (3,)
        ambient: Ambient light intensity
        has_vertex_normals: Whether vertex normals are available
    """
    # Project vertices to 2D
    v1 = project_vertex(vertices[0], canvas_width, canvas_height, viewport_width, viewport_height)
    v2 = project_vertex(vertices[1], canvas_width, canvas_height, viewport_width, viewport_height)
    v3 = project_vertex(vertices[2], canvas_width, canvas_height, viewport_width, viewport_height)
    
    # Skip if any vertex is behind camera
    if v1[0] == 0.0 and v1[1] == 0.0 and v1[2] == 0.0 or v2[0] == 0.0 and v2[1] == 0.0 and v2[2] == 0.0 or v3[0] == 0.0 and v3[1] == 0.0 and v3[2] == 0.0:
        return
    
    # Convert vertices to screen space coordinates
    x1, y1, z1 = int(v1[0]), int(v1[1]), v1[2]
    x2, y2, z2 = int(v2[0]), int(v2[1]), v2[2]
    x3, y3, z3 = int(v3[0]), int(v3[1]), v3[2]
    
    # Draw triangle based on shading mode
    if shading_mode == 'wireframe':
        draw_triangle(x1, y1, z1, x2, y2, z2, x3, y3, z3,
                     canvas_buffer, depth_buffer,
                     center_x, center_y,
                     color[0], color[1], color[2],
                     canvas_width, canvas_height)
    else:
        if shading_mode == 'gouraud' and has_vertex_normals:
            # Compute vertex intensities for Gouraud shading
            intensities = triangle_gouraud_shading(vertex_normals, light_dir, ambient)
            draw_shaded_triangle(x1, y1, z1, x2, y2, z2, x3, y3, z3,
                               canvas_buffer, depth_buffer,
                               center_x, center_y,
                               color[0], color[1], color[2],
                               intensities[0], intensities[1], intensities[2],
                               canvas_width, canvas_height)
        else:  # FLAT shading
            flat_shaded_color = triangle_flat_shading(vertex_normals[0], light_dir, color, ambient)
            draw_flat_triangle(x1, y1, z1, x2, y2, z2, x3, y3, z3,
                             canvas_buffer, depth_buffer,
                             center_x, center_y,
                             int(flat_shaded_color[0]), int(flat_shaded_color[1]), int(flat_shaded_color[2]),
                             canvas_width, canvas_height)
