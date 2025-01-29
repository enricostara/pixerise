"""
JIT-compiled kernel functions for the Pixerise rasterizer.
These functions are optimized using Numba's JIT compilation for better performance.
"""

import numpy as np
from numba import njit

from .rasterizing_mod import (draw_triangle, draw_flat_triangle, draw_shaded_triangle)
from .shading_mod import triangle_flat_shading, triangle_gouraud_shading
from .transforming_mod import project_vertex
from .clipping_mod import clip_triangle, clip_triangle_and_normals


@njit(cache=True)
def process_and_render_triangle(triangle_vertices: np.ndarray,
                              transformed_vertices: np.ndarray,
                              triangle_normals: np.ndarray,
                              transformed_normals: np.ndarray,
                              triangle_idx: int,
                              shading_mode: str,
                              has_vertex_normals: bool,
                              fully_visible: bool,
                              frustum_planes: np.ndarray,
                              canvas_width: int, canvas_height: int,
                              viewport_width: float, viewport_height: float,
                              canvas_buffer: np.ndarray, depth_buffer: np.ndarray,
                              center_x: int, center_y: int,
                              color: np.ndarray,
                              light_dir: np.ndarray,
                              ambient: float) -> None:
    """
    JIT-compiled function to process and render a triangle, handling clipping and shading.
    
    Args:
        triangle_vertices: Array of shape (3,) containing vertex indices
        transformed_vertices: Array of all transformed vertices
        triangle_normals: Array of face normals
        transformed_normals: Array of all transformed vertex normals
        triangle_idx: Index of the current triangle
        shading_mode: String indicating shading mode ('wireframe', 'flat', or 'gouraud')
        has_vertex_normals: Whether vertex normals are available
        fully_visible: Whether the triangle is fully visible or needs clipping
        frustum_planes: Array of shape (N, 4) containing plane equations (normal + distance)
        canvas_width, canvas_height: Canvas dimensions
        viewport_width, viewport_height: Viewport dimensions
        canvas_buffer: RGB color buffer
        depth_buffer: Depth buffer
        center_x, center_y: Canvas center coordinates
        color: RGB color array
        light_dir: Directional light vector
        ambient: Ambient light intensity
    """
    # Get triangle vertices as numpy array
    vertices = np.zeros((3, 3), dtype=np.float32)
    vertices[0] = transformed_vertices[triangle_vertices[0]]
    vertices[1] = transformed_vertices[triangle_vertices[1]]
    vertices[2] = transformed_vertices[triangle_vertices[2]]

    # Get corresponding transformed normals for this triangle
    if shading_mode == 'gouraud' and has_vertex_normals:
        vertex_normals = np.zeros((3, 3), dtype=np.float32)
        vertex_normals[0] = transformed_normals[triangle_vertices[0]]
        vertex_normals[1] = transformed_normals[triangle_vertices[1]]
        vertex_normals[2] = transformed_normals[triangle_vertices[2]]
    else:
        # If no vertex normals, use face normal for all vertices
        vertex_normals = np.zeros((3, 3), dtype=np.float32)
        vertex_normals[0] = triangle_normals[triangle_idx]
        vertex_normals[1] = triangle_normals[triangle_idx]
        vertex_normals[2] = triangle_normals[triangle_idx]

    if not fully_visible:
        # Process each frustum plane
        current_triangles = np.zeros((4, 3, 3), dtype=np.float32)  # Increased to 4 triangles
        current_triangles[0] = vertices
        num_triangles = 1
        
        if shading_mode == 'gouraud' and has_vertex_normals:
            current_normals = np.zeros((4, 3, 3), dtype=np.float32)  # Increased to 4 triangles
            current_normals[0] = vertex_normals
            
            # Clip against each frustum plane
            for i in range(len(frustum_planes)):
                plane_normal = frustum_planes[i, :3]
                plane_dist = frustum_planes[i, 3]
                
                # Process each triangle from previous iteration
                next_triangles = np.zeros((4, 3, 3), dtype=np.float32)  # Max 4 triangles after split
                next_normals = np.zeros((4, 3, 3), dtype=np.float32)
                next_num_triangles = 0
                
                for j in range(num_triangles):
                    # Clip triangle against current plane
                    result_triangles, result_normals, num_clipped = clip_triangle_and_normals(
                        current_triangles[j], current_normals[j], plane_normal, plane_dist
                    )
                    
                    # Add resulting triangles if any space left
                    for k in range(num_clipped):
                        if next_num_triangles < 4:
                            next_triangles[next_num_triangles] = result_triangles[k]
                            next_normals[next_num_triangles] = result_normals[k]
                            next_num_triangles += 1
                
                if next_num_triangles == 0:  # Triangle completely clipped away
                    return
                
                # Update for next iteration
                num_triangles = next_num_triangles
                for j in range(num_triangles):  # Copy triangles one by one
                    current_triangles[j] = next_triangles[j]
                    current_normals[j] = next_normals[j]

            # Project and draw the clipped triangles
            for i in range(num_triangles):
                project_and_draw_triangle(
                    current_triangles[i], current_normals[i], shading_mode,
                    canvas_width, canvas_height,
                    viewport_width, viewport_height,
                    canvas_buffer, depth_buffer,
                    center_x, center_y,
                    color,
                    light_dir,
                    ambient,
                    has_vertex_normals)
        else:
            # Clip against each frustum plane
            for i in range(len(frustum_planes)):
                plane_normal = frustum_planes[i, :3]
                plane_dist = frustum_planes[i, 3]
                
                # Process each triangle from previous iteration
                next_triangles = np.zeros((4, 3, 3), dtype=np.float32)  # Max 4 triangles after split
                next_num_triangles = 0
                
                for j in range(num_triangles):
                    # Clip triangle against current plane
                    result_triangles, num_clipped = clip_triangle(
                        current_triangles[j], plane_normal, plane_dist
                    )
                    
                    # Add resulting triangles if any space left
                    for k in range(num_clipped):
                        if next_num_triangles < 4:
                            next_triangles[next_num_triangles] = result_triangles[k]
                            next_num_triangles += 1
                
                if next_num_triangles == 0:  # Triangle completely clipped away
                    return
                
                # Update for next iteration
                num_triangles = next_num_triangles
                for j in range(num_triangles):  # Copy triangles one by one
                    current_triangles[j] = next_triangles[j]

            # Project and draw the clipped triangles
            for i in range(num_triangles):
                project_and_draw_triangle(
                    current_triangles[i], vertex_normals, shading_mode,
                    canvas_width, canvas_height,
                    viewport_width, viewport_height,
                    canvas_buffer, depth_buffer,
                    center_x, center_y,
                    color,
                    light_dir,
                    ambient,
                    has_vertex_normals)
    else:
        # For fully visible instances, direct kernel call
        project_and_draw_triangle(
            vertices, vertex_normals, shading_mode,
            canvas_width, canvas_height,
            viewport_width, viewport_height,
            canvas_buffer, depth_buffer,
            center_x, center_y,
            color,
            light_dir,
            ambient,
            has_vertex_normals)


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


@njit(cache=True)
def process_triangles_batch(triangles_array: np.ndarray,
                          transformed_vertices: np.ndarray,
                          triangle_normals: np.ndarray,
                          transformed_normals: np.ndarray,
                          shading_mode: str,
                          has_vertex_normals: bool,
                          fully_visible: bool,
                          frustum_planes: np.ndarray,
                          canvas_width: int, canvas_height: int,
                          viewport_width: float, viewport_height: float,
                          canvas_buffer: np.ndarray, depth_buffer: np.ndarray,
                          center_x: int, center_y: int,
                          color: np.ndarray,
                          light_dir: np.ndarray,
                          ambient: float) -> None:
    """
    JIT-compiled function to process and render a batch of triangles.
    
    Args:
        triangles_array: Array of shape (N, 3) containing vertex indices for N triangles
        transformed_vertices: Array of all transformed vertices
        triangle_normals: Array of face normals
        transformed_normals: Array of all transformed vertex normals
        shading_mode: String indicating shading mode ('wireframe', 'flat', or 'gouraud')
        has_vertex_normals: Whether vertex normals are available
        fully_visible: Whether triangles are fully visible or need clipping
        frustum_planes: Array of shape (N, 4) containing plane equations (normal + distance)
        canvas_width, canvas_height: Canvas dimensions
        viewport_width, viewport_height: Viewport dimensions
        canvas_buffer: RGB color buffer
        depth_buffer: Depth buffer
        center_x, center_y: Canvas center coordinates
        color: RGB color array
        light_dir: Directional light vector
        ambient: Ambient light intensity
    """
    for i in range(len(triangles_array)):
        process_and_render_triangle(
            triangles_array[i],
            transformed_vertices,
            triangle_normals,
            transformed_normals,
            i,
            shading_mode,
            has_vertex_normals,
            fully_visible,
            frustum_planes,
            canvas_width, canvas_height,
            viewport_width, viewport_height,
            canvas_buffer, depth_buffer,
            center_x, center_y,
            color,
            light_dir,
            ambient
        )
