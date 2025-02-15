"""
JIT-compiled kernel functions for the Pixerise rasterizer.
These functions are optimized using Numba's JIT compilation for better performance.
"""

import numpy as np
from numba import njit

from .rasterizing_mod import draw_triangle, draw_flat_triangle, draw_shaded_triangle
from .shading_mod import triangle_flat_shading, triangle_gouraud_shading
from .transforming_mod import project_vertex
from .clipping_mod import clip_triangle, clip_triangle_and_normals


@njit(cache=True)
def process_and_draw_triangles(
    triangles_array: np.ndarray,
    transformed_vertices: np.ndarray,
    triangle_normals: np.ndarray,
    transformed_normals: np.ndarray,
    shading_mode: str,
    has_vertex_normals: bool,
    fully_visible: bool,
    frustum_planes: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    viewport_width: float,
    viewport_height: float,
    canvas_buffer: np.ndarray,
    depth_buffer: np.ndarray,
    center_x: int,
    center_y: int,
    color: np.ndarray,
    light_dir: np.ndarray,
    ambient: float,
) -> None:
    """
    JIT-compiled function to process and render a batch of triangles.
    This is the main rendering kernel that handles triangle clipping, shading, and rasterization.

    Args:
        triangles_array: Array of shape (N, 3) containing vertex indices for N triangles
        transformed_vertices: Array of shape (M, 3) containing view-space vertex positions
        triangle_normals: Array of shape (N, 3) containing face normals for flat shading
        transformed_normals: Array of shape (M, 3) containing view-space vertex normals
        shading_mode: String indicating shading mode ('wireframe', 'flat', or 'gouraud')
        has_vertex_normals: Whether vertex normals are available for Gouraud shading
        fully_visible: Whether triangles are fully visible or need view frustum clipping
        frustum_planes: Array of shape (N, 4) containing plane equations (normal + distance)
        canvas_width, canvas_height: Canvas dimensions in pixels
        viewport_width, viewport_height: Viewport dimensions in world units
        canvas_buffer: RGB color buffer of shape (width, height, 3)
        depth_buffer: Z-buffer of shape (width, height) for depth testing
        center_x, center_y: Canvas center coordinates for screen-space mapping
        color: Base RGB color array of shape (3,) for the model
        light_dir: Normalized directional light vector of shape (3,)
        ambient: Ambient light intensity in range [0, 1]
    """
    for triangle_idx in range(len(triangles_array)):
        # Extract current triangle vertices from index array
        triangle_vertices = triangles_array[triangle_idx]

        # Construct triangle vertices array from transformed positions
        vertices = np.zeros((3, 3), dtype=np.float32)
        vertices[0] = transformed_vertices[triangle_vertices[0]]
        vertices[1] = transformed_vertices[triangle_vertices[1]]
        vertices[2] = transformed_vertices[triangle_vertices[2]]

        # Setup vertex normals based on shading mode
        vertex_normals = np.zeros((3, 3), dtype=np.float32)
        if shading_mode == "gouraud" and has_vertex_normals:
            # For Gouraud shading, use per-vertex normals
            vertex_normals[0] = transformed_normals[triangle_vertices[0]]
            vertex_normals[1] = transformed_normals[triangle_vertices[1]]
            vertex_normals[2] = transformed_normals[triangle_vertices[2]]
        else:
            # For flat shading, use the same face normal for all vertices
            vertex_normals[0] = triangle_normals[triangle_idx]
            vertex_normals[1] = triangle_normals[triangle_idx]
            vertex_normals[2] = triangle_normals[triangle_idx]

        if not fully_visible:
            # Perform view frustum clipping for triangles that may intersect frustum planes
            max_no_clipped_triangles = 8  # Maximum number of triangles after clipping

            # Initialize clipping buffers with the original triangle
            current_triangles = np.zeros(
                (max_no_clipped_triangles, 3, 3), dtype=np.float32
            )
            current_triangles[0] = vertices

            if shading_mode == "gouraud" and has_vertex_normals:
                # Additional buffer for vertex normals when using Gouraud shading
                current_normals = np.zeros(
                    (max_no_clipped_triangles, 3, 3), dtype=np.float32
                )
                current_normals[0] = vertex_normals

                is_clipped_away = False
                num_triangles = 1
                # Clip against each frustum plane sequentially
                for i in range(len(frustum_planes)):
                    plane_normal = frustum_planes[i, :3]
                    plane_dist = frustum_planes[i, 3]

                    # Temporary buffers for the next iteration
                    next_triangles = np.zeros(
                        (max_no_clipped_triangles, 3, 3), dtype=np.float32
                    )
                    next_normals = np.zeros(
                        (max_no_clipped_triangles, 3, 3), dtype=np.float32
                    )
                    next_num_triangles = 0

                    # Process each triangle from the previous clipping iteration
                    for j in range(num_triangles):
                        # Clip triangle and its normals against current plane
                        result_triangles, result_normals, num_clipped = (
                            clip_triangle_and_normals(
                                current_triangles[j],
                                current_normals[j],
                                plane_normal,
                                plane_dist,
                            )
                        )

                        # Store resulting triangles if buffer space available
                        for k in range(num_clipped):
                            if next_num_triangles < max_no_clipped_triangles:
                                next_triangles[next_num_triangles] = result_triangles[k]
                                next_normals[next_num_triangles] = result_normals[k]
                                next_num_triangles += 1

                    if next_num_triangles == 0:  # Triangle completely outside frustum
                        is_clipped_away = True
                        break

                    # Update buffers for next plane
                    num_triangles = next_num_triangles
                    for j in range(num_triangles):
                        current_triangles[j] = next_triangles[j]
                        current_normals[j] = next_normals[j]

                if is_clipped_away:  # Skip completely clipped triangles
                    continue

                # Project and draw all resulting triangles after clipping
                for i in range(num_triangles):
                    project_and_draw_triangle(
                        current_triangles[i],
                        current_normals[i],
                        shading_mode,
                        canvas_width,
                        canvas_height,
                        viewport_width,
                        viewport_height,
                        canvas_buffer,
                        depth_buffer,
                        center_x,
                        center_y,
                        color,
                        light_dir,
                        ambient,
                        has_vertex_normals,
                    )
            else:
                # Simpler clipping path when only flat shading is needed
                is_clipped_away = False
                num_triangles = 1
                for i in range(len(frustum_planes)):
                    plane_normal = frustum_planes[i, :3]
                    plane_dist = frustum_planes[i, 3]

                    next_triangles = np.zeros(
                        (max_no_clipped_triangles, 3, 3), dtype=np.float32
                    )
                    next_num_triangles = 0

                    for j in range(num_triangles):
                        # Clip triangle geometry only (no normals)
                        result_triangles, num_clipped = clip_triangle(
                            current_triangles[j], plane_normal, plane_dist
                        )

                        for k in range(num_clipped):
                            if next_num_triangles < max_no_clipped_triangles:
                                next_triangles[next_num_triangles] = result_triangles[k]
                                next_num_triangles += 1

                    if next_num_triangles == 0:
                        is_clipped_away = True
                        break

                    num_triangles = next_num_triangles
                    for j in range(num_triangles):
                        current_triangles[j] = next_triangles[j]

                if is_clipped_away:
                    continue

                # Project and draw clipped triangles with flat shading
                for i in range(num_triangles):
                    project_and_draw_triangle(
                        current_triangles[i],
                        vertex_normals,
                        shading_mode,
                        canvas_width,
                        canvas_height,
                        viewport_width,
                        viewport_height,
                        canvas_buffer,
                        depth_buffer,
                        center_x,
                        center_y,
                        color,
                        light_dir,
                        ambient,
                        has_vertex_normals,
                    )
        else:
            # Fast path for fully visible triangles (no clipping needed)
            project_and_draw_triangle(
                vertices,
                vertex_normals,
                shading_mode,
                canvas_width,
                canvas_height,
                viewport_width,
                viewport_height,
                canvas_buffer,
                depth_buffer,
                center_x,
                center_y,
                color,
                light_dir,
                ambient,
                has_vertex_normals,
            )


@njit(cache=True)
def project_and_draw_triangle(
    vertices: np.ndarray,
    vertex_normals: np.ndarray,
    shading_mode: str,
    canvas_width: int,
    canvas_height: int,
    viewport_width: float,
    viewport_height: float,
    canvas_buffer: np.ndarray,
    depth_buffer: np.ndarray,
    center_x: int,
    center_y: int,
    color: np.ndarray,
    light_dir: np.ndarray,
    ambient: float,
    has_vertex_normals: bool,
) -> None:
    """
    JIT-compiled function to project and draw a triangle with various shading modes.
    This is a low-level kernel function that handles vertex projection and triangle rasterization.

    Args:
        vertices: Array of shape (3, 3) containing view-space triangle vertices
        vertex_normals: Array of shape (3, 3) containing vertex or face normals
        shading_mode: String indicating shading mode ('wireframe', 'flat', or 'gouraud')
        canvas_width, canvas_height: Canvas dimensions in pixels
        viewport_width, viewport_height: Viewport dimensions in world units
        canvas_buffer: RGB color buffer of shape (width, height, 3)
        depth_buffer: Z-buffer of shape (width, height) for depth testing
        center_x, center_y: Canvas center coordinates for screen-space mapping
        color: Base RGB color array of shape (3,) for the triangle
        light_dir: Normalized directional light vector of shape (3,)
        ambient: Ambient light intensity in range [0, 1]
        has_vertex_normals: Whether vertex normals are available for Gouraud shading
    """
    # Project vertices from view space to screen space
    v1 = project_vertex(
        vertices[0], canvas_width, canvas_height, viewport_width, viewport_height
    )
    v2 = project_vertex(
        vertices[1], canvas_width, canvas_height, viewport_width, viewport_height
    )
    v3 = project_vertex(
        vertices[2], canvas_width, canvas_height, viewport_width, viewport_height
    )

    # Skip degenerate triangles (any vertex behind camera)
    if (
        v1[0] == 0.0
        and v1[1] == 0.0
        and v1[2] == 0.0
        or v2[0] == 0.0
        and v2[1] == 0.0
        and v2[2] == 0.0
        or v3[0] == 0.0
        and v3[1] == 0.0
        and v3[2] == 0.0
    ):
        return

    # Convert floating-point coordinates to integer screen coordinates
    x1, y1, z1 = int(v1[0]), int(v1[1]), v1[2]
    x2, y2, z2 = int(v2[0]), int(v2[1]), v2[2]
    x3, y3, z3 = int(v3[0]), int(v3[1]), v3[2]

    # Select appropriate drawing mode based on shading type
    if shading_mode == "wireframe":
        # Draw triangle edges only
        draw_triangle(
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
            x3,
            y3,
            z3,
            canvas_buffer,
            depth_buffer,
            center_x,
            center_y,
            color[0],
            color[1],
            color[2],
            canvas_width,
            canvas_height,
        )
    else:
        if shading_mode == "gouraud" and has_vertex_normals:
            # Compute per-vertex lighting for smooth shading
            intensities = triangle_gouraud_shading(vertex_normals, light_dir, ambient)
            draw_shaded_triangle(
                x1,
                y1,
                z1,
                x2,
                y2,
                z2,
                x3,
                y3,
                z3,
                canvas_buffer,
                depth_buffer,
                center_x,
                center_y,
                color[0],
                color[1],
                color[2],
                intensities[0],
                intensities[1],
                intensities[2],
                canvas_width,
                canvas_height,
            )
        else:  # Flat shading (single color per triangle)
            flat_shaded_color = triangle_flat_shading(
                vertex_normals[0], light_dir, color, ambient
            )
            draw_flat_triangle(
                x1,
                y1,
                z1,
                x2,
                y2,
                z2,
                x3,
                y3,
                z3,
                canvas_buffer,
                depth_buffer,
                center_x,
                center_y,
                int(flat_shaded_color[0]),
                int(flat_shaded_color[1]),
                int(flat_shaded_color[2]),
                canvas_width,
                canvas_height,
            )
