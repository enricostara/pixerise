import numpy as np
from numba import njit


@njit(cache=True)
def cull_back_faces(
    vertices: np.ndarray, triangle_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cull back-facing triangles based on view direction.

    This function implements back-face culling, a fundamental optimization in 3D graphics
    that removes triangles facing away from the camera. It works by:
    1. Computing each triangle's normal vector using cross product
    2. Checking if the normal points towards or away from camera

    Args:
        vertices: Array of vertices in camera space coordinates
                 Shape (N, 3) where each row is [x, y, z]:
                 - x: right direction
                 - y: up direction
                 - z: forward direction (positive = away from camera)

        triangle_indices: Array of triangle indices into vertices array
                        Shape (M, 3) where each row defines one triangle
                        Vertices must be in counter-clockwise order

    Returns:
        Tuple containing:
        - Array of triangle indices for visible triangles only
        - Array of triangle normals for visible triangles

    Note:
        The input vertices are assumed to be in camera space, where the camera
        is at the origin looking down the positive z-axis. A triangle is considered
        back-facing if its normal points away from the origin (camera position).

        The function uses numba's @njit decorator for performance and assumes a
        right-handed coordinate system with counter-clockwise triangle winding.
    """
    # Initialize arrays for visibility flags and computed normals
    num_triangles = len(triangle_indices)
    visible = np.zeros(num_triangles, dtype=np.bool_)
    normals = np.zeros((num_triangles, 3), dtype=np.float32)

    # Process each triangle
    for i in range(num_triangles):
        # Get triangle vertices: v0 = anchor point, v1/v2 define edges
        v0 = vertices[triangle_indices[i, 0]]
        v1 = vertices[triangle_indices[i, 1]]
        v2 = vertices[triangle_indices[i, 2]]

        # Calculate triangle normal using cross product of edges
        # Cross product direction follows right-hand rule
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)

        # Normalize the normal vector to unit length for consistent dot products
        length = np.sqrt(
            normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]
        )
        if length > 0:
            normal = normal / length

        # Store normalized normal for later use
        normals[i] = normal

        # Triangle is visible if it faces the camera (dot product with view vector is negative)
        # In camera space, the view vector from any vertex to camera is just -vertex
        # Using v0, but any vertex would work since they're all in camera space
        view_vec = -v0  # Points from vertex towards camera (origin)
        visible[i] = np.dot(normal, view_vec) < 0

    # Return only the visible triangles and their corresponding normals
    return triangle_indices[visible], normals[visible]
