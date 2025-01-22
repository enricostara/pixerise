import numpy as np
from numba import njit


@njit(cache=True)
def cull_back_faces(vertices: np.ndarray, triangle_indices: np.ndarray):
    """Cull back-facing triangles based on view direction.
    
    Args:
        vertices: Array of vertices in camera space coordinates
        triangle_indices: Array of triangle indices into vertices array
        
    Returns:
        Tuple containing:
        - Array of boolean flags indicating which triangles are visible
        - Array of triangle normals for visible triangles
        
    Note:
        The input vertices are assumed to be in camera space, where the camera
        is at the origin looking down the positive z-axis. A triangle is considered
        back-facing if its normal points away from the origin (camera position).
    """
    num_triangles = len(triangle_indices)
    visible = np.zeros(num_triangles, dtype=np.bool_)
    normals = np.zeros((num_triangles, 3), dtype=np.float32)
    
    # Process each triangle
    for i in range(num_triangles):
        # Get triangle vertices
        v0 = vertices[triangle_indices[i, 0]]
        v1 = vertices[triangle_indices[i, 1]]
        v2 = vertices[triangle_indices[i, 2]]
        
        # Calculate triangle normal using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        
        # Normalize the normal vector
        length = np.sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2])
        if length > 0:
            normal = normal / length
        
        # Store normalized normal
        normals[i] = normal
        
        # Triangle is visible if it faces the camera (dot product with view vector is negative)
        # In camera space, the view vector from any vertex to camera is just -vertex
        view_vec = -v0  # Use any vertex, they're all in camera space
        visible[i] = np.dot(normal, view_vec) < 0
    
    return triangle_indices[visible], normals[visible]
