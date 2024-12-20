import numpy as np
from numba import njit


@njit(cache=True)
def cull_back_faces(vertices, indices, camera_pos):
    """Cull back-facing triangles based on camera position.
    
    Args:
        vertices: Nx3 array of vertex positions
        indices: Mx3 array of triangle indices into vertices
        camera_pos: 3D camera position
        
    Returns:
        ndarray: Boolean mask of length M indicating which triangles are visible
    """
    num_triangles = len(indices)
    visible = np.ones(num_triangles, dtype=np.bool_)
    
    for i in range(num_triangles):
        # Get triangle vertices
        p1 = vertices[indices[i, 0]]
        p2 = vertices[indices[i, 1]]
        p3 = vertices[indices[i, 2]]
        
        # Check if triangle faces camera
        visible[i] = is_triangle_back_facing(p1, p2, p3, camera_pos)
    
    return visible


@njit(cache=True)
def is_triangle_back_facing(p1, p2, p3, camera_pos):
    """Determine if triangle is facing away from the camera.
    
    Args:
        p1, p2, p3: Triangle vertices in world space
        camera_pos: Camera position in world space
        
    Returns:
        bool: True if triangle is back-facing (should be culled), False if front-facing
    """
    # Calculate triangle normal using cross product of edges
    v1 = p2 - p1  # First edge
    v2 = p3 - p1  # Second edge
    normal = np.cross(v1, v2)
    
    # Normalize the normal vector
    length = np.sqrt(np.sum(normal * normal))
    if length > 0:
        normal = normal / length
    
    # Calculate vector from any vertex to camera
    view_vector = camera_pos - p1
    
    # If dot product is negative, triangle is facing away from camera
    return np.dot(normal, view_vector) < 0
