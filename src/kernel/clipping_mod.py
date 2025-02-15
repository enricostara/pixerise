"""
JIT-compiled kernel functions for the Pixerise rasterizer.
These functions are optimized using Numba's JIT compilation for better performance.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def calculate_bounding_sphere(vertices: np.ndarray) -> tuple:
    """
    Calculate the center and radius of a bounding sphere containing all given vertices.
    Uses a simple bounding box approach optimized for JIT compilation.

    Args:
        vertices: Numpy array of shape (N, 3) containing N 3D points

    Returns:
        tuple: (center, radius) where center is a numpy array of shape (3,) and radius is a float

    Raises:
        IndexError: If vertices array is empty
    """
    if vertices.shape[0] == 0:
        raise IndexError("Cannot calculate bounding sphere for empty array")

    # Find the bounding box
    min_coords = np.empty(3)
    max_coords = np.empty(3)

    # Initialize with first vertex
    for i in range(3):
        min_coords[i] = vertices[0, i]
        max_coords[i] = vertices[0, i]

    # Find min and max for each coordinate
    for i in range(vertices.shape[0]):
        for j in range(3):
            if vertices[i, j] < min_coords[j]:
                min_coords[j] = vertices[i, j]
            if vertices[i, j] > max_coords[j]:
                max_coords[j] = vertices[i, j]

    # Calculate center as the middle of the bounding box
    center = np.empty(3)
    for i in range(3):
        center[i] = (min_coords[i] + max_coords[i]) / 2.0

    # Calculate radius as the distance to the furthest vertex
    radius = 0.0
    for i in range(vertices.shape[0]):
        distance_sq = 0.0
        for j in range(3):
            diff = vertices[i, j] - center[j]
            distance_sq += diff * diff
        distance = np.sqrt(distance_sq)
        if distance > radius:
            radius = distance

    return center, radius


@njit(cache=True)
def clip_triangle(
    vertices: np.ndarray, plane_normal: np.ndarray, plane_d: float = 0
) -> tuple:
    """
    Clip a triangle against a plane using the Sutherland-Hodgman algorithm.

    This function implements triangle clipping against a plane, handling various cases:
    1. Triangle completely above plane (preserved)
    2. Triangle completely below plane (removed)
    3. Vertex exactly on plane (special cases)
    4. One vertex above plane, two below (creates one triangle)
    5. Two vertices above plane, one below (creates two triangles)

    Args:
        vertices: Numpy array of shape (3, 3) containing the triangle vertices
        plane_normal: Numpy array of shape (3,) representing the plane normal vector (should be normalized)
        plane_d: float, the constant term in the plane equation

    Returns:
        tuple: (triangles, num_triangles) where:
               - triangles is a numpy array of shape (2, 3, 3) containing up to 2 triangles
               - num_triangles is the number of triangles after clipping (0, 1, or 2)

    Note:
        The function uses signed distances to determine vertex positions relative to the plane:
        - Positive distance: vertex is above/in front of the plane
        - Negative distance: vertex is below/behind the plane
        - Zero distance: vertex lies exactly on the plane
    """
    # Calculate signed distances from each vertex to the plane
    d0 = np.dot(vertices[0], plane_normal) + plane_d
    d1 = np.dot(vertices[1], plane_normal) + plane_d
    d2 = np.dot(vertices[2], plane_normal) + plane_d

    # Initialize result array to store up to 2 triangles
    triangles = np.empty((2, 3, 3), dtype=np.float32)

    # Case 1: All vertices on or above the plane
    # No clipping needed, return original triangle
    if d0 >= 0 and d1 >= 0 and d2 >= 0:
        triangles[0] = vertices
        return triangles, 1

    # Case 2: All vertices below the plane
    # Triangle is completely clipped away
    if d0 < 0 and d1 < 0 and d2 < 0:
        return triangles, 0

    # Special cases: Handle vertices exactly on the plane
    # These cases require special handling to avoid division by zero
    # For each vertex on the plane, we check the positions of other vertices
    # and create appropriate triangles based on their positions

    # Case 3a: Vertex 0 exactly on plane
    if abs(d0) < 1e-6:  # Using small epsilon for floating-point comparison
        if d1 >= 0 and d2 < 0:  # v1 above, v2 below
            # Calculate intersection point using linear interpolation
            t = d1 / (d1 - d2)  # Interpolation factor
            i = vertices[1] + t * (vertices[2] - vertices[1])  # Intersection point
            triangles[0, 0] = vertices[1]  # Keep clockwise order
            triangles[0, 1] = vertices[0]
            triangles[0, 2] = i
            return triangles, 1
        if d1 < 0 and d2 >= 0:  # v1 below, v2 above
            t = d2 / (d2 - d1)
            i = vertices[2] + t * (vertices[1] - vertices[2])
            triangles[0, 0] = vertices[2]  # Keep clockwise order
            triangles[0, 1] = vertices[0]
            triangles[0, 2] = i
            return triangles, 1

    if abs(d1) < 1e-6:  # v1 on plane
        if d0 >= 0 and d2 < 0:  # v0 above, v2 below
            t = d0 / (d0 - d2)
            i = vertices[0] + t * (vertices[2] - vertices[0])
            triangles[0, 0] = vertices[0]  # Keep clockwise order
            triangles[0, 1] = vertices[1]
            triangles[0, 2] = i
            return triangles, 1
        if d0 < 0 and d2 >= 0:  # v0 below, v2 above
            t = d2 / (d2 - d0)
            i = vertices[2] + t * (vertices[0] - vertices[2])
            triangles[0, 0] = vertices[2]  # Keep clockwise order
            triangles[0, 1] = vertices[1]
            triangles[0, 2] = i
            return triangles, 1

    if abs(d2) < 1e-6:  # v2 on plane
        if d0 >= 0 and d1 < 0:  # v0 above, v1 below
            t = d0 / (d0 - d1)
            i = vertices[0] + t * (vertices[1] - vertices[0])
            triangles[0, 0] = vertices[0]  # Keep clockwise order
            triangles[0, 1] = i
            triangles[0, 2] = vertices[2]
            return triangles, 1
        if d0 < 0 and d1 >= 0:  # v0 below, v1 above
            t = d1 / (d1 - d0)
            i = vertices[1] + t * (vertices[0] - vertices[1])
            triangles[0, 0] = vertices[1]  # Keep clockwise order
            triangles[0, 1] = i
            triangles[0, 2] = vertices[2]
            return triangles, 1

    # Case 4: One vertex above plane, two below
    # Results in one triangle formed by the above vertex and two intersection points
    if d0 >= 0 and d1 < 0 and d2 < 0:
        # Calculate intersection points using linear interpolation
        t1 = d0 / (d0 - d1)  # Interpolation factor for edge v0-v1
        t2 = d0 / (d0 - d2)  # Interpolation factor for edge v0-v2
        i1 = vertices[0] + t1 * (vertices[1] - vertices[0])  # First intersection point
        i2 = vertices[0] + t2 * (vertices[2] - vertices[0])  # Second intersection point
        # Form new triangle using the above vertex and intersection points
        triangles[0, 0] = vertices[0]  # Keep clockwise order
        triangles[0, 1] = i1
        triangles[0, 2] = i2
        return triangles, 1

    if d1 >= 0 and d0 < 0 and d2 < 0:
        t1 = d1 / (d1 - d0)
        t2 = d1 / (d1 - d2)
        i1 = vertices[1] + t1 * (vertices[0] - vertices[1])
        i2 = vertices[1] + t2 * (vertices[2] - vertices[1])
        triangles[0, 0] = vertices[1]  # Keep clockwise order
        triangles[0, 1] = i2
        triangles[0, 2] = i1
        return triangles, 1

    if d2 >= 0 and d0 < 0 and d1 < 0:
        t1 = d2 / (d2 - d0)
        t2 = d2 / (d2 - d1)
        i1 = vertices[2] + t1 * (vertices[0] - vertices[2])
        i2 = vertices[2] + t2 * (vertices[1] - vertices[2])
        triangles[0, 0] = vertices[2]  # Keep clockwise order
        triangles[0, 1] = i2
        triangles[0, 2] = i1
        return triangles, 1

    # Case 5: One vertex below plane, two above
    # Results in two triangles that form a quad
    if d0 < 0 and d1 >= 0 and d2 >= 0:
        # Calculate intersection points using linear interpolation
        t1 = d1 / (d1 - d0)  # Interpolation factor for edge v1-v0
        t2 = d2 / (d2 - d0)  # Interpolation factor for edge v2-v0
        i1 = vertices[1] + t1 * (vertices[0] - vertices[1])  # First intersection point
        i2 = vertices[2] + t2 * (vertices[0] - vertices[2])  # Second intersection point
        # Form two triangles to represent the clipped quad
        triangles[0, 0] = vertices[1]  # First triangle, keep clockwise
        triangles[0, 1] = vertices[2]
        triangles[0, 2] = i2
        triangles[1, 0] = vertices[1]  # Second triangle, keep clockwise
        triangles[1, 1] = i2
        triangles[1, 2] = i1
        return triangles, 2

    if d1 < 0 and d0 >= 0 and d2 >= 0:
        t1 = d0 / (d0 - d1)
        t2 = d2 / (d2 - d1)
        i1 = vertices[0] + t1 * (vertices[1] - vertices[0])
        i2 = vertices[2] + t2 * (vertices[1] - vertices[2])
        triangles[0, 0] = vertices[0]  # First triangle, keep clockwise
        triangles[0, 1] = vertices[2]
        triangles[0, 2] = i2
        triangles[1, 0] = vertices[0]  # Second triangle, keep clockwise
        triangles[1, 1] = i2
        triangles[1, 2] = i1
        return triangles, 2

    if d2 < 0 and d0 >= 0 and d1 >= 0:
        t1 = d0 / (d0 - d2)
        t2 = d1 / (d1 - d2)
        i1 = vertices[0] + t1 * (vertices[2] - vertices[0])
        i2 = vertices[1] + t2 * (vertices[2] - vertices[1])
        triangles[0, 0] = vertices[0]  # First triangle, keep clockwise
        triangles[0, 1] = vertices[1]
        triangles[0, 2] = i2
        triangles[1, 0] = vertices[0]  # Second triangle, keep clockwise
        triangles[1, 1] = i2
        triangles[1, 2] = i1
        return triangles, 2

    return triangles, 1  # Should never reach here


@njit(cache=True)
def normalize_vector(v):
    """Normalize a vector"""
    norm = np.sqrt(np.sum(v * v))
    if norm > 0:
        return v / norm
    return v


@njit(cache=True)
def clip_triangle_and_normals(
    vertices: np.ndarray,
    vertex_normals: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float = 0,
) -> tuple:
    """
    Clip a triangle against a plane using the Sutherland-Hodgman algorithm.

    This function implements triangle clipping against a plane, handling various cases:
    1. Triangle completely above plane (preserved)
    2. Triangle completely below plane (removed)
    3. Vertex exactly on plane (special cases)
    4. One vertex above plane, two below (creates one triangle)
    5. Two vertices above plane, one below (creates two triangles)

    Args:
        vertices: Numpy array of shape (3, 3) containing the triangle vertices
        vertex_normals: Numpy array of shape (3, 3) containing the vertex normals
        plane_normal: Numpy array of shape (3,) representing the plane normal vector (should be normalized)
        plane_d: float, the constant term in the plane equation

    Returns:
        tuple: (triangles, normals, num_triangles) where:
            - triangles is a numpy array of shape (2, 3, 3) containing up to 2 triangles
            - normals is a numpy array of shape (2, 3, 3) containing interpolated vertex normals
            - num_triangles is the number of triangles after clipping (0, 1, or 2)
    """
    # Calculate signed distances from each vertex to the plane
    d0 = np.dot(vertices[0], plane_normal) + plane_d
    d1 = np.dot(vertices[1], plane_normal) + plane_d
    d2 = np.dot(vertices[2], plane_normal) + plane_d

    # Initialize result arrays to store up to 2 triangles and their normals
    triangles = np.empty((2, 3, 3), dtype=np.float32)
    normals = np.empty((2, 3, 3), dtype=np.float32)

    # Case 1: All vertices on or above the plane
    # No clipping needed, return original triangle and normals
    if d0 >= 0 and d1 >= 0 and d2 >= 0:
        triangles[0] = vertices
        normals[0] = vertex_normals
        return triangles, normals, 1

    # Case 2: All vertices below the plane
    # Triangle is completely clipped away
    if d0 < 0 and d1 < 0 and d2 < 0:
        return triangles, normals, 0

    # Special cases: Handle vertices exactly on the plane
    # These cases require special handling to avoid division by zero
    # For each vertex on the plane, we check the positions of other vertices
    # and create appropriate triangles based on their positions

    # Case 3a: Vertex 0 exactly on plane
    if abs(d0) < 1e-6:  # Using small epsilon for floating-point comparison
        if d1 >= 0 and d2 < 0:  # v1 above, v2 below
            # Calculate intersection point using linear interpolation
            t = d1 / (d1 - d2)  # Interpolation factor
            i = vertices[1] + t * (vertices[2] - vertices[1])  # Intersection point
            i_normal = normalize_vector(
                vertex_normals[1] + t * (vertex_normals[2] - vertex_normals[1])
            )  # Interpolated normal
            triangles[0, 0] = vertices[1]  # Keep clockwise order
            triangles[0, 1] = vertices[0]
            triangles[0, 2] = i
            normals[0, 0] = vertex_normals[1]  # Keep corresponding normals
            normals[0, 1] = vertex_normals[0]
            normals[0, 2] = i_normal
            return triangles, normals, 1
        if d1 < 0 and d2 >= 0:  # v1 below, v2 above
            t = d2 / (d2 - d1)
            i = vertices[2] + t * (vertices[1] - vertices[2])
            i_normal = normalize_vector(
                vertex_normals[2] + t * (vertex_normals[1] - vertex_normals[2])
            )
            triangles[0, 0] = vertices[2]  # Keep clockwise order
            triangles[0, 1] = vertices[0]
            triangles[0, 2] = i
            normals[0, 0] = vertex_normals[2]
            normals[0, 1] = vertex_normals[0]
            normals[0, 2] = i_normal
            return triangles, normals, 1

    if abs(d1) < 1e-6:  # v1 on plane
        if d0 >= 0 and d2 < 0:  # v0 above, v2 below
            t = d0 / (d0 - d2)
            i = vertices[0] + t * (vertices[2] - vertices[0])
            i_normal = normalize_vector(
                vertex_normals[0] + t * (vertex_normals[2] - vertex_normals[0])
            )
            triangles[0, 0] = vertices[0]  # Keep clockwise order
            triangles[0, 1] = vertices[1]
            triangles[0, 2] = i
            normals[0, 0] = vertex_normals[0]
            normals[0, 1] = vertex_normals[1]
            normals[0, 2] = i_normal
            return triangles, normals, 1
        if d0 < 0 and d2 >= 0:  # v0 below, v2 above
            t = d2 / (d2 - d0)
            i = vertices[2] + t * (vertices[0] - vertices[2])
            i_normal = normalize_vector(
                vertex_normals[2] + t * (vertex_normals[0] - vertex_normals[2])
            )
            triangles[0, 0] = vertices[2]  # Keep clockwise order
            triangles[0, 1] = vertices[1]
            triangles[0, 2] = i
            normals[0, 0] = vertex_normals[2]
            normals[0, 1] = vertex_normals[1]
            normals[0, 2] = i_normal
            return triangles, normals, 1

    if abs(d2) < 1e-6:  # v2 on plane
        if d0 >= 0 and d1 < 0:  # v0 above, v1 below
            t = d0 / (d0 - d1)
            i = vertices[0] + t * (vertices[1] - vertices[0])
            i_normal = normalize_vector(
                vertex_normals[0] + t * (vertex_normals[1] - vertex_normals[0])
            )
            triangles[0, 0] = vertices[0]  # Keep clockwise order
            triangles[0, 1] = i
            triangles[0, 2] = vertices[2]
            normals[0, 0] = vertex_normals[0]
            normals[0, 1] = i_normal
            normals[0, 2] = vertex_normals[2]
            return triangles, normals, 1
        if d0 < 0 and d1 >= 0:  # v0 below, v1 above
            t = d1 / (d1 - d0)
            i = vertices[1] + t * (vertices[0] - vertices[1])
            i_normal = normalize_vector(
                vertex_normals[1] + t * (vertex_normals[0] - vertex_normals[1])
            )
            triangles[0, 0] = vertices[1]  # Keep clockwise order
            triangles[0, 1] = i
            triangles[0, 2] = vertices[2]
            normals[0, 0] = vertex_normals[1]
            normals[0, 1] = i_normal
            normals[0, 2] = vertex_normals[2]
            return triangles, normals, 1

    # Case 4: One vertex above plane, two below
    # Results in one triangle formed by the above vertex and two intersection points
    if d0 >= 0 and d1 < 0 and d2 < 0:
        # Calculate intersection points using linear interpolation
        t1 = d0 / (d0 - d1)  # Interpolation factor for edge v0-v1
        t2 = d0 / (d0 - d2)  # Interpolation factor for edge v0-v2
        i1 = vertices[0] + t1 * (vertices[1] - vertices[0])  # First intersection point
        i2 = vertices[0] + t2 * (vertices[2] - vertices[0])  # Second intersection point
        # Interpolate normals
        i1_normal = normalize_vector(
            vertex_normals[0] + t1 * (vertex_normals[1] - vertex_normals[0])
        )
        i2_normal = normalize_vector(
            vertex_normals[0] + t2 * (vertex_normals[2] - vertex_normals[0])
        )
        # Form new triangle using the above vertex and intersection points
        triangles[0, 0] = vertices[0]  # Keep clockwise order
        triangles[0, 1] = i1
        triangles[0, 2] = i2
        normals[0, 0] = vertex_normals[0]
        normals[0, 1] = i1_normal
        normals[0, 2] = i2_normal
        return triangles, normals, 1

    if d1 >= 0 and d0 < 0 and d2 < 0:
        t1 = d1 / (d1 - d0)
        t2 = d1 / (d1 - d2)
        i1 = vertices[1] + t1 * (vertices[0] - vertices[1])
        i2 = vertices[1] + t2 * (vertices[2] - vertices[1])
        i1_normal = normalize_vector(
            vertex_normals[1] + t1 * (vertex_normals[0] - vertex_normals[1])
        )
        i2_normal = normalize_vector(
            vertex_normals[1] + t2 * (vertex_normals[2] - vertex_normals[1])
        )
        triangles[0, 0] = vertices[1]  # Keep clockwise order
        triangles[0, 1] = i2
        triangles[0, 2] = i1
        normals[0, 0] = vertex_normals[1]
        normals[0, 1] = i2_normal
        normals[0, 2] = i1_normal
        return triangles, normals, 1

    if d2 >= 0 and d0 < 0 and d1 < 0:
        t1 = d2 / (d2 - d0)
        t2 = d2 / (d2 - d1)
        i1 = vertices[2] + t1 * (vertices[0] - vertices[2])
        i2 = vertices[2] + t2 * (vertices[1] - vertices[2])
        i1_normal = normalize_vector(
            vertex_normals[2] + t1 * (vertex_normals[0] - vertex_normals[2])
        )
        i2_normal = normalize_vector(
            vertex_normals[2] + t2 * (vertex_normals[1] - vertex_normals[2])
        )
        triangles[0, 0] = vertices[2]  # Keep clockwise order
        triangles[0, 1] = i2
        triangles[0, 2] = i1
        normals[0, 0] = vertex_normals[2]
        normals[0, 1] = i2_normal
        normals[0, 2] = i1_normal
        return triangles, normals, 1

    # Case 5: One vertex below plane, two above
    # Results in two triangles that form a quad
    if d0 < 0 and d1 >= 0 and d2 >= 0:
        t1 = d1 / (d1 - d0)  # Interpolation factor for edge v1-v0
        t2 = d2 / (d2 - d0)  # Interpolation factor for edge v2-v0
        i1 = vertices[1] + t1 * (vertices[0] - vertices[1])  # First intersection point
        i2 = vertices[2] + t2 * (vertices[0] - vertices[2])  # Second intersection point
        i1_normal = normalize_vector(
            vertex_normals[1] + t1 * (vertex_normals[0] - vertex_normals[1])
        )
        i2_normal = normalize_vector(
            vertex_normals[2] + t2 * (vertex_normals[0] - vertex_normals[2])
        )
        # Form two triangles
        triangles[0, 0] = vertices[1]  # First triangle
        triangles[0, 1] = vertices[2]
        triangles[0, 2] = i1
        triangles[1, 0] = i1  # Second triangle
        triangles[1, 1] = vertices[2]
        triangles[1, 2] = i2
        normals[0, 0] = vertex_normals[1]
        normals[0, 1] = vertex_normals[2]
        normals[0, 2] = i1_normal
        normals[1, 0] = i1_normal
        normals[1, 1] = vertex_normals[2]
        normals[1, 2] = i2_normal
        return triangles, normals, 2

    if d1 < 0 and d0 >= 0 and d2 >= 0:
        t1 = d0 / (d0 - d1)
        t2 = d2 / (d2 - d1)
        i1 = vertices[0] + t1 * (vertices[1] - vertices[0])
        i2 = vertices[2] + t2 * (vertices[1] - vertices[2])
        i1_normal = normalize_vector(
            vertex_normals[0] + t1 * (vertex_normals[1] - vertex_normals[0])
        )
        i2_normal = normalize_vector(
            vertex_normals[2] + t2 * (vertex_normals[1] - vertex_normals[2])
        )
        triangles[0, 0] = vertices[0]
        triangles[0, 1] = i1
        triangles[0, 2] = i2
        triangles[1, 0] = vertices[0]
        triangles[1, 1] = i2
        triangles[1, 2] = vertices[2]
        normals[0, 0] = vertex_normals[0]
        normals[0, 1] = i1_normal
        normals[0, 2] = i2_normal
        normals[1, 0] = vertex_normals[0]
        normals[1, 1] = i2_normal
        normals[1, 2] = vertex_normals[2]
        return triangles, normals, 2

    if d2 < 0 and d0 >= 0 and d1 >= 0:
        t1 = d0 / (d0 - d2)
        t2 = d1 / (d1 - d2)
        i1 = vertices[0] + t1 * (vertices[2] - vertices[0])
        i2 = vertices[1] + t2 * (vertices[2] - vertices[1])
        i1_normal = normalize_vector(
            vertex_normals[0] + t1 * (vertex_normals[2] - vertex_normals[0])
        )
        i2_normal = normalize_vector(
            vertex_normals[1] + t2 * (vertex_normals[2] - vertex_normals[1])
        )
        triangles[0, 0] = vertices[0]
        triangles[0, 1] = vertices[1]
        triangles[0, 2] = i1
        triangles[1, 0] = i1
        triangles[1, 1] = vertices[1]
        triangles[1, 2] = i2
        normals[0, 0] = vertex_normals[0]
        normals[0, 1] = vertex_normals[1]
        normals[0, 2] = i1_normal
        normals[1, 0] = i1_normal
        normals[1, 1] = vertex_normals[1]
        normals[1, 2] = i2_normal
        return triangles, normals, 2

    # Should never reach here
    return triangles, normals, 0
