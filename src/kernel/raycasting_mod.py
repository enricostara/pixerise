"""
Ray casting operations optimized with Numba.
Implements efficient ray-triangle intersection testing using the Möller–Trumbore algorithm.
"""

import numpy as np
from numba import njit

EPSILON = 1e-7


@njit(cache=True)
def check_ray_triangle_intersection(
    ray_origin: np.ndarray,  # Origin of ray in camera space
    ray_direction: np.ndarray,  # Direction of ray in camera space (normalized)
    v0: np.ndarray,  # First vertex of triangle in camera space
    v1: np.ndarray,  # Second vertex of triangle in camera space
    v2: np.ndarray,  # Third vertex of triangle in camera space
) -> tuple[bool, float, float, float]:
    """Test if a ray intersects a triangle using the Möller–Trumbore algorithm.

    The algorithm works in 3 steps:
    1. Compute vectors and determinant
    2. Calculate barycentric coordinates (u,v)
    3. Calculate intersection distance t

    In camera space:
    - Origin is at (0,0,0)
    - Looking down -Z axis
    - Right-handed coordinate system
    - Counter-clockwise winding when looking at front face

    Args:
        ray_origin: Origin point of ray
        ray_direction: Direction vector of ray (normalized)
        v0, v1, v2: Vertices of triangle to test against

    Returns:
        Tuple of:
        - bool: True if ray intersects triangle
        - float: Distance t along ray to intersection
        - float: Barycentric coordinate u
        - float: Barycentric coordinate v
    """
    # Edge vectors (counter-clockwise winding)
    edge1 = v1 - v0  # Edge from v0 to v1
    edge2 = v2 - v0  # Edge from v0 to v2

    # In camera space we're looking down -Z, so negate ray direction
    ray_dir = -ray_direction

    # Calculate determinant
    h = np.cross(ray_dir, edge2)
    a = np.dot(edge1, h)  # Determinant

    # If determinant is near zero, ray lies in plane of triangle
    if abs(a) < EPSILON:
        return False, 0.0, 0.0, 0.0

    f = 1.0 / a
    s = ray_origin - v0  # Vector from v0 to ray origin

    # Calculate u parameter
    u = f * np.dot(s, h)

    # Test bounds for U
    if u < -EPSILON or u > 1.0 + EPSILON:
        return False, 0.0, 0.0, 0.0

    # Calculate v parameter
    q = np.cross(s, edge1)
    v = f * np.dot(ray_dir, q)

    # Test bounds for V and U+V
    if v < -EPSILON or u + v > 1.0 + EPSILON:
        return False, 0.0, 0.0, 0.0

    # Calculate t - distance from ray origin to intersection
    t = f * np.dot(edge2, q)

    # Ensure intersection is in front of ray origin
    # We're in camera space looking down -Z, so t should be positive
    if t < EPSILON:
        return False, 0.0, 0.0, 0.0

    return True, t, u, v


@njit(cache=True)
def check_ray_triangles_intersection(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    transformed_vertices: np.ndarray,
    triangles: np.ndarray,
) -> tuple[bool, float]:
    """Test ray intersection against all triangles in a group.

    Args:
        ray_origin: Origin of ray in camera space
        ray_dir: Direction of ray in camera space (normalized)
        transformed_vertices: Array of shape (N, 3) containing pre-transformed vertex positions in camera space
        triangles: Array of shape (M, 3) containing triangle vertex indices

    Returns:
        Tuple of (hit, t) where:
        - hit: True if ray intersects any triangle
        - t: Distance to closest intersection
    """
    closest_t = np.inf
    hit = False

    for triangle in triangles:
        v0 = transformed_vertices[triangle[0]]
        v1 = transformed_vertices[triangle[1]]
        v2 = transformed_vertices[triangle[2]]

        triangle_hit, t, _, _ = check_ray_triangle_intersection(
            ray_origin, ray_dir, v0, v1, v2
        )
        if triangle_hit and t < closest_t:
            hit = True
            closest_t = t

    return hit, closest_t
