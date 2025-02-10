"""
Ray casting operations optimized with Numba.
Implements efficient ray-triangle intersection testing using the Möller–Trumbore algorithm.
"""

import numpy as np
from numba import njit

EPSILON = 1e-7

@njit(cache=True)
def rayIntersectsTriangle(
    ray_origin: np.ndarray,    # Origin of ray in camera space
    ray_direction: np.ndarray, # Direction of ray in camera space (normalized)
    v0: np.ndarray,           # First vertex of triangle in camera space
    v1: np.ndarray,           # Second vertex of triangle in camera space
    v2: np.ndarray            # Third vertex of triangle in camera space
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
    
    print("Determinant a =", a)
    
    # If determinant is near zero, ray lies in plane of triangle
    if abs(a) < EPSILON:
        print("  Ray parallel to triangle")
        return False, 0.0, 0.0, 0.0
        
    f = 1.0 / a
    s = ray_origin - v0  # Vector from v0 to ray origin
    
    # Calculate u parameter
    u = f * np.dot(s, h)
    
    print("  Barycentric u =", u)
    
    # Test bounds for U
    if u < -EPSILON or u > 1.0 + EPSILON:
        print("  u out of bounds")
        return False, 0.0, 0.0, 0.0
        
    # Calculate v parameter
    q = np.cross(s, edge1)
    v = f * np.dot(ray_dir, q)
    
    print("  Barycentric v =", v)
    
    # Test bounds for V and U+V
    if v < -EPSILON or u + v > 1.0 + EPSILON:
        print("  v out of bounds or u+v > 1")
        return False, 0.0, 0.0, 0.0
        
    # Calculate t - distance from ray origin to intersection
    t = f * np.dot(edge2, q)
    
    print("  Intersection distance t =", t)
    
    # Ensure intersection is in front of ray origin
    # We're in camera space looking down -Z, so t should be positive
    if t < EPSILON:
        print("  Intersection behind ray origin")
        return False, 0.0, 0.0, 0.0
    
    # Calculate and print intersection point for debugging
    intersect_point = ray_origin + ray_dir * t
    print("  Intersection point:", intersect_point)
        
    return True, t, u, v
