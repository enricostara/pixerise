"""
Ray casting operations optimized with Numba.
Implements efficient ray-triangle intersection testing using the Möller–Trumbore algorithm.
"""

import numpy as np
from numba import njit

EPSILON = 1e-7

@njit(cache=True)
def rayIntersectsTriangle(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray
) -> tuple[bool, float, float, float]:
    """
    Compute ray-triangle intersection using the Möller–Trumbore algorithm.
    
    Args:
        ray_origin: Origin point of the ray in 3D space
        ray_direction: Direction vector of the ray (should be normalized)
        v0, v1, v2: Vertices of the triangle in 3D space
        
    Returns:
        tuple containing:
        - bool: True if intersection occurs within triangle
        - float: Distance t along ray to intersection point
        - float: Barycentric coordinate u
        - float: Barycentric coordinate v
        
    Note:
        - If no intersection occurs, returns (False, 0.0, 0.0, 0.0)
        - Barycentric coordinates (u,v) determine where in triangle intersection occurs:
          P = (1-u-v)*v0 + u*v1 + v*v2
    """
    # Compute edges of triangle
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # Begin calculating determinant
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    
    print("Determinant a =", a)
    
    # If determinant is near zero, ray is parallel to triangle
    if a < EPSILON and a > -EPSILON:
        print("  Ray parallel to triangle (determinant near zero)")
        return False, 0.0, 0.0, 0.0
        
    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)
    
    print("  Barycentric u =", u)
    
    # Test bounds for U
    if u < -EPSILON or u > 1.0 + EPSILON:
        print("  u out of bounds")
        return False, 0.0, 0.0, 0.0
        
    # Prepare to test V parameter
    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)
    
    print("  Barycentric v =", v)
    
    # Test bounds for V and U+V
    if v < -EPSILON or u + v > 1.0 + EPSILON:
        print("  v out of bounds or u+v > 1")
        return False, 0.0, 0.0, 0.0
        
    # Calculate t - ray intersection distance
    t = f * np.dot(edge2, q)
    
    print("  Intersection distance t =", t)
    
    # Ensure intersection is in front of ray origin
    if t <= EPSILON:
        print("  Intersection behind ray origin")
        return False, 0.0, 0.0, 0.0
        
    intersect_point = ray_origin + ray_direction * t   
    print("  Intersection at", intersect_point[0], intersect_point[1], intersect_point[2]) 
    return True, t, u, v
