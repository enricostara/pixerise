"""
JIT-compiled kernel functions for the Pixerise rasterizer.
These functions are optimized using Numba's JIT compilation for better performance.
"""

import numpy as np
from numba import jit


@jit(nopython=True)
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


@jit(nopython=True)
def calculate_signed_distance(plane_normal: np.ndarray, vertex: np.ndarray) -> float:
    """
    Calculate the signed distance between a plane passing through the origin and a vertex.
    
    Args:
        plane_normal: Numpy array of shape (3,) representing the plane normal vector (should be normalized)
        vertex: Numpy array of shape (3,) representing the vertex to calculate distance from
        
    Returns:
        float: The signed distance from the plane to the vertex. 
              Positive if vertex is on the same side as the normal,
              negative if on the opposite side, zero if on the plane.
    """
    
    # The signed distance is the dot product of this vector with the normal
    return np.dot(vertex, plane_normal)


@jit(nopython=True)
def calculate_segment_plane_intersection(plane_normal: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple:
    """
    Calculate the intersection point between a line segment and a plane passing through the origin.
    
    Args:
        plane_normal: Numpy array of shape (3,) representing the plane normal vector (should be normalized)
        start: Numpy array of shape (3,) representing the start point of the line segment
        end: Numpy array of shape (3,) representing the end point of the line segment
        
    Returns:
        tuple: (intersection_point, t) where:
               - intersection_point is a numpy array of shape (3,) representing the intersection point
               - t is a float between 0 and 1 indicating where along the segment the intersection occurs
               Returns (None, None) if the segment is parallel to the plane or doesn't intersect
    """
    # Calculate direction vector of the line segment
    direction = end - start
    
    # Calculate denominator for intersection check
    denom = np.dot(direction, plane_normal)
    
    # Check if line is parallel to plane (or very close to parallel)
    if abs(denom) < 1e-6:
        return None, None
        
    # Calculate the parameter t where the intersection occurs
    t = -np.dot(start, plane_normal) / denom
    
    # Check if intersection occurs within the segment bounds
    if t < 0.0 or t > 1.0:
        return None, None
        
    # Calculate the intersection point
    intersection = start + t * direction
    
    return intersection, t
