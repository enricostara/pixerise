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
