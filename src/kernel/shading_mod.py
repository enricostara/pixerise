import numpy as np
from numba import njit


@njit(cache=True)
def compute_flat_shading(vertices, indices, normals, light_dir, light_color, material_color, ambient: float = 0.1) -> np.ndarray:
    """Compute flat shading for triangles.
    
    Args:
        vertices: Nx3 array of vertex positions
        indices: Mx3 array of triangle indices into vertices
        normals: Mx3 array of pre-computed face normals
        light_dir: Normalized 3D vector for light direction
        light_color: RGB color of the light (3D array)
        material_color: RGB base color of the material (3D array)
        ambient: Ambient light intensity (default: 0.1)
        
    Returns:
        ndarray: Mx3 array of RGB colors for each triangle
    """
    num_triangles = len(indices)
    colors = np.zeros((num_triangles, 3))
    
    for i in range(num_triangles):
        # Calculate diffuse intensity using dot product
        intensity = np.maximum(np.dot(normals[i], light_dir), 0.0)
        
        # Calculate final color with ambient term
        colors[i] = material_color * (ambient + (1.0 - ambient) * intensity * light_color)
        
        # Clamp colors to valid range
        colors[i] = np.clip(colors[i], 0.0, 1.0)
    
    return colors
