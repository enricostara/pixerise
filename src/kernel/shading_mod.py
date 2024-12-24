import numpy as np
from numba import njit


@njit(cache=True)
def compute_flat_shading(vertices, indices, normals, light_dir, material_color, ambient: float = 0.1) -> np.ndarray:
    """Compute flat shading for triangles.
    
    Args:
        vertices: Nx3 array of vertex positions
        indices: Mx3 array of triangle indices into vertices
        normals: Mx3 array of pre-computed face normals
        light_dir: Normalized 3D vector for light direction
        material_color: RGB base color of the material (3D array, 8-bit values)
        ambient: Ambient light intensity (default: 0.1)
        
    Returns:
        ndarray: Mx3 array of RGB colors for each triangle (8-bit values)
    """
    num_triangles = len(indices)
    colors = np.zeros((num_triangles, 3), dtype=np.float64)
    
    for i in range(num_triangles):
        # Calculate diffuse intensity using dot product
        intensity = np.maximum(np.dot(normals[i], light_dir), 0.0)
        
        # Calculate final color with ambient term
        colors[i] = material_color * (ambient + (1.0 - ambient) * intensity)
        
        # Clamp colors to valid range
        colors[i] = np.clip(colors[i], 0.0, 255.0)
    
    return colors.astype(np.uint8)
