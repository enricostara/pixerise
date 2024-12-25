import numpy as np
from numba import njit


@njit(cache=True)
def triangle_flat_shading(vertices, normal, light_dir, material_color, ambient: float = 0.1) -> np.ndarray:
    """Compute flat shading for triangles.
    
    Args:
        vertices: Nx3 array of vertex positions
        normals: 3D array of pre-computed face normal
        light_dir: Normalized 3D vector for light direction
        material_color: RGB base color of the material (3D array, 8-bit values)
        ambient: Ambient light intensity (0-1)
        
    Returns:
        ndarray: RGB color for the triangle (8-bit values)
    """
    
    # Calculate diffuse intensity using dot product
    intensity = np.maximum(np.dot(normal, light_dir), 0.0)
    
    # Calculate final color with ambient term
    color = material_color * (ambient + (1.0 - ambient) * intensity)
    
    # Clamp colors to valid range
    color = np.clip(color, 0.0, 255.0)
    
    return color.astype(np.uint8)
