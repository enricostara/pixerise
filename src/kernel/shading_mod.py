import numpy as np
from numba import njit


@njit(cache=True)
def triangle_flat_shading(vertices, normal, light_dir, material_color, ambient: float = 0.1) -> np.ndarray:
    """Calculate flat shading color for a triangle.
    
    Args:
        vertices: Triangle vertices (3x3 array)
        normal: Triangle normal vector (3D array)
        light_dir: Light direction vector (3D array)
        material_color: Base material color (RGB array)
        ambient: Ambient light intensity (float)
        
    Returns:
        ndarray: Final RGB color after shading
    """
    
    # Calculate diffuse intensity using dot product
    intensity = np.maximum(np.dot(normal, light_dir), 0.0)
    
    # Calculate final color with ambient term
    color = material_color * (ambient + (1.0 - ambient) * intensity)
    
    # Clamp colors to valid range
    color = np.clip(color, 0.0, 255.0)
    
    return color.astype(np.uint8)
