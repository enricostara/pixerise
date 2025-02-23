import numpy as np
from numba import njit


@njit(cache=True)
def triangle_flat_shading(
    normal, light_dir, material_color, ambient: float = 0.1
) -> np.ndarray:
    """Calculate flat shading color for a triangle.

    Args:
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


@njit(cache=True)
def triangle_gouraud_shading(
    vertex_normals, light_dir, ambient: float = 0.1
) -> np.ndarray:
    """Calculate Gouraud shading intensities for triangle vertices.

    Args:
        vertex_normals: Normal vectors for each vertex (3x3 array, pre-normalized)
        light_dir: Light direction vector (3D array, pre-normalized)
        ambient: Ambient light intensity (float)

    Returns:
        ndarray: Array of vertex intensities (3,) for interpolation
    """
    # Initialize vertex intensities
    vertex_intensities = np.zeros(3)

    # Calculate lighting for each vertex
    for i in range(3):
        # Calculate diffuse intensity using dot product
        intensity = np.maximum(np.dot(vertex_normals[i], light_dir), 0.0)

        # Add ambient term
        vertex_intensities[i] = ambient + (1.0 - ambient) * intensity

    return vertex_intensities
