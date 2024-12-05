"""
JIT-compiled kernel functions for the Pixerise rasterizer.
These functions are optimized using Numba's JIT compilation for better performance.
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def transform_vertex(vertex: np.ndarray, 
                        translation: np.ndarray, rotation: np.ndarray, scale: np.ndarray,
                        camera_translation: np.ndarray, camera_rotation: np.ndarray,
                        has_camera: bool) -> np.ndarray:
    """JIT-compiled vertex transformation using homogeneous coordinates."""
    # Create transformation matrices inline to avoid matrix creation overhead
    x, y, z = vertex
    
    # Apply scale
    x *= scale[0]
    y *= scale[1]
    z *= scale[2]
    
    # Apply rotation (Y * X * Z order)
    # Z rotation
    rx, ry, rz = rotation
    cz, sz = np.cos(rz), np.sin(rz)
    x_new = x * cz - y * sz
    y_new = x * sz + y * cz
    x, y = x_new, y_new
    
    # X rotation
    cx, sx = np.cos(rx), np.sin(rx)
    y_new = y * cx - z * sx
    z_new = y * sx + z * cx
    y, z = y_new, z_new
    
    # Y rotation
    cy, sy = np.cos(ry), np.sin(ry)
    x_new = x * cy + z * sy
    z_new = -x * sy + z * cy
    x, z = x_new, z_new
    
    # Apply translation
    x += translation[0]
    y += translation[1]
    z += translation[2]
    
    # Apply camera transform if present
    if has_camera:
        # First translate to camera space
        x -= camera_translation[0]
        y -= camera_translation[1]
        z -= camera_translation[2]
        
        # Apply camera rotation (inverse of normal rotation)
        # Y rotation
        crx, cry, crz = camera_rotation
        ccy, csy = np.cos(cry), np.sin(cry)
        x_new = x * ccy - z * csy
        z_new = x * csy + z * ccy
        x, z = x_new, z_new
        
        # X rotation
        ccx, csx = np.cos(crx), np.sin(crx)
        y_new = y * ccx + z * csx
        z_new = -y * csx + z * ccx
        y, z = y_new, z_new
        
        # Z rotation
        ccz, csz = np.cos(crz), np.sin(crz)
        x_new = x * ccz + y * csz
        y_new = -x * csz + y * ccz
        x, y = x_new, y_new
    
    return np.array([x, y, z])
