"""
JIT-compiled kernel functions for the Pixerise rasterizer.
These functions are optimized using Numba's JIT compilation for better performance.
"""

import numpy as np
from numba import njit
from typing import Tuple


@njit(cache=True)
def transform_vertex(
    vertex: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
    scale: np.ndarray,
    camera_translation: np.ndarray,
    camera_rotation: np.ndarray,
    has_camera: bool,
) -> np.ndarray:
    """
    Transform a vertex in 3D space using scale, rotation, translation, and optional camera transformations.
    This function is optimized using Numba's JIT compilation and avoids matrix creation overhead
    by performing transformations inline.

    The transformations are applied in the following order:
    1. Scale
    2. Model Rotation (Y * X * Z order)
    3. Translation
    4. Camera Transform (if has_camera is True):
       - Camera Translation (inverse)
       - Camera Rotation (inverse, Z * X * Y order)

    Args:
        vertex: Numpy array of shape (3,) representing the vertex position (x, y, z)
        translation: Numpy array of shape (3,) representing the translation vector (tx, ty, tz)
        rotation: Numpy array of shape (3,) containing rotation angles in radians (rx, ry, rz)
                 around X, Y, and Z axes respectively
        scale: Numpy array of shape (3,) containing scale factors (sx, sy, sz)
        camera_translation: Numpy array of shape (3,) representing camera position (cx, cy, cz)
        camera_rotation: Numpy array of shape (3,) containing camera rotation angles in radians
                        (crx, cry, crz) around X, Y, and Z axes respectively
        has_camera: Boolean flag indicating whether to apply camera transformation

    Returns:
        Numpy array of shape (3,) containing the transformed vertex position

    Note:
        - Rotations use right-hand rule: positive angles rotate counterclockwise when looking
          along the positive axis towards the origin
        - Camera transformations are applied as inverse operations to transform vertices into
          camera space
        - The implementation avoids matrix multiplication by directly computing the transformed
          coordinates, which is more efficient for single vertex transformations
    """
    # Extract vertex components for direct manipulation
    # This avoids repeated array indexing operations
    x, y, z = vertex

    # Step 1: Apply non-uniform scale
    # Scale each component independently to support different scale factors per axis
    x *= scale[0]  # Scale X component
    y *= scale[1]  # Scale Y component
    z *= scale[2]  # Scale Z component

    # Step 2: Apply model rotations in Y * X * Z order
    # This order minimizes gimbal lock for most common use cases

    # 2a: Z-axis rotation (around Z-axis)
    # [cos(z)  -sin(z)  0]   [x]
    # [sin(z)   cos(z)  0] * [y]
    # [  0       0      1]   [z]
    rx, ry, rz = rotation
    cz, sz = np.cos(rz), np.sin(rz)
    x_new = x * cz - y * sz  # x' = x*cos(z) - y*sin(z)
    y_new = x * sz + y * cz  # y' = x*sin(z) + y*cos(z)
    x, y = x_new, y_new  # z remains unchanged

    # 2b: X-axis rotation (around X-axis)
    # [1     0        0  ]   [x]
    # [0   cos(x)  -sin(x)] * [y]
    # [0   sin(x)   cos(x)]   [z]
    cx, sx = np.cos(rx), np.sin(rx)
    y_new = y * cx - z * sx  # y' = y*cos(x) - z*sin(x)
    z_new = y * sx + z * cx  # z' = y*sin(x) + z*cos(x)
    y, z = y_new, z_new  # x remains unchanged

    # 2c: Y-axis rotation (around Y-axis)
    # [ cos(y)  0  sin(y)]   [x]
    # [   0     1    0   ] * [y]
    # [-sin(y)  0  cos(y)]   [z]
    cy, sy = np.cos(ry), np.sin(ry)
    x_new = x * cy + z * sy  # x' = x*cos(y) + z*sin(y)
    z_new = -x * sy + z * cy  # z' = -x*sin(y) + z*cos(y)
    x, z = x_new, z_new  # y remains unchanged

    # Step 3: Apply translation
    # Simple addition of translation vector components
    x += translation[0]  # Translate along X
    y += translation[1]  # Translate along Y
    z += translation[2]  # Translate along Z

    # Step 4: Apply camera transformation if enabled
    if has_camera:
        # 4a: Transform to camera space by subtracting camera position
        # This effectively moves the world relative to the camera
        x -= camera_translation[0]
        y -= camera_translation[1]
        z -= camera_translation[2]

        # 4b: Apply inverse camera rotations in Z * X * Y order
        # This is the inverse of model rotation, applied in reverse order

        # Y-axis camera rotation (inverse)
        crx, cry, crz = camera_rotation
        ccy, csy = np.cos(cry), np.sin(cry)
        x_new = x * ccy - z * csy  # Inverse of Y rotation matrix
        z_new = x * csy + z * ccy
        x, z = x_new, z_new

        # X-axis camera rotation (inverse)
        ccx, csx = np.cos(crx), np.sin(crx)
        y_new = y * ccx + z * csx  # Inverse of X rotation matrix
        z_new = -y * csx + z * ccx
        y, z = y_new, z_new

        # Z-axis camera rotation (inverse)
        ccz, csz = np.cos(crz), np.sin(crz)
        x_new = x * ccz + y * csz  # Inverse of Z rotation matrix
        y_new = -x * csz + y * ccz
        x, y = x_new, y_new

    # Return the transformed vertex as a new array
    # This ensures the original vertex array is not modified
    return np.array([x, y, z], dtype=np.float32)


@njit(cache=True)
def transform_vertex_normal(
    normal: np.ndarray,
    rotation: np.ndarray,
    camera_rotation: np.ndarray,
    has_camera: bool,
) -> np.ndarray:
    """
    Transform a vertex normal in 3D space using rotation and optional camera transformations.
    This function is optimized using Numba's JIT compilation and avoids matrix creation overhead
    by performing transformations inline.

    The transformations are applied in the following order:
    1. Model Rotation (Y * X * Z order)
    2. Camera Rotation (if has_camera is True):
       - Camera Rotation (inverse, Z * X * Y order)

    Args:
        normal: Numpy array of shape (3,) representing the normal vector (nx, ny, nz)
        rotation: Numpy array of shape (3,) containing rotation angles in radians (rx, ry, rz)
                 around X, Y, and Z axes respectively
        camera_rotation: Numpy array of shape (3,) containing camera rotation angles in radians
                        (crx, cry, crz) around X, Y, and Z axes respectively
        has_camera: Boolean flag indicating whether to apply camera transformation

    Returns:
        Numpy array of shape (3,) containing the transformed normal vector

    Note:
        - Rotations use right-hand rule: positive angles rotate counterclockwise when looking
          along the positive axis towards the origin
        - Camera transformations are applied as inverse operations
        - The implementation avoids matrix multiplication by directly computing the transformed
          coordinates, which is more efficient for single normal transformations
    """
    # Extract normal components for direct manipulation
    x, y, z = normal

    # Step 1: Apply model rotations in Y * X * Z order
    # This order minimizes gimbal lock for most common use cases

    # 1a: Z-axis rotation (around Z-axis)
    rx, ry, rz = rotation
    cz, sz = np.cos(rz), np.sin(rz)
    x_new = x * cz - y * sz
    y_new = x * sz + y * cz
    x, y = x_new, y_new

    # 1b: X-axis rotation (around X-axis)
    cx, sx = np.cos(rx), np.sin(rx)
    y_new = y * cx - z * sx
    z_new = y * sx + z * cx
    y, z = y_new, z_new

    # 1c: Y-axis rotation (around Y-axis)
    cy, sy = np.cos(ry), np.sin(ry)
    x_new = x * cy - z * sy  # Changed from: x * cy + z * sy
    z_new = x * sy + z * cy  # Changed from: -x * sy + z * cy
    x, z = x_new, z_new

    # Step 2: Apply camera rotation if enabled
    if has_camera:
        # Apply inverse camera rotations in Z * X * Y order

        # Y-axis camera rotation (inverse)
        crx, cry, crz = camera_rotation
        ccy, csy = np.cos(cry), np.sin(cry)
        x_new = x * ccy - z * csy  # Inverse of Y rotation matrix
        z_new = x * csy + z * ccy
        x, z = x_new, z_new

        # X-axis camera rotation (inverse)
        ccx, csx = np.cos(crx), np.sin(crx)
        y_new = y * ccx + z * csx  # Inverse of X rotation matrix
        z_new = -y * csx + z * ccx
        y, z = y_new, z_new

        # Z-axis camera rotation (inverse)
        ccz, csz = np.cos(crz), np.sin(crz)
        x_new = x * ccz + y * csz  # Inverse of Z rotation matrix
        y_new = -x * csz + y * ccz
        x, y = x_new, y_new

    # Normalize the resulting normal vector
    length = np.sqrt(x * x + y * y + z * z)
    if length > 0:
        x /= length
        y /= length
        z /= length

    return np.array([x, y, z], dtype=np.float32)


@njit(cache=True)
def project_vertex(
    vertex: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    viewport_width: float,
    viewport_height: float,
) -> Tuple[float, float, float]:
    """
    Project a vertex from 3D to 2D screen space.

    Args:
        vertex: 3D vertex coordinates
        canvas_width: Width of the canvas in pixels
        canvas_height: Height of the canvas in pixels
        viewport_width: Width of the viewport
        viewport_height: Height of the viewport

    Returns:
        tuple: Projected coordinates (x, y, z) or (None, None, None) if behind camera
    """
    if vertex[2] <= 0:  # Behind camera
        return (0.0, 0.0, 0.0)  # Return dummy values when behind camera

    # Perspective projection
    x = vertex[0] / vertex[2]
    y = vertex[1] / vertex[2]

    # Convert to screen space
    x = x * canvas_width / viewport_width
    y = y * canvas_height / viewport_height

    return (x, y, vertex[2])


@njit(cache=True)
def transform_vertices_and_normals(
    vertices: np.ndarray,
    vertex_normals: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
    scale: np.ndarray,
    camera_translation: np.ndarray,
    camera_rotation: np.ndarray,
    shading_mode: str,
    has_vertex_normals: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform a batch of vertices and normals in 3D space using scale, rotation, translation, and optional camera transformations.
    This function is optimized using Numba's JIT compilation.

    Args:
        vertices: Numpy array of shape (N, 3) representing the vertex positions
        vertex_normals: Numpy array of shape (N, 3) representing the vertex normals
        translation: Numpy array of shape (3,) representing the translation vector
        rotation: Numpy array of shape (3,) containing rotation angles in radians
        scale: Numpy array of shape (3,) containing scale factors
        camera_translation: Numpy array of shape (3,) representing camera position
        camera_rotation: Numpy array of shape (3,) containing camera rotation angles in radians
        shading_mode: Integer representing the shading mode (e.g., GOURAUD)
        has_vertex_normals: Boolean indicating if vertex normals are provided

    Returns:
        Tuple of Numpy arrays containing the transformed vertices and normals
    """
    num_vertices = vertices.shape[0]
    transformed_vertices = np.zeros((num_vertices, 3), dtype=np.float32)
    transformed_normals = np.zeros((num_vertices, 3), dtype=np.float32)

    for i in range(num_vertices):
        # Transform vertex
        transformed_vertices[i] = transform_vertex(
            vertices[i],
            translation,
            rotation,
            scale,
            camera_translation,
            camera_rotation,
            True,
        )

        # Transform normal if shading mode is GOURAUD and normals are available
        if shading_mode == "gouraud" and has_vertex_normals:
            transformed_normals[i] = transform_vertex_normal(
                vertex_normals[i], -rotation, camera_rotation, True
            )

    return transformed_vertices, transformed_normals
