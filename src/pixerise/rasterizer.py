import numpy as np
from numba import jit
from pixerise.canvas import Canvas
from pixerise.viewport import ViewPort
from pixerise.kernel import (draw_line, draw_triangle, draw_shaded_triangle,
                           transform_vertex_jit)


class Rasterizer:
    def __init__(self, canvas: Canvas, viewport: ViewPort, scene: dict, background_color=(32, 32, 32)):
        self._canvas = canvas
        self._viewport = viewport
        self._scene = scene
        self._background_color = np.array(background_color, dtype=int)

    def _create_rotation_matrix(self, angles: np.ndarray) -> np.ndarray:
        """Create a homogeneous 4x4 rotation matrix from euler angles (x, y, z) in radians."""
        # Extract angles
        rx, ry, rz = angles
        
        # X rotation
        cx, sx = np.cos(rx), np.sin(rx)
        Rx = np.array([
            [1, 0, 0, 0],
            [0, cx, -sx, 0],
            [0, sx, cx, 0],
            [0, 0, 0, 1]
        ])
        
        # Y rotation
        cy, sy = np.cos(ry), np.sin(ry)
        Ry = np.array([
            [cy, 0, sy, 0],
            [0, 1, 0, 0],
            [-sy, 0, cy, 0],
            [0, 0, 0, 1]
        ])
        
        # Z rotation
        cz, sz = np.cos(rz), np.sin(rz)
        Rz = np.array([
            [cz, -sz, 0, 0],
            [sz, cz, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Combined rotation matrix (order: Y * X * Z)
        return Ry @ Rx @ Rz

    def _create_scale_matrix(self, scale: np.ndarray) -> np.ndarray:
        """Create a homogeneous 4x4 scale matrix."""
        return np.array([
            [scale[0], 0, 0, 0],
            [0, scale[1], 0, 0],
            [0, 0, scale[2], 0],
            [0, 0, 0, 1]
        ])

    def _create_translation_matrix(self, translation: np.ndarray) -> np.ndarray:
        """Create a homogeneous 4x4 translation matrix."""
        return np.array([
            [1, 0, 0, translation[0]],
            [0, 1, 0, translation[1]],
            [0, 0, 1, translation[2]],
            [0, 0, 0, 1]
        ])

    def _create_camera_matrix(self, transform: dict) -> np.ndarray:
        """Create a homogeneous 4x4 camera matrix from position and orientation.
        The camera matrix transforms vertices from world space to camera space.
        It's the inverse of the camera's model matrix (position and orientation in world space).
        """
        # Get camera transform components
        translation = transform.get('translation', np.zeros(3))
        rotation = transform.get('rotation', np.zeros(3))
        
        # Create rotation matrix for camera orientation
        R = self._create_rotation_matrix(rotation)
        
        # Create translation matrix for camera position
        T = self._create_translation_matrix(-translation)  # Negative translation for camera space
        
        # Camera matrix is the inverse of view transform: R^T * T
        # R^T is the transpose of R, which is the inverse for orthogonal rotation matrices
        R_transpose = R.T
        
        # Camera matrix combines rotation and translation: R^T * T
        return R_transpose @ T

    def _transform_vertex(self, vertex: np.ndarray, transform: dict) -> np.ndarray:
        """Apply transformation to a vertex using homogeneous coordinates."""
        # Get transform components
        translation = transform.get('translation', np.zeros(3))
        rotation = transform.get('rotation', np.zeros(3))
        scale = transform.get('scale', np.ones(3))
        
        # Get camera transform if it exists
        has_camera = 'camera' in self._scene and 'transform' in self._scene['camera']
        camera_translation = np.zeros(3)
        camera_rotation = np.zeros(3)
        if has_camera:
            camera_transform = self._scene['camera']['transform']
            camera_translation = camera_transform.get('translation', np.zeros(3))
            camera_rotation = camera_transform.get('rotation', np.zeros(3))
        
        # Call JIT-compiled function
        return transform_vertex_jit(
            vertex, translation, rotation, scale,
            camera_translation, camera_rotation, has_camera
        )

    def _project_vertex(self, vertex, position=None):
        # Apply translation if position is provided
        if position is not None:
            vertex = vertex + position
            
        x, y, z = vertex
        d = self._viewport._plane_distance / z
        return self._viewport.viewport_to_canvas(x * d, y * d)

    def draw_line(self, start: (float, float), end: (float, float), color: (int, int, int)):
        """Draw a line using Bresenham's algorithm for better performance."""
        draw_line(
            int(start[0]), int(start[1]), 
            int(end[0]), int(end[1]), 
            self._canvas.grid, 
            self._canvas._center[0], self._canvas._center[1],
            color[0], color[1], color[2],
            self._canvas.width, self._canvas.height
        )

    def draw_triangle(self, p1: (float, float), p2: (float, float), p3: (float, float), color: (int, int, int), fill: bool = True):
        """Draw a triangle defined by three points. If fill is True, the triangle will be filled,
        otherwise only the outline will be drawn."""
        if fill:
            draw_triangle(
                int(p1[0]), int(p1[1]), 
                int(p2[0]), int(p2[1]), 
                int(p3[0]), int(p3[1]), 
                self._canvas.grid,
                self._canvas._center[0], self._canvas._center[1],
                color[0], color[1], color[2],
                self._canvas.width, self._canvas.height
            )
        else:
            # Draw the three edges of the triangle
            self.draw_line(p1, p2, color)
            self.draw_line(p2, p3, color)
            self.draw_line(p3, p1, color)

    def draw_shaded_triangle(self, p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float],
                           color: tuple[int, int, int],
                           intensity1: float, intensity2: float, intensity3: float) -> None:
        """
        Draw a triangle with interpolated intensities for a single color.
        
        Args:
            p1, p2, p3: Vertex positions as (x, y) tuples
            color: RGB color as (r, g, b) tuple
            intensity1, intensity2, intensity3: Intensity values (0.0 to 1.0) for each vertex
        """
        # Clamp intensities to valid range
        i1 = max(0.0, min(1.0, intensity1))
        i2 = max(0.0, min(1.0, intensity2))
        i3 = max(0.0, min(1.0, intensity3))
        
        draw_shaded_triangle(
            int(p1[0]), int(p1[1]), 
            int(p2[0]), int(p2[1]), 
            int(p3[0]), int(p3[1]), 
            self._canvas.grid, self._canvas._center[0], self._canvas._center[1],
            color[0], color[1], color[2],
            i1, i2, i3,
            self._canvas.width, self._canvas.height
        )

    def render(self, scene: dict):
        """Render a scene containing models and their instances."""
        models = scene['models']
        
        # Render each instance
        for instance in scene['instances']:
            model_name = instance['model']
            transform = instance.get('transform', {
                'translation': np.zeros(3),
                'rotation': np.zeros(3),
                'scale': np.ones(3)
            })
            
            # Get the model data
            model = models[model_name]
            vertices = model['vertices']
            triangles = model['triangles']
            
            # Transform and project vertices
            transformed_vertices = [self._transform_vertex(np.array(v), transform) for v in vertices]
            projected_vertices = [self._project_vertex(v) for v in transformed_vertices]
            
            # Draw each triangle
            for triangle in triangles:
                v1, v2, v3 = triangle
                self.draw_triangle(
                    projected_vertices[v1],
                    projected_vertices[v2],
                    projected_vertices[v3],
                    (255, 255, 255),  # White color for now
                    fill=False)