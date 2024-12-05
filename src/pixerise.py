"""
Core components of the Pixerise rendering engine.
This module contains the main classes for rendering: Canvas, ViewPort, and Renderer.
"""

import numpy as np
from numba import jit
import pygame
from kernel.rasterizing_mod import (draw_pixel, draw_line, draw_triangle, draw_shaded_triangle)
from kernel.transforming_mod import transform_vertex


class Canvas:
    """A 2D canvas for drawing pixels and managing the drawing surface."""
    
    def __init__(self, size: (int, int) = (800, 600)):
        self.size = size
        self.width = size[0]
        self.height = size[1]
        self.grid = np.ones((self.width, self.height, 3), dtype=np.uint8) * 32  # Back to column-major order (width, height)
        self.half_width = self.width // 2
        self.half_height = self.height // 2
        self._center = (self.half_width, self.half_height)


class ViewPort:
    """Manages the view frustum and coordinate transformations from viewport to canvas space."""

    def __init__(self, size: (float, float), plane_distance: float, canvas: Canvas):
        self._width = size[0]
        self._height = size[1]
        self._plane_distance = plane_distance
        self._canvas = canvas

    def viewport_to_canvas(self, x, y) -> (float, float):
        return x * self._canvas.width / self._width, y * self._canvas.height / self._height


class Renderer:
    """Main renderer class that handles rendering of 3D scenes."""
    
    def __init__(self, canvas: Canvas, viewport: ViewPort, scene: dict, background_color=(32, 32, 32)):
        self._canvas = canvas
        self._viewport = viewport
        self._scene = scene
        self._background_color = np.array(background_color, dtype=int)

    def _transform_vertex(self, vertex: np.ndarray, transform: dict) -> np.ndarray:
        """Apply transformation to a vertex using homogeneous coordinates."""
        # Get transform components
        translation = transform.get('translation', np.zeros(3))
        rotation = transform.get('rotation', np.zeros(3))
        scale = transform.get('scale', np.ones(3))
        
        # Get camera transform if present
        has_camera = 'camera' in self._scene
        if has_camera:
            camera_transform = self._scene['camera'].get('transform', {})
            camera_translation = camera_transform.get('translation', np.zeros(3))
            camera_rotation = camera_transform.get('rotation', np.zeros(3))
        else:
            camera_translation = np.zeros(3)
            camera_rotation = np.zeros(3)
        
        # Call JIT-compiled function
        return transform_vertex(
            vertex, translation, rotation, scale,
            camera_translation, camera_rotation, has_camera
        )

    def _project_vertex(self, vertex, position=None):
        """Project a vertex from 3D to 2D screen coordinates."""
        if position is None:
            x, y, z = vertex
        else:
            x, y, z = position
        
        # Early exit if behind camera
        if z <= 0:
            return None
        
        # Project to viewport
        x_proj = x / z
        y_proj = y / z
        
        # Convert to canvas coordinates
        x_canvas, y_canvas = self._viewport.viewport_to_canvas(x_proj, y_proj)
        return int(x_canvas), int(y_canvas)

    def draw_line(self, start: (float, float), end: (float, float), color: (int, int, int)):
        """Draw a line using Bresenham's algorithm for better performance."""
        draw_line(
            int(start[0]), int(start[1]), 
            int(end[0]), int(end[1]), 
            self._canvas.grid, 
            self._canvas._center[0], self._canvas._center[1],
            color[0], color[1], color[2],
            self._canvas.width, self._canvas.height)

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
                self._canvas.width, self._canvas.height)
        else:
            # Draw outline using lines
            self.draw_line(p1, p2, color)
            self.draw_line(p2, p3, color)
            self.draw_line(p3, p1, color)

    def draw_shaded_triangle(self, p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float],
                           color: tuple[int, int, int],
                           intensity1: float, intensity2: float, intensity3: float):
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
            self._canvas.grid, 
            self._canvas._center[0], self._canvas._center[1],
            color[0], color[1], color[2],
            i1, i2, i3,
            self._canvas.width, self._canvas.height)

    def render(self, scene: dict):
        """Render a scene containing models and their instances."""
        # Clear canvas
        self._canvas.grid[:] = self._background_color

        # Get camera transform if present
        camera_transform = scene.get('camera', {}).get('transform', {})

        # Render each model instance
        for model_name, model in scene.get('models', {}).items():
            # Get model data
            vertices = model.get('vertices', [])
            triangles = model.get('triangles', [])
            
            # Render each instance of the model
            for instance in scene.get('instances', []):
                if instance.get('model') != model_name:
                    continue
                    
                # Get instance transform and color
                transform = instance.get('transform', {})
                color = instance.get('color', (255, 255, 255))
                
                # Transform vertices
                transformed_vertices = []
                for vertex in vertices:
                    # Apply model transform
                    transformed = self._transform_vertex(vertex, transform)
                    
                    # Project to 2D
                    projected = self._project_vertex(transformed)
                    if projected is not None:
                        transformed_vertices.append(projected)
                    else:
                        transformed_vertices.append(None)
                
                # Draw triangles
                for triangle in triangles:
                    # Get triangle vertices
                    v1 = transformed_vertices[triangle[0]]
                    v2 = transformed_vertices[triangle[1]]
                    v3 = transformed_vertices[triangle[2]]
                    
                    # Skip if any vertex is behind camera
                    if v1 is None or v2 is None or v3 is None:
                        continue
                    
                    # Draw triangle
                    self.draw_triangle(v1, v2, v3, color, fill=False)
