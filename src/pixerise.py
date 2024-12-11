"""
Core components of the Pixerise rendering engine.
This module contains the main classes for rendering: Canvas, ViewPort, and Renderer.
"""

import numpy as np
from numba import jit
import pygame
from kernel.rasterizing_mod import (draw_pixel, draw_line, draw_triangle, draw_shaded_triangle)
from kernel.transforming_mod import transform_vertex
from kernel.clipping_mod import (clip_triangle, calculate_bounding_sphere)

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
        
        # Pre-calculate frustum planes for efficiency
        self._calculate_frustum_planes()
        
    def _calculate_frustum_planes(self):
        """Calculate the view frustum plane normals (pointing inward)."""
        # Calculate half-dimensions at the near plane
        half_width = self._width / 2
        half_height = self._height / 2
        
        # Calculate plane normals (pointing inward)
        self._left_plane = np.array([1, 0, half_width / self._plane_distance], dtype=np.float64)
        self._right_plane = np.array([-1, 0, half_width / self._plane_distance], dtype=np.float64)
        self._top_plane = np.array([0, -1, half_height / self._plane_distance], dtype=np.float64)
        self._bottom_plane = np.array([0, 1, half_height / self._plane_distance], dtype=np.float64)

        # Define the near plane
        self._near_plane = np.array([0, 0, 1], dtype=np.float64)  # Normal pointing towards the viewer
        
        # Normalize the plane normals
        self._left_plane /= np.linalg.norm(self._left_plane)
        self._right_plane /= np.linalg.norm(self._right_plane)
        self._top_plane /= np.linalg.norm(self._top_plane)
        self._bottom_plane /= np.linalg.norm(self._bottom_plane)
        self._near_plane /= np.linalg.norm(self._near_plane)

        # Define frustum planes as a list of tuples with the D constant term
        self.frustum_planes = [
            (self._left_plane, 0),
            (self._right_plane, 0),
            (self._top_plane, 0),
            (self._bottom_plane, 0),
            (self._near_plane, -self._plane_distance)
        ]
    
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
        Draw a triangle with smooth shading using per-vertex intensity interpolation.
        This method implements Gouraud shading by interpolating intensity values across
        the triangle's surface.
        
        Args:
            p1, p2, p3: Vertex positions as (x, y) tuples in screen space coordinates.
                       The vertices can be in any order, they will be sorted internally.
            color: Base RGB color as (r, g, b) tuple, where each component is in range [0, 255].
                  This color will be modulated by the interpolated intensities.
            intensity1, intensity2, intensity3: Light intensity values for each vertex in range [0.0, 1.0].
                                              These values determine how bright the color appears at each vertex
                                              and are linearly interpolated across the triangle.
        
        Note:
            - Intensity values are automatically clamped to the valid range [0.0, 1.0] to ensure correct color modulation
            - The final color at each pixel is computed as: final_rgb = base_rgb * interpolated_intensity
            - The implementation uses a scanline algorithm with linear interpolation for efficiency
            - Triangles completely outside the canvas or with zero intensity are skipped
        """
        # Clamp intensities to valid range [0.0, 1.0] to ensure correct color modulation
        i1 = max(0.0, min(1.0, intensity1))
        i2 = max(0.0, min(1.0, intensity2))
        i3 = max(0.0, min(1.0, intensity3))
        
        # Delegate to the optimized JIT-compiled implementation
        draw_shaded_triangle(
            int(p1[0]), int(p1[1]),  # Convert vertex coordinates to integers
            int(p2[0]), int(p2[1]), 
            int(p3[0]), int(p3[1]), 
            self._canvas.grid,  # Target canvas buffer
            self._canvas._center[0], self._canvas._center[1],  # Canvas center for coordinate transformation
            color[0], color[1], color[2],  # RGB components
            i1, i2, i3,  # Clamped intensity values
            self._canvas.width, self._canvas.height)  # Canvas dimensions

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
                    transformed_vertices.append(transformed)
                
                # Convert transformed vertices to numpy array for bounding sphere calculation
                vertices_array = np.array(transformed_vertices, dtype=np.float64)
                
                # Calculate bounding sphere for the entire instance
                sphere_center, sphere_radius = calculate_bounding_sphere(vertices_array)
                
                # Check visibility against each frustum plane
                fully_visible = True
                fully_invisible = False
                
                for plane, plane_d in self._viewport.frustum_planes:
                    # Calculate signed distance from sphere center to plane
                    center_distance = np.dot(plane, sphere_center) + plane_d
                    
                    # If center distance is less than -radius, sphere is completely behind plane
                    if center_distance < -sphere_radius:
                        fully_invisible = True
                        break
                        
                    # If center distance is less than radius, sphere intersects plane
                    if abs(center_distance) < sphere_radius:
                        fully_visible = False
                
                # Skip if instance is completely invisible
                if fully_invisible:
                    continue
                
                # Function to project and draw a triangle
                def project_and_draw_triangle(vertices):
                    # Project vertices to 2D
                    v1 = self._project_vertex(vertices[0])
                    v2 = self._project_vertex(vertices[1])
                    v3 = self._project_vertex(vertices[2])
                    
                    # Skip if any vertex is behind camera
                    if v1 is None or v2 is None or v3 is None:
                        return
                    
                    # Draw triangle
                    self.draw_triangle(v1, v2, v3, color, fill=False)

                # Draw triangles
                for triangle in triangles:
                    # Get triangle vertices as numpy array
                    triangle_vertices = np.array([
                        transformed_vertices[triangle[0]],
                        transformed_vertices[triangle[1]],
                        transformed_vertices[triangle[2]]
                    ], dtype=np.float64)

                    if not fully_visible:
                        # Clip against each frustum plane
                        planes = self._viewport.frustum_planes
                        clipped_triangles = [triangle_vertices]
                        
                        for plane in planes:
                            next_triangles = []
                            for tri in clipped_triangles:
                                # Clip triangle against current plane
                                result_triangles, num_triangles = clip_triangle(tri, plane[0], plane[1])
                                # Add resulting triangles
                                for i in range(num_triangles):
                                    next_triangles.append(result_triangles[i])
                            clipped_triangles = next_triangles
                            if not clipped_triangles:  # Triangle completely clipped away
                                break
                        
                        # Project and draw the clipped triangles
                        for clipped_tri in clipped_triangles:
                            project_and_draw_triangle(clipped_tri)
                    else:
                        # For fully visible instances, still need to check if vertices are behind camera
                        project_and_draw_triangle(triangle_vertices)
