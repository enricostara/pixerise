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
from kernel.culling_mod import cull_back_faces
from kernel.shading_mod import triangle_flat_shading
from typing import Tuple
from enum import Enum

class ShadingMode(Enum):
    """Enum defining different shading modes for 3D rendering.
    
    Available modes:
    - FLAT: Uses a single normal per face for constant shading across the triangle
    - WIREFRAME: Renders only the edges of triangles without filling
    """
    FLAT = "flat"
    WIREFRAME = "wireframe"

class Canvas:
    """A 2D canvas for drawing pixels and managing the drawing surface.
    
    The Canvas class provides a fundamental drawing surface for the rendering engine.
    It manages both the color buffer and depth buffer (zbuffer) for proper
    3D rendering with depth testing.
    
    Attributes:
        size (Tuple[int, int]): Canvas dimensions as (width, height)
        width (int): Canvas width in pixels
        height (int): Canvas height in pixels
        color_buffer (np.ndarray): 3D array of shape (width, height, 3) storing RGB values
        depth_buffer (np.ndarray): 2D array of shape (width, height) storing depth values
        half_width (int): Half of canvas width, used for center-based coordinates
        half_height (int): Half of canvas height, used for center-based coordinates
        _center (Tuple[int, int]): Canvas center point coordinates
    """
    
    def __init__(self, size: Tuple[int, int] = (800, 600)):
        """Initialize a new Canvas instance.
        
        Args:
            size (Tuple[int, int], optional): Canvas dimensions (width, height).
                Defaults to (800, 600).
        """
        self.size = size
        self.width = size[0]
        self.height = size[1]
        # Initialize color buffer with dark gray background (column-major order)
        self.color_buffer = np.ones((self.width, self.height, 3), dtype=np.uint8) * 32
        # Initialize z-buffer with infinity for depth testing
        self.depth_buffer = np.full((self.width, self.height), np.inf, dtype=np.float32)
        # Calculate center-based coordinates
        self.half_width = self.width // 2
        self.half_height = self.height // 2
        self._center = (self.half_width, self.half_height)

    def clear(self, color: Tuple[int, int, int] = (32, 32, 32)):
        """Clear the canvas and reset the z-buffer.
        
        Resets both the color buffer to the specified color and the depth-buffer
        to infinity, preparing the canvas for a new frame.
        
        Args:
            color (Tuple[int, int, int], optional): RGB color to fill the canvas.
                Each component should be in range [0, 255].
                Defaults to dark gray (32, 32, 32).
        """
        self.color_buffer.fill(0)
        self.color_buffer[:, :] = color
        self.depth_buffer.fill(np.inf)  # Reset z-buffer for new frame


class ViewPort:
    """Manages the view frustum and coordinate transformations from viewport to canvas space.
    
    The ViewPort class handles the 3D viewing volume (frustum) and provides methods for
    transforming coordinates between viewport and canvas space. It pre-calculates the
    frustum planes for efficient view frustum culling during rendering.
    
    The view frustum is defined by five planes:
        - Left and Right planes
        - Top and Bottom planes
        - Near plane (at the specified plane distance)
    
    Each frustum plane is represented by its normal vector (pointing inward) and distance
    from origin, stored in the format (normal_vector, distance).
    
    Attributes:
        _width (float): Width of the viewport
        _height (float): Height of the viewport
        _plane_distance (float): Distance to the near plane
        _canvas (Canvas): Reference to the target canvas
        frustum_planes (List[Tuple[np.ndarray, float]]): List of frustum planes,
            each defined by (normal_vector, distance)
    """

    def __init__(self, size: Tuple[float, float], plane_distance: float, canvas: Canvas):
        """Initialize a new ViewPort instance.
        
        Args:
            size (Tuple[float, float]): Dimensions of the viewport (width, height)
            plane_distance (float): Distance to the near plane from the camera
            canvas (Canvas): Target canvas for rendering
        """
        self._width = size[0]
        self._height = size[1]
        self._plane_distance = max(1.0, plane_distance) # Ensure positive plane distance
        self._canvas = canvas
        
        # Initialize frustum planes for view frustum culling
        self._calculate_frustum_planes()
        
    def _calculate_frustum_planes(self):
        """Calculate the view frustum plane normals and distances.
        
        This method computes the normal vectors and distances for all frustum planes.
        The normals point inward and are normalized for efficient plane-point tests.
        The frustum is defined in view space, where:
            - X-axis points right
            - Y-axis points up
            - Z-axis points away from the viewer (into the screen)
        """
        # Calculate half-dimensions at the near plane for plane equations
        half_width = self._width / 2
        half_height = self._height / 2
        
        # Calculate plane normals with correct orientation (pointing inward)
        # Each normal is computed based on the plane's orientation in view space
        self._left_plane = np.array([1, 0, half_width / self._plane_distance], dtype=np.float64)
        self._right_plane = np.array([-1, 0, half_width / self._plane_distance], dtype=np.float64)
        self._top_plane = np.array([0, -1, half_height / self._plane_distance], dtype=np.float64)
        self._bottom_plane = np.array([0, 1, half_height / self._plane_distance], dtype=np.float64)
        self._near_plane = np.array([0, 0, 1], dtype=np.float64)  # Points towards viewer
        
        # Normalize all plane normals for consistent distance calculations
        self._left_plane /= np.linalg.norm(self._left_plane)
        self._right_plane /= np.linalg.norm(self._right_plane)
        self._top_plane /= np.linalg.norm(self._top_plane)
        self._bottom_plane /= np.linalg.norm(self._bottom_plane)
        self._near_plane /= np.linalg.norm(self._near_plane)

        # Store frustum planes as (normal, distance) pairs
        # Distance is 0 for side planes (they pass through origin)
        # Near plane distance is negative as it's measured from origin
        self.frustum_planes = [
            (self._left_plane, 0),
            (self._right_plane, 0),
            (self._top_plane, 0),
            (self._bottom_plane, 0),
            (self._near_plane, -self._plane_distance)
        ]
    
    def viewport_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        """Transform viewport coordinates to canvas coordinates.
        
        Converts coordinates from viewport space to canvas space by applying
        appropriate scaling based on the relative dimensions of viewport and canvas.
        
        Args:
            x (float): X-coordinate in viewport space
            y (float): Y-coordinate in viewport space
            
        Returns:
            Tuple[float, float]: Transformed coordinates in canvas space
        """
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

    def _project_vertex(self, vertex):
        """Project a vertex from 3D to 2D screen coordinates."""
        x, y, z = vertex
        
        # Early exit if behind camera
        if z <= 0:
            return None
        
        # Project to viewport
        x_proj = x / z
        y_proj = y / z
        
        # Convert to canvas coordinates
        x_canvas, y_canvas = self._viewport.viewport_to_canvas(x_proj, y_proj)
        return x_canvas, y_canvas, z

    def draw_line(self, start: Tuple[float, float, float], end: Tuple[float, float, float], color: Tuple[int, int, int]):
        """Draw a line using Bresenham's algorithm with depth buffering.
        
        Args:
            start: Starting point as (x, y, z) tuple
            end: Ending point as (x, y, z) tuple
            color: RGB color as (r, g, b) tuple
        """
        draw_line(
            int(start[0]), int(start[1]), float(start[2]),
            int(end[0]), int(end[1]), float(end[2]),
            self._canvas.color_buffer, self._canvas.depth_buffer,
            self._canvas._center[0], self._canvas._center[1],
            color[0], color[1], color[2],
            self._canvas.width, self._canvas.height)

    def draw_triangle(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float], p3: Tuple[float, float, float], color: Tuple[int, int, int], fill: bool = True):
        """Draw a triangle defined by three points. If fill is True, the triangle will be filled,
        otherwise only the outline will be drawn."""
        if fill:
            draw_triangle(
                int(p1[0]), int(p1[1]), p1[2],
                int(p2[0]), int(p2[1]), p2[2],
                int(p3[0]), int(p3[1]), p3[2],
                self._canvas.color_buffer, self._canvas.depth_buffer,
                self._canvas._center[0], self._canvas._center[1],
                color[0], color[1], color[2],
                self._canvas.width, self._canvas.height)
        else:
            # Draw outline using lines
            self.draw_line(p1, p2, color)
            self.draw_line(p2, p3,  color)
            self.draw_line(p3, p1, color)

    def draw_shaded_triangle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float],
                           color: Tuple[int, int, int],
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
            int(p1[0]), int(p1[1]), 0.0,  # Convert vertex coordinates to integers, add z=0
            int(p2[0]), int(p2[1]), 0.0,  # Add z=0 for second vertex
            int(p3[0]), int(p3[1]), 0.0,  # Add z=0 for third vertex
            self._canvas.color_buffer, self._canvas.depth_buffer,  # Target canvas buffers
            self._canvas._center[0], self._canvas._center[1],  # Canvas center for coordinate transformation
            color[0], color[1], color[2],  # RGB components
            i1, i2, i3,  # Clamped intensity values
            self._canvas.width, self._canvas.height)  # Canvas dimensions

    def render(self, scene: dict, shading_mode: ShadingMode = ShadingMode.WIREFRAME):
        """Render a scene containing models and their instances.
        
        Args:
            scene (dict): Scene data containing models and camera information
            shading_mode (ShadingMode): Rendering mode to use (FLAT or WIREFRAME)
        """
        # Clear canvas
        self._canvas.clear(tuple(self._background_color))

        # Get camera transform if present
        camera_transform = scene.get('camera', {}).get('transform', {})
        camera_pos = camera_transform.get('position', [0, 0, 0])
        camera_pos = np.array(camera_pos, dtype=np.float64)

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
                def project_and_draw_triangle(vertices, normal):
                    # Project vertices to 2D
                    v1 = self._project_vertex(vertices[0])
                    v2 = self._project_vertex(vertices[1])
                    v3 = self._project_vertex(vertices[2])
                    
                    # Skip if any vertex is behind camera
                    if v1 is None or v2 is None or v3 is None:
                        return
                    
                    # Draw triangle
                    if shading_mode == ShadingMode.WIREFRAME:
                        self.draw_triangle(v1, v2, v3, color, fill=False)
                    else:
                        # Compute shading using the provided normal
                        directional_light = scene['lights']['directional']
                        light_dir = -np.array(directional_light['direction'], dtype=np.float32)
                        color_array = np.array(color, dtype=np.float32)
                        shaded_color = triangle_flat_shading(vertices, normal,
                                                          light_dir, 
                                                          color_array, 
                                                          directional_light.get('ambient', 0.1))
                        
                        # Use the computed color for the triangle
                        final_color = tuple(shaded_color.astype(np.uint8))
                        self.draw_triangle(v1, v2, v3, final_color, fill=True)

                # Convert triangle indices to numpy array
                triangles_array = np.array(triangles, dtype=np.int32)
                # Perform backface culling and get normals
                visible_triangles, triangle_normals = cull_back_faces(vertices_array, triangles_array, camera_pos)
                # Filter out invisible triangles
                triangles_array = triangles_array[visible_triangles]

                # Draw triangles
                for i, triangle in enumerate(triangles_array):
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
                                for j in range(num_triangles):
                                    next_triangles.append(result_triangles[j])
                            clipped_triangles = next_triangles
                            if not clipped_triangles:  # Triangle completely clipped away
                                break
                        
                        # Project and draw the clipped triangles
                        for clipped_tri in clipped_triangles:
                            project_and_draw_triangle(clipped_tri, triangle_normals[i])
                    else:
                        # For fully visible instances, still need to check if vertices are behind camera
                        project_and_draw_triangle(triangle_vertices, triangle_normals[i])
