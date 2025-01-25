"""
Core components of the Pixerise rendering engine.
This module contains the main classes for rendering: Canvas, ViewPort, and Renderer.
"""

from typing import List, Tuple
import numpy as np
from kernel.rasterizing_mod import draw_line, draw_pixel, draw_flat_triangle, draw_shaded_triangle, draw_triangle
from kernel.transforming_mod import transform_vertex, transform_vertex_normal, project_vertex
from kernel.clipping_mod import (clip_triangle, calculate_bounding_sphere, clip_triangle_and_normals)
from kernel.culling_mod import cull_back_faces
from kernel.shading_mod import triangle_flat_shading, triangle_gouraud_shading
from enum import Enum
from scene import Scene

class ShadingMode(Enum):
    """Enum defining different shading modes for 3D rendering.
    
    Available modes:
    - FLAT: Uses a single normal per face for constant shading across the triangle
    - GOURAUD: Interpolates shading across the triangle using vertex normals
    - WIREFRAME: Renders only the edges of triangles without filling
    """
    FLAT = "flat"
    GOURAUD = "gouraud"
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
        self.width = size[0]
        self.height = size[1]
        self.plane_distance = max(1.0, plane_distance) # Ensure positive plane distance
        self.canvas = canvas
        
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
        half_width = self.width / 2
        half_height = self.height / 2
        
        # Calculate plane normals with correct orientation (pointing inward)
        # Each normal is computed based on the plane's orientation in view space
        self._left_plane = np.array([1, 0, half_width / self.plane_distance], dtype=np.float32)
        self._right_plane = np.array([-1, 0, half_width / self.plane_distance], dtype=np.float32)
        self._top_plane = np.array([0, -1, half_height / self.plane_distance], dtype=np.float32)
        self._bottom_plane = np.array([0, 1, half_height / self.plane_distance], dtype=np.float32)
        self._near_plane = np.array([0, 0, 1], dtype=np.float32)  # Points towards viewer
        
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
            (self._near_plane, -self.plane_distance)
        ]


class Renderer:
    """Main renderer class that handles rendering of 3D scenes."""
    
    def __init__(self, canvas: Canvas, viewport: ViewPort, background_color=(32, 32, 32)):
        self._canvas = canvas
        self._viewport = viewport
        self._background_color = np.array(background_color, dtype=int)

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
            draw_flat_triangle(
                int(p1[0]), int(p1[1]), p1[2],
                int(p2[0]), int(p2[1]), p2[2],
                int(p3[0]), int(p3[1]), p3[2],
                self._canvas.color_buffer, self._canvas.depth_buffer,
                self._canvas._center[0], self._canvas._center[1],
                color[0], color[1], color[2],
                self._canvas.width, self._canvas.height)
        else:
            draw_triangle(
                int(p1[0]), int(p1[1]), p1[2],
                int(p2[0]), int(p2[1]), p2[2],
                int(p3[0]), int(p3[1]), p3[2],
                self._canvas.color_buffer, self._canvas.depth_buffer,
                self._canvas._center[0], self._canvas._center[1],
                color[0], color[1], color[2],
                self._canvas.width, self._canvas.height)

    def draw_shaded_triangle(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float], p3: Tuple[float, float, float],
                           color: Tuple[int, int, int],
                           intensity1: float, intensity2: float, intensity3: float):
        """
        Draw a triangle with smooth shading using per-vertex intensity interpolation.
        This method implements Gouraud shading by interpolating intensity values across
        the triangle's surface.
        
        Args:
            p1, p2, p3: Vertex positions as (x, y, z) tuples in screen space coordinates.
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
            - Z-coordinates are used for depth testing to ensure correct visibility
        """
        # Clamp intensities to valid range [0.0, 1.0] to ensure correct color modulation
        i1 = max(0.0, min(1.0, intensity1))
        i2 = max(0.0, min(1.0, intensity2))
        i3 = max(0.0, min(1.0, intensity3))
        
        # Delegate to the optimized JIT-compiled implementation
        draw_shaded_triangle(
            int(p1[0]), int(p1[1]), p1[2],  # Convert x,y to integers, pass z as float
            int(p2[0]), int(p2[1]), p2[2],  # Pass z-coordinate for second vertex
            int(p3[0]), int(p3[1]), p3[2],  # Pass z-coordinate for third vertex
            self._canvas.color_buffer, self._canvas.depth_buffer,  # Target canvas buffers
            self._canvas._center[0], self._canvas._center[1],  # Canvas center for coordinate transformation
            color[0], color[1], color[2],  # RGB components
            i1, i2, i3,  # Clamped intensity values
            self._canvas.width, self._canvas.height)  # Canvas dimensions

    def render(self, scene: Scene, shading_mode: ShadingMode = ShadingMode.WIREFRAME):
        """Render a scene containing models and their instances.
        
        Args:
            scene (Scene): Scene object containing models and camera information
            shading_mode (ShadingMode): Rendering mode to use (FLAT, GOURAUD, or WIREFRAME)
        """
        # Clear canvas
        self._canvas.clear(tuple(self._background_color))

        # Render each instance
        for instance in scene.instances.values():
            model = scene.get_model(instance.model)
            if model is None:
                continue

            # Get instance transform and color
            color = instance.color
            
            # Get transform components
            translation = instance.translation
            rotation = instance.rotation
            scale = instance.scale
            
            # Get camera transform
            camera_translation = scene.camera.translation
            camera_rotation = scene.camera.rotation
            
            # Transform vertices for each model group
            for group in model.groups.values():
                vertices = group.vertices
                triangles = group.triangles
                vertex_normals = group.vertex_normals
                
                transformed_vertices = []
                transformed_normals = []  # Store transformed normals
                has_vertex_normals = len(vertex_normals) == len(vertices)
                
                for i, vertex in enumerate(vertices):
                    # Apply model transform to vertex
                    transformed = transform_vertex(vertex, translation, rotation, scale,
                                                camera_translation, camera_rotation, True)
                    transformed_vertices.append(transformed)
                    
                    # Transform normal using rotation only if available
                    if shading_mode == ShadingMode.GOURAUD and has_vertex_normals:
                        normal = vertex_normals[i]
                        transformed_normal = transform_vertex_normal(normal, rotation, camera_rotation, True)
                        transformed_normals.append(transformed_normal)
                
                # Convert transformed vertices to numpy array for bounding sphere calculation
                vertices_array = np.array(transformed_vertices, dtype=np.float32)
                
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
                
                # if not fully_visible:
                    # print(f"{group_name} fully visible: {fully_visible}")

                # Skip if instance is completely invisible
                if fully_invisible:
                    continue
                
                # Function to project and draw a triangle
                def project_and_draw_triangle(vertices, vertex_normals, shading_mode: ShadingMode):
                    # Project vertices to 2D
                    v1 = project_vertex(vertices[0], self._canvas.width, self._canvas.height,
                                             self._viewport.width, self._viewport.height)
                    v2 = project_vertex(vertices[1], self._canvas.width, self._canvas.height,
                                             self._viewport.width, self._viewport.height)
                    v3 = project_vertex(vertices[2], self._canvas.width, self._canvas.height,
                                             self._viewport.width, self._viewport.height)
                    
                    # Skip if any vertex is behind camera
                    if v1 is None or v2 is None or v3 is None:
                        return
                    
                    # Draw triangle
                    if shading_mode == ShadingMode.WIREFRAME:
                        draw_triangle(
                            int(v1[0]), int(v1[1]), v1[2],
                            int(v2[0]), int(v2[1]), v2[2],
                            int(v3[0]), int(v3[1]), v3[2],
                            self._canvas.color_buffer, self._canvas.depth_buffer,
                            self._canvas._center[0], self._canvas._center[1],
                            color[0], color[1], color[2],
                            self._canvas.width, self._canvas.height)
                    else:
                        light_dir = -scene.directional_light.direction
                        color_array = np.array(color, dtype=np.float32)
                        ambient = scene.directional_light.ambient
                        
                        if shading_mode == ShadingMode.GOURAUD and has_vertex_normals:
                            # Compute vertex intensities using vertex normals
                            intensities = triangle_gouraud_shading(vertex_normals, light_dir, ambient)
                            draw_shaded_triangle(
                                int(v1[0]), int(v1[1]), v1[2],
                                int(v2[0]), int(v2[1]), v2[2],
                                int(v3[0]), int(v3[1]), v3[2],
                                self._canvas.color_buffer, self._canvas.depth_buffer,
                                self._canvas._center[0], self._canvas._center[1],
                                color[0], color[1], color[2],
                                intensities[0], intensities[1], intensities[2],
                                self._canvas.width, self._canvas.height)
                        else:  # FLAT shading
                            flat_shaded_color = triangle_flat_shading(vertex_normals[0], light_dir, color_array, ambient)
                            draw_flat_triangle(
                                int(v1[0]), int(v1[1]), v1[2],
                                int(v2[0]), int(v2[1]), v2[2],
                                int(v3[0]), int(v3[1]), v3[2],
                                self._canvas.color_buffer, self._canvas.depth_buffer,
                                self._canvas._center[0], self._canvas._center[1],
                                int(flat_shaded_color[0]), int(flat_shaded_color[1]), int(flat_shaded_color[2]),
                                self._canvas.width, self._canvas.height)

                # Convert triangle indices to numpy array
                triangles_array = np.array(triangles, dtype=np.int32)
                # Perform backface culling and get normals
                triangles_array, triangle_normals = cull_back_faces(vertices_array, triangles_array)

                # Draw triangles
                for i, triangle in enumerate(triangles_array):
                    # Get triangle vertices as numpy array
                    triangle_vertices = np.array([
                        transformed_vertices[triangle[0]],
                        transformed_vertices[triangle[1]],
                        transformed_vertices[triangle[2]]
                    ], dtype=np.float32)

                    # Get corresponding transformed normals for this triangle
                    if shading_mode == ShadingMode.GOURAUD and has_vertex_normals:
                        triangle_transformed_normals = np.array([
                            transformed_normals[triangle[0]],
                            transformed_normals[triangle[1]],
                            transformed_normals[triangle[2]]
                        ], dtype=np.float32)
                    else:
                        # If no vertex normals, use face normal for all vertices
                        triangle_transformed_normals = np.array([
                            triangle_normals[i],
                            triangle_normals[i],
                            triangle_normals[i]
                        ], dtype=np.float32)

                    if not fully_visible:
                        # Clip against each frustum plane
                        planes = self._viewport.frustum_planes
                        clipped_triangles = [triangle_vertices]
                        clipped_normals = [triangle_transformed_normals]
                        
                        if shading_mode == ShadingMode.GOURAUD and has_vertex_normals:
                            for plane in planes:
                                next_triangles = []
                                next_normals = []
                                for tri_idx, tri in enumerate(clipped_triangles):
                                    # Clip triangle against current plane
                                    result_triangles, result_normals, num_triangles = clip_triangle_and_normals(
                                        tri, clipped_normals[tri_idx], plane[0], plane[1]
                                    )
                                    # Add resulting triangles and their normals
                                    for j in range(num_triangles):
                                        next_triangles.append(result_triangles[j])
                                        next_normals.append(result_normals[j])
                                clipped_triangles = next_triangles
                                clipped_normals = next_normals
                                if not clipped_triangles:  # Triangle completely clipped away
                                    break
                            # Project and draw the clipped triangles
                            for i, clipped_tri in enumerate(clipped_triangles):
                                project_and_draw_triangle(clipped_tri, clipped_normals[i], shading_mode)

                        else:
                            for plane in planes:
                                next_triangles = []
                                for tri in clipped_triangles:
                                    clipped, num_triangles = clip_triangle(tri, plane[0], plane[1])
                                    for j in range(num_triangles):
                                        next_triangles.append(clipped[j])
                                clipped_triangles = next_triangles
                                if not clipped_triangles:  # Triangle completely clipped away
                                    break
                            # Project and draw the clipped triangles
                            for clipped_tri in clipped_triangles:
                                project_and_draw_triangle(clipped_tri, triangle_transformed_normals, shading_mode)

                    else:
                        # For fully visible instances, still need to check if vertices are behind camera
                        project_and_draw_triangle(triangle_vertices, triangle_transformed_normals, shading_mode)
