"""
Core components of the Pixerise rendering engine.
This module contains the main classes for rendering: Canvas, ViewPort, and Renderer.
"""

from typing import Tuple, Optional
import numpy as np
from kernel.rasterizing_mod import (
    draw_line,
    draw_flat_triangle,
    draw_shaded_triangle,
    draw_triangle,
)
from kernel.transforming_mod import (
    transform_vertex_normal,
    transform_vertices_and_normals,
)
from kernel.clipping_mod import calculate_bounding_sphere
from kernel.culling_mod import cull_back_faces
from kernel.rendering_mod import process_and_draw_triangles
from enum import Enum
from scene import Scene
from kernel.raycasting_mod import check_ray_triangles_intersection


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
        # Initialize buffers
        self.color_buffer = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        # Initialize depth buffer with 0 (1/∞) for 1/z depth testing
        self.depth_buffer = np.zeros((self.width, self.height), dtype=np.float32)
        # Calculate center-based coordinates
        self.half_width = self.width // 2
        self.half_height = self.height // 2
        self._center = (self.half_width, self.half_height)

    def clear(self, color: Tuple[int, int, int]):
        """Clear the canvas and reset the z-buffer.

        Resets both the color buffer to the specified color and the depth-buffer
        to infinity, preparing the canvas for a new frame.

        Args:
            color (Tuple[int, int, int], optional): RGB color to fill the canvas.
                Each component should be in range [0, 255].
        """
        self.depth_buffer.fill(0)  # Reset depth buffer to 0 (1/∞) for 1/z depth testing
        self.color_buffer[:, :] = color


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
    from origin, stored in a numpy array format for efficient JIT processing.

    Attributes:
        _width (float): Width of the viewport
        _height (float): Height of the viewport
        _plane_distance (float): Distance to the near plane
        _canvas (Canvas): Reference to the target canvas
        frustum_planes (np.ndarray): Array of shape (N, 4) containing frustum planes,
            where each row is [nx, ny, nz, d] representing the plane equation nx*x + ny*y + nz*z + d = 0
    """

    def __init__(
        self, size: Tuple[float, float], plane_distance: float, canvas: Canvas
    ):
        """Initialize a new ViewPort instance.

        Args:
            size (Tuple[float, float]): Dimensions of the viewport (width, height)
            plane_distance (float): Distance to the near plane from the camera
            canvas (Canvas): Target canvas for rendering
        """
        self.width = size[0]
        self.height = size[1]
        self.plane_distance = max(1.0, plane_distance)  # Ensure positive plane distance
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

        The planes are stored in a numpy array format for efficient JIT processing,
        where each row is [nx, ny, nz, d] representing the plane equation nx*x + ny*y + nz*z + d = 0
        """
        # Calculate half-dimensions at the near plane for plane equations
        half_width = self.width / 2
        half_height = self.height / 2

        # Calculate plane normals with correct orientation (pointing inward)
        # Each normal is computed based on the plane's orientation in view space
        self._left_plane = np.array(
            [1, 0, half_width / self.plane_distance], dtype=np.float32
        )
        self._right_plane = np.array(
            [-1, 0, half_width / self.plane_distance], dtype=np.float32
        )
        self._top_plane = np.array(
            [0, -1, half_height / self.plane_distance], dtype=np.float32
        )
        self._bottom_plane = np.array(
            [0, 1, half_height / self.plane_distance], dtype=np.float32
        )
        self._near_plane = np.array(
            [0, 0, 1], dtype=np.float32
        )  # Points towards viewer

        # Normalize all plane normals for consistent distance calculations
        self._left_plane /= np.linalg.norm(self._left_plane)
        self._right_plane /= np.linalg.norm(self._right_plane)
        self._top_plane /= np.linalg.norm(self._top_plane)
        self._bottom_plane /= np.linalg.norm(self._bottom_plane)
        self._near_plane /= np.linalg.norm(self._near_plane)

        # Create a numpy array to store all frustum planes in the format [nx, ny, nz, d]
        self.frustum_planes = np.zeros((5, 4), dtype=np.float32)

        # Store the planes in the optimized format
        planes = [
            (self._left_plane, 0),
            (self._right_plane, 0),
            (self._top_plane, 0),
            (self._bottom_plane, 0),
            (self._near_plane, -self.plane_distance),
        ]

        for i, (normal, distance) in enumerate(planes):
            self.frustum_planes[i, :3] = normal
            self.frustum_planes[i, 3] = distance


class Renderer:
    """A high-performance 3D renderer using NumPy and Numba JIT compilation.

    The Renderer class implements a complete 3D rendering pipeline with the following features:
    - Multiple shading modes (Wireframe, Flat, Gouraud)
    - View frustum culling with bounding spheres
    - Backface culling for performance optimization
    - Directional lighting with ambient and diffuse components
    - Efficient batch processing of vertices and normals
    - JIT-compiled core functions for maximum performance

    The rendering pipeline follows these main steps:
    1. Scene Setup:
       - Process scene graph with models, instances, camera, and lights
       - Configure viewport and canvas for output

    2. Geometry Processing:
       - Transform vertices from model to world space
       - Apply camera transformations to reach camera space
       - Perform view frustum culling using bounding spheres

    3. Rasterization:
       - Project visible triangles to screen space
       - Apply backface culling
       - Rasterize triangles with the selected shading mode

    4. Shading:
       - Calculate lighting based on surface normals and light direction
       - Apply shading model (flat or smooth)
       - Handle depth testing and pixel output

    Performance Optimizations:
    - Pre-computed frustum planes in optimized format
    - Batch processing of vertex transformations
    - JIT-compiled core rendering functions
    - Early culling of invisible geometry
    - Efficient memory layout for vertex data

    Attributes:
        _canvas (Canvas): Target canvas for rendering output
        _viewport (ViewPort): Viewport configuration and frustum planes
        _background_color (Tuple[int, int, int]): RGB color for canvas clear
    """

    def __init__(
        self, canvas: Canvas, viewport: ViewPort, background_color=(24, 24, 24)
    ):
        self._canvas = canvas
        self._viewport = viewport
        self._background_color = np.array(background_color, dtype=int)

    def draw_line(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        color: Tuple[int, int, int],
    ):
        """Draw a line using Bresenham's algorithm with depth buffering.

        Args:
            start: Starting point as (x, y, z) tuple
            end: Ending point as (x, y, z) tuple
            color: RGB color as (r, g, b) tuple
        """
        draw_line(
            int(start[0]),
            int(start[1]),
            float(start[2]),
            int(end[0]),
            int(end[1]),
            float(end[2]),
            self._canvas.color_buffer,
            self._canvas.depth_buffer,
            self._canvas._center[0],
            self._canvas._center[1],
            color[0],
            color[1],
            color[2],
            self._canvas.width,
            self._canvas.height,
        )

    def draw_triangle(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        p3: Tuple[float, float, float],
        color: Tuple[int, int, int],
        fill: bool = True,
    ):
        """Draw a triangle defined by three points. If fill is True, the triangle will be filled,
        otherwise only the outline will be drawn."""
        if fill:
            draw_flat_triangle(
                int(p1[0]),
                int(p1[1]),
                p1[2],
                int(p2[0]),
                int(p2[1]),
                p2[2],
                int(p3[0]),
                int(p3[1]),
                p3[2],
                self._canvas.color_buffer,
                self._canvas.depth_buffer,
                self._canvas._center[0],
                self._canvas._center[1],
                color[0],
                color[1],
                color[2],
                self._canvas.width,
                self._canvas.height,
            )
        else:
            draw_triangle(
                int(p1[0]),
                int(p1[1]),
                p1[2],
                int(p2[0]),
                int(p2[1]),
                p2[2],
                int(p3[0]),
                int(p3[1]),
                p3[2],
                self._canvas.color_buffer,
                self._canvas.depth_buffer,
                self._canvas._center[0],
                self._canvas._center[1],
                color[0],
                color[1],
                color[2],
                self._canvas.width,
                self._canvas.height,
            )

    def draw_shaded_triangle(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        p3: Tuple[float, float, float],
        color: Tuple[int, int, int],
        intensity1: float,
        intensity2: float,
        intensity3: float,
    ):
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
        # Delegate to the optimized JIT-compiled implementation
        draw_shaded_triangle(
            int(p1[0]),
            int(p1[1]),
            p1[2],  # Convert x,y to integers, pass z as float
            int(p2[0]),
            int(p2[1]),
            p2[2],  # Pass z-coordinate for second vertex
            int(p3[0]),
            int(p3[1]),
            p3[2],  # Pass z-coordinate for third vertex
            self._canvas.color_buffer,
            self._canvas.depth_buffer,  # Target canvas buffers
            self._canvas._center[0],
            self._canvas._center[1],  # Canvas center for coordinate transformation
            color[0],
            color[1],
            color[2],  # RGB components
            intensity1,
            intensity2,
            intensity3,  # Clamped intensity values
            self._canvas.width,
            self._canvas.height,
        )  # Canvas dimensions

    def cast_ray(
        self, screen_x: int, screen_y: int, scene: Scene
    ) -> Optional[Tuple[str, str]]:
        """Cast a ray from the camera through a screen point and find the first hit triangle.

        The ray casting process involves these steps:
        1. Screen Space -> Viewport Space: Convert pixel coordinates to normalized viewport coordinates
        2. Create ray in camera space (origin at 0,0,0, direction through viewport point)
        3. For each instance:
           - Transform vertices to camera space
           - Check bounding sphere intersection for early culling
           - Test ray intersection with transformed triangles
           - Track closest intersection

        Args:
            screen_x: X coordinate in screen space (pixels from left)
            screen_y: Y coordinate in screen space (pixels from top)
            scene: Scene containing models and instances to test against

        Returns:
            Optional tuple of (instance_name, group_name) of the first hit triangle,
            or None if no triangle was hit
        """

        # Check if screen coordinates are within canvas bounds
        if (
            screen_x < 0
            or screen_x >= self._canvas.width
            or screen_y < 0
            or screen_y >= self._canvas.height
        ):
            print(f"Screen coordinates ({screen_x}, {screen_y}) outside canvas bounds")
            return None

        # Convert screen coordinates to viewport space
        # X coordinate needs to be inverted to match viewport orientation
        viewport_x = ((self._canvas.width - screen_x) / self._canvas.width) * 2 - 1
        viewport_y = (screen_y / self._canvas.height) * 2 - 1

        # Create ray direction in camera space
        ray_dir = np.array(
            [
                viewport_x * (self._viewport.width / 2),
                viewport_y * (self._viewport.height / 2),
                -1.0,
            ],
            dtype=np.float32,
        )
        ray_dir /= np.linalg.norm(ray_dir)  # Normalize direction

        # Ray origin is camera position (0,0,0 in camera space)
        ray_origin = np.zeros(3, dtype=np.float32)

        # Track closest hit
        closest_hit = None
        closest_t = np.inf

        # Test all instances in scene
        for instance_name, instance in scene._instances.items():
            model = scene.get_model(instance.model)
            if model is None:
                continue

            # Process each group in the model
            for group_name, group in model.groups.items():
                vertices = group.vertices
                vertex_normals = group.vertex_normals

                # Pad vertex normals with zeros if they are None
                vertex_normals = (
                    np.zeros((0, 3), dtype=np.float32)
                    if vertex_normals is None or len(vertex_normals) == 0
                    else vertex_normals
                )

                # Transform group vertices to camera space
                # Transform vertices and normals using the new JIT-compiled function
                transformed_vertices, _ = transform_vertices_and_normals(
                    vertices,
                    vertex_normals,
                    instance.translation,
                    instance.rotation,
                    instance.scale,
                    scene._camera.translation,
                    scene._camera.rotation,
                    "",
                    False,
                )
                # Calculate bounding sphere for the transformed group vertices
                sphere_center, sphere_radius = calculate_bounding_sphere(
                    transformed_vertices
                )

                # Check visibility against each frustum plane
                fully_invisible = False

                for plane, plane_d in zip(
                    self._viewport.frustum_planes[:, :3],
                    self._viewport.frustum_planes[:, 3],
                ):
                    # Calculate signed distance from sphere center to plane
                    center_distance = np.dot(plane, sphere_center) + plane_d

                    # If center distance is less than -radius, sphere is completely behind plane
                    if center_distance < -sphere_radius:
                        fully_invisible = True
                        break

                # Skip if group is completely invisible
                if fully_invisible:
                    continue

                # Check intersection with triangles in this group
                hit, t = check_ray_triangles_intersection(
                    ray_origin, ray_dir, transformed_vertices, group.triangles
                )

                if hit and t < closest_t:
                    closest_hit = (instance_name, group_name)
                    closest_t = t

        return closest_hit

    def render(self, scene: Scene, shading_mode: ShadingMode = ShadingMode.WIREFRAME):
        """Render a 3D scene using the specified shading mode.

        This method performs the complete rendering pipeline for a 3D scene:
        1. Transforms vertices and normals from model to camera space
        2. Performs view frustum culling using bounding spheres
        3. Applies backface culling to optimize rendering
        4. Projects visible triangles to screen space
        5. Applies the specified shading mode with directional lighting

        The rendering process is optimized using JIT-compiled functions for:
        - Batch processing of vertex and normal transformations
        - Efficient view frustum culling with pre-computed planes
        - Fast triangle processing and rasterization

        Args:
            scene (Scene): Scene object containing models, instances, camera, and lighting
            shading_mode (ShadingMode): Rendering mode to use. Options are:
                - WIREFRAME: Only render triangle edges
                - FLAT: Single color per triangle with basic lighting
                - GOURAUD: Smooth shading with per-vertex lighting interpolation
        """
        # Clear canvas
        self._canvas.clear(tuple(self._background_color))

        # Transform light direction into camera space
        light_dir = transform_vertex_normal(
            -scene._directional_light.direction,
            np.zeros(3, dtype=np.float32),
            scene._camera.rotation,
            True,
        )

        # Render each instance
        for instance in scene._instances.values():
            model = scene.get_model(instance.model)
            if model is None:
                continue

            # Transform vertices for each model group
            for group_name, group in model.groups.items():
                # Skip invisible groups
                if not instance.get_group_visibility(group_name):
                    continue

                vertices = group.vertices
                triangles = group.triangles
                vertex_normals = group.vertex_normals
                has_vertex_normals = vertex_normals is not None and len(
                    vertex_normals
                ) == len(vertices)

                # Get group-specific color if set, otherwise use instance color
                color = instance.get_group_color(group_name)
                if color is None:
                    color = instance.color

                # Pad vertex normals with zeros if they are None
                vertex_normals = (
                    np.zeros((0, 3), dtype=np.float32)
                    if vertex_normals is None or len(vertex_normals) == 0
                    else vertex_normals
                )

                # Transform vertices and normals using the new JIT-compiled function
                transformed_vertices, transformed_normals = (
                    transform_vertices_and_normals(
                        vertices,
                        vertex_normals,
                        instance.translation,
                        instance.rotation,
                        instance.scale,
                        scene._camera.translation,
                        scene._camera.rotation,
                        shading_mode.value,
                        has_vertex_normals,
                    )
                )

                # Calculate bounding sphere for the entire instance
                sphere_center, sphere_radius = calculate_bounding_sphere(
                    transformed_vertices
                )

                # Check visibility against each frustum plane
                fully_visible = True
                fully_invisible = False

                for plane, plane_d in zip(
                    self._viewport.frustum_planes[:, :3],
                    self._viewport.frustum_planes[:, 3],
                ):
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

                # Convert triangle indices to numpy array
                triangles_array = np.array(triangles, dtype=np.int32)

                # Perform backface culling and get normals
                triangles_array, triangle_normals = cull_back_faces(
                    transformed_vertices, triangles_array
                )

                # Process all triangles in a batch using JIT-compiled function
                process_and_draw_triangles(
                    triangles_array,
                    transformed_vertices,
                    triangle_normals,
                    transformed_normals,
                    shading_mode.value,
                    has_vertex_normals,
                    fully_visible,
                    self._viewport.frustum_planes,
                    self._canvas.width,
                    self._canvas.height,
                    self._viewport.width,
                    self._viewport.height,
                    self._canvas.color_buffer,
                    self._canvas.depth_buffer,
                    self._canvas._center[0],
                    self._canvas._center[1],
                    color,  # Use group-specific color if available
                    light_dir,
                    scene._directional_light.ambient,
                )
