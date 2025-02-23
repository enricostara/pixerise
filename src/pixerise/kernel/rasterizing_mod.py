"""
JIT-compiled kernel functions for the Pixerise rasterizer.
These functions are optimized using Numba's JIT compilation for better performance.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def draw_pixel(
    color_buffer: np.ndarray,
    depth_buffer: np.ndarray,
    x: int,
    y: int,
    inv_z: float,
    center_x: int,
    center_y: int,
    color_r: int,
    color_g: int,
    color_b: int,
    width: int,
    height: int,
) -> None:
    """JIT-compiled pixel drawing function with depth testing using 1/z values.

    Args:
        color_buffer: RGB color buffer array of shape (width, height, 3)
        depth_buffer: Depth buffer array of shape (width, height)
        x, y: Pixel coordinates relative to canvas center
        inv_z: 1/z depth value of the pixel (larger values are closer)
        center_x, center_y: Canvas center coordinates
        color_r, color_g, color_b: RGB color components (0-255)
        width, height: Canvas dimensions

    Note:
        The depth buffer stores 1/z values instead of z values.
        Larger 1/z values mean the pixel is closer to the camera.
    """
    px = center_x + x
    py = center_y - y  # Flip y coordinate
    if 0 <= px < width and 0 <= py < height:
        if inv_z > depth_buffer[px, py]:  # Larger 1/z means closer to camera
            depth_buffer[px, py] = inv_z  # Update depth buffer
            color_buffer[px, py, 0] = (
                color_r  # Back to column-major order for pygame compatibility
            )
            color_buffer[px, py, 1] = color_g
            color_buffer[px, py, 2] = color_b


@njit(cache=True)
def draw_line(
    x0: int,
    y0: int,
    z0: float,
    x1: int,
    y1: int,
    z1: float,
    canvas_grid: np.ndarray,
    depth_buffer: np.ndarray,
    center_x: int,
    center_y: int,
    color_r: int,
    color_g: int,
    color_b: int,
    canvas_width: int,
    canvas_height: int,
) -> None:
    """
    Draw a line using Bresenham's line algorithm with integer arithmetic and depth buffering.
    This is a low-level, JIT-compiled implementation optimized for performance.

    The algorithm works by:
    1. Determining the primary direction (x or y) based on line slope
    2. Using integer arithmetic to track the error term for the secondary axis
    3. Interpolating z-values along the line for depth testing
    4. Drawing pixels with minimal floating-point operations
    5. Handling all octants with a single implementation

    Args:
        x0, y0, z0: Starting point coordinates and depth in screen space
        x1, y1, z1: Ending point coordinates and depth in screen space
        canvas_grid: Target numpy array for drawing (shape: [height, width, 3] for RGB)
        depth_buffer: Depth buffer array of shape (width, height)
        center_x, center_y: Canvas center coordinates for coordinate system transformation
        color_r, color_g, color_b: RGB color components (0-255)
        canvas_width, canvas_height: Dimensions of the canvas

    Implementation Notes:
        - Uses Bresenham's algorithm to avoid floating-point arithmetic
        - Handles all octants without special cases by swapping axes when needed
        - Interpolates z-values linearly along the line
        - Clips lines to canvas bounds for efficiency
        - Optimized for JIT compilation with numba
    """
    # Calculate absolute differences and determine primary direction
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    # Determine if x or y is the driving axis
    steep = dy > dx

    # If y is the driving axis, swap x and y coordinates
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        dx, dy = dy, dx

    # Ensure we always draw from left to right
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        z0, z1 = z1, z0

    # Calculate step direction for the secondary axis
    y_step = 1 if y0 < y1 else -1

    # Initialize Bresenham's error term
    error = dx // 2
    y = y0

    # Calculate total line length for z-interpolation
    total_steps = dx + 1 if dx > 0 else 1
    z_step = (z1 - z0) / total_steps if total_steps > 1 else 0

    # Main line drawing loop
    for i, x in enumerate(range(x0, x1 + 1)):
        # Interpolate z-value based on current position
        z = z0 + z_step * i
        # Convert to 1/z for depth testing
        eps = 1e-6  # Small epsilon to avoid division by zero
        inv_z = 1.0 / (z + eps)

        # If steep (y is primary), swap back coordinates for pixel drawing
        if steep:
            draw_pixel(
                canvas_grid,
                depth_buffer,
                y,
                x,
                inv_z,
                center_x,
                center_y,
                color_r,
                color_g,
                color_b,
                canvas_width,
                canvas_height,
            )
        else:
            draw_pixel(
                canvas_grid,
                depth_buffer,
                x,
                y,
                inv_z,
                center_x,
                center_y,
                color_r,
                color_g,
                color_b,
                canvas_width,
                canvas_height,
            )

        # Update error term and step y if needed
        error -= dy
        if error < 0:
            y += y_step
            error += dx


@njit(cache=True)
def draw_triangle(
    x0: int,
    y0: int,
    z0: float,
    x1: int,
    y1: int,
    z1: float,
    x2: int,
    y2: int,
    z2: float,
    canvas_grid: np.ndarray,
    depth_buffer: np.ndarray,
    center_x: int,
    center_y: int,
    color_r: int,
    color_g: int,
    color_b: int,
    canvas_width: int,
    canvas_height: int,
) -> None:
    """
    Draw a triangle outline using three line segments.
    This is a low-level, JIT-compiled implementation optimized for performance.

    Args:
        x0, y0, z0: First vertex coordinates and depth
        x1, y1, z1: Second vertex coordinates and depth
        x2, y2, z2: Third vertex coordinates and depth
        canvas_grid: Target numpy array for drawing (shape: [height, width, 3] for RGB)
        depth_buffer: Depth buffer array of shape (width, height)
        center_x, center_y: Canvas center coordinates for coordinate system transformation
        color_r, color_g, color_b: RGB color components (0-255)
        canvas_width, canvas_height: Dimensions of the canvas
    """
    # Draw three lines to form the triangle outline
    draw_line(
        x0,
        y0,
        z0,
        x1,
        y1,
        z1,
        canvas_grid,
        depth_buffer,
        center_x,
        center_y,
        color_r,
        color_g,
        color_b,
        canvas_width,
        canvas_height,
    )
    draw_line(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        canvas_grid,
        depth_buffer,
        center_x,
        center_y,
        color_r,
        color_g,
        color_b,
        canvas_width,
        canvas_height,
    )
    draw_line(
        x2,
        y2,
        z2,
        x0,
        y0,
        z0,
        canvas_grid,
        depth_buffer,
        center_x,
        center_y,
        color_r,
        color_g,
        color_b,
        canvas_width,
        canvas_height,
    )


@njit(cache=True)
def draw_flat_triangle(
    x0: int,
    y0: int,
    z0: float,
    x1: int,
    y1: int,
    z1: float,
    x2: int,
    y2: int,
    z2: float,
    canvas_grid: np.ndarray,
    depth_buffer: np.ndarray,
    center_x: int,
    center_y: int,
    color_r: int,
    color_g: int,
    color_b: int,
    canvas_width: int,
    canvas_height: int,
) -> None:
    """
    Draw a solid-colored triangle using a scanline rasterization algorithm with depth buffering.
    This is a low-level, JIT-compiled implementation optimized for performance.

    The algorithm works by:
    1. Converting to screen coordinates and checking canvas bounds
    2. Sorting vertices by y-coordinate for consistent edge traversal
    3. Interpolating depth values along edges and scanlines
    4. Rasterizing the triangle in two parts (upper and lower) using a scanline approach
    5. Using fixed-point arithmetic for sub-pixel precision

    Args:
        x0, y0, z0: First vertex coordinates and depth
        x1, y1, z1: Second vertex coordinates and depth
        x2, y2, z2: Third vertex coordinates and depth
        canvas_grid: Target numpy array for drawing (shape: [height, width, 3] for RGB)
        depth_buffer: Depth buffer array of shape (width, height)
        center_x, center_y: Canvas center coordinates for coordinate system transformation
        color_r, color_g, color_b: RGB color components (0-255)
        canvas_width, canvas_height: Dimensions of the canvas

    Implementation Notes:
        - Uses 16.16 fixed-point arithmetic for edge traversal to avoid floating-point errors
        - Handles edge cases like zero-height triangles and ensures non-zero denominators
        - Implements linear interpolation for smooth intensity gradients
        - Clips triangles to canvas bounds for efficiency
        - Automatically sorts vertices for consistent edge traversal
        - Includes early rejection tests for improved performance
    """
    # Transform from world space to screen space coordinates:
    # - Add center_x to shift from [-width/2, width/2] to [0, width]
    # - Subtract from center_y to flip Y axis (screen Y grows downward)
    sx0, sy0 = center_x + x0, center_y - y0
    sx1, sy1 = center_x + x1, center_y - y1
    sx2, sy2 = center_x + x2, center_y - y2

    # Compute triangle bounds for canvas clipping:
    # - If triangle is completely outside canvas bounds, we can skip it entirely
    # - This is a conservative test (bounding box may be larger than actual triangle)
    min_x = min(sx0, sx1, sx2)
    max_x = max(sx0, sx1, sx2)
    min_y = min(sy0, sy1, sy2)
    max_y = max(sy0, sy1, sy2)

    if max_x < 0 or min_x >= canvas_width or max_y < 0 or min_y >= canvas_height:
        return

    # Sort vertices by Y coordinate, keeping depth values aligned
    if y1 < y0:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        z0, z1 = z1, z0
    if y2 < y0:
        x0, x2 = x2, x0
        y0, y2 = y2, y0
        z0, z2 = z2, z0
    if y2 < y1:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        z1, z2 = z2, z1

    # Convert z values to 1/z for linear interpolation
    # Add small epsilon to avoid division by zero
    eps = 1e-6
    inv_z0 = 1.0 / (z0 + eps)
    inv_z1 = 1.0 / (z1 + eps)
    inv_z2 = 1.0 / (z2 + eps)

    # Initialize edge traversal with reciprocal depth values
    dx1 = x1 - x0
    dy1 = y1 - y0
    dinv_z1 = inv_z1 - inv_z0
    x_left = x0 << 16
    step_left = (dx1 << 16) // max(1, dy1)
    inv_z_left = inv_z0
    inv_z_step_left = dinv_z1 / max(1, dy1) if dy1 > 0 else 0.0

    dx2 = x2 - x0
    dy2 = y2 - y0
    dinv_z2 = inv_z2 - inv_z0
    x_right = x0 << 16
    step_right = (dx2 << 16) // max(1, dy2)
    inv_z_right = inv_z0
    inv_z_step_right = dinv_z2 / max(1, dy2) if dy2 > 0 else 0.0

    # Fill the upper triangle section
    for y in range(y0, y1):  # Changed to be top-inclusive, bottom-exclusive
        start_x = x_left >> 16
        end_x = x_right >> 16

        if start_x > end_x:
            start_x, end_x = end_x, start_x
            inv_z_scan = inv_z_right
            inv_z_step = (inv_z_left - inv_z_right) / max(1, end_x - start_x)
        else:
            inv_z_scan = inv_z_left
            inv_z_step = (inv_z_right - inv_z_left) / max(1, end_x - start_x)

        # Changed to be left-inclusive, right-exclusive
        for x in range(start_x, end_x):
            # Convert back to z for depth testing
            draw_pixel(
                canvas_grid,
                depth_buffer,
                x,
                y,
                inv_z_scan,
                center_x,
                center_y,
                color_r,
                color_g,
                color_b,
                canvas_width,
                canvas_height,
            )
            inv_z_scan += inv_z_step

        if y1 > y0:
            x_left += step_left
            x_right += step_right
            inv_z_left += inv_z_step_left
            inv_z_right += inv_z_step_right

    # Initialize edge traversal for lower triangle with reciprocal depth
    dx3 = x2 - x1
    dy3 = y2 - y1
    dinv_z3 = inv_z2 - inv_z1
    x_left = x1 << 16
    step_left = (dx3 << 16) // max(1, dy3)
    inv_z_left = inv_z1
    inv_z_step_left = dinv_z3 / max(1, dy3) if dy3 > 0 else 0.0

    # Fill the lower triangle section
    for y in range(y1, y2):  # Changed to be top-inclusive, bottom-exclusive
        start_x = x_left >> 16
        end_x = x_right >> 16

        if start_x > end_x:
            start_x, end_x = end_x, start_x
            inv_z_scan = inv_z_right
            inv_z_step = (inv_z_left - inv_z_right) / max(1, end_x - start_x)
        else:
            inv_z_scan = inv_z_left
            inv_z_step = (inv_z_right - inv_z_left) / max(1, end_x - start_x)

        # Changed to be left-inclusive, right-exclusive
        for x in range(start_x, end_x):
            # Convert back to z for depth testing
            draw_pixel(
                canvas_grid,
                depth_buffer,
                x,
                y,
                inv_z_scan,
                center_x,
                center_y,
                color_r,
                color_g,
                color_b,
                canvas_width,
                canvas_height,
            )
            inv_z_scan += inv_z_step

        if y2 > y1:
            x_left += step_left
            x_right += step_right
            inv_z_left += inv_z_step_left
            inv_z_right += inv_z_step_right


@njit(cache=True)
def draw_shaded_triangle(
    x0: int,
    y0: int,
    z0: float,
    x1: int,
    y1: int,
    z1: float,
    x2: int,
    y2: int,
    z2: float,
    canvas_grid: np.ndarray,
    depth_buffer: np.ndarray,
    center_x: int,
    center_y: int,
    color_r: int,
    color_g: int,
    color_b: int,
    i0: float,
    i1: float,
    i2: float,
    canvas_width: int,
    canvas_height: int,
) -> None:
    """
    Draw a shaded triangle using a scanline algorithm with linear interpolation for intensities.
    This is a low-level, JIT-compiled implementation optimized for performance.

    The algorithm works by:
    1. Early rejection of degenerate cases (zero intensity, black color, out of bounds)
    2. Converting to screen coordinates and checking canvas bounds
    3. Sorting vertices by y-coordinate for consistent edge traversal
    4. Rasterizing the triangle in two parts (upper and lower) using a scanline approach
    5. Interpolating intensities along edges and scanlines using fixed-point arithmetic

    Args:
        x0, y0, z0: First vertex coordinates and depth
        x1, y1, z1: Second vertex coordinates and depth
        x2, y2, z2: Third vertex coordinates and depth
        canvas_grid: Target numpy array for drawing (shape: [height, width, 3] for RGB)
        depth_buffer: Depth buffer array of shape (width, height)
        center_x, center_y: Canvas center coordinates for coordinate system transformation
        color_r, color_g, color_b: Base RGB color components (0-255)
        i0, i1, i2: Light intensity values for each vertex (0.0-1.0)
        canvas_width, canvas_height: Dimensions of the canvas

    Implementation Notes:
        - Uses 16.16 fixed-point arithmetic for edge traversal to avoid floating-point errors
        - Handles edge cases like zero-height triangles and ensures non-zero denominators
        - Implements linear interpolation for smooth intensity gradients
        - Clips triangles to canvas bounds for efficiency
        - Automatically sorts vertices for consistent edge traversal
        - Includes early rejection tests for improved performance
    """
    # Early rejection test for degenerate cases:
    # - Skip if all vertices have zero or near-zero intensity (would result in black triangle)
    # - Skip if the base color is black (would result in black triangle regardless of intensity)
    if max(i0, i1, i2) <= 0.001 or (color_r == 0 and color_g == 0 and color_b == 0):
        return

    # Clamp intensities to valid range [0.0, 1.0] to ensure correct color modulation
    i0 = max(0.0, min(1.0, i0))
    i1 = max(0.0, min(1.0, i1))
    i2 = max(0.0, min(1.0, i2))

    # Transform from world space to screen space coordinates:
    # - Add center_x to shift from [-width/2, width/2] to [0, width]
    # - Subtract from center_y to flip Y axis (screen Y grows downward)
    sx0, sy0 = center_x + x0, center_y - y0
    sx1, sy1 = center_x + x1, center_y - y1
    sx2, sy2 = center_x + x2, center_y - y2

    # Compute triangle bounds for canvas clipping:
    # - If triangle is completely outside canvas bounds, we can skip it entirely
    # - This is a conservative test (bounding box may be larger than actual triangle)
    min_x = min(sx0, sx1, sx2)
    max_x = max(sx0, sx1, sx2)
    min_y = min(sy0, sy1, sy2)
    max_y = max(sy0, sy1, sy2)

    if max_x < 0 or min_x >= canvas_width or max_y < 0 or min_y >= canvas_height:
        return

    # Sort vertices by Y coordinate to split triangle into upper and lower parts:
    # - This creates a consistent traversal order regardless of input vertex order
    # - Each swap must also swap the corresponding intensity values to maintain mapping
    if y1 < y0:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        z0, z1 = z1, z0
        i0, i1 = i1, i0
    if y2 < y0:
        x0, x2 = x2, x0
        y0, y2 = y2, y0
        z0, z2 = z2, z0
        i0, i2 = i2, i0
    if y2 < y1:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        z1, z2 = z2, z1
        i1, i2 = i2, i1

    # Convert z values to 1/z for linear interpolation
    # Add small epsilon to avoid division by zero
    eps = 1e-6
    inv_z0 = 1.0 / (z0 + eps)
    inv_z1 = 1.0 / (z1 + eps)
    inv_z2 = 1.0 / (z2 + eps)

    # Initialize edge traversal for the first two edges (from top vertex)
    # Edge 1: y0 to y1 (left or right edge of upper triangle)
    dx1 = x1 - x0
    dy1 = y1 - y0
    dinv_z1 = inv_z1 - inv_z0
    # Convert x-coordinate to 16.16 fixed-point for sub-pixel precision
    x_left = x0 << 16
    # Calculate x-step in fixed-point, ensuring non-zero denominator
    step_left = (dx1 << 16) // max(1, dy1)
    # Initialize intensity interpolation for this edge
    i_left = i0
    i_step_left = (i1 - i0) / max(1, dy1)
    inv_z_left = inv_z0
    inv_z_step_left = dinv_z1 / max(1, dy1) if dy1 > 0 else 0.0

    # Edge 2: y0 to y2 (spans full height of triangle)
    dx2 = x2 - x0
    dy2 = y2 - y0
    dinv_z2 = inv_z2 - inv_z0
    x_right = x0 << 16
    step_right = (dx2 << 16) // max(1, dy2)
    i_right = i0
    i_step_right = (i2 - i0) / max(1, dy2)
    inv_z_right = inv_z0
    inv_z_step_right = dinv_z2 / max(1, dy2) if dy2 > 0 else 0.0

    # Fill the upper triangle section (from y0 to y1):
    # - Always draw at least one scanline even for zero-height sections
    # - This handles degenerate cases where vertices have same y-coordinate
    for y in range(y0, y1):  # Changed to be top-inclusive, bottom-exclusive
        # Convert fixed-point x-coordinates back to integers for this scanline
        start_x = x_left >> 16
        end_x = x_right >> 16

        # Ensure correct left-to-right drawing order:
        # - Swap x-coordinates and intensities if right edge is actually on the left
        # - This maintains consistent interpolation direction
        if start_x > end_x:
            start_x, end_x = end_x, start_x
            i_curr, i_end = i_right, i_left
            inv_z_scan = inv_z_right
            inv_z_step = (inv_z_left - inv_z_right) / max(1, end_x - start_x + 1)
        else:
            i_curr, i_end = i_left, i_right
            inv_z_scan = inv_z_left
            inv_z_step = (inv_z_right - inv_z_left) / max(1, end_x - start_x + 1)

        # Calculate intensity step for this scanline:
        # - Add 1 to span to include both endpoints
        # - Ensure non-zero denominator to prevent division by zero
        i_step = (i_end - i_curr) / max(1, end_x - start_x + 1)

        # Draw the scanline pixels with interpolated intensity:
        # - Skip pixels with near-zero intensity for efficiency
        # - Multiply base color by intensity for final pixel color
        for x in range(start_x, end_x):
            if i_curr > 0.001:
                r = int(color_r * i_curr)
                g = int(color_g * i_curr)
                b = int(color_b * i_curr)
                # Convert back to z for depth testing
                draw_pixel(
                    canvas_grid,
                    depth_buffer,
                    x,
                    y,
                    inv_z_scan,
                    center_x,
                    center_y,
                    r,
                    g,
                    b,
                    canvas_width,
                    canvas_height,
                )
            i_curr += i_step
            inv_z_scan += inv_z_step

        # Update edge coordinates and intensities:
        # - Only update if actually moving in y-direction
        # - This handles zero-height triangle sections
        if y1 > y0:
            x_left += step_left
            x_right += step_right
            i_left += i_step_left
            i_right += i_step_right
            inv_z_left += inv_z_step_left
            inv_z_right += inv_z_step_right

    # Initialize edge traversal for the third edge (y1 to y2):
    # - This replaces the shorter edge (y0 to y1) for lower triangle section
    dx3 = x2 - x1
    dy3 = y2 - y1
    dinv_z3 = inv_z2 - inv_z1
    x_left = x1 << 16
    step_left = (dx3 << 16) // max(1, dy3)
    i_left = i1
    i_step_left = (i2 - i1) / max(1, dy3)
    inv_z_left = inv_z1
    inv_z_step_left = dinv_z3 / max(1, dy3) if dy3 > 0 else 0.0

    # Fill the lower triangle section:
    # - Similar to upper section but uses the third edge
    # - Always draw at least one scanline even for zero-height sections
    for y in range(y1, y2):  # Changed to be top-inclusive, bottom-exclusive
        start_x = x_left >> 16
        end_x = x_right >> 16

        if start_x > end_x:
            start_x, end_x = end_x, start_x
            i_curr, i_end = i_right, i_left
            inv_z_scan = inv_z_right
            inv_z_step = (inv_z_left - inv_z_right) / max(1, end_x - start_x + 1)
        else:
            i_curr, i_end = i_left, i_right
            inv_z_scan = inv_z_left
            inv_z_step = (inv_z_right - inv_z_left) / max(1, end_x - start_x + 1)

        i_step = (i_end - i_curr) / max(1, end_x - start_x + 1)

        for x in range(start_x, end_x):
            if i_curr > 0.001:
                r = int(color_r * i_curr)
                g = int(color_g * i_curr)
                b = int(color_b * i_curr)
                # Convert back to z for depth testing
                draw_pixel(
                    canvas_grid,
                    depth_buffer,
                    x,
                    y,
                    inv_z_scan,
                    center_x,
                    center_y,
                    r,
                    g,
                    b,
                    canvas_width,
                    canvas_height,
                )
            i_curr += i_step
            inv_z_scan += inv_z_step

        if y2 > y1:
            x_left += step_left
            x_right += step_right
            i_left += i_step_left
            i_right += i_step_right
            inv_z_left += inv_z_step_left
            inv_z_right += inv_z_step_right
