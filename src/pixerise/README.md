# Table of Contents

* [pixerise](#pixerise)
  * [ShadingMode](#pixerise.ShadingMode)
  * [Canvas](#pixerise.Canvas)
    * [\_\_init\_\_](#pixerise.Canvas.__init__)
    * [clear](#pixerise.Canvas.clear)
  * [ViewPort](#pixerise.ViewPort)
    * [\_\_init\_\_](#pixerise.ViewPort.__init__)
  * [Renderer](#pixerise.Renderer)
    * [draw\_line](#pixerise.Renderer.draw_line)
    * [draw\_triangle](#pixerise.Renderer.draw_triangle)
    * [draw\_shaded\_triangle](#pixerise.Renderer.draw_shaded_triangle)
    * [cast\_ray](#pixerise.Renderer.cast_ray)
    * [render](#pixerise.Renderer.render)
* [scene](#scene)
  * [Model](#scene.Model)
    * [Group](#scene.Model.Group)
    * [groups](#scene.Model.groups)
    * [add\_group](#scene.Model.add_group)
    * [from\_dict](#scene.Model.from_dict)
    * [to\_dict](#scene.Model.to_dict)
  * [Instance](#scene.Instance)
    * [Group](#scene.Instance.Group)
    * [\_\_post\_init\_\_](#scene.Instance.__post_init__)
    * [model](#scene.Instance.model)
    * [model](#scene.Instance.model)
    * [get\_group\_color](#scene.Instance.get_group_color)
    * [set\_group\_color](#scene.Instance.set_group_color)
    * [get\_group\_visibility](#scene.Instance.get_group_visibility)
    * [set\_group\_visibility](#scene.Instance.set_group_visibility)
    * [translation](#scene.Instance.translation)
    * [translation](#scene.Instance.translation)
    * [rotation](#scene.Instance.rotation)
    * [rotation](#scene.Instance.rotation)
    * [scale](#scene.Instance.scale)
    * [scale](#scene.Instance.scale)
    * [color](#scene.Instance.color)
    * [color](#scene.Instance.color)
    * [set\_translation](#scene.Instance.set_translation)
    * [set\_rotation](#scene.Instance.set_rotation)
    * [set\_scale](#scene.Instance.set_scale)
    * [set\_color](#scene.Instance.set_color)
    * [from\_dict](#scene.Instance.from_dict)
    * [to\_dict](#scene.Instance.to_dict)
  * [Camera](#scene.Camera)
    * [translation](#scene.Camera.translation)
    * [translation](#scene.Camera.translation)
    * [rotation](#scene.Camera.rotation)
    * [rotation](#scene.Camera.rotation)
    * [set\_translation](#scene.Camera.set_translation)
    * [set\_rotation](#scene.Camera.set_rotation)
    * [from\_dict](#scene.Camera.from_dict)
    * [to\_dict](#scene.Camera.to_dict)
  * [DirectionalLight](#scene.DirectionalLight)
    * [direction](#scene.DirectionalLight.direction)
    * [direction](#scene.DirectionalLight.direction)
    * [ambient](#scene.DirectionalLight.ambient)
    * [ambient](#scene.DirectionalLight.ambient)
    * [from\_dict](#scene.DirectionalLight.from_dict)
    * [to\_dict](#scene.DirectionalLight.to_dict)
  * [Scene](#scene.Scene)
    * [add\_model](#scene.Scene.add_model)
    * [get\_model](#scene.Scene.get_model)
    * [add\_instance](#scene.Scene.add_instance)
    * [get\_instance](#scene.Scene.get_instance)
    * [set\_camera](#scene.Scene.set_camera)
    * [set\_directional\_light](#scene.Scene.set_directional_light)
    * [from\_dict](#scene.Scene.from_dict)
    * [to\_dict](#scene.Scene.to_dict)

<a id="pixerise"></a>

# pixerise

Core components of the Pixerise rendering engine.
This module contains the main classes for rendering: Canvas, ViewPort, and Renderer.

<a id="pixerise.ShadingMode"></a>

## ShadingMode Objects

```python
class ShadingMode(Enum)
```

Enum defining different shading modes for 3D rendering.

Available modes:
- FLAT: Uses a single normal per face for constant shading across the triangle
- GOURAUD: Interpolates shading across the triangle using vertex normals
- WIREFRAME: Renders only the edges of triangles without filling

<a id="pixerise.Canvas"></a>

## Canvas Objects

```python
class Canvas()
```

A 2D canvas for drawing pixels and managing the drawing surface.

The Canvas class provides a fundamental drawing surface for the rendering engine.
It manages both the color buffer and depth buffer (zbuffer) for proper
3D rendering with depth testing.

**Attributes**:

- `size` _Tuple[int, int]_ - Canvas dimensions as (width, height)
- `width` _int_ - Canvas width in pixels
- `height` _int_ - Canvas height in pixels
- `color_buffer` _np.ndarray_ - 3D array of shape (width, height, 3) storing RGB values
- `depth_buffer` _np.ndarray_ - 2D array of shape (width, height) storing depth values
- `half_width` _int_ - Half of canvas width, used for center-based coordinates
- `half_height` _int_ - Half of canvas height, used for center-based coordinates
- `_center` _Tuple[int, int]_ - Canvas center point coordinates

<a id="pixerise.Canvas.__init__"></a>

#### \_\_init\_\_

```python
def __init__(size: Tuple[int, int] = (800, 600))
```

Initialize a new Canvas instance.

**Arguments**:

- `size` _Tuple[int, int], optional_ - Canvas dimensions (width, height).
  Defaults to (800, 600).

<a id="pixerise.Canvas.clear"></a>

#### clear

```python
def clear(color: Tuple[int, int, int] = (32, 32, 32))
```

Clear the canvas and reset the z-buffer.

Resets both the color buffer to the specified color and the depth-buffer
to infinity, preparing the canvas for a new frame.

**Arguments**:

- `color` _Tuple[int, int, int], optional_ - RGB color to fill the canvas.
  Each component should be in range [0, 255].
  Defaults to dark gray (32, 32, 32).

<a id="pixerise.ViewPort"></a>

## ViewPort Objects

```python
class ViewPort()
```

Manages the view frustum and coordinate transformations from viewport to canvas space.

The ViewPort class handles the 3D viewing volume (frustum) and provides methods for
transforming coordinates between viewport and canvas space. It pre-calculates the
frustum planes for efficient view frustum culling during rendering.

The view frustum is defined by five planes:
- Left and Right planes
- Top and Bottom planes
- Near plane (at the specified plane distance)

Each frustum plane is represented by its normal vector (pointing inward) and distance
from origin, stored in a numpy array format for efficient JIT processing.

**Attributes**:

- `_width` _float_ - Width of the viewport
- `_height` _float_ - Height of the viewport
- `_plane_distance` _float_ - Distance to the near plane
- `_canvas` _Canvas_ - Reference to the target canvas
- `frustum_planes` _np.ndarray_ - Array of shape (N, 4) containing frustum planes,
  where each row is [nx, ny, nz, d] representing the plane equation nx*x + ny*y + nz*z + d = 0

<a id="pixerise.ViewPort.__init__"></a>

#### \_\_init\_\_

```python
def __init__(size: Tuple[float, float], plane_distance: float, canvas: Canvas)
```

Initialize a new ViewPort instance.

**Arguments**:

- `size` _Tuple[float, float]_ - Dimensions of the viewport (width, height)
- `plane_distance` _float_ - Distance to the near plane from the camera
- `canvas` _Canvas_ - Target canvas for rendering

<a id="pixerise.Renderer"></a>

## Renderer Objects

```python
class Renderer()
```

A high-performance 3D renderer using NumPy and Numba JIT compilation.

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

**Attributes**:

- `_canvas` _Canvas_ - Target canvas for rendering output
- `_viewport` _ViewPort_ - Viewport configuration and frustum planes
- `_background_color` _Tuple[int, int, int]_ - RGB color for canvas clear

<a id="pixerise.Renderer.draw_line"></a>

#### draw\_line

```python
def draw_line(start: Tuple[float, float, float],
              end: Tuple[float, float, float], color: Tuple[int, int, int])
```

Draw a line using Bresenham's algorithm with depth buffering.

**Arguments**:

- `start` - Starting point as (x, y, z) tuple
- `end` - Ending point as (x, y, z) tuple
- `color` - RGB color as (r, g, b) tuple

<a id="pixerise.Renderer.draw_triangle"></a>

#### draw\_triangle

```python
def draw_triangle(p1: Tuple[float, float, float],
                  p2: Tuple[float, float, float],
                  p3: Tuple[float, float, float],
                  color: Tuple[int, int, int],
                  fill: bool = True)
```

Draw a triangle defined by three points. If fill is True, the triangle will be filled,
otherwise only the outline will be drawn.

<a id="pixerise.Renderer.draw_shaded_triangle"></a>

#### draw\_shaded\_triangle

```python
def draw_shaded_triangle(p1: Tuple[float, float,
                                   float], p2: Tuple[float, float, float],
                         p3: Tuple[float, float, float],
                         color: Tuple[int, int, int], intensity1: float,
                         intensity2: float, intensity3: float)
```

Draw a triangle with smooth shading using per-vertex intensity interpolation.
This method implements Gouraud shading by interpolating intensity values across
the triangle's surface.

**Arguments**:

  p1, p2, p3: Vertex positions as (x, y, z) tuples in screen space coordinates.
  The vertices can be in any order, they will be sorted internally.
- `color` - Base RGB color as (r, g, b) tuple, where each component is in range [0, 255].
  This color will be modulated by the interpolated intensities.
  intensity1, intensity2, intensity3: Light intensity values for each vertex in range [0.0, 1.0].
  These values determine how bright the color appears at each vertex
  and are linearly interpolated across the triangle.
  

**Notes**:

  - Intensity values are automatically clamped to the valid range [0.0, 1.0] to ensure correct color modulation
  - The final color at each pixel is computed as: final_rgb = base_rgb * interpolated_intensity
  - The implementation uses a scanline algorithm with linear interpolation for efficiency
  - Triangles completely outside the canvas or with zero intensity are skipped
  - Z-coordinates are used for depth testing to ensure correct visibility

<a id="pixerise.Renderer.cast_ray"></a>

#### cast\_ray

```python
def cast_ray(screen_x: int, screen_y: int,
             scene: Scene) -> Optional[Tuple[str, str]]
```

Cast a ray from the camera through a screen point and find the first hit triangle.

The ray casting process involves these steps:
1. Screen Space -> Viewport Space: Convert pixel coordinates to normalized viewport coordinates
2. Create ray in camera space (origin at 0,0,0, direction through viewport point)
3. For each instance:
- Transform vertices to camera space
- Check bounding sphere intersection for early culling
- Test ray intersection with transformed triangles
- Track closest intersection

**Arguments**:

- `screen_x` - X coordinate in screen space (pixels from left)
- `screen_y` - Y coordinate in screen space (pixels from top)
- `scene` - Scene containing models and instances to test against
  

**Returns**:

  Optional tuple of (instance_name, group_name) of the first hit triangle,
  or None if no triangle was hit

<a id="pixerise.Renderer.render"></a>

#### render

```python
def render(scene: Scene, shading_mode: ShadingMode = ShadingMode.WIREFRAME)
```

Render a 3D scene using the specified shading mode.

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

**Arguments**:

- `scene` _Scene_ - Scene object containing models, instances, camera, and lighting
- `shading_mode` _ShadingMode_ - Rendering mode to use. Options are:
  - WIREFRAME: Only render triangle edges
  - FLAT: Single color per triangle with basic lighting
  - GOURAUD: Smooth shading with per-vertex lighting interpolation

<a id="scene"></a>

# scene

Scene management module for the Pixerise rendering engine.

This module provides a complete scene graph implementation for 3D rendering, including:
- Model management with support for multiple geometry groups
- Instance transformation and material properties
- Camera positioning and orientation
- Lighting configuration with directional lights

The scene graph is organized hierarchically:
1. Scene: Top-level container managing all scene elements
2. Models: Reusable geometry definitions
3. Instances: Concrete occurrences of models with unique transforms
4. Groups: Sub-components within models for organized geometry

Key Features:
- Efficient geometry storage using NumPy arrays
- Support for vertex normals and color properties
- Serialization to/from dictionary format
- Memory-efficient implementation using slots
- Type hints and dataclass decorators for clean APIs

**Example**:

    ```python
    # Create a scene with a model and instance
    scene = Scene()

    # Add a model with vertices and triangles
    model = Model()
    model.add_group(
        vertices=np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=np.float32),
        triangles=np.array([[0,1,2]], dtype=np.int32)
    )
    scene.add_model("triangle", model)

    # Create an instance of the model
    instance = Instance(
        model="triangle",
        translation=np.array([0,0,-5], dtype=np.float32),
        color=np.array([255,0,0], dtype=np.int32)
    )
    scene.add_instance("triangle1", instance)
    ```

<a id="scene.Model"></a>

## Model Objects

```python
@dataclass(slots=True)
class Model()
```

A 3D model containing one or more groups of geometry.

Models are reusable geometry definitions that can be instantiated multiple times
in a scene. Each model can contain multiple groups, allowing for organized
geometry with different materials or properties.

The geometry data is stored efficiently using NumPy arrays, and the model
supports both flat shading (without normals) and smooth shading (with vertex
normals).

**Attributes**:

- `_groups` _Dict[str, Group]_ - Named groups containing geometry data

<a id="scene.Model.Group"></a>

## Group Objects

```python
@dataclass(slots=True)
class Group()
```

A group within a model containing geometry data.

Groups allow models to be organized into logical components, each with its own
geometry data. This is useful for complex models where different parts may need
different materials or may need to be manipulated independently.

**Attributes**:

- `_vertices` _np.ndarray_ - Array of shape (N, 3) containing vertex positions
- `_triangles` _np.ndarray_ - Array of shape (M, 3) containing vertex indices
- `_vertex_normals` _Optional[np.ndarray]_ - Array of shape (N, 3) containing vertex normals
  If None, flat shading will be used for this group

<a id="scene.Model.Group.vertices"></a>

#### vertices

```python
@property
def vertices() -> np.ndarray
```

Get the vertex positions (read-only).

**Returns**:

- `np.ndarray` - Array of shape (N, 3) containing vertex positions

<a id="scene.Model.Group.triangles"></a>

#### triangles

```python
@property
def triangles() -> np.ndarray
```

Get the triangle indices (read-only).

**Returns**:

- `np.ndarray` - Array of shape (M, 3) containing vertex indices

<a id="scene.Model.Group.vertex_normals"></a>

#### vertex\_normals

```python
@property
def vertex_normals() -> Optional[np.ndarray]
```

Get the vertex normals if they exist (read-only).

**Returns**:

- `Optional[np.ndarray]` - Array of shape (N, 3) containing vertex normals
  If None, flat shading will be used for this group

<a id="scene.Model.Group.from_dict"></a>

#### from\_dict

```python
@classmethod
def from_dict(cls, data: dict) -> "Model.Group"
```

Create a Model.Group from a dictionary representation.

**Arguments**:

- `data` _dict_ - Dictionary containing 'vertices', 'triangles', and optionally
  'vertex_normals' arrays
  

**Returns**:

- `Model.Group` - New instance with the specified geometry data

<a id="scene.Model.groups"></a>

#### groups

```python
@property
def groups() -> Dict[str, Group]
```

Get the model's groups (read-only).

**Returns**:

  Dict[str, Group]: Dictionary of named groups containing geometry data

<a id="scene.Model.add_group"></a>

#### add\_group

```python
def add_group(vertices: np.ndarray,
              triangles: np.ndarray,
              vertex_normals: Optional[np.ndarray] = None,
              name: str = "default") -> None
```

Add a new geometry group to the model.

**Arguments**:

- `vertices` _np.ndarray_ - Array of shape (N, 3) containing vertex positions
- `triangles` _np.ndarray_ - Array of shape (M, 3) containing vertex indices
  forming triangles
- `vertex_normals` _Optional[np.ndarray]_ - Array of shape (N, 3) containing
  vertex normals for smooth shading. If None, flat shading will be used
- `name` _str_ - Unique identifier for the group (default: 'default')

<a id="scene.Model.from_dict"></a>

#### from\_dict

```python
@classmethod
def from_dict(cls, data: dict) -> "Model"
```

Create a Model from a dictionary representation.

The dictionary can either contain a flat structure with direct geometry data
(which will be placed in a 'default' group) or a nested structure with
multiple named groups.

**Arguments**:

- `data` _dict_ - Dictionary containing model data
  

**Returns**:

- `Model` - New model instance with the specified geometry

<a id="scene.Model.to_dict"></a>

#### to\_dict

```python
def to_dict() -> dict
```

Convert the Model to a dictionary representation.

**Returns**:

- `dict` - Dictionary containing model data. If the model has only a default
  group, returns a flat structure. Otherwise, returns a nested structure
  with named groups.

<a id="scene.Instance"></a>

## Instance Objects

```python
@dataclass(slots=True)
class Instance()
```

An instance of a model with transformation and color properties.

Instances represent concrete occurrences of models in the scene, each with its
own position, orientation, scale, and color. Multiple instances can reference
the same model, allowing for efficient memory usage when the same geometry
appears multiple times in the scene.

**Attributes**:

- `_model` _str_ - Name of the model this instance references
- `_translation` _np.ndarray_ - 3D vector specifying position
- `_rotation` _np.ndarray_ - 3D vector specifying rotation in radians
- `_scale` _np.ndarray_ - 3D vector specifying scale in each axis
- `_color` _np.ndarray_ - RGB color values as integers in range [0, 255]
- `_groups` _defaultdict[str, Group]_ - Group-specific states containing
  color and visibility information

<a id="scene.Instance.Group"></a>

## Group Objects

```python
@dataclass(slots=True)
class Group()
```

State information for a model group within an instance.

Groups allow instances to override properties of specific model groups,
such as color and visibility. This enables fine-grained control over
how different parts of a model are rendered in each instance.

**Attributes**:

- `color` _Optional[np.ndarray]_ - RGB color values as integers in range [0, 255],
  or None to use instance color
- `visible` _bool_ - Whether the group is visible

<a id="scene.Instance.Group.to_dict"></a>

#### to\_dict

```python
def to_dict() -> dict
```

Convert to dictionary representation.

**Returns**:

- `dict` - Dictionary with color and visibility information

<a id="scene.Instance.Group.from_dict"></a>

#### from\_dict

```python
@classmethod
def from_dict(cls, data: dict) -> "Instance.Group"
```

Create from dictionary representation.

**Arguments**:

- `data` _dict_ - Dictionary with color and visibility information
  

**Returns**:

- `Instance.Group` - New state with the specified properties

<a id="scene.Instance.__post_init__"></a>

#### \_\_post\_init\_\_

```python
def __post_init__()
```

Initialize defaultdict after instance creation.

<a id="scene.Instance.model"></a>

#### model

```python
@property
def model() -> str
```

Get the model name this instance references.

**Returns**:

- `str` - Name of the model

<a id="scene.Instance.model"></a>

#### model

```python
@model.setter
def model(value: str) -> None
```

Set the model name this instance references.

**Arguments**:

- `value` _str_ - New model name

<a id="scene.Instance.get_group_color"></a>

#### get\_group\_color

```python
def get_group_color(group_name: str) -> Optional[np.ndarray]
```

Get the color for a specific group.

**Arguments**:

- `group_name` _str_ - Name of the group
  

**Returns**:

- `Optional[np.ndarray]` - RGB color values as integers in range [0, 255],
  or None if no specific color is set for this group

<a id="scene.Instance.set_group_color"></a>

#### set\_group\_color

```python
def set_group_color(group_name: str, color: Optional[np.ndarray]) -> None
```

Set the color for a specific group.

**Arguments**:

- `group_name` _str_ - Name of the group
- `color` _Optional[np.ndarray]_ - RGB color values as integers in range [0, 255],
  or None to remove the group-specific color

<a id="scene.Instance.get_group_visibility"></a>

#### get\_group\_visibility

```python
def get_group_visibility(group_name: str) -> bool
```

Get the visibility state for a specific group.

**Arguments**:

- `group_name` _str_ - Name of the group
  

**Returns**:

- `bool` - True if the group is visible, False otherwise.
  If no state is set, returns True as default.

<a id="scene.Instance.set_group_visibility"></a>

#### set\_group\_visibility

```python
def set_group_visibility(group_name: str, visible: bool) -> None
```

Set the visibility state for a specific group.

**Arguments**:

- `group_name` _str_ - Name of the group
- `visible` _bool_ - True to make the group visible, False to hide it

<a id="scene.Instance.translation"></a>

#### translation

```python
@property
def translation() -> np.ndarray
```

Get the instance's translation vector.

**Returns**:

- `np.ndarray` - 3D vector specifying position

<a id="scene.Instance.translation"></a>

#### translation

```python
@translation.setter
def translation(value: np.ndarray) -> None
```

Set the instance's translation vector.

**Arguments**:

- `value` _np.ndarray_ - New 3D vector specifying position

<a id="scene.Instance.rotation"></a>

#### rotation

```python
@property
def rotation() -> np.ndarray
```

Get the instance's rotation vector.

**Returns**:

- `np.ndarray` - 3D vector specifying rotation in radians

<a id="scene.Instance.rotation"></a>

#### rotation

```python
@rotation.setter
def rotation(value: np.ndarray) -> None
```

Set the instance's rotation vector.

**Arguments**:

- `value` _np.ndarray_ - New 3D vector specifying rotation in radians

<a id="scene.Instance.scale"></a>

#### scale

```python
@property
def scale() -> np.ndarray
```

Get the instance's scale vector.

**Returns**:

- `np.ndarray` - 3D vector specifying scale in each axis

<a id="scene.Instance.scale"></a>

#### scale

```python
@scale.setter
def scale(value: np.ndarray) -> None
```

Set the instance's scale vector.

**Arguments**:

- `value` _np.ndarray_ - New 3D vector specifying scale in each axis

<a id="scene.Instance.color"></a>

#### color

```python
@property
def color() -> np.ndarray
```

Get the instance's color values.

**Returns**:

- `np.ndarray` - RGB color values as integers in range [0, 255]

<a id="scene.Instance.color"></a>

#### color

```python
@color.setter
def color(value: np.ndarray) -> None
```

Set the instance's color values.

**Arguments**:

- `value` _np.ndarray_ - New RGB color values as integers in range [0, 255]

<a id="scene.Instance.set_translation"></a>

#### set\_translation

```python
def set_translation(x: float, y: float, z: float) -> None
```

Set the translation of this instance.

**Arguments**:

- `x` _float_ - X-coordinate in world space
- `y` _float_ - Y-coordinate in world space
- `z` _float_ - Z-coordinate in world space

<a id="scene.Instance.set_rotation"></a>

#### set\_rotation

```python
def set_rotation(x: float, y: float, z: float) -> None
```

Set the rotation of this instance in radians.

**Arguments**:

- `x` _float_ - Rotation around X-axis in radians
- `y` _float_ - Rotation around Y-axis in radians
- `z` _float_ - Roll (rotation around Z-axis) in radians

<a id="scene.Instance.set_scale"></a>

#### set\_scale

```python
def set_scale(x: float, y: float, z: float) -> None
```

Set the scale of this instance.

**Arguments**:

- `x` _float_ - Scale factor along X-axis
- `y` _float_ - Scale factor along Y-axis
- `z` _float_ - Scale factor along Z-axis

<a id="scene.Instance.set_color"></a>

#### set\_color

```python
def set_color(r: int, g: int, b: int) -> None
```

Set the color of this instance.

**Arguments**:

- `r` _int_ - Red component (0-255)
- `g` _int_ - Green component (0-255)
- `b` _int_ - Blue component (0-255)

<a id="scene.Instance.from_dict"></a>

#### from\_dict

```python
@classmethod
def from_dict(cls, data: dict) -> "Instance"
```

Create an Instance from a dictionary representation.

**Arguments**:

- `data` _dict_ - Dictionary containing instance data with model reference
  and optional transform and color information
  

**Returns**:

- `Instance` - New instance with the specified properties

<a id="scene.Instance.to_dict"></a>

#### to\_dict

```python
def to_dict() -> dict
```

Convert this instance to a dictionary representation.

**Returns**:

- `dict` - Dictionary containing the instance's model reference,
  transformation data, color information, and group states

<a id="scene.Camera"></a>

## Camera Objects

```python
@dataclass(slots=True)
class Camera()
```

Camera settings for the scene.

The camera defines the viewpoint from which the scene is rendered. It supports
positioning and orientation through translation and rotation transforms.

The camera uses a right-handed coordinate system where:
- X-axis points right
- Y-axis points up
- Z-axis points away from the viewer (into the screen)

**Attributes**:

- `_translation` _np.ndarray_ - 3D vector specifying camera position
- `_rotation` _np.ndarray_ - 3D vector specifying camera rotation in radians

<a id="scene.Camera.translation"></a>

#### translation

```python
@property
def translation() -> np.ndarray
```

Get the camera's translation vector [x, y, z] in world space.

**Returns**:

- `np.ndarray` - 3D vector [x, y, z] in world space

<a id="scene.Camera.translation"></a>

#### translation

```python
@translation.setter
def translation(value: np.ndarray) -> None
```

Set the camera's translation vector [x, y, z] in world space.

**Arguments**:

- `value` _np.ndarray_ - 3D vector [x, y, z] in world space

<a id="scene.Camera.rotation"></a>

#### rotation

```python
@property
def rotation() -> np.ndarray
```

Get the camera's rotation vector [x, y, z] in radians.

**Returns**:

- `np.ndarray` - 3D vector [x, y, z] in radians

<a id="scene.Camera.rotation"></a>

#### rotation

```python
@rotation.setter
def rotation(value: np.ndarray) -> None
```

Set the camera's rotation vector [x, y, z] in radians.

**Arguments**:

- `value` _np.ndarray_ - 3D vector [x, y, z] in radians

<a id="scene.Camera.set_translation"></a>

#### set\_translation

```python
def set_translation(x: float, y: float, z: float) -> None
```

Set the camera's position.

**Arguments**:

- `x` _float_ - X-coordinate in world space
- `y` _float_ - Y-coordinate in world space
- `z` _float_ - Z-coordinate in world space

<a id="scene.Camera.set_rotation"></a>

#### set\_rotation

```python
def set_rotation(x: float, y: float, z: float) -> None
```

Set the camera's rotation in radians.

**Arguments**:

- `x` _float_ - Pitch (rotation around X-axis) in radians
- `y` _float_ - Yaw (rotation around Y-axis) in radians
- `z` _float_ - Roll (rotation around Z-axis) in radians

<a id="scene.Camera.from_dict"></a>

#### from\_dict

```python
@classmethod
def from_dict(cls, data: dict) -> "Camera"
```

Create a Camera from a dictionary representation.

**Arguments**:

- `data` _dict_ - Dictionary containing camera transform data
  

**Returns**:

- `Camera` - New camera instance with the specified position and orientation

<a id="scene.Camera.to_dict"></a>

#### to\_dict

```python
def to_dict() -> dict
```

Convert the Camera to a dictionary representation.

**Returns**:

- `dict` - Dictionary containing the camera's transformation data

<a id="scene.DirectionalLight"></a>

## DirectionalLight Objects

```python
@dataclass(slots=True)
class DirectionalLight()
```

A directional light in the scene.

A directional light simulates a light source that is infinitely far away,
producing parallel light rays. It is defined by a direction vector and
ambient light intensity.

**Attributes**:

- `_direction` _np.ndarray_ - 3D vector specifying light direction
- `_ambient` _float_ - Ambient light intensity in range [0, 1]

<a id="scene.DirectionalLight.direction"></a>

#### direction

```python
@property
def direction() -> np.ndarray
```

Get the light's direction vector.

The direction vector points from the light source towards the scene.
For example, [0, 0, -1] represents a light shining straight down
along the negative Z-axis.

**Returns**:

- `np.ndarray` - Light direction vector (normalized)

<a id="scene.DirectionalLight.direction"></a>

#### direction

```python
@direction.setter
def direction(value: np.ndarray) -> None
```

Set the light's direction vector.

The direction vector should point from the light source towards the scene.
For example, [0, 0, -1] represents a light shining straight down
along the negative Z-axis.

**Arguments**:

- `value` _np.ndarray_ - New light direction vector (will be normalized)

<a id="scene.DirectionalLight.ambient"></a>

#### ambient

```python
@property
def ambient() -> float
```

Get the light's ambient intensity.

The ambient intensity determines how much light is present in areas
not directly lit by the directional light. A value of 0 means complete
darkness in shadows, while 1 means no shadows at all.

**Returns**:

- `float` - Ambient light intensity in range [0, 1]

<a id="scene.DirectionalLight.ambient"></a>

#### ambient

```python
@ambient.setter
def ambient(value: float) -> None
```

Set the light's ambient intensity.

The ambient intensity determines how much light is present in areas
not directly lit by the directional light. A value of 0 means complete
darkness in shadows, while 1 means no shadows at all.

**Arguments**:

- `value` _float_ - New ambient light intensity in range [0, 1]

<a id="scene.DirectionalLight.from_dict"></a>

#### from\_dict

```python
@classmethod
def from_dict(cls, data: dict) -> "DirectionalLight"
```

Create a DirectionalLight from a dictionary representation.

**Arguments**:

- `data` _dict_ - Dictionary containing light direction and ambient intensity
  

**Returns**:

- `DirectionalLight` - New light instance with the specified properties

<a id="scene.DirectionalLight.to_dict"></a>

#### to\_dict

```python
def to_dict() -> dict
```

Convert the DirectionalLight to a dictionary representation.

**Returns**:

- `dict` - Dictionary containing the light's direction and ambient intensity

<a id="scene.Scene"></a>

## Scene Objects

```python
@dataclass(slots=True)
class Scene()
```

A 3D scene containing models, instances, camera settings, and lights.

The Scene class is the top-level container for all elements in a 3D scene.
It manages:
- A collection of reusable 3D models
- Multiple instances of those models with unique transforms
- Camera position and orientation
- Lighting configuration

The scene supports serialization to/from dictionary format for easy saving
and loading of scene configurations.

**Attributes**:

- `models` _Dict[str, Model]_ - Named collection of 3D models
- `instances` _Dict[str, Instance]_ - Named collection of model instances
- `camera` _Camera_ - Scene camera configuration
- `directional_light` _DirectionalLight_ - Main directional light source

<a id="scene.Scene.add_model"></a>

#### add\_model

```python
def add_model(name: str, model: Model) -> None
```

Add a model to the scene.

**Arguments**:

- `name` _str_ - Unique identifier for the model
- `model` _Model_ - Model instance to add

<a id="scene.Scene.get_model"></a>

#### get\_model

```python
def get_model(model_name: str) -> Optional[Model]
```

Get a model by name.

**Arguments**:

- `model_name` _str_ - Name of the model to get
  

**Returns**:

- `Model` - Model instance if found, None otherwise

<a id="scene.Scene.add_instance"></a>

#### add\_instance

```python
def add_instance(name: str, instance: Instance) -> None
```

Add a model instance to the scene.

**Arguments**:

- `name` _str_ - Unique identifier for the instance
- `instance` _Instance_ - Instance to add

<a id="scene.Scene.get_instance"></a>

#### get\_instance

```python
def get_instance(instance_name: str) -> Optional[Instance]
```

Get an instance by name.

**Arguments**:

- `instance_name` _str_ - Name of the instance to get
  

**Returns**:

- `Instance` - Instance if found, None otherwise

<a id="scene.Scene.set_camera"></a>

#### set\_camera

```python
def set_camera(camera: Camera) -> None
```

Set the scene's camera.

**Arguments**:

- `camera` _Camera_ - Camera instance to set

<a id="scene.Scene.set_directional_light"></a>

#### set\_directional\_light

```python
def set_directional_light(light: DirectionalLight) -> None
```

Set the scene's directional light.

**Arguments**:

- `light` _DirectionalLight_ - Light instance to set

<a id="scene.Scene.from_dict"></a>

#### from\_dict

```python
@classmethod
def from_dict(cls, data: dict) -> "Scene"
```

Create a Scene from a dictionary representation.

**Arguments**:

- `data` _dict_ - Dictionary containing scene data
  

**Returns**:

- `Scene` - New scene instance with the specified elements

<a id="scene.Scene.to_dict"></a>

#### to\_dict

```python
def to_dict() -> dict
```

Convert the Scene to a dictionary representation.

**Returns**:

- `dict` - Dictionary containing scene data

