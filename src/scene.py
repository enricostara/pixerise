"""
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

Example:
    ```python
    # Create a scene with a model and instance
    scene = Scene()

    # Add a model with vertices and triangles
    model = Model()
    model.add_group(
        "default",
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
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass(slots=True)
class ModelInnerGroup:
    """A group within a model containing geometry data.

    Groups allow models to be organized into logical components, each with its own
    geometry data. This is useful for complex models where different parts may need
    different materials or may need to be manipulated independently.

    Attributes:
        _vertices (np.ndarray): Array of shape (N, 3) containing vertex positions
        _triangles (np.ndarray): Array of shape (M, 3) containing vertex indices
        _vertex_normals (Optional[np.ndarray]): Array of shape (N, 3) containing vertex normals
            If None, flat shading will be used for this group
    """

    _vertices: np.ndarray
    _triangles: np.ndarray
    _vertex_normals: Optional[np.ndarray] = None

    @property
    def vertices(self) -> np.ndarray:
        """Get the vertex positions (read-only)."""
        return self._vertices.copy()

    @property
    def triangles(self) -> np.ndarray:
        """Get the triangle indices (read-only)."""
        return self._triangles.copy()

    @property
    def vertex_normals(self) -> Optional[np.ndarray]:
        """Get the vertex normals if they exist (read-only)."""
        return self._vertex_normals.copy() if self._vertex_normals is not None else None

    @classmethod
    def from_dict(cls, data: dict) -> "ModelInnerGroup":
        """Create a ModelGroup from a dictionary representation.

        Args:
            data (dict): Dictionary containing 'vertices', 'triangles', and optionally
                'vertex_normals' arrays

        Returns:
            ModelInnerGroup: New instance with the specified geometry data
        """
        return cls(
            _vertices=np.array(data.get("vertices", []), dtype=np.float32),
            _triangles=np.array(data.get("triangles", []), dtype=np.int32),
            _vertex_normals=np.array(data.get("vertex_normals", []), dtype=np.float32)
            if "vertex_normals" in data
            else None,
        )


@dataclass(slots=True)
class Model:
    """A 3D model containing one or more groups of geometry.

    Models are reusable geometry definitions that can be instantiated multiple times
    in a scene. Each model can contain multiple groups, allowing for organized
    geometry with different materials or properties.

    The geometry data is stored efficiently using NumPy arrays, and the model
    supports both flat shading (without normals) and smooth shading (with vertex
    normals).

    Attributes:
        _groups (Dict[str, ModelInnerGroup]): Named groups containing geometry data
    """

    _groups: Dict[str, ModelInnerGroup] = field(default_factory=dict)

    @property
    def groups(self) -> Dict[str, ModelInnerGroup]:
        """Get the model's groups (read-only)."""
        return self._groups.copy()

    def add_group(
        self,
        name: str,
        vertices: np.ndarray,
        triangles: np.ndarray,
        vertex_normals: Optional[np.ndarray] = None,
    ) -> None:
        """Add a new geometry group to the model.

        Args:
            name (str): Unique identifier for the group
            vertices (np.ndarray): Array of shape (N, 3) containing vertex positions
            triangles (np.ndarray): Array of shape (M, 3) containing vertex indices
                forming triangles
            vertex_normals (Optional[np.ndarray]): Array of shape (N, 3) containing
                vertex normals for smooth shading. If None, flat shading will be used
        """
        self._groups[name] = ModelInnerGroup(
            _vertices=vertices.astype(np.float32),
            _triangles=triangles.astype(np.int32),
            _vertex_normals=vertex_normals.astype(np.float32)
            if vertex_normals is not None
            else None,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "Model":
        """Create a Model from a dictionary representation.

        The dictionary can either contain a flat structure with direct geometry data
        (which will be placed in a 'default' group) or a nested structure with
        multiple named groups.

        Args:
            data (dict): Dictionary containing model data

        Returns:
            Model: New model instance with the specified geometry
        """
        model = cls()
        groups = data.get("groups")

        # If no groups are defined, create a default group with model data
        if not groups:
            groups = {
                "default": {
                    "vertices": data.get("vertices", []),
                    "triangles": data.get("triangles", []),
                    "vertex_normals": data.get("vertex_normals", []),
                }
            }

        for name, group_data in groups.items():
            model._groups[name] = ModelInnerGroup.from_dict(group_data)

        return model

    def to_dict(self) -> dict:
        """Convert the Model to a dictionary representation.

        Returns:
            dict: Dictionary containing model data. If the model has only a default
                group, returns a flat structure. Otherwise, returns a nested structure
                with named groups.
        """
        if len(self._groups) == 1 and "default" in self._groups:
            # If there's only a default group, use flat structure
            group = self._groups["default"]
            return {
                "vertices": group.vertices.tolist(),
                "triangles": group.triangles.tolist(),
                "vertex_normals": group.vertex_normals.tolist()
                if group.vertex_normals is not None
                else None,
            }
        else:
            # Otherwise, use groups structure
            return {
                "groups": {
                    name: {
                        "vertices": group.vertices.tolist(),
                        "triangles": group.triangles.tolist(),
                        "vertex_normals": group.vertex_normals.tolist()
                        if group.vertex_normals is not None
                        else None,
                    }
                    for name, group in self._groups.items()
                }
            }


@dataclass(slots=True)
class Instance:
    """An instance of a model with transformation and color properties.

    Instances represent concrete occurrences of models in the scene, each with its
    own position, orientation, scale, and color. Multiple instances can reference
    the same model, allowing for efficient memory usage when the same geometry
    appears multiple times in the scene.

    Attributes:
        _model (str): Name of the model this instance references
        _translation (np.ndarray): 3D vector specifying position
        _rotation (np.ndarray): 3D vector specifying rotation in radians
        _scale (np.ndarray): 3D vector specifying scale in each axis
        _color (np.ndarray): RGB color values as integers in range [0, 255]
    """

    _model: str
    _translation: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    _rotation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    _scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))
    _color: np.ndarray = field(
        default_factory=lambda: np.array([200, 200, 200], dtype=np.int32)
    )  # Default gray color

    @property
    def model(self) -> str:
        """Get the model name this instance references."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Set the model name this instance references."""
        self._model = value

    # Properties
    @property
    def translation(self) -> np.ndarray:
        """Get the instance's translation vector."""
        return self._translation

    @translation.setter
    def translation(self, value: np.ndarray) -> None:
        """Set the instance's translation vector."""
        self._translation = np.array(value, dtype=np.float32)

    @property
    def rotation(self) -> np.ndarray:
        """Get the instance's rotation vector."""
        return self._rotation

    @rotation.setter
    def rotation(self, value: np.ndarray) -> None:
        """Set the instance's rotation vector."""
        self._rotation = np.array(value, dtype=np.float32)

    @property
    def scale(self) -> np.ndarray:
        """Get the instance's scale vector."""
        return self._scale

    @scale.setter
    def scale(self, value: np.ndarray) -> None:
        """Set the instance's scale vector."""
        self._scale = np.array(value, dtype=np.float32)

    @property
    def color(self) -> np.ndarray:
        """Get the instance's color values."""
        return self._color

    @color.setter
    def color(self, value: np.ndarray) -> None:
        """Set the instance's color values."""
        self._color = np.array(value, dtype=np.int32)

    def set_translation(self, x: float, y: float, z: float) -> None:
        """Set the translation of this instance.

        Args:
            x (float): X-coordinate in world space
            y (float): Y-coordinate in world space
            z (float): Z-coordinate in world space
        """
        self._translation = np.array([x, y, z], dtype=np.float32)

    def set_rotation(self, x: float, y: float, z: float) -> None:
        """Set the rotation of this instance in radians.

        Args:
            x (float): Rotation around X-axis in radians
            y (float): Rotation around Y-axis in radians
            z (float): Roll (rotation around Z-axis) in radians
        """
        self._rotation = np.array([x, y, z], dtype=np.float32)

    def set_scale(self, x: float, y: float, z: float) -> None:
        """Set the scale of this instance.

        Args:
            x (float): Scale factor along X-axis
            y (float): Scale factor along Y-axis
            z (float): Scale factor along Z-axis
        """
        self._scale = np.array([x, y, z], dtype=np.float32)

    @classmethod
    def from_dict(cls, data: dict) -> "Instance":
        """Create an Instance from a dictionary representation.

        Args:
            data (dict): Dictionary containing instance data with model reference
                and optional transform and color information

        Returns:
            Instance: New instance with the specified properties
        """
        instance = cls(_model=data["model"])
        if "transform" in data:
            transform = data["transform"]
            if "translation" in transform:
                instance._translation = np.array(
                    transform["translation"], dtype=np.float32
                )
            if "rotation" in transform:
                instance._rotation = np.array(transform["rotation"], dtype=np.float32)
            if "scale" in transform:
                instance._scale = np.array(transform["scale"], dtype=np.float32)
        if "color" in data:
            instance._color = np.array(data["color"], dtype=np.int32)
        return instance

    def to_dict(self) -> dict:
        """Convert this instance to a dictionary representation.

        Returns:
            dict: Dictionary containing the instance's model reference,
                transformation data, and color information
        """
        return {
            "model": self._model,
            "transform": {
                "translation": self._translation.tolist(),
                "rotation": self._rotation.tolist(),
                "scale": self._scale.tolist(),
            },
            "color": self._color.tolist(),
        }


@dataclass(slots=True)
class Camera:
    """Camera settings for the scene.

    The camera defines the viewpoint from which the scene is rendered. It supports
    positioning and orientation through translation and rotation transforms.

    The camera uses a right-handed coordinate system where:
    - X-axis points right
    - Y-axis points up
    - Z-axis points away from the viewer (into the screen)

    Attributes:
        _translation (np.ndarray): 3D vector specifying camera position
        _rotation (np.ndarray): 3D vector specifying camera rotation in radians
    """

    _translation: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    _rotation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    @property
    def translation(self) -> np.ndarray:
        """Get the camera's translation vector [x, y, z] in world space.

        Returns:
            np.ndarray: 3D vector [x, y, z] in world space
        """
        return self._translation

    @translation.setter
    def translation(self, value: np.ndarray) -> None:
        """Set the camera's translation vector [x, y, z] in world space.

        Args:
            value (np.ndarray): 3D vector [x, y, z] in world space
        """
        self._translation = np.array(value, dtype=np.float32)

    @property
    def rotation(self) -> np.ndarray:
        """Get the camera's rotation vector [x, y, z] in radians.

        Returns:
            np.ndarray: 3D vector [x, y, z] in radians
        """
        return self._rotation

    @rotation.setter
    def rotation(self, value: np.ndarray) -> None:
        """Set the camera's rotation vector [x, y, z] in radians.

        Args:
            value (np.ndarray): 3D vector [x, y, z] in radians
        """
        self._rotation = np.array(value, dtype=np.float32)

    def set_translation(self, x: float, y: float, z: float) -> None:
        """Set the camera's position.

        Args:
            x (float): X-coordinate in world space
            y (float): Y-coordinate in world space
            z (float): Z-coordinate in world space
        """
        self._translation = np.array([x, y, z], dtype=np.float32)

    def set_rotation(self, x: float, y: float, z: float) -> None:
        """Set the camera's rotation in radians.

        Args:
            x (float): Pitch (rotation around X-axis) in radians
            y (float): Yaw (rotation around Y-axis) in radians
            z (float): Roll (rotation around Z-axis) in radians
        """
        self._rotation = np.array([x, y, z], dtype=np.float32)

    @classmethod
    def from_dict(cls, data: dict) -> "Camera":
        """Create a Camera from a dictionary representation.

        Args:
            data (dict): Dictionary containing camera transform data

        Returns:
            Camera: New camera instance with the specified position and orientation
        """
        transform = data.get("transform", {})
        return cls(
            _translation=np.array(
                transform.get("translation", [0, 0, 0]), dtype=np.float32
            ),
            _rotation=np.array(transform.get("rotation", [0, 0, 0]), dtype=np.float32),
        )

    def to_dict(self) -> dict:
        """Convert the Camera to a dictionary representation.

        Returns:
            dict: Dictionary containing the camera's transformation data
        """
        return {
            "transform": {
                "translation": self._translation.tolist(),
                "rotation": self._rotation.tolist(),
            }
        }


@dataclass(slots=True)
class DirectionalLight:
    """A directional light in the scene.

    A directional light simulates a light source that is infinitely far away,
    producing parallel light rays. It is defined by a direction vector and
    ambient light intensity.

    Attributes:
        _direction (np.ndarray): 3D vector specifying light direction
        _ambient (float): Ambient light intensity in range [0, 1]
    """

    _direction: np.ndarray  # 3D vector
    _ambient: float = 0.1

    @property
    def direction(self) -> np.ndarray:
        """3D vector specifying light direction

        Returns:
            np.ndarray: Light direction vector
        """
        return self._direction

    @direction.setter
    def direction(self, value: np.ndarray) -> None:
        """Set the light's direction vector

        Args:
            value (np.ndarray): New light direction vector
        """
        self._direction = np.array(value, dtype=np.float32)

    @property
    def ambient(self) -> float:
        """Ambient light intensity in range [0, 1]

        Returns:
            float: Ambient light intensity
        """
        return self._ambient

    @ambient.setter
    def ambient(self, value: float) -> None:
        """Set the light's ambient intensity

        Args:
            value (float): New ambient light intensity
        """
        self._ambient = value

    @classmethod
    def from_dict(cls, data: dict) -> "DirectionalLight":
        """Create a DirectionalLight from a dictionary representation.

        Args:
            data (dict): Dictionary containing light direction and ambient intensity

        Returns:
            DirectionalLight: New light instance with the specified properties
        """
        return cls(
            _direction=np.array(data["direction"], dtype=np.float32),
            _ambient=data.get("ambient", 0.1),
        )

    def to_dict(self) -> dict:
        """Convert the DirectionalLight to a dictionary representation.

        Returns:
            dict: Dictionary containing the light's direction and ambient intensity
        """
        return {"direction": self.direction.tolist(), "ambient": self.ambient}


@dataclass(slots=True)
class Scene:
    """A 3D scene containing models, instances, camera settings, and lights.

    The Scene class is the top-level container for all elements in a 3D scene.
    It manages:
    - A collection of reusable 3D models
    - Multiple instances of those models with unique transforms
    - Camera position and orientation
    - Lighting configuration

    The scene supports serialization to/from dictionary format for easy saving
    and loading of scene configurations.

    Attributes:
        models (Dict[str, Model]): Named collection of 3D models
        instances (Dict[str, Instance]): Named collection of model instances
        camera (Camera): Scene camera configuration
        directional_light (DirectionalLight): Main directional light source
    """

    _models: Dict[str, Model] = field(default_factory=dict)
    _instances: Dict[str, Instance] = field(default_factory=dict)
    _camera: Camera = field(default_factory=Camera)
    _directional_light: DirectionalLight = field(
        default_factory=lambda: DirectionalLight(
            _direction=np.array([0, 0, -1], dtype=np.float32)
        )
    )

    def add_model(self, name: str, model: Model) -> None:
        """Add a model to the scene.

        Args:
            name (str): Unique identifier for the model
            model (Model): Model instance to add
        """
        self._models[name] = model

    def get_model(self, model_name: str) -> Optional[Model]:
        """Get a model by name.

        Args:
            model_name (str): Name of the model to get

        Returns:
            Model: Model instance if found, None otherwise
        """
        return self._models.get(model_name)

    def add_instance(self, name: str, instance: Instance) -> None:
        """Add a model instance to the scene.

        Args:
            name (str): Unique identifier for the instance
            instance (Instance): Instance to add
        """
        self._instances[name] = instance

    def get_instance(self, instance_name: str) -> Optional[Instance]:
        """Get an instance by name.

        Args:
            instance_name (str): Name of the instance to get

        Returns:
            Instance: Instance if found, None otherwise
        """
        return self._instances.get(instance_name)

    def set_camera(self, camera: Camera) -> None:
        """Set the scene's camera.

        Args:
            camera (Camera): Camera instance to set
        """
        self._camera = camera

    def set_directional_light(self, light: DirectionalLight) -> None:
        """Set the scene's directional light.

        Args:
            light (DirectionalLight): Light instance to set
        """
        self._directional_light = light

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        """Create a Scene from a dictionary representation.

        Args:
            data (dict): Dictionary containing scene data

        Returns:
            Scene: New scene instance with the specified elements
        """
        scene = cls()

        # Load camera
        if "camera" in data:
            scene._camera = Camera.from_dict(data["camera"])

        # Load models
        if "models" in data:
            for name, model_data in data["models"].items():
                scene._models[name] = Model.from_dict(model_data)

        # Load instances
        if "instances" in data:
            for instance_data in data["instances"]:
                name = instance_data.get("name", f"instance_{len(scene._instances)}")
                scene._instances[name] = Instance.from_dict(instance_data)

        # Load directional light
        if "lights" in data and "directional" in data["lights"]:
            scene._directional_light = DirectionalLight.from_dict(
                data["lights"]["directional"]
            )

        return scene

    def to_dict(self) -> dict:
        """Convert the Scene to a dictionary representation.

        Returns:
            dict: Dictionary containing scene data
        """
        return {
            "camera": self._camera.to_dict(),
            "models": {name: model.to_dict() for name, model in self._models.items()},
            "instances": [instance.to_dict() for instance in self._instances.values()],
            "directional_light": self._directional_light.to_dict(),
        }
