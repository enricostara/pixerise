"""
Scene management module for the Pixerise rendering engine.
This module contains the Scene class which manages 3D models and their instances.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass(slots=True)
class ModelInnerGroup:
    """A group within a model containing geometry data."""
    vertices: np.ndarray
    triangles: np.ndarray
    vertex_normals: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelInnerGroup':
        """Create a ModelGroup from a dictionary representation."""
        return cls(
            vertices=np.array(data.get('vertices', []), dtype=np.float32),
            triangles=np.array(data.get('triangles', []), dtype=np.int32),
            vertex_normals=np.array(data.get('vertex_normals', []), dtype=np.float32) if 'vertex_normals' in data else None
        )


@dataclass(slots=True)
class Model:
    """A 3D model containing one or more groups of geometry."""
    groups: Dict[str, ModelInnerGroup] = field(default_factory=dict)

    def add_group(self, name: str, vertices: List[List[float]], triangles: List[List[int]], 
                 vertex_normals: Optional[List[List[float]]] = None) -> None:
        """Add a new group to the model."""
        self.groups[name] = ModelInnerGroup(
            vertices=np.array(vertices, dtype=np.float32),
            triangles=np.array(triangles, dtype=np.int32),
            vertex_normals=np.array(vertex_normals, dtype=np.float32) if vertex_normals else None
        )

    @classmethod
    def from_dict(cls, data: dict) -> 'Model':
        """Create a Model from a dictionary representation."""
        model = cls()
        groups = data.get('groups')
        
        # If no groups are defined, create a default group with model data
        if not groups:
            groups = {
                'default': {
                    'vertices': data.get('vertices', []),
                    'triangles': data.get('triangles', []),
                    'vertex_normals': data.get('vertex_normals', [])
                }
            }
        
        for name, group_data in groups.items():
            model.groups[name] = ModelInnerGroup.from_dict(group_data)
        
        return model


@dataclass(slots=True)
class Instance:
    """An instance of a model with transformation and color properties."""
    model_name: str
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))
    color: Tuple[int, int, int] = (255, 255, 255)

    def set_translation(self, x: float, y: float, z: float) -> None:
        """Set the translation of this instance."""
        self.translation = np.array([x, y, z], dtype=np.float32)

    def set_rotation(self, x: float, y: float, z: float) -> None:
        """Set the rotation of this instance in radians."""
        self.rotation = np.array([x, y, z], dtype=np.float32)

    def set_scale(self, x: float, y: float, z: float) -> None:
        """Set the scale of this instance."""
        self.scale = np.array([x, y, z], dtype=np.float32)

    @classmethod
    def from_dict(cls, data: dict) -> 'Instance':
        """Create an Instance from a dictionary representation."""
        instance = cls(model_name=data['model'])
        if 'translation' in data:
            instance.translation = np.array(data['translation'], dtype=np.float32)
        if 'rotation' in data:
            instance.rotation = np.array(data['rotation'], dtype=np.float32)
        if 'scale' in data:
            instance.scale = np.array(data['scale'], dtype=np.float32)
        if 'color' in data:
            instance.color = tuple(data['color'])
        return instance


@dataclass(slots=True)
class Camera:
    """Camera settings for the scene."""
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def set_translation(self, x: float, y: float, z: float) -> None:
        """Set the camera's translation."""
        self.translation = np.array([x, y, z], dtype=np.float32)

    def set_rotation(self, x: float, y: float, z: float) -> None:
        """Set the camera's rotation in radians."""
        self.rotation = np.array([x, y, z], dtype=np.float32)

    @classmethod
    def from_dict(cls, data: dict) -> 'Camera':
        """Create a Camera from a dictionary representation."""
        camera = cls()
        if 'translation' in data:
            camera.translation = np.array(data['translation'], dtype=np.float32)
        if 'rotation' in data:
            camera.rotation = np.array(data['rotation'], dtype=np.float32)
        return camera


@dataclass(slots=True)
class DirectionalLight:
    """A directional light in the scene."""
    direction: np.ndarray  # 3D vector
    ambient: float = 0.1

    @classmethod
    def from_dict(cls, data: dict) -> 'DirectionalLight':
        """Create a DirectionalLight from a dictionary representation."""
        return cls(
            direction=np.array(data['direction'], dtype=np.float32),
            ambient=data.get('ambient', 0.1)
        )

    def to_dict(self) -> dict:
        """Convert the DirectionalLight to a dictionary representation."""
        return {
            'direction': self.direction.tolist(),
            'ambient': self.ambient
        }


@dataclass(slots=True)
class Scene:
    """A 3D scene containing models, instances, camera settings, and lights."""
    models: Dict[str, Model] = field(default_factory=dict)
    instances: List[Instance] = field(default_factory=list)
    camera: Camera = field(default_factory=Camera)
    directional_light: DirectionalLight = field(default_factory=lambda: DirectionalLight(direction=np.array([0, 0, -1], dtype=np.float32)))

    def add_model(self, name: str, model: Model) -> None:
        """Add a model to the scene."""
        self.models[name] = model

    def get_model(self, model_name: str) -> Optional[Model]:
        """Get a model by name.
        
        Args:
            model_name: Name of the model to get
            
        Returns:
            Model if found, None otherwise
        """
        return self.models.get(model_name)

    def add_instance(self, instance: Instance) -> None:
        """Add a model instance to the scene."""
        self.instances.append(instance)

    def set_camera(self, camera: Camera) -> None:
        """Set the scene's camera."""
        self.camera = camera

    def set_directional_light(self, light: DirectionalLight) -> None:
        """Set the scene's directional light."""
        self.directional_light = light

    @classmethod
    def from_dict(cls, data: dict) -> 'Scene':
        """Create a Scene from a dictionary representation."""
        scene = cls()
        
        # Add models
        for model_name, model_data in data.get('models', {}).items():
            model = Model.from_dict(model_data)
            scene.add_model(model_name, model)
        
        # Add instances
        for instance_data in data.get('instances', []):
            scene.add_instance(Instance.from_dict(instance_data))
        
        # Set camera if present
        if 'camera' in data:
            scene.set_camera(Camera.from_dict(data['camera']))

        # Set directional light if present
        if 'lights' in data and 'directional' in data['lights']:
            scene.set_directional_light(DirectionalLight.from_dict(data['lights']['directional']))
        
        return scene

    def to_dict(self) -> dict:
        """Convert the Scene to a dictionary representation."""
        return {
            'models': {name: model.to_dict() for name, model in self.models.items()},
            'instances': [instance.to_dict() for instance in self.instances],
            'camera': self.camera.to_dict(),
            'lights': {
                'directional': self.directional_light.to_dict()
            }
        }
