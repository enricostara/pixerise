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

    def to_dict(self) -> dict:
        """Convert the Model to a dictionary representation."""
        if len(self.groups) == 1 and 'default' in self.groups:
            # If there's only a default group, use flat structure
            group = self.groups['default']
            return {
                'vertices': group.vertices.tolist(),
                'triangles': group.triangles.tolist(),
                'vertex_normals': group.vertex_normals.tolist() if group.vertex_normals is not None else None
            }
        else:
            # Otherwise, use groups structure
            return {
                'groups': {
                    name: {
                        'vertices': group.vertices.tolist(),
                        'triangles': group.triangles.tolist(),
                        'vertex_normals': group.vertex_normals.tolist() if group.vertex_normals is not None else None
                    }
                    for name, group in self.groups.items()
                }
            }


@dataclass(slots=True)
class Instance:
    """An instance of a model with transformation and color properties."""
    model: str
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))
    color: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float32))

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
        instance = cls(model=data['model'])
        if 'transform' in data:
            transform = data['transform']
            if 'translation' in transform:
                instance.translation = np.array(transform['translation'], dtype=np.float32)
            if 'rotation' in transform:
                instance.rotation = np.array(transform['rotation'], dtype=np.float32)
            if 'scale' in transform:
                instance.scale = np.array(transform['scale'], dtype=np.float32)
        if 'color' in data:
            instance.color = np.array(data['color'], dtype=np.float32)
        return instance

    def to_dict(self) -> dict:
        """Convert the Instance to a dictionary representation."""
        return {
            'model': self.model,
            'transform': {
                'translation': self.translation.tolist(),
                'rotation': self.rotation.tolist(),
                'scale': self.scale.tolist()
            },
            'color': self.color.tolist()
        }


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
        transform = data.get('transform', {})
        return cls(
            translation=np.array(transform.get('translation', [0, 0, 0]), dtype=np.float32),
            rotation=np.array(transform.get('rotation', [0, 0, 0]), dtype=np.float32)
        )

    def to_dict(self) -> dict:
        """Convert the Camera to a dictionary representation."""
        return {
            'transform': {
                'translation': self.translation.tolist(),
                'rotation': self.rotation.tolist()
            }
        }


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
    instances: Dict[str, Instance] = field(default_factory=dict)
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

    def add_instance(self, name: str, instance: Instance) -> None:
        """Add a model instance to the scene.
        
        Args:
            name: Unique name for the instance
            instance: Instance to add
        """
        self.instances[name] = instance

    def get_instance(self, instance_name: str) -> Optional[Instance]:
        """Get an instance by name.
        
        Args:
            instance_name: Name of the instance to get
            
        Returns:
            Instance if found, None otherwise
        """
        return self.instances.get(instance_name)

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
        
        # Load camera
        if 'camera' in data:
            scene.camera = Camera.from_dict(data['camera'])
        
        # Load models
        if 'models' in data:
            for name, model_data in data['models'].items():
                scene.models[name] = Model.from_dict(model_data)
        
        # Load instances
        if 'instances' in data:
            for instance_data in data['instances']:
                name = instance_data.get('name', f'instance_{len(scene.instances)}')
                scene.instances[name] = Instance.from_dict(instance_data)
        
        # Load directional light
        if 'lights' in data and 'directional' in data['lights']:
            scene.directional_light = DirectionalLight.from_dict(data['lights']['directional'])
        
        return scene

    def to_dict(self) -> dict:
        """Convert the Scene to a dictionary representation."""
        return {
            'camera': self.camera.to_dict(),
            'models': {name: model.to_dict() for name, model in self.models.items()},
            'instances': [instance.to_dict() for instance in self.instances.values()],
            'directional_light': self.directional_light.to_dict()
        }
