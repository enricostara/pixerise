"""
Test suite for the Scene management module.
Tests Scene class and related classes (Model, Instance, Camera, DirectionalLight).
"""

import pytest
import numpy as np
from scene import Scene, Model, Instance, Camera, DirectionalLight, ModelInnerGroup


def test_model_inner_group():
    """Test ModelInnerGroup creation and dictionary conversion."""
    # Test creation with vertices and triangles
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    group = ModelInnerGroup(vertices=vertices, triangles=triangles)
    
    assert np.array_equal(group.vertices, vertices)
    assert np.array_equal(group.triangles, triangles)
    assert group.vertex_normals is None
    
    # Test creation with vertex normals
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
    group_with_normals = ModelInnerGroup(vertices=vertices, triangles=triangles, vertex_normals=normals)
    assert np.array_equal(group_with_normals.vertex_normals, normals)
    
    # Test from_dict
    data = {
        'vertices': vertices.tolist(),
        'triangles': triangles.tolist(),
        'vertex_normals': normals.tolist()
    }
    group_from_dict = ModelInnerGroup.from_dict(data)
    assert np.array_equal(group_from_dict.vertices, vertices)
    assert np.array_equal(group_from_dict.triangles, triangles)
    assert np.array_equal(group_from_dict.vertex_normals, normals)


def test_model():
    """Test Model creation and manipulation."""
    model = Model()
    
    # Test adding a group
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    model.add_group('test_group', vertices, triangles)
    
    assert 'test_group' in model.groups
    assert np.array_equal(model.groups['test_group'].vertices, vertices)
    assert np.array_equal(model.groups['test_group'].triangles, triangles)
    
    # Test to_dict with single default group
    model = Model()
    model.add_group('default', vertices, triangles)
    data = model.to_dict()
    assert 'vertices' in data  # Flat structure for default group
    assert 'triangles' in data
    
    # Test to_dict with multiple groups
    model = Model()
    model.add_group('group1', vertices, triangles)
    model.add_group('group2', vertices, triangles)
    data = model.to_dict()
    assert 'groups' in data  # Grouped structure for multiple groups
    assert 'group1' in data['groups']
    assert 'group2' in data['groups']
    
    # Test from_dict with flat structure
    flat_data = {
        'vertices': vertices.tolist(),
        'triangles': triangles.tolist()
    }
    model = Model.from_dict(flat_data)
    assert 'default' in model.groups
    
    # Test from_dict with groups structure
    grouped_data = {
        'groups': {
            'group1': {'vertices': vertices.tolist(), 'triangles': triangles.tolist()},
            'group2': {'vertices': vertices.tolist(), 'triangles': triangles.tolist()}
        }
    }
    model = Model.from_dict(grouped_data)
    assert 'group1' in model.groups
    assert 'group2' in model.groups


def test_instance():
    """Test Instance creation and manipulation."""
    # Test default creation
    instance = Instance(model='test_model')
    assert instance.model == 'test_model'
    assert np.array_equal(instance.translation, np.zeros(3))
    assert np.array_equal(instance.rotation, np.zeros(3))
    assert np.array_equal(instance.scale, np.ones(3))
    assert np.array_equal(instance.color, np.array([200, 200, 200], dtype=np.int32))  # Default gray color
    
    # Test setters
    instance.set_translation(1, 2, 3)
    instance.set_rotation(np.pi/2, 0, np.pi/4)
    instance.set_scale(2, 2, 2)
    instance.color = [128, 179, 255]  # Light blue color as RGB integers
    
    assert np.array_equal(instance.translation, np.array([1, 2, 3], dtype=np.float32))
    assert np.array_equal(instance.rotation, np.array([np.pi/2, 0, np.pi/4], dtype=np.float32))
    assert np.array_equal(instance.scale, np.array([2, 2, 2], dtype=np.float32))
    assert np.array_equal(instance.color, np.array([128, 179, 255], dtype=np.int32))
    
    # Test to_dict and from_dict
    data = instance.to_dict()
    assert data['model'] == 'test_model'
    assert data['transform']['translation'] == [1, 2, 3]
    assert np.allclose(data['transform']['rotation'], [np.pi/2, 0, np.pi/4])
    assert data['transform']['scale'] == [2, 2, 2]
    assert data['color'] == [128, 179, 255]  # Color as RGB integers
    
    new_instance = Instance.from_dict(data)
    assert new_instance.model == instance.model
    assert np.array_equal(new_instance.translation, instance.translation)
    assert np.array_equal(new_instance.rotation, instance.rotation)
    assert np.array_equal(new_instance.scale, instance.scale)
    assert np.array_equal(new_instance.color, instance.color)


def test_camera():
    """Test Camera creation and manipulation."""
    # Test default creation
    camera = Camera()
    assert np.array_equal(camera.translation, np.zeros(3))
    assert np.array_equal(camera.rotation, np.zeros(3))
    
    # Test setters
    camera.set_translation(1, 2, 3)
    camera.set_rotation(np.pi/2, 0, np.pi/4)
    
    assert np.array_equal(camera.translation, np.array([1, 2, 3], dtype=np.float32))
    assert np.array_equal(camera.rotation, np.array([np.pi/2, 0, np.pi/4], dtype=np.float32))
    
    # Test to_dict and from_dict
    data = camera.to_dict()
    assert data['transform']['translation'] == [1, 2, 3]
    assert np.allclose(data['transform']['rotation'], [np.pi/2, 0, np.pi/4])
    
    new_camera = Camera.from_dict(data)
    assert np.array_equal(new_camera.translation, camera.translation)
    assert np.array_equal(new_camera.rotation, camera.rotation)


def test_directional_light():
    """Test DirectionalLight creation and manipulation."""
    # Test creation
    direction = np.array([-1, -1, -1], dtype=np.float32)
    light = DirectionalLight(direction=direction, ambient=0.2)
    
    assert np.array_equal(light.direction, direction)
    assert light.ambient == 0.2
    
    # Test to_dict and from_dict
    data = light.to_dict()
    assert data['direction'] == direction.tolist()
    assert data['ambient'] == 0.2
    
    new_light = DirectionalLight.from_dict(data)
    assert np.array_equal(new_light.direction, light.direction)
    assert new_light.ambient == light.ambient


def test_scene():
    """Test Scene creation and manipulation."""
    scene = Scene()
    
    # Test default state
    assert len(scene.models) == 0
    assert len(scene.instances) == 0
    assert isinstance(scene.camera, Camera)
    assert isinstance(scene.directional_light, DirectionalLight)
    
    # Test adding and getting models
    model = Model()
    model.add_group('default', np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32), np.array([[0, 1, 2]], dtype=np.int32))
    scene.add_model('test_model', model)
    
    assert scene.get_model('test_model') == model
    assert scene.get_model('nonexistent') is None
    
    # Test adding and getting instances
    instance = Instance(model='test_model')
    scene.add_instance('test_instance', instance)
    
    assert scene.get_instance('test_instance') == instance
    assert scene.get_instance('nonexistent') is None
    
    # Test setting camera and light
    camera = Camera()
    camera.set_translation(1, 2, 3)
    scene.set_camera(camera)
    
    light = DirectionalLight(direction=np.array([-1, -1, -1], dtype=np.float32), ambient=0.2)
    scene.set_directional_light(light)
    
    assert scene.camera == camera
    assert scene.directional_light == light
    
    # Test to_dict and from_dict
    scene_data = {
        'camera': {
            'transform': {
                'translation': [1, 2, 3],
                'rotation': [0, 0, 0]
            }
        },
        'models': {
            'cube': {
                'vertices': [[-1, -1, -1], [1, -1, -1], [1, 1, -1]],
                'triangles': [[0, 1, 2]]
            }
        },
        'instances': [
            {
                'name': 'cube_instance',
                'model': 'cube',
                'transform': {
                    'translation': [0, 0, 5],
                    'rotation': [0, 0, 0],
                    'scale': [1, 1, 1]
                },
                'color': [255, 0, 0]  # Color as RGB integers
            }
        ],
        'lights': {
            'directional': {
                'direction': [-1, -1, -1],
                'ambient': 0.2
            }
        }
    }
    
    scene = Scene.from_dict(scene_data)
    assert 'cube' in scene.models
    assert 'cube_instance' in scene.instances
    assert np.array_equal(scene.camera.translation, np.array([1, 2, 3], dtype=np.float32))
    assert np.array_equal(scene.directional_light.direction, np.array([-1, -1, -1], dtype=np.float32))
    assert scene.directional_light.ambient == 0.2
    
    # Test round-trip serialization
    new_data = scene.to_dict()
    assert 'camera' in new_data
    assert 'models' in new_data
    assert 'instances' in new_data
    assert 'directional_light' in new_data
